"""통합 실행 진입점.
    python run.py --dataset {docx,pdf,srt} --task {retrieval,summary,translate}
                   --chunker {fixed,semantic,smart,all} [--limit N] [--fake-llm]
결과는 results/<dataset>_<task>_<chunker>.json(retrieval/summary) 또는
results/translate_pairs/<dataset>_<chunker>_<문서명>.json(translate)에 저장 - 이미
있으면 그 조합/문서는 건너뛴다(이어하기). srt는 기존 subtitle_pipeline을 그대로 호출."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from config import ALLGANIZE_DIR, RESULTS_DIR, VECTARA_DIR

CHUNKERS = ["fixed", "semantic", "smart"]


def setup_models(fake_llm, need_llm):
    from langchain_huggingface import HuggingFaceEmbeddings
    from transformers import AutoTokenizer

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings": True})
    bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    count_tokens = lambda s: len(bge_tokenizer.encode(s, add_special_tokens=False))

    tokenizer = model = device = None
    if need_llm and not fake_llm:
        import torch
        from config import TOKENIZER
        from llm import load_llm
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer, model, device = load_llm(TOKENIZER, device)
    return embeddings, count_tokens, tokenizer, model, device


def run_retrieval(dataset, chunker, embeddings, count_tokens, limit):
    import retrieval
    out_path = RESULTS_DIR / f"{dataset}_retrieval_{chunker}.json"
    if out_path.exists():
        print(f"  {out_path.name} 이미 있음, 건너뜀(검색은 재사용 위해 다시 계산은 함 - 요약이 필요할 수 있어서)")
    result = retrieval.run(dataset, chunker, embeddings, count_tokens, limit=limit)
    save = {"overall": result["overall"], "table": result["table"],
            "n_chunks": result["n_chunks"], "n_docs": result["n_docs"]}
    out_path.write_text(json.dumps(save, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"검색: {save['overall']}", flush=True)
    return result


def run_summary(dataset, chunker, embeddings, count_tokens, tokenizer, model, device, limit):
    import summary
    out_path = RESULTS_DIR / f"{dataset}_summary_{chunker}.json"
    retrieval_result = run_retrieval(dataset, chunker, embeddings, count_tokens, limit)
    result = summary.run(retrieval_result["retriever"], retrieval_result["qas"], tokenizer, model, device)
    save = {"coverage": result["coverage"], "n": result["n"], "examples": result["rows"][:5]}
    out_path.write_text(json.dumps(save, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"요약 커버리지: {result['coverage']}", flush=True)
    return result


def _files_for(dataset):
    if dataset == "docx":
        return sorted(ALLGANIZE_DIR.glob("docx/**/*.docx"))
    return sorted(VECTARA_DIR.glob("*.pdf"))


def _splitter_for(chunker, dataset, embeddings, count_tokens):
    from langchain_text_splitters import CharacterTextSplitter
    from smart_chunker import SemanticMaxSplitter, SmartChunker, SmartTextSplitter
    if chunker == "fixed":
        return CharacterTextSplitter(separator="", chunk_size=500, chunk_overlap=0, add_start_index=True)
    if chunker == "semantic":
        return SemanticMaxSplitter(embeddings, count_tokens)
    lang = "ko" if dataset == "docx" else "en"
    return SmartTextSplitter(SmartChunker(embeddings._client, count_tokens, mode=dataset, lang=lang))


def run_translate(dataset, chunker, embeddings, count_tokens, tokenizer, model, device, fake_llm, limit):
    import translate
    from docx_track.loader import load_docx
    from pdf_track.loader import load_pdf
    from llm import translate_batch

    load_fn = load_docx if dataset == "docx" else load_pdf
    files = _files_for(dataset)[:limit] if limit else _files_for(dataset)
    batch_fn = translate.fake_translate_batch if fake_llm else \
        (lambda prompts: translate_batch(tokenizer, model, device, prompts))

    pairs_dir = RESULTS_DIR / "translate_pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    all_stats = []
    for i, f in enumerate(files, 1):
        pairs_path = pairs_dir / f"{dataset}_{chunker}_{f.stem}.json"
        if pairs_path.exists():
            print(f"[{i}/{len(files)}] {f.name}: 이미 완료, 건너뜀", flush=True)
            continue
        flat_text = load_fn(str(f)).page_content
        splitter = _splitter_for(chunker, dataset, embeddings, count_tokens)
        docs = splitter.create_documents([flat_text])
        chunk_spans = [(d.metadata["start_index"], d.metadata["start_index"] + len(d.page_content)) for d in docs]

        t0 = time.monotonic()
        pairs, stats = translate.translate_document(flat_text, dataset, chunk_spans, batch_fn)
        stats["time"], stats["file"] = time.monotonic() - t0, f.name
        pairs_path.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")
        all_stats.append(stats)
        print(f"[{i}/{len(files)}] {f.name}: {stats}", flush=True)
    return all_stats


def run_srt(fake_llm, limit):
    """자막은 기존 subtitle_pipeline.main()을 그대로 호출(재구현 안 함)."""
    from srt.subtitle_pipeline import main as srt_main
    if fake_llm:
        print("srt 트랙은 --fake-llm 없이 기존 파이프라인을 그대로 돕니다(자체 채점 완성본).")
    srt_main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["docx", "pdf", "srt"])
    ap.add_argument("--task", required=True, choices=["retrieval", "summary", "translate"])
    ap.add_argument("--chunker", required=True, choices=[*CHUNKERS, "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fake-llm", action="store_true")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.dataset == "srt":
        run_srt(args.fake_llm, args.limit)
        return

    need_llm = args.task in ("summary", "translate")
    print(f"모델 준비 중(fake_llm={args.fake_llm})...", flush=True)
    embeddings, count_tokens, tokenizer, model, device = setup_models(args.fake_llm, need_llm)

    chunkers = CHUNKERS if args.chunker == "all" else [args.chunker]
    for chunker in chunkers:
        print(f"\n=== {args.dataset} / {args.task} / {chunker} ===", flush=True)
        if args.task == "retrieval":
            run_retrieval(args.dataset, chunker, embeddings, count_tokens, args.limit)
        elif args.task == "summary":
            run_summary(args.dataset, chunker, embeddings, count_tokens, tokenizer, model, device, args.limit)
        else:
            run_translate(args.dataset, chunker, embeddings, count_tokens, tokenizer, model, device,
                           args.fake_llm, args.limit)


if __name__ == "__main__":
    main()
