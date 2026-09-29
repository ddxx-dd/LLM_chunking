"""docx 트랙 - Allganize QA(45개 문서, 211개 질문) + 세 청커로 검색/요약/번역(한→영).

CHUNKERS: 이 데이터셋에 쓸 청커 설정값(튜닝 시작값, min 100 / max 400 토큰)."""
import json
import time

import chunkers
import retrieval
import summary
import translate
from config import ALLGANIZE_DIR, RESULTS_DIR
from docx_track.loader import load_docx
from llm import translate_batch

DATASET = "docx"
DIRECTION = "ko2en"  # 한국어 원문 -> 영어 번역


def build_chunkers(embeddings, count_tokens):
    return {
        "fixed":    chunkers.make_fixed(chunk_size=500),
        "semantic": chunkers.make_semantic(embeddings, count_tokens, percentile=90,
                                            min_tokens=100, max_tokens=400),
        "smart":    chunkers.make_smart(embeddings._client, count_tokens, mode="docx", lang="ko",
                                         min_tokens=100, max_tokens=400),
    }


def _files():
    return sorted(ALLGANIZE_DIR.glob("docx/**/*.docx"))


def run_retrieval(chunker, embeddings, count_tokens, limit):
    out_path = RESULTS_DIR / f"{DATASET}_retrieval_{chunker}.json"
    splitter = build_chunkers(embeddings, count_tokens)[chunker]
    result = retrieval.run(DATASET, chunker, splitter, embeddings, limit=limit)
    save = {"overall": result["overall"], "table": result["table"],
            "n_chunks": result["n_chunks"], "n_docs": result["n_docs"]}
    out_path.write_text(json.dumps(save, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"검색: {save['overall']}", flush=True)
    return result


def run_summary(chunker, embeddings, count_tokens, tokenizer, model, device, limit):
    out_path = RESULTS_DIR / f"{DATASET}_summary_{chunker}.json"
    retrieval_result = run_retrieval(chunker, embeddings, count_tokens, limit)
    result = summary.run(retrieval_result["retriever"], retrieval_result["qas"], tokenizer, model, device)
    save = {"coverage": result["coverage"], "n": result["n"], "examples": result["rows"][:5]}
    out_path.write_text(json.dumps(save, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"요약 커버리지: {result['coverage']}", flush=True)
    return result


def run_translate(chunker, embeddings, count_tokens, tokenizer, model, device, fake_llm, limit):
    splitter = build_chunkers(embeddings, count_tokens)[chunker]
    batch_fn = translate.fake_translate_batch if fake_llm else \
        (lambda prompts: translate_batch(tokenizer, model, device, prompts))

    pairs_dir = RESULTS_DIR / "translate_pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    files = _files()[:limit] if limit else _files()
    all_stats = []
    for i, f in enumerate(files, 1):
        pairs_path = pairs_dir / f"{DATASET}_{chunker}_{f.stem}.json"
        if pairs_path.exists():
            print(f"[{i}/{len(files)}] {f.name}: 이미 완료, 건너뜀", flush=True)
            continue
        flat_text = load_docx(str(f)).page_content
        docs = splitter.create_documents([flat_text])
        chunk_spans = [(d.metadata["start_index"], d.metadata["start_index"] + len(d.page_content)) for d in docs]

        t0 = time.monotonic()
        pairs, stats = translate.translate_document(flat_text, DATASET, chunk_spans, batch_fn)
        stats["time"], stats["file"] = time.monotonic() - t0, f.name
        pairs_path.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")
        all_stats.append(stats)
        print(f"[{i}/{len(files)}] {f.name}: {stats}", flush=True)
    return all_stats


def run(task, chunker, limit=None, fake_llm=False):
    """run.py가 부르는 진입점 - task: retrieval/summary/translate. chunker="all"이면
    모델을 한 번만 불러온 뒤 fixed/semantic/smart를 순서대로 전부 돈다."""
    from llm import setup_embeddings_and_llm
    embeddings, count_tokens, tokenizer, model, device = \
        setup_embeddings_and_llm(fake_llm, need_llm=task in ("summary", "translate"))

    chunker_names = ["fixed", "semantic", "smart"] if chunker == "all" else [chunker]

    results = {}
    for name in chunker_names:
        if task == "retrieval":
            results[name] = run_retrieval(name, embeddings, count_tokens, limit)
        elif task == "summary":
            results[name] = run_summary(name, embeddings, count_tokens, tokenizer, model, device, limit)
        else:
            results[name] = run_translate(name, embeddings, count_tokens, tokenizer, model, device, fake_llm, limit)
    return results
