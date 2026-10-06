"""평가 실행 (docs/restructure_plan.md 6.3·10절). 지금은 --task retrieval 만 있다.

  source env.sh
  .venv311/bin/python scripts/run.py --task retrieval --dataset vectara --tag all [--split eval] [--k 5] [--force]

- --tag: data/chunks/<dataset>/ 의 청크 파일 이름(확장자 제외), 쉼표로 여러 개, 또는 all.
- 질문은 청크 파일에 들어 있는 문서의 것만 쓴다(표본 청킹이면 그 문서들의 질문만; 코퍼스가 작아 전체 평가보다 쉬움).
- 결과: results/<dataset>/retrieval/<태그>[@split].json (이미 있으면 건너뜀, --force로 덮어쓰기).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.config import CHUNKS_DIR, DATASETS, RESULTS_DIR  # noqa: E402

MAX_SEQ = 8192   # bge-m3 최대 입력 토큰


def run_retrieval(dataset, tags, split, k, ngram, force):
    from indexing.index import get_index, load_chunks, make_encoder
    from qa_build.build_qa import load_qa
    from common.models import get_embeddings
    from tasks.retrieval import PRIMARY, evaluate

    encode = make_encoder(get_embeddings()._client)
    out_dir = RESULTS_DIR / dataset / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for tag in tags:
        out = out_dir / f"{tag}{'@' + split if split else ''}.json"
        if out.exists() and not force:
            print(f"[{dataset}/{tag}] 결과가 이미 있음, 건너뜀 (--force로 다시)")
            continue
        docs = sorted({c["doc_id"] for c in load_chunks(dataset, tag)})
        qas = load_qa(dataset, split=split, doc_ids=docs)
        idx = get_index(dataset, tag, encode, doc_ids=docs, force=force, max_seq_length=MAX_SEQ)
        res = evaluate(idx, qas, k=k, ngram=ngram)
        res.update(dataset=dataset, tag=tag, split=split, n_docs=len(docs), n_questions=len(qas), ngram=ngram,
                   chunks_over_max_seq_length=getattr(idx, "truncated", None))
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        o = res["overall"]
        prim = PRIMARY.get(dataset)
        print(f"[{dataset}/{tag}] 문서 {len(docs)} 질문 {len(qas)} 청크 {res['index']['n_chunks']} "
              f"(평균 {res['index']['chunk_tokens_mean']:.0f}토큰, 최대 {res['index']['chunk_tokens_max']}, "
              f"{MAX_SEQ} 초과 {res['chunks_over_max_seq_length']}) | hit@{k} {o['hit@k']:.3f} MRR {o['mrr']:.3f} F1 {o['f1']:.3f}"
              + (f" {prim}@{k} {o[prim + '@k']:.3f}" if prim and prim + "@k" in o else "") + f" | top-{k} 평균 {o['top_k_tokens_mean']:.0f}토큰")
        summary.append(res)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["retrieval"])
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", choices=["tune", "eval"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ngram", type=int, default=1, help="section_hit 의 포함 비율 단위 (계획서 정의 = 1)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    tags = sorted(p.stem for p in (CHUNKS_DIR / args.dataset).glob("*.jsonl")) if args.tag == "all" else args.tag.split(",")
    run_retrieval(args.dataset, tags, args.split, args.k, args.ngram, args.force)


if __name__ == "__main__":
    main()
