"""청크 파일 통계 (docs/restructure_plan.md 5.5절).

  .venv311/bin/python scripts/chunk_stats.py --dataset vectara [--tag smart__p90-min100-max400] [--min-tokens N --max-tokens N]

--tag를 안 주면 data/chunks/<dataset>/ 의 모든 청크 파일을 처리한다.
태그는 청크 파일 이름(확장자 제외)이다. 예: fixed__c300, semantic__p90-min100-max400, smart__p90-min100-max400
(srt의 legacy smart는 smart__min128-max512-w2 형식).
--min-tokens/--max-tokens는 under_min_rate/over_max_rate의 기준이다. 안 주면 청크 파일 안 params의 값을 쓰고,
없으면(fixed) 그 두 지표는 빼므로, 청커를 같은 기준으로 비교하려면 같은 값을 명시할 것.
결과: results/<dataset>/chunk_stats/<tag>.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunking.stats import compute_stats  # noqa: E402
from common.config import CHUNKS_DIR, DATASETS, PROCESSED_DIR, RESULTS_DIR  # noqa: E402
from common.docjson import load_doc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--tag")
    ap.add_argument("--min-tokens", type=int)
    ap.add_argument("--max-tokens", type=int)
    args = ap.parse_args()

    files = [CHUNKS_DIR / args.dataset / f"{args.tag}.jsonl"] if args.tag else sorted((CHUNKS_DIR / args.dataset).glob("*.jsonl"))
    if not files or not all(f.exists() for f in files):
        sys.exit(f"청크 파일이 없다: {CHUNKS_DIR / args.dataset} (먼저 scripts/chunk.py)")
    out_dir = RESULTS_DIR / args.dataset / "chunk_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        chunks = [json.loads(line) for line in f.read_text(encoding="utf-8").split("\n") if line.strip()]   # splitlines()는 U+0085 등에서도 나눈다
        docs = {i: load_doc(PROCESSED_DIR / args.dataset / f"{i}.json") for i in {c["doc_id"] for c in chunks}}
        p = chunks[0]["params"] if chunks else {}
        st = compute_stats(chunks, docs, min_tokens=args.min_tokens or p.get("min_tokens"), max_tokens=args.max_tokens or p.get("max_tokens"))
        st.update(dataset=args.dataset, tag=f.stem, n_docs=len(docs))
        (out_dir / f"{f.stem}.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{f.stem}]", {k: v for k, v in st.items() if k not in ("dataset", "tag")})


if __name__ == "__main__":
    main()
