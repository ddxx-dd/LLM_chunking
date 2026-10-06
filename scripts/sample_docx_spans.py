"""docx_corpus QA의 근거 구간 후보 샘플링 (docs/restructure_plan.md 7.3절 1단계, LLM 없음).

  .venv311/bin/python scripts/sample_docx_spans.py [--per-doc 5] [--limit N]
  -> data/raw/docx_corpus/_meta/candidates.json
실행 전에 `source env.sh`(캐시된 bge-m3 토크나이저 사용).
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.models import get_token_counter  # noqa: E402
from qa_build import docx_candidates as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-doc", type=int, default=D.PER_DOC)
    ap.add_argument("--limit", type=int, help="앞에서 N개 문서만(시험용)")
    ap.add_argument("--out", help="기본: data/raw/docx_corpus/_meta/candidates.json")
    args = ap.parse_args()
    ids = None
    if args.limit:
        ids = {p.stem for p in sorted((D.PROCESSED_DIR / "docx_corpus").glob("*.json")) if not p.name.startswith("_")}
        ids = set(sorted(ids)[:args.limit])
    cands, problems, rejected = D.sample_corpus(get_token_counter("bge-m3"), doc_ids=ids, per_doc=args.per_doc)
    path = D.write_candidates(cands, Path(args.out) if args.out else D.CANDIDATES_PATH)
    toks = sorted(c["n_tokens"] for c in cands.values())
    print(f"후보 {len(cands)}개 / 문서 {len({c['doc_id'] for c in cands.values()})}개 -> {path}")
    print(f"  토큰 min {toks[0]} / 중앙 {toks[len(toks)//2]} / max {toks[-1]}")
    print("  표 시작 앵커:", sum(c["anchor"] == "table" for c in cands.values()), "| 표 포함:", sum(c["has_table"] for c in cands.values()))
    print("  주제:", dict(sorted(Counter(c["topic"] for c in cands.values()).items())))
    print("  거절 사유(시도 중):", rejected or "없음", "| 후보 부족 문서:", problems or "없음")


if __name__ == "__main__":
    main()
