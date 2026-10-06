"""raw/_meta 의 질문·정답 -> data/qa/<dataset>.jsonl (docs/restructure_plan.md 7절).

  .venv311/bin/python scripts/build_qa.py --dataset vectara      # 또는 allganize
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qa_build.build_qa import BUILDERS, build, write_qa  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(BUILDERS))
    args = ap.parse_args()
    recs, problems = build(args.dataset)
    path = write_qa(args.dataset, recs)
    print(f"[{args.dataset}] 질문 {len(recs)}개 -> {path}")
    print("  qtype:", dict(Counter(r["qtype"] for r in recs)))
    print("  제외(excluded):", sum(r["excluded"] for r in recs), "| split:", dict(Counter(r["split"] for r in recs)))
    print("  문서 수(질문 있는):", len({r["doc_id"] for r in recs}), "| 문제:", problems or "없음")


if __name__ == "__main__":
    main()
