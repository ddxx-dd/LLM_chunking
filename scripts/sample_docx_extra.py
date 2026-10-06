"""docx_corpus QA 보강용 추가 후보 샘플링: 이미 쓴 구간과 겹치지 않게, 표 시작 후보 또는 일반 후보를 더 뽑는다.

  .venv311/bin/python scripts/sample_docx_extra.py --tag t --table-only --per-doc 2 [--dry]
  .venv311/bin/python scripts/sample_docx_extra.py --tag b --big-tables [--dry]   # 600토큰 넘는 표를 통째로
  .venv311/bin/python scripts/sample_docx_extra.py --tag a --per-doc 2 --max-questions 4 [--dry]
후보 id는 `<doc>#<tag><k>`(예: `x#t0`). candidates.json 에 합쳐 쓴다(--dry면 저장하지 않고 집계만).
--max-questions N: 현재 통과 질문이 N개 이하인 문서에서만 뽑는다. 실행 전에 `source env.sh`.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.config import MANIFEST_DIR, PROCESSED_DIR  # noqa: E402
from common.docjson import load_doc  # noqa: E402
from common.models import get_token_counter  # noqa: E402
from qa_build import docx_candidates as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--per-doc", type=int, default=2)
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--max-questions", type=int, help="현재 통과 질문 수가 이 값 이하인 문서에서만")
    ap.add_argument("--big-tables", action="store_true", help="600토큰을 넘는 표(최대 1600)를 통째로 후보로")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    cands = json.loads(D.CANDIDATES_PATH.read_text(encoding="utf-8"))
    if any(f"#{args.tag}" in k for k in cands):
        sys.exit(f"태그 {args.tag!r}는 이미 쓰였다. 다른 태그를 쓸 것")
    qrels = json.loads((D.META_DIR / "qrels.json").read_text(encoding="utf-8"))
    n_q = Counter(v["doc_id"] for v in qrels.values())
    used = {}
    for c in cands.values():
        used.setdefault(c["doc_id"], []).append((c["start"], c["end"]))
    meta = {Path(m["file"]).stem: m for m in json.loads((MANIFEST_DIR / "docx_corpus.json").read_text(encoding="utf-8"))}
    count = get_token_counter("bge-m3")
    new, rej = {}, Counter()
    for path in sorted((PROCESSED_DIR / "docx_corpus").glob("*.json")):
        if path.name.startswith("_") or path.stem not in meta:
            continue
        doc = load_doc(path)
        if args.max_questions is not None and n_q.get(doc["doc_id"], 0) > args.max_questions:
            continue
        m = meta[doc["doc_id"]]
        if args.big_tables:
            got, r = D.big_tables(doc, count, topic=m["topic"], doc_type=m["type"], avoid=used.get(doc["doc_id"], ()), cid_tag=args.tag), {}
        else:
            got, r = D.sample_doc(doc, count, topic=m["topic"], doc_type=m["type"], per_doc=args.per_doc,
                                  avoid=used.get(doc["doc_id"], ()), table_only=args.table_only, cid_tag=args.tag)
        new.update({c["cid"]: c for c in got})
        rej.update(r)
    toks = sorted(c["n_tokens"] for c in new.values()) or [0]
    print(f"추가 후보 {len(new)}개 / 문서 {len({c['doc_id'] for c in new.values()})}개 | 토큰 min {toks[0]} 중앙 {toks[len(toks)//2]} max {toks[-1]}")
    print("  표 시작:", sum(c["anchor"] == "table" for c in new.values()), "| 표 포함:", sum(c["has_table"] for c in new.values()))
    print("  주제:", dict(sorted(Counter(c["topic"] for c in new.values()).items())), "| 거절:", dict(rej))
    if not args.dry:
        D.write_candidates({**cands, **new})
        print("candidates.json 에 합쳐 저장:", len(cands), "->", len(cands) + len(new))


if __name__ == "__main__":
    main()
