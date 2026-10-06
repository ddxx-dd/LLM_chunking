"""전처리 결과(_report.json)의 실측값을 매니페스트에 새 필드로 덧붙인다(기존 필드는 덮어쓰지 않음).
  .venv311/bin/python scripts/annotate_manifests.py
추가 필드: n_blocks, n_tables_docling, n_footnotes, chars_json, n_pages_docling(pdf).
vectara의 기존 n_tables는 선별 당시의 옛 값이라 실제 변환과 다르다(2단계에서 확인) - 비교는 n_tables_docling으로.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.config import MANIFEST_DIR, PROCESSED_DIR  # noqa: E402

ID_KEY = {"vectara": "paper_id", "allganize": "doc_id", "docx_corpus": None}


def main():
    for ds, key in ID_KEY.items():
        path = MANIFEST_DIR / f"{ds}.json"
        report = json.loads((PROCESSED_DIR / ds / "_report.json").read_text(encoding="utf-8"))["docs"]
        docs = json.loads(path.read_text(encoding="utf-8"))
        hit = 0
        for e in docs:
            doc_id = e[key] if key else Path(e["file"]).stem
            r = report.get(doc_id)
            if not r or "stats" not in r:
                continue
            s = r["stats"]
            e.update(n_blocks=s["n_blocks"], n_tables_docling=s["n_tables"], n_footnotes=s["n_footnotes"],
                     chars_json=s["total_chars"])
            if ds != "docx_corpus":
                doc = json.loads((PROCESSED_DIR / ds / f"{doc_id}.json").read_text(encoding="utf-8"))
                e["n_pages_docling"] = doc.get("n_pages")
            hit += 1
        path.write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{ds}] {hit}/{len(docs)}개 문서에 실측 필드 추가 → {path}")


if __name__ == "__main__":
    main()
