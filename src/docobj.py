"""문서 객체 공통 스키마 - docx/pdf 파서가 만드는 elements를 저장/재사용한다.
elements: [{"id":, "label":, "level":, "text":, "loc":{...}}, ...] - 한 번 파싱해두면
청킹/검색/요약/번역/병합이 전부 이 하나의 리스트만 보고 동작한다(docx/pdf는 파서만 다름,
모양은 같음)."""
import json
from pathlib import Path


def save_elements(doc_name, fmt, elements, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{doc_name}.json"
    path.write_text(
        json.dumps({"name": doc_name, "fmt": fmt, "elements": elements}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_elements(doc_name, out_dir):
    path = Path(out_dir) / f"{doc_name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["elements"]
