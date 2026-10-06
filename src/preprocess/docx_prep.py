"""docx -> 공통 JSON (docs/restructure_plan.md 4.3절). 기존 legacy/src/docx_track/loader.py와 같은 기본
DocumentConverter를 쓰고, 글자 없는 제목은 만들지 않는다. docx에는 쪽 정보가 없어 page는 None이다.
"""
from functools import lru_cache
from importlib.metadata import version

from docling_core.types.doc import DocItemLabel

from preprocess._docling import build_blocks


@lru_cache(maxsize=1)
def get_converter():
    from docling.document_converter import DocumentConverter

    return DocumentConverter()


def make_heading_level(dl):
    """Docling이 docx에서 주는 제목 수준을 쓴다(Word 스타일 Heading 1~5). TITLE이 있는 문서는 TITLE을
    1단계로 두고 SECTION_HEADER를 한 단계 내린다(기존 Docling 마크다운 내보내기와 같은 위계). 최대 6."""
    has_title = any(it.label == DocItemLabel.TITLE for it, _ in dl.iterate_items())

    def level(item):
        if item.label == DocItemLabel.TITLE:
            return 1
        lv = int(getattr(item, "level", 1) or 1)
        return min(lv + 1 if has_title else lv, 6)
    return level


def convert_docx(entry, converter=None):
    converter = converter or get_converter()
    dl = converter.convert(str(entry["path"])).document
    blocks, log = build_blocks(dl, make_heading_level(dl))
    title_block = next((b for b in blocks if b["type"] == "heading"), None)
    return {
        "doc_id": entry["doc_id"], "dataset": entry["dataset"], "fmt": "docx", "lang": entry["lang"],
        "title": entry.get("title") or (title_block["text"] if title_block else entry["doc_id"]),
        "source_file": entry["source_file"],
        "parser": f"docling-{version('docling')}",
        "parser_options": {"converter": "DocumentConverter(default)"},
        "n_pages": None,
        "log": log,
        "blocks": blocks,
    }
