"""docx -> elements 파서 (python-docx). Allganize docx는 제목 스타일이 있는 일반 문서라는
게 실측 확인됨(45개 중 43개가 Heading/Title 스타일 보유) - 그래서 글자 크기/굵기로 제목을
추측하는 규칙은 안 쓰고 스타일/outline level만 본다."""
from pathlib import Path

from docx import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph


def _iter_block_items(doc):
    """document.paragraphs/.tables는 두 종류가 섞인 원래 순서를 안 보존한다 -
    body XML을 직접 순회해서 문단/표가 실제 나온 순서 그대로 얻는다(표준 python-docx 레시피)."""
    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


def _label_for_paragraph(p):
    style_name = p.style.name if p.style is not None else ""
    # python-docx 1.2.0엔 ParagraphFormat.outline_level가 없음(확인됨) - 스타일 기반 감지만
    # 사용(45개 중 43개가 Heading/Title 스타일 보유, 실측 확인) - 나머지 2개는 알려진 한계.
    if style_name == "Title":
        return "section_header", 0
    if style_name.startswith("Heading"):
        try:
            level = int(style_name.split()[-1])
        except ValueError:
            level = 1
        return "section_header", level
    if style_name == "Caption":
        return "caption", None
    if style_name == "List Paragraph":
        return "list_item", None
    return "text", None


def parse_docx(filepath):
    doc = DocxDocument(str(filepath))
    elements = []
    eid = 0
    seen_cell_ids = set()  # 병합 셀은 같은 <w:tc>를 여러 grid 위치에서 가리키므로 한 번만 기록

    for block_i, block in enumerate(_iter_block_items(doc)):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            label, level = _label_for_paragraph(block)
            elements.append({"id": eid, "label": label, "level": level, "text": text,
                              "loc": {"block": block_i}})
            eid += 1
        elif isinstance(block, Table):
            for r, row in enumerate(block.rows):
                for c, cell in enumerate(row.cells):
                    tc_id = id(cell._tc)
                    if tc_id in seen_cell_ids:
                        continue
                    seen_cell_ids.add(tc_id)
                    text = cell.text.strip()
                    if not text:
                        continue
                    elements.append({"id": eid, "label": "table_cell", "level": None, "text": text,
                                      "loc": {"block": block_i, "row": r, "col": c}})
                    eid += 1
    return elements


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from docobj import save_elements
    from collections import Counter

    for f in sys.argv[1:]:
        elements = parse_docx(f)
        counts = Counter(e["label"] for e in elements)
        print(f"\n=== {Path(f).name} ({len(elements)}개 요소) ===")
        print("label별 개수:", dict(counts))
        for e in elements[:8]:
            print(" ", e["id"], e["label"], e["level"], "|", e["loc"], "|", e["text"][:40])
