"""pdf -> elements 파서 (Docling). loader.py의 컨버터(do_ocr=False, FAST 표 모드)를
그대로 재사용해서 모델을 두 번 로드하지 않는다. label은 Docling이 이미 판별한 걸 그대로
쓰고(제목 감지 규칙을 직접 만들지 않음), loc은 page_no+bbox(top-left 좌표계로 통일)."""
from pathlib import Path

from pdf_track.loader import _converter, _TEXT_LABELS


def _bbox_to_dict(bbox, page_height):
    """텍스트 아이템은 BOTTOMLEFT, 표 셀 bbox는 이미 TOPLEFT로 나오는 게 실측 확인됨 -
    아이템마다 coord_origin을 확인하고 필요할 때만 변환한다."""
    if bbox.coord_origin.name != "TOPLEFT":
        bbox = bbox.to_top_left_origin(page_height)
    return {"l": bbox.l, "t": bbox.t, "r": bbox.r, "b": bbox.b}


def parse_pdf(filepath):
    result = _converter.convert(str(filepath))
    doc = result.document
    elements = []
    eid = 0

    for item in doc.texts:
        if item.label not in _TEXT_LABELS:
            continue
        text = item.text.strip()
        if not text or not item.prov:
            continue
        p = item.prov[0]
        page_h = doc.pages[p.page_no].size.height
        elements.append({
            "id": eid, "label": str(item.label), "level": getattr(item, "level", None),
            "text": text, "loc": {"page": p.page_no, "bbox": _bbox_to_dict(p.bbox, page_h)},
        })
        eid += 1

    for table in doc.tables:
        if not table.prov:
            continue
        page_no = table.prov[0].page_no
        page_h = doc.pages[page_no].size.height
        for cell in table.data.table_cells:
            text = cell.text.strip()
            if not text:
                continue
            elements.append({
                "id": eid, "label": "table_cell", "level": None, "text": text,
                "loc": {"page": page_no, "bbox": _bbox_to_dict(cell.bbox, page_h),
                        "row": cell.start_row_offset_idx, "col": cell.start_col_offset_idx},
            })
            eid += 1
    return elements


if __name__ == "__main__":
    import sys
    import torch
    torch.backends.cudnn.enabled = False  # 이 서버의 cuDNN/드라이버 버전 불일치 우회 (실측 확인된 문제)
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from docobj import save_elements
    from collections import Counter

    for f in sys.argv[1:]:
        elements = parse_pdf(f)
        counts = Counter(e["label"] for e in elements)
        print(f"\n=== {Path(f).name} ({len(elements)}개 요소) ===")
        print("label별 개수:", dict(counts))
        for e in elements[:8]:
            print(" ", e["id"], e["label"], e["level"], "|", e["loc"], "|", e["text"][:40])
