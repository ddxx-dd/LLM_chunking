"""pdf -> elements 파서 (Docling). smart_chunk.md 4절.
loader.py의 컨버터(do_ocr=False, FAST 표 모드)를 그대로 재사용해서 모델을 두 번 로드하지
않는다. doc.texts(평면 리스트, 표 위치가 문서 순서와 안 맞을 수 있음) 대신
doc.iterate_items()(진짜 문서 순서, (item, level) 튜플, 실측 확인)로 순회 - docx와 같은
이유로 표가 본문 흐름과 다른 위치로 밀려서 smart의 제목/캡션 묶기가 깨지는 것 방지."""
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pymupdf

from docobj import NUM_ONLY, build_table_markdown
from pdf_track.loader import _converter

FURNITURE_LABELS = {"page_header", "page_footer"}
# 진짜 수식은 기본적으로 Docling formula 라벨로 거르고(아래 parse_pdf), 이 글꼴 판정은
# 보조로만 쓴다. ★ 실측 확인(2026-09-26): 원본 PDFMathTranslate는 글자(span) 단위로
# 이 패턴을 적용하지만 우리는 요소(문단) 단위로 적용한다 - 원본 패턴의 CM[^R]는 "CM으로
# 시작하고 세 번째 글자가 R만 아니면 전부 수식"이라 CMBX(볼드)/CMTI(이탤릭) 같은 본문
# 강조체도 걸려버려서, 문단 단위로 적용하면 강조 단어 하나만 있어도 문단 전체가 skip
# 처리되는 문제가 실측됨(샘플 논문 196개 중 151개, 77% 오탐). 그래서 CM[^R] 계열을 빼고
# 진짜 수식 전용 글꼴 패밀리만 남겼고, 요소 단위 판정이니 "글자 수 대비 매치 비율"
# 기준(아래 _math_font_ratio, 0.5 이상)을 추가로 둔다.
MATH_FONT_RE = re.compile(
    r"^(CMMI|CMSY|CMEX|MSAM|MSBM|EUFM|EUSM|EURM|RSFS|STMARY|WASY|TXSY|.*Math|.*Sym)")
MATH_FONT_RATIO_THRESHOLD = 0.5


def _bbox_to_dict(bbox, page_height):
    """텍스트 아이템은 BOTTOMLEFT, 표 셀 bbox는 이미 TOPLEFT로 나오는 게 실측 확인됨 -
    아이템마다 coord_origin을 확인하고 필요할 때만 변환한다."""
    if bbox.coord_origin.name != "TOPLEFT":
        bbox = bbox.to_top_left_origin(page_height)
    return {"l": bbox.l, "t": bbox.t, "r": bbox.r, "b": bbox.b}


def _provs_for(item, doc):
    """prov 전체를 저장(문단이 페이지/단을 넘으면 여러 개 나옴 - item.prov[0]만 쓰던 이전
    버전과의 차이, 4절)."""
    provs = []
    for p in item.prov or []:
        page_h = doc.pages[p.page_no].size.height
        provs.append({"page": p.page_no, "bbox": _bbox_to_dict(p.bbox, page_h),
                      "charspan": list(p.charspan)})
    return provs


def _math_font_ratio(pymupdf_doc, page_no, bbox):
    """요소 bbox 안 전체 글자 수 대비 수식 글꼴 글자 수 비율(0~1). 서브셋 접두어
    ("ABCDEF+CMR10")는 "+" 기준으로 잘라내고 re.match(문자열 시작 기준)로 판정 - 이 둘을
    안 지키면 "ArialMT"가 "MT"에 우연히 걸리는 등 오탐이 남(실측 확인)."""
    page = pymupdf_doc[page_no - 1]
    rect = pymupdf.Rect(bbox["l"], bbox["t"], bbox["r"], bbox["b"])
    text_dict = page.get_text("dict", clip=rect)
    total = 0
    math_chars = 0
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                n = len(span.get("text", ""))
                total += n
                font = span.get("font", "").split("+")[-1]
                if MATH_FONT_RE.match(font):
                    math_chars += n
    return math_chars / total if total else 0.0


def _build_table_element(item, eid, pymupdf_doc):
    """표 = 요소 1개 + cells 목록(6절 두 층 구조, docx와 동일 원칙). 셀 loc은 일반 요소와
    같은 모양: {"prov": [{"page":, "bbox": {...}}]} - bbox는 table_cells[].bbox에서
    (TOPLEFT로 변환), page는 표 자체의 prov[0].page_no에서 가져온다(4절)."""
    n_rows, n_cols = item.data.num_rows, item.data.num_cols
    raw_cells = list(item.data.table_cells)
    cell_texts = {}
    for c in raw_cells:
        r, col = c.start_row_offset_idx, c.start_col_offset_idx
        cell_texts[(r, col)] = (c.text or "").strip()
    md, span_map = build_table_markdown(n_rows, n_cols, cell_texts)

    page_no = item.prov[0].page_no if item.prov else None
    page_h = pymupdf_doc[page_no - 1].rect.height if page_no is not None else None

    cells = []
    for cid, c in enumerate(raw_cells):
        r, col = c.start_row_offset_idx, c.start_col_offset_idx
        text = cell_texts[(r, col)]
        loc = None
        if page_no is not None and c.bbox is not None:
            loc = {"prov": [{"page": page_no, "bbox": _bbox_to_dict(c.bbox, page_h)}]}
        cell = {
            "cell_id": cid, "row": r, "col": col,
            "row_span": c.row_span, "col_span": c.col_span,
            "header": bool(c.column_header or c.row_header),
            "text": text, "span_in_table": span_map[(r, col)], "loc": loc,
        }
        if NUM_ONLY.fullmatch(text):
            cell["skip"] = "num_only"
        cells.append(cell)

    return {
        "id": eid, "label": "table", "level": None, "text": md, "loc": None, "skip": None,
        "table_id": eid, "caption_ids": [], "n_rows": n_rows, "n_cols": n_cols, "cells": cells,
    }


def parse_pdf(filepath):
    result = _converter.convert(str(filepath))
    doc = result.document
    pymupdf_doc = pymupdf.open(str(filepath))

    elements = []
    eid = 0
    for item, _level in doc.iterate_items():
        label = str(item.label)
        if label in FURNITURE_LABELS:
            continue
        if label == "table":
            elements.append(_build_table_element(item, eid, pymupdf_doc))
            eid += 1
            continue
        if label == "picture":
            elements.append({"id": eid, "label": "picture", "level": None, "text": "",
                              "loc": None, "skip": "picture"})
            eid += 1
            continue

        text = (getattr(item, "text", "") or "").strip()
        if not text or not item.prov:
            continue
        provs = _provs_for(item, doc)
        item_level = getattr(item, "level", None) if label == "section_header" else None

        skip = "formula" if label == "formula" else None
        if skip is None:
            ratio = _math_font_ratio(pymupdf_doc, provs[0]["page"], provs[0]["bbox"])
            if ratio >= MATH_FONT_RATIO_THRESHOLD:
                skip = "math_font"

        elements.append({"id": eid, "label": label, "level": item_level, "text": text,
                          "loc": {"prov": provs}, "skip": skip})
        eid += 1

    pymupdf_doc.close()

    # 표 앞뒤 caption 연결(간단한 인접 검사, 6절 caption_ids)
    for i, e in enumerate(elements):
        if e["label"] != "table":
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(elements) and elements[j]["label"] == "caption":
                e["caption_ids"].append(elements[j]["id"])

    print(f"[pdf parse] {Path(filepath).name}: 요소 {len(elements)}개", flush=True)
    return elements


if __name__ == "__main__":
    import torch
    torch.backends.cudnn.enabled = False  # 이 서버의 cuDNN/드라이버 버전 불일치 우회(실측 확인된 문제)

    from collections import Counter

    from docobj import add_spans, save_elements

    for f in sys.argv[1:]:
        elements = parse_pdf(f)
        flat_text = add_spans(elements)
        counts = Counter(e["label"] for e in elements)
        skip_counts = Counter(e.get("skip") for e in elements if e.get("skip"))
        print(f"\n=== {Path(f).name} ({len(elements)}개 요소, flat_text {len(flat_text)}자) ===")
        print("label별 개수:", dict(counts))
        print("skip 사유별 개수:", dict(skip_counts))
        for e in elements[:10]:
            preview = e["text"][:40].replace("\n", "\\n")
            print(" ", e["id"], e["label"], e["level"], "|", e["loc"], "| skip=", e.get("skip"),
                  "| span=", e["span"], "|", preview)
        out = save_elements(Path(f).stem, "pdf", elements, "data/processed")
        print("saved:", out)
