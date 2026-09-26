"""항등 테스트(pdf, smart_chunk.md 8절/10절 2번) - 요소/셀 자기 자신의 text를 자기 bbox에
그대로 되돌려 써서 원본과 비교한다. 7절 pdf writer 순서(test_fit -> redact -> apply ->
insert)를 그대로 따르되 번역문 대신 요소 자기 자신의 text를 쓴다. 8절 기준 (a)(b)(c)를
그대로 구현: (a) bbox 안쪽 재추출 텍스트 일치, (b) 그림/도형 개수 redaction 전후 동일,
(c) bbox 바깥 영역 픽셀 동일(스크린샷 비교)."""
import html
import sys
from pathlib import Path

import numpy as np
import pymupdf

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pdf_track.parse import parse_pdf

# 샘플이 영문 논문이라 한글 폰트(NotoSansKR) 불필요 - 7절 CSS는 번역(한글 출력)용,
# 항등 테스트는 원문(영문) 되돌려쓰기라 기본 sans-serif로 충분.
CSS = "body { font-family: sans-serif; font-size: 8pt; }"


def flatten_targets(elements):
    out = []
    for e in elements:
        if e["label"] == "table":
            for c in e["cells"]:
                if c.get("skip") or not c.get("loc"):
                    continue
                prov = c["loc"]["prov"][0]
                out.append({"owner": ("cell", e["id"], c["cell_id"]), "page": prov["page"],
                            "bbox": prov["bbox"], "text": c["text"]})
            continue
        if e.get("skip") or e["label"] == "picture" or not e.get("loc"):
            continue
        prov = e["loc"]["prov"][0]
        out.append({"owner": ("el", e["id"]), "page": prov["page"], "bbox": prov["bbox"], "text": e["text"]})
    return out


def _rect(bbox, shrink=1):
    return pymupdf.Rect(bbox["l"] + shrink, bbox["t"] + shrink, bbox["r"] - shrink, bbox["b"] - shrink)


def _norm(s):
    return "".join(s.split())


def run_identity_test(src_path):
    src_path = Path(src_path)
    elements = parse_pdf(src_path)
    targets = flatten_targets(elements)

    pdf = pymupdf.open(str(src_path))
    by_page = {}
    for t in targets:
        by_page.setdefault(t["page"], []).append(t)

    scratch = pymupdf.open()
    written = failed = 0
    drawings_before = {}

    for page_no, page_targets in by_page.items():
        page = pdf[page_no - 1]
        drawings_before[page_no] = len(page.get_drawings())

        while len(scratch) > 0:  # 빈 스크래치 페이지(같은 크기)로 test_fit
            scratch.delete_page(0)
        scratch.new_page(width=page.rect.width, height=page.rect.height)
        sp = scratch[0]

        fits = []
        for t in page_targets:
            safe_text = html.escape(t["text"])
            rect = _rect(t["bbox"])
            spare, _scale = sp.insert_htmlbox(rect, safe_text, css=CSS, scale_low=0.4)
            if spare >= 0:
                fits.append((t, safe_text))
            else:
                failed += 1

        for t, _ in fits:  # (a) 들어가는 것만 redact 표시
            page.add_redact_annot(_rect(t["bbox"]))
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)

        for t, safe_text in fits:  # (b) 한 번에 적용 후, (c) 전부 쓰기
            spare, _scale = page.insert_htmlbox(_rect(t["bbox"]), safe_text, css=CSS, scale_low=0.4)
            if spare < 0:
                failed += 1
            else:
                written += 1

    out_path = src_path.with_name(src_path.stem + ".identitytest.pdf")
    pdf.save(str(out_path))
    pdf.close()

    out_pdf = pymupdf.open(str(out_path))
    orig_pdf = pymupdf.open(str(src_path))

    # 기준 (a): bbox 안쪽 재추출 텍스트가 원문과 일치
    text_mismatches = []
    for t in targets:
        page_out = out_pdf[t["page"] - 1]
        extracted = page_out.get_text(clip=_rect(t["bbox"]))
        if _norm(extracted) != _norm(t["text"]):
            text_mismatches.append((t["owner"], t["text"][:40], extracted[:40]))

    # 기준 (b): 그림/도형(벡터 라인아트) 개수가 redaction 전후 동일
    drawing_mismatches = []
    for page_no in by_page:
        n_after = len(out_pdf[page_no - 1].get_drawings())
        if n_after != drawings_before[page_no]:
            drawing_mismatches.append((page_no, drawings_before[page_no], n_after))

    # 기준 (c): 모든 요소 bbox 바깥 영역은 전후 픽셀이 그대로인지(스크린샷 비교)
    outside_diffs = []
    for page_no in by_page:
        p_orig, p_new = orig_pdf[page_no - 1], out_pdf[page_no - 1]
        pix_o, pix_n = p_orig.get_pixmap(), p_new.get_pixmap()
        if (pix_o.width, pix_o.height) != (pix_n.width, pix_n.height):
            outside_diffs.append((page_no, "크기 다름"))
            continue
        arr_o = np.frombuffer(pix_o.samples, dtype=np.uint8).reshape(pix_o.height, pix_o.width, pix_o.n)
        arr_n = np.frombuffer(pix_n.samples, dtype=np.uint8).reshape(pix_n.height, pix_n.width, pix_n.n)
        mask = np.ones((pix_o.height, pix_o.width), dtype=bool)
        for t in by_page[page_no]:
            b = t["bbox"]
            x0, y0, x1, y1 = int(b["l"]), int(b["t"]), int(b["r"]), int(b["b"])
            mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False
        diff = (arr_o != arr_n).any(axis=2) & mask
        n_diff = int(diff.sum())
        if n_diff > 0:
            outside_diffs.append((page_no, n_diff))

    print(f"[identity pdf] {src_path.name}")
    print(f"  대상 요소/셀 수: {len(targets)}, 삽입 성공: {written}, 실패(안 들어감): {failed}")
    print(f"  (a) bbox 안쪽 텍스트 재추출 일치: {not text_mismatches} (불일치 {len(text_mismatches)}건)")
    for m in text_mismatches[:5]:
        print("   ", m)
    print(f"  (b) 그림/도형 개수 redaction 전후 동일: {not drawing_mismatches} (불일치 {len(drawing_mismatches)}건)")
    for m in drawing_mismatches[:5]:
        print("   ", m)
    print(f"  (c) bbox 바깥 영역 픽셀 동일: {not outside_diffs} (차이 있는 페이지 {len(outside_diffs)}개)")
    for m in outside_diffs[:5]:
        print("   ", m)

    passed = not text_mismatches and not drawing_mismatches and not outside_diffs
    print(f"  ▶ 항등 테스트 {'통과' if passed else '실패'}")
    return {"passed": passed, "written": written, "failed": failed,
            "text_mismatches": len(text_mismatches), "drawing_mismatches": len(drawing_mismatches),
            "outside_diffs": len(outside_diffs)}


if __name__ == "__main__":
    import torch
    torch.backends.cudnn.enabled = False
    for f in sys.argv[1:]:
        run_identity_test(f)
