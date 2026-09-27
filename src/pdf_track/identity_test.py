"""항등 테스트(pdf, smart_chunk.md 8절/10절 2번) - 요소/셀 자기 자신의 text를 자기 bbox에
그대로 되돌려 써서 원본과 비교한다. 실제 쓰기는 pdf_track/writer.py를 그대로 씀(docx와
동일 구조 - 표시/항등 테스트와 실제 병합이 같은 코드 경로를 타야 검증 의미가 있다).

8절 기준 최종판(2026-09-26 갱신, 실측 반영) - **비율 기준**(문서 수준 통과 여부):
(a) 쓴 요소의 재추출 텍스트 일치율 ≥ 99%
(b) 그림/도형(벡터 라인아트) 보존율 = 100%
(c) 안 쓴 요소(skip/merge_skip이거나 넣기 실패)의 원문 보존율 ≥ 99%
    + bbox를 3px 넓힌 마스크 바깥 픽셀 변화율(%)은 참고 지표로만 기록(pass/fail 아님).
넣기 성공률은 목표 90%로 두고 문서별로 보고(통과 기준에는 안 씀).

★ 실측 확인 - (a)(c) 재추출은 `get_text(clip=bbox)` 대신 `get_text("words")`에서 **단어
중심점이 bbox 안에 있는 단어만** 모으는 방식으로 바꿨다 - clip 방식은 조밀한 표에서
이웃 셀 글자가 섞여 들어오는 문제가 있었음(3단계 스캔에서 발견, 예: "0.1099"가 이웃
셀 글자와 섞여 "0934 0"으로 추출됨 - 우리가 손도 안 댄 영역인데도 검증에서만 불일치로
잡혔었음)."""
import sys
from pathlib import Path

import numpy as np
import pymupdf

sys.path.append(str(Path(__file__).resolve().parent.parent))
from docobj import checks_output_path
from pdf_track.parse import parse_pdf
from pdf_track.writer import _rect, flatten_targets, normalize_text, write_translations

TEXT_MATCH_TARGET = 0.99
PRESERVED_TARGET = 0.99
FIT_RATE_TARGET = 0.90


def _split_by_writability(elements):
    """번역(쓰기) 대상(skip/merge_skip 없고 loc 있음)과, 쓰지 않을 대상(skip/merge_skip
    이지만 loc은 있어서 "안 건드렸는지" 확인 가능한 것)을 나눈다."""
    writable, untouched = {}, {}
    for e in elements:
        if e["label"] == "table":
            for c in e["cells"]:
                owner = ("cell", e["id"], c["cell_id"])
                reason = c.get("skip") or c.get("merge_skip") or e.get("skip") or e.get("merge_skip")
                if reason:
                    if c.get("loc"):
                        untouched[owner] = c["text"]
                else:
                    writable[owner] = c["text"]
            continue
        owner = ("el", e["id"])
        reason = "picture" if e["label"] == "picture" else (e.get("skip") or e.get("merge_skip"))
        if reason:
            if e.get("loc"):
                untouched[owner] = e["text"]
        else:
            writable[owner] = e["text"]
    return writable, untouched


def _extract_in_rect(words, rect):
    """rect 안에 중심점이 있는 단어만 모아 순서대로(줄→단어 순) 이어붙인다 - clip 방식의
    이웃 셀 글자 섞임 문제 회피(모듈 docstring 참고). words는 페이지당 1번만 뽑아서
    재사용(★ 실측 확인 - 대상마다 get_text("words")를 새로 부르면 대상 많은 페이지에서
    문서 하나 항등 테스트가 150초 넘게 걸림, pdf_track/writer.py의 같은 최적화 참고)."""
    picked = []
    for x0, y0, x1, y1, word, block_no, line_no, word_no in words:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if rect.x0 <= cx <= rect.x1 and rect.y0 <= cy <= rect.y1:
            picked.append((block_no, line_no, word_no, word))
    picked.sort()
    return " ".join(w[3] for w in picked)


def run_identity_test(src_path, elements=None):
    src_path = Path(src_path)
    if elements is None:  # 3단계 전체 스캔처럼 이미 파싱한 elements가 있으면 재파싱 안 함
        elements = parse_pdf(src_path)
    translations, skip_translations = _split_by_writability(elements)

    pdf = pymupdf.open(str(src_path))

    written_owners, failed_owners = [], []
    stats = write_translations(pdf, elements, translations,
                                on_failed=failed_owners.append,
                                on_scale=lambda o, s: written_owners.append(o))

    out_path = checks_output_path(src_path, ".identitytest.pdf")  # 원본 데이터 폴더에는 안 씀
    # ★ 실측 확인(2026-09-27) - insert_htmlbox(archive=...)가 호출마다 폰트를 새로 임베드해서
    # 일반 save()로는 파일이 수백 배 부풀어 오른다(garbage=4가 중복 객체를 병합해서 없앰).
    pdf.ez_save(str(out_path), garbage=4, deflate=True)
    pdf.close()

    out_pdf = pymupdf.open(str(out_path))
    orig_pdf = pymupdf.open(str(src_path))

    written_targets = flatten_targets(elements, {o: translations[o] for o in written_owners})
    by_page_written = {}
    for t in written_targets:
        by_page_written.setdefault(t["page_no"], []).append(t)

    words_cache = {}

    def words_for(page_no):
        if page_no not in words_cache:
            words_cache[page_no] = out_pdf[page_no - 1].get_text("words")
        return words_cache[page_no]

    # 기준 (a): 쓴 요소의 bbox 안 단어 중심점 기준 재추출 텍스트가 원문과 일치(정규화 비교)
    text_mismatches = []
    for t in written_targets:
        extracted = _extract_in_rect(words_for(t["page_no"]), _rect(t["bbox"], shrink=1))
        if normalize_text(t["text"]) != normalize_text(extracted):
            text_mismatches.append((t["owner"], t["text"][:40], extracted[:40]))
    text_match_rate = 1 - len(text_mismatches) / len(written_targets) if written_targets else 1.0

    # 기준 (b): 그림/도형(벡터 라인아트) 개수가 redaction 전후 동일
    drawing_mismatches = []
    for page_no in by_page_written:
        n_before = len(orig_pdf[page_no - 1].get_drawings())
        n_after = len(out_pdf[page_no - 1].get_drawings())
        if n_after != n_before:
            drawing_mismatches.append((page_no, n_before, n_after))

    # 기준 (c) 앞부분: 안 쓴 요소(skip/merge_skip + 넣기 실패)의 텍스트가 그대로 남아있는지
    untouched = dict(skip_translations)
    for o in failed_owners:
        untouched[o] = translations[o]
    untouched_targets = flatten_targets(elements, untouched)
    preserved_mismatches = []
    for t in untouched_targets:
        extracted = _extract_in_rect(words_for(t["page_no"]), _rect(t["bbox"], shrink=1))
        if normalize_text(t["text"]) != normalize_text(extracted):
            preserved_mismatches.append((t["owner"], t["text"][:40], extracted[:40]))
    preserved_rate = 1 - len(preserved_mismatches) / len(untouched_targets) if untouched_targets else 1.0

    # 기준 (c) 뒷부분(정보용, pass/fail 아님): bbox를 3px 넓힌 마스크 바깥 픽셀 변화율(%)
    outside_change_pct = {}
    for page_no, page_written in by_page_written.items():
        p_orig, p_new = orig_pdf[page_no - 1], out_pdf[page_no - 1]
        pix_o, pix_n = p_orig.get_pixmap(), p_new.get_pixmap()
        if (pix_o.width, pix_o.height) != (pix_n.width, pix_n.height):
            outside_change_pct[page_no] = None
            continue
        arr_o = np.frombuffer(pix_o.samples, dtype=np.uint8).reshape(pix_o.height, pix_o.width, pix_o.n)
        arr_n = np.frombuffer(pix_n.samples, dtype=np.uint8).reshape(pix_n.height, pix_n.width, pix_n.n)
        mask = np.ones((pix_o.height, pix_o.width), dtype=bool)
        for t in page_written:  # redaction이 경계에 걸친 이웃 글자를 지우는 특성 반영해 3px 확장
            b = t["bbox"]
            x0, y0 = int(b["l"]) - 3, int(b["t"]) - 3
            x1, y1 = int(b["r"]) + 3, int(b["b"]) + 3
            mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False
        diff = (arr_o != arr_n).any(axis=2) & mask
        total = int(mask.sum())
        outside_change_pct[page_no] = (100 * int(diff.sum()) / total) if total else 0.0

    fit_by_kind = stats["fit_by_kind"]
    fit_rate = stats["written"] / (stats["written"] + stats["failed"]) if (stats["written"] + stats["failed"]) else 1.0
    drawing_ok = not drawing_mismatches

    print(f"[identity pdf] {src_path.name}")
    print(f"  쓰기 시도 대상: {len(translations)}, 성공: {stats['written']}, 실패(넣기 실패): {stats['failed']}"
          f" (성공률 {fit_rate:.1%}, 목표 {FIT_RATE_TARGET:.0%})")
    for kind, label in (("cell", "표 셀"), ("el", "본문 요소")):
        w, fa = fit_by_kind[kind]["written"], fit_by_kind[kind]["failed"]
        tot = w + fa
        print(f"    {label}: 성공 {w}/{tot} ({w/tot:.1%})" if tot else f"    {label}: 대상 없음")
    print(f"  (a) 쓴 요소 텍스트 일치율: {text_match_rate:.1%} (목표 {TEXT_MATCH_TARGET:.0%}, "
          f"불일치 {len(text_mismatches)}/{len(written_targets)}건)")
    for m in text_mismatches[:5]:
        print("   ", m)
    print(f"  (b) 그림/도형 보존: {'100%' if drawing_ok else '불일치 ' + str(len(drawing_mismatches)) + '건'}")
    for m in drawing_mismatches[:5]:
        print("   ", m)
    print(f"  (c) 안 쓴 요소 원문 보존율: {preserved_rate:.1%} (목표 {PRESERVED_TARGET:.0%}, "
          f"불일치 {len(preserved_mismatches)}/{len(untouched_targets)}건)")
    for m in preserved_mismatches[:5]:
        print("   ", m)
    print("  (c) 참고: bbox+3px 바깥 픽셀 변화율(%, pass/fail 아님) - 페이지별:")
    for page_no in sorted(outside_change_pct):
        pct = outside_change_pct[page_no]
        print(f"    page {page_no}: {pct:.3f}%" if pct is not None else f"    page {page_no}: 크기 다름")

    passed = drawing_ok and text_match_rate >= TEXT_MATCH_TARGET and preserved_rate >= PRESERVED_TARGET
    print(f"  ▶ 항등 테스트 {'통과' if passed else '실패'}")
    return {"passed": passed, "written": stats["written"], "failed": stats["failed"], "fit_rate": fit_rate,
            "fit_by_kind": fit_by_kind, "text_match_rate": text_match_rate, "drawing_ok": drawing_ok,
            "preserved_rate": preserved_rate, "text_mismatches": len(text_mismatches),
            "drawing_mismatches": len(drawing_mismatches), "preserved_mismatches": len(preserved_mismatches),
            "outside_change_pct": outside_change_pct}


if __name__ == "__main__":
    import torch
    torch.backends.cudnn.enabled = False
    for f in sys.argv[1:]:
        run_identity_test(f)
