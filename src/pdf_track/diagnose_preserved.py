"""[1] 진단(2026-09-26) - (c) 안 쓴 요소 불일치가 "실제 손상"(redaction이 이웃 요소를
지움)인지 "측정 오류"(Docling bbox가 원래 겹쳐서 원본에서도 같은 방식으로 추출하면
어긋남)인지 구분한다. 같은 추출 방식(단어 중심점 + normalize_text)을 원본 pdf에도
적용해서 비교."""
import sys
from pathlib import Path

import pymupdf

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pdf_track.identity_test import _extract_in_rect, _split_by_writability
from pdf_track.parse import parse_pdf
from pdf_track.writer import _rect, flatten_targets, normalize_text, write_translations


def diagnose(src_path, elements=None):
    src_path = Path(src_path)
    if elements is None:
        elements = parse_pdf(src_path)
    translations, skip_translations = _split_by_writability(elements)

    pdf = pymupdf.open(str(src_path))
    failed_owners = []
    write_translations(pdf, elements, translations, on_failed=failed_owners.append)
    out_path = src_path.with_name(src_path.stem + ".diag.pdf")
    # ★ 실측 확인(2026-09-27) - insert_htmlbox(archive=...)가 호출마다 폰트를 새로 임베드해서
    # 일반 save()로는 파일이 수백 배 부풀어 오른다(garbage=4가 중복 객체를 병합해서 없앰).
    pdf.ez_save(str(out_path), garbage=4, deflate=True)
    pdf.close()

    out_pdf = pymupdf.open(str(out_path))
    orig_pdf = pymupdf.open(str(src_path))

    untouched = dict(skip_translations)
    for o in failed_owners:
        untouched[o] = translations[o]
    untouched_targets = flatten_targets(elements, untouched)

    words_cache_out, words_cache_orig = {}, {}

    def words_for(cache, doc, page_no):
        if page_no not in cache:
            cache[page_no] = doc[page_no - 1].get_text("words")
        return cache[page_no]

    real_damage, measurement_error = [], []
    for t in untouched_targets:
        rect = _rect(t["bbox"], shrink=1)
        extracted_out = _extract_in_rect(words_for(words_cache_out, out_pdf, t["page_no"]), rect)
        extracted_orig = _extract_in_rect(words_for(words_cache_orig, orig_pdf, t["page_no"]), rect)
        expected = normalize_text(t["text"])
        out_ok = normalize_text(extracted_out) == expected
        orig_ok = normalize_text(extracted_orig) == expected
        if out_ok:
            continue  # 불일치 없음
        if orig_ok:
            real_damage.append((t["owner"], t["text"][:40], extracted_out[:40]))
        else:
            measurement_error.append((t["owner"], t["text"][:40], extracted_orig[:40], extracted_out[:40]))

    out_path.unlink(missing_ok=True)
    print(f"[진단] {src_path.name}: 실제 손상 {len(real_damage)}건, 측정 오류 {len(measurement_error)}건")
    return real_damage, measurement_error


if __name__ == "__main__":
    import torch
    torch.backends.cudnn.enabled = False
    all_damage, all_error = [], []
    for f in sys.argv[1:]:
        d, e = diagnose(f)
        all_damage.extend((Path(f).name, *x) for x in d)
        all_error.extend((Path(f).name, *x) for x in e)
    print(f"\n=== 합계: 실제 손상 {len(all_damage)}건, 측정 오류 {len(all_error)}건 ===")
    print("\n실제 손상 예시 5개:")
    for x in all_damage[:5]:
        print(" ", x)
    print("\n측정 오류 예시 5개:")
    for x in all_error[:5]:
        print(" ", x)
