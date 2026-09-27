"""pdf 병합 writer 핵심(smart_chunk.md 7절) - test_fit -> redact -> apply -> insert 순서.
identity_test.py(자기 텍스트 되돌려쓰기)가 이 모듈을 그대로 씀(docx_track/writer.py와
동일 구조 - 표시/항등 테스트와 실제 병합이 같은 코드 경로를 타야 검증 의미가 있다)."""
import difflib
import html
import re
import unicodedata
from pathlib import Path
from statistics import median

import pymupdf

# ★ 실측 확인(2026-09-26, 항등 테스트에서 발견, 11절 트러블슈팅): 설계서 7절 원안은
# NotoSansKR이었는데, Google Fonts에서 받은 NotoSansKR-Regular.ttf(v39)로 렌더링하면
# 숫자 바로 뒤에 공백 없이 알파벳이 붙는 특정 패턴("(3.1)is" 등)에서 숫자 글자가 무작위
# 한자로 깨지는 폰트 결함이 재현됨(기본 내장 폰트, NanumGothic 둘 다 같은 텍스트를 정상
# 렌더링 - NotoSansKR 파일 자체의 문제로 확인). 실제 번역 숫자가 깨질 수 있는 심각한
# 문제라 NanumGothic으로 교체.
FONT_DIR = str(Path(__file__).resolve().parent.parent.parent / "fonts")
CSS_BASE = "@font-face {font-family: NanumGothic; src: url(NanumGothic.ttf);} body {font-family: NanumGothic;"

FIT_COVERAGE_THRESHOLD = 0.95
DEFAULT_FONT_SIZE = 9  # 원문 bbox 안에서 글자 크기를 못 읽었을 때(예: 빈 영역) 쓰는 기본값(pt)
LONG_TOKEN_LEN = 20  # 공백 없는 토큰이 이 길이를 넘으면 줄바꿈 지점(zero-width space) 삽입
LONG_TOKEN_BREAK_EVERY = 6  # 몇 글자마다 삽입할지

# ★ 실측 확인(2026-09-27, pipeline.py A/B/C 진단) - 원문을 1.3배로 늘려서 넣어보면 셀
# 성공률이 0%(0/63)까지 떨어짐(el은 94%로 큰 차이 없음) - 표 셀은 여백이 거의 없어서
# 번역문이 조금만 길어져도(실제 번역은 원문보다 길어지는 경우가 흔함) scale_low=0.6로는
# 못 줄여서 못 들어간다. 셀만 더 많이 줄이는 걸 허용(글자가 작아져도 안 들어가는 것보단
# 낫다는 판단) - el은 문단이라 상대적으로 여유가 있어 기존 값 유지.
SCALE_LOW_BY_KIND = {"cell": 0.4, "el": 0.6}


def normalize_text(s):
    """★ 실측 확인(2026-09-26) - 이전엔 NFKC + 대시류 통일 + lookalike 치환표를 썼는데,
    새 글자가 나올 때마다 표를 늘려야 해서(‖∥, 〈⟨ 등) 한계가 있었다. 대신 NFKC로 정규화한
    뒤 결합 문자(악센트 등, 유니코드 Mn 카테고리)를 지우고 **글자·숫자만** 남겨서 비교한다
    - 기호 모양 차이(대시 종류, 수학 기호 lookalike 등)는 전부 판정에서 제외된다."""
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return "".join(ch for ch in s if ch.isalnum())


def coverage_ratio(orig_norm, extracted_norm):
    """정규화한 원문 글자 중 몇 %가 추출된 텍스트에 순서대로(SequenceMatcher 매칭 블록
    기준) 들어있는지. spare(삽입 높이 여유)만 보면 가로로 넘치는 긴 토큰이 통째로 잘려도
    "들어감"으로 나올 수 있어서(실측 확인) 추가한 검사."""
    if not orig_norm:
        return 1.0
    sm = difflib.SequenceMatcher(None, orig_norm, extracted_norm, autojunk=False)
    matched = sum(block.size for block in sm.get_matching_blocks())
    return matched / len(orig_norm)


def get_target(elements, owner):
    """owner 키(튜플)로 실제 요소 또는 셀 dict를 찾는다(5절) - ("el", id) 또는
    ("cell", table_id, cell_id)."""
    if owner[0] == "el":
        return elements[owner[1]]
    table = elements[owner[1]]
    return next(c for c in table["cells"] if c["cell_id"] == owner[2])


_LONG_TOKEN_RE = re.compile(r"\S{%d,}" % LONG_TOKEN_LEN)


def _break_long_tokens(text):
    """★ 실측 확인 - PyMuPDF의 insert_htmlbox는 CSS word-break/overflow-wrap을 지원하지
    않아서, 공백 없이 긴 토큰(20자 이상)이 있으면 줄바꿈이 안 돼 가로로 넘쳐버린다
    (좁은 표 셀에서 넣기 실패의 원인 중 하나로 의심됨). 실제로 몇 글자마다 U+200B(zero-width
    space)를 끼워 넣으면 그 지점에서 줄바꿈이 되는 걸 확인함(spare -1 -> 성공)."""
    def repl(m):
        tok = m.group(0)
        return "​".join(tok[i:i + LONG_TOKEN_BREAK_EVERY] for i in range(0, len(tok), LONG_TOKEN_BREAK_EVERY))
    return _LONG_TOKEN_RE.sub(repl, text)


def flatten_targets(elements, translations):
    """표는 셀 단위로 펼쳐서, 일반 요소와 셀을 같은 모양의 (owner, page_no, bbox, text,
    safe_text) 리스트로 만든다(7절) - html.escape는 여기서 한 번만. text(원문/번역문
    그대로, 줄바꿈 삽입 전)는 넣기 판정(coverage_ratio)과 항등 테스트 비교에 쓰고,
    safe_text(긴 토큰에 줄바꿈 지점 삽입 후 escape)는 실제 렌더링에 쓴다."""
    out = []
    for owner, translated_text in translations.items():
        target = get_target(elements, owner)
        bbox = target["loc"]["prov"][0]["bbox"]  # 1차 규칙: 여러 칸 걸치면 첫 칸만(7절 하단)
        page_no = target["loc"]["prov"][0]["page"]
        out.append({"owner": owner, "page_no": page_no, "bbox": bbox, "text": translated_text,
                    "safe_text": html.escape(_break_long_tokens(translated_text))})
    return out


def _rect(bbox, shrink=0):
    return pymupdf.Rect(bbox["l"] + shrink, bbox["t"] + shrink, bbox["r"] - shrink, bbox["b"] - shrink)


def _page_span_sizes(page):
    """페이지 전체의 (글자 중심점, 크기) 목록을 한 번만 뽑아둔다 - ★ 실측 확인: 대상마다
    get_text("dict", clip=rect)를 새로 부르면 대상 많은 페이지에서 문서 하나에 150초
    넘게 걸림(속도 병목). 페이지당 1번만 호출하고 대상마다는 이 리스트에서 필터링만
    한다(get_text 호출 횟수: 대상 수 -> 페이지 수)."""
    try:
        d = page.get_text("dict")
    except Exception:
        return []
    out = []
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    x0, y0, x1, y1 = span["bbox"]
                    out.append(((x0 + x1) / 2, (y0 + y1) / 2, span["size"]))
    return out


def _median_font_size(page_span_sizes, rect):
    """rect 영역 안(글자 중심점 기준) 원문 span들의 글자 크기 중앙값(pt). 원문 크기에
    맞춰야 좁은 셀에서 기본 폰트 크기(보통 12pt 안팎)로 렌더링돼 안 들어가는 문제를
    줄일 수 있다(★ 실측 의도 - 좁은 표 셀은 원문 글자 크기가 훨씬 작은 경우가 많음)."""
    sizes = [s for cx, cy, s in page_span_sizes if rect.x0 <= cx <= rect.x1 and rect.y0 <= cy <= rect.y1]
    return median(sizes) if sizes else DEFAULT_FONT_SIZE


def _css_for(font_size):
    return f"{CSS_BASE} font-size: {font_size:.1f}pt; line-height: 1.15;}}"


def write_translations(pdf, elements, translations, on_failed=None, on_scale=None):
    """translations({owner: 번역문 또는 원문})을 pdf에 병합한다(7절 순서: 페이지마다
    test_fit -> 들어가는 것만 redact 표시 -> 한 번에 apply -> 전부 insert - bbox 겹침 시
    방금 쓴 걸 지우는 사고 방지). ★ 실측 확인 - redact_annot은 1pt 안쪽으로 줄이지만
    insert_htmlbox는 **원래 bbox 그대로** 쓴다(줄였던 게 좁은 셀에서 한 줄도 안 들어가는
    원인으로 의심됐음 - redact는 경계선 잔여물 방지용으로 줄이는 게 맞지만, 텍스트 삽입은
    가용 공간을 최대로 써야 함). 반환: {"written":, "failed":, "fit_by_kind": {...}}."""
    targets = flatten_targets(elements, translations)
    by_page = {}
    for t in targets:
        by_page.setdefault(t["page_no"], []).append(t)

    archive = pymupdf.Archive(FONT_DIR)
    scratch = pymupdf.open()
    written = failed = 0
    fit_by_kind = {"cell": [0, 0], "el": [0, 0]}  # owner[0] -> [성공, 실패]

    for page_no, page_targets in by_page.items():
        page = pdf[page_no - 1]
        page_span_sizes = _page_span_sizes(page)  # 페이지당 1번만(위 함수 docstring 참고)

        fits = []
        for t in page_targets:  # (a) 먼저 넣어보고 들어가는지 확인 - 실제 쓰기와 같은 CSS·텍스트
            insert_rect = _rect(t["bbox"], shrink=0)
            font_size = _median_font_size(page_span_sizes, insert_rect)
            css = _css_for(font_size)
            # ★ 실측 확인(2026-09-26, 항등 테스트에서 발견, 11절) - 스크래치 페이지를
            # 페이지당 한 번만 비우고 그 안에서 대상 여러 개를 연달아 insert_htmlbox하면,
            # 앞 대상이 넘친 내용이 뒤 대상의 사각형과 겹쳐서 test_fit 결과가 오염된다
            # (격리 상태에선 spare=-1로 정확히 판정되던 문단이, 스크래치 페이지를
            # 공유하면 spare>=0으로 잘못 판정되고 실제로는 70%가 잘려나감). 그래서
            # 대상마다 스크래치 페이지를 새로 만든다.
            while len(scratch) > 0:
                scratch.delete_page(0)
            scratch.new_page(width=page.rect.width, height=page.rect.height)
            sp = scratch[0]
            scale_low = SCALE_LOW_BY_KIND[t["owner"][0]]
            spare, _scale = sp.insert_htmlbox(insert_rect, t["safe_text"], css=css,
                                               archive=archive, scale_low=scale_low)
            ok = spare >= 0
            if ok:
                # ★ 실측 확인 - spare는 삽입 높이 여유만 보므로, 가로로 넘치는 긴 토큰
                # (줄바꿈 안 되는 긴 단어/숫자열 등)이 통째로 잘려도 "들어감"으로 나올 수
                # 있다. 실제로 추출해서 원문 글자의 95% 이상이 들어갔는지까지 확인한다.
                extracted = sp.get_text(clip=insert_rect)
                ok = coverage_ratio(normalize_text(t["text"]), normalize_text(extracted)) >= FIT_COVERAGE_THRESHOLD
            kind = fit_by_kind[t["owner"][0]]
            if ok:
                fits.append((t, css))
                kind[0] += 1
            else:
                kind[1] += 1
                failed += 1
                if on_failed:
                    on_failed(t["owner"])

        for t, _css in fits:  # (b) 들어가는 것만 전부 redact 표시(1pt 안쪽 - 경계선 잔여물 방지)
            page.add_redact_annot(_rect(t["bbox"], shrink=1))
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)  # (c) 한 번에 적용

        for t, css in fits:  # (d) 그다음에 전부 쓰기 - insert는 원래 bbox 그대로
            scale_low = SCALE_LOW_BY_KIND[t["owner"][0]]
            spare, scale = page.insert_htmlbox(_rect(t["bbox"], shrink=0), t["safe_text"], css=css,
                                                archive=archive, scale_low=scale_low)
            if spare < 0:
                failed += 1
                fit_by_kind[t["owner"][0]][0] -= 1
                fit_by_kind[t["owner"][0]][1] += 1
                if on_failed:
                    on_failed(t["owner"])
            else:
                written += 1
                if on_scale:
                    on_scale(t["owner"], scale)

    pdf.subset_fonts()  # 맨 마지막에 한 번
    return {"written": written, "failed": failed,
            "fit_by_kind": {k: {"written": v[0], "failed": v[1]} for k, v in fit_by_kind.items()}}
