"""전처리 결과(공통 JSON) 문서별 점검 (docs/restructure_plan.md 12절 2단계 통과 기준).
errors는 변환이 틀렸다는 뜻(문서를 실패로 센다), warnings는 사람이 보면 좋은 것(문서는 통과)이다.
"""
from collections import Counter

BLOCK_TYPES = {"heading", "paragraph", "list_item", "table", "formula", "code", "footnote", "cue"}
LOW_TEXT_PAGE_CHARS = 50          # 쪽당 글자 수가 이보다 적으면 "글이 빈 쪽"(스캔·그림 쪽 의심)
MIN_TOTAL_CHARS = 2000


def validate_doc(doc):
    errors, warnings = [], []
    blocks = doc.get("blocks") or []
    fmt = doc.get("fmt")
    log = doc.get("log") or {}

    if not blocks:
        errors.append("블록이 없다")
    ids = [b.get("id") for b in blocks]
    if len(set(ids)) != len(ids):
        errors.append("블록 id가 중복된다")
    bad_types = sorted({b.get("type") for b in blocks} - BLOCK_TYPES)
    if bad_types:
        errors.append(f"알 수 없는 블록 유형: {bad_types}")
    empty = sum(1 for b in blocks if not (b.get("text") or "").strip())
    if empty:
        errors.append(f"텍스트가 빈 블록 {empty}개")

    types = Counter(b.get("type") for b in blocks)
    total_chars = sum(len(b.get("text") or "") for b in blocks)
    stats = {"n_blocks": len(blocks), "by_type": dict(types), "total_chars": total_chars}

    if fmt == "srt":
        from srt.validate import validate_srt

        validate_srt(blocks, types, log, errors, warnings, stats)
        return {"errors": errors, "warnings": warnings, "stats": stats}

    # --- pdf / docx
    headings = [b for b in blocks if b.get("type") == "heading"]
    levels = [b["level"] for b in headings]
    if any((not isinstance(lv, int)) or not 1 <= lv <= 6 for lv in levels):
        errors.append("제목 수준이 1~6 정수가 아니다")
    stats.update(n_headings=len(headings), levels=sorted(set(levels)), n_tables=types.get("table", 0),
                 n_footnotes=types.get("footnote", 0))

    # 각주는 모두 블록 목록 맨 끝에 있어야 한다(A안)
    kinds = [b.get("type") == "footnote" for b in blocks]
    if any(kinds) and kinds != sorted(kinds):
        errors.append("각주 블록이 맨 끝에 모여 있지 않다")

    # 표 개수: Docling이 낸 표(빈 표 제외) = 표 블록
    if "docling_tables" in log:
        want = log["docling_tables"] - log.get("empty_tables", 0)
        if types.get("table", 0) != want:
            errors.append(f"표 개수 불일치: 블록 {types.get('table', 0)} vs Docling {want}")

    # 캡션이 표 텍스트 안에도 들어 있으면 중복이다(표는 캡션 없는 마크다운이어야 한다)
    by_id = {b.get("id"): b for b in blocks}
    dup = [b["id"] for b in blocks if b.get("role") == "caption" and b.get("for") in by_id
           and b["text"][:40] in by_id[b["for"]]["text"]]
    if dup:
        errors.append(f"표 텍스트에 캡션이 중복된다: {dup[:5]}")

    # 쪽 번호 (pdf)
    n_pages = doc.get("n_pages")
    if fmt == "pdf":
        pages = [b.get("page") for b in blocks]
        if n_pages and any(p is not None and not 1 <= p <= n_pages for p in pages):
            errors.append("page가 1..n_pages 범위를 벗어난다")
        if any(p is None for p in pages):
            warnings.append(f"page가 없는 블록 {sum(p is None for p in pages)}개")
        if n_pages:
            per_page = Counter()
            for b in blocks:
                if b.get("page"):
                    per_page[b["page"]] += len(b.get("text") or "")
            low = [p for p in range(1, n_pages + 1) if per_page[p] < LOW_TEXT_PAGE_CHARS]
            stats["low_text_pages"] = len(low)
            if len(low) > 0.2 * n_pages:
                warnings.append(f"글이 거의 없는 쪽이 {len(low)}/{n_pages} (스캔·그림 쪽 의심)")

    # 제목 구조
    if not headings:
        warnings.append("제목 블록이 없다")
    else:
        if levels[0] != 1:
            warnings.append(f"첫 제목 수준이 {levels[0]}이다")
        if len(headings) >= 5 and len(set(levels)) == 1:
            warnings.append("제목이 5개 이상인데 수준이 전부 같다")
        jumps = sum(1 for a, b in zip(levels, levels[1:]) if b - a > 1)
        if jumps:
            warnings.append(f"제목 수준이 2단계 이상 건너뛴 곳 {jumps}회")
    if total_chars < MIN_TOTAL_CHARS:
        warnings.append(f"글자 수가 적다({total_chars})")
    return {"errors": errors, "warnings": warnings, "stats": stats}
