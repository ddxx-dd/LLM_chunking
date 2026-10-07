"""의미 분할 계열(semantic, A, B, E)이 쓰는 단위 만들기와 구조 규칙.

구조 규칙은 3개뿐이다(옛 6개 중 이번 실험에서 쓰는 것만. 옛 규칙 전체는 chunking/v1/units.py에 보존):
  heading_glue   제목 조각을 뒤따르는 첫 단위에 붙인다(제목 바로 뒤는 못 자름. 제목은 문장 정규식으로 쪼개지 않음, 연속 제목은 함께)
  table_atomic   표 조각 전체를 단위 하나로 만든다(행마다 쪼개지 않음)
  caption_glue   표와 캡션 사이 경계를 못 자르게 두 단위를 합친다
kind "sent"는 규칙이 없고(평문 전체를 문장 정규식으로 나눔, semantic), kind "sentg"는 3개 규칙을 모두 적용한다(A, B, E).
"""
from chunking.base import sentence_spans


def caption_glue(doc, pieces):
    """표와 캡션이 붙어 있어야 하는 위치 -> {그 경계 뒤 조각의 start}.
    (a) 표 블록의 caption 필드: 렌더가 캡션 조각 + 표 조각을 만들고 둘이 같은 블록 id를 가진다.
    (b) Docling 캡션: role="caption", for=<표 블록 id>인 문단이 표 바로 앞이나 바로 뒤에 있다."""
    by_id = {b["id"]: b for b in doc["blocks"]}
    glue = set()
    for a, b in zip(pieces, pieces[1:]):
        if a.kind == "para" and b.kind == "table" and set(a.block_ids) & set(b.block_ids):
            glue.add(b.start)
        ba = by_id.get(a.block_ids[0]) if a.block_ids else None
        bb = by_id.get(b.block_ids[0]) if b.block_ids else None
        if ba and ba.get("role") == "caption" and b.kind == "table" and ba.get("for") in b.block_ids:
            glue.add(b.start)                       # 캡션 -> 표
        if bb and bb.get("role") == "caption" and a.kind == "table" and bb.get("for") in a.block_ids:
            glue.add(b.start)                       # 표 -> 캡션
    return glue


def make_units(ctx, kind):
    """DocContext -> 단위 구간 [(start, end)] (평문 안의 연속 구간, 겹침 없음)."""
    if kind == "sent":
        return sentence_spans(ctx.text)
    if kind != "sentg":
        raise ValueError(f"모르는 단위 종류: {kind!r} (sent | sentg)")
    text, pieces, glue = ctx.text, ctx.pieces, ctx.glue
    units = []
    for k, p in enumerate(pieces):
        if p.kind in ("heading", "table"):
            spans = [(p.start, p.end)]                          # 제목·표는 쪼개지 않는다
        else:
            spans = sentence_spans(text, p.start, p.end)
        if not spans:
            continue
        stuck = k > 0 and (pieces[k - 1].kind == "heading" or p.start in glue)
        if stuck and units:                                      # 이 조각 앞 경계는 자를 수 없다 -> 앞 단위와 합친다
            units[-1] = (units[-1][0], spans[0][1])
            spans = spans[1:]
        units.extend(spans)
    return units
