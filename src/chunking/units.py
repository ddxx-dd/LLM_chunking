"""구조 제약 규칙 -> 의미 분할 엔진에 넘길 단위 (docs/restructure_plan.md 5절).

엔진(semantic_core)은 구조를 모른다. 제약은 "단위를 어떻게 만드느냐"와 "반드시 자를 경계(forced)"로만 표현한다.

  규칙             켜면                                                              끄면
  heading_forced   단위가 제목으로 시작하는 경계를 forced에 넣는다(제목 앞은 반드시 자름)      forced 없음
  heading_glue     제목 조각을 뒤따르는 첫 단위에 붙인다(제목 뒤는 자르지 않음, 연속 제목은 함께)  제목도 평문의 한 문장
  table_atomic     표 조각 전체를 단위 하나로 만든다(표는 쪼개지 않음)                       표 텍스트를 문장 정규식으로 쪼갠다(행마다 단위)
  caption_glue     표와 캡션 사이 경계를 자르지 못하게 두 단위를 합친다                      따로 둔다

  semantic = 모든 규칙 off (조각 정보를 아예 쓰지 않고 평문 전체를 문장 정규식으로 나눔 - LangChain과 같은 문장 나누기)
  smart    = 모든 규칙 on
일부 규칙만 켜는 실험(어블레이션)은 설정만 바꾸면 된다.
"""
from dataclasses import dataclass, fields

from chunking.semantic_core import sentence_spans


@dataclass(frozen=True)
class Rules:
    heading_forced: bool = False
    heading_glue: bool = False
    table_atomic: bool = False
    caption_glue: bool = False

    @classmethod
    def from_dict(cls, d):
        return cls(**{f.name: bool(d.get(f.name, False)) for f in fields(cls)})

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def any(self):
        return any(self.as_dict().values())

    def code(self):
        """파일 이름용 약어: H=제목 앞 강제, G=제목 붙임, T=표 한 덩어리, C=캡션 붙임. 없으면 'none'."""
        letters = [("heading_forced", "H"), ("heading_glue", "G"), ("table_atomic", "T"), ("caption_glue", "C")]
        return "".join(k for name, k in letters if getattr(self, name)) or "none"


NO_RULES = Rules()
ALL_RULES = Rules(True, True, True, True)
DEFAULT_RULES = {"semantic": NO_RULES, "smart": ALL_RULES}


def rules_for(chunker, cfg):
    """cfg["rules"]가 있으면 그것, 없으면 청커의 기본 규칙 세트(semantic=전부 off, smart=전부 on)."""
    return Rules.from_dict(cfg["rules"]) if "rules" in cfg else DEFAULT_RULES[chunker]


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


def build_units(text, pieces, glue=(), rules=NO_RULES):
    """-> (units [(start, end)], forced [경계 번호]). pieces: common.docjson.render(doc)의 Piece 목록,
    glue: caption_glue(doc, pieces)가 돌려준 집합."""
    if not rules.any():
        return sentence_spans(text), []
    glue = set(glue)
    heading_starts = {p.start for p in pieces if p.kind == "heading"}
    units = []
    for k, p in enumerate(pieces):
        if p.kind == "heading" and (rules.heading_glue or rules.heading_forced):
            spans = [(p.start, p.end)]                           # 제목은 쪼개지 않는다("1. 서론"의 마침표에서 나뉘지 않게)
        elif p.kind == "table" and rules.table_atomic:
            spans = [(p.start, p.end)]
        else:
            spans = sentence_spans(text, p.start, p.end)
        if not spans:
            continue
        stuck = k > 0 and ((rules.heading_glue and pieces[k - 1].kind == "heading") or (rules.caption_glue and p.start in glue))
        if stuck and units:                                      # 이 조각 앞 경계는 자를 수 없다 -> 앞 단위와 합친다
            units[-1] = (units[-1][0], spans[0][1])
            spans = spans[1:]
        units.extend(spans)
    forced = [i for i in range(len(units) - 1) if rules.heading_forced and units[i + 1][0] in heading_starts]
    return units, forced
