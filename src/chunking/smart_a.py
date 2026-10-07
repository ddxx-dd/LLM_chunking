"""A (구조 -> 의미): 최상위 섹션(H1) 경계는 거리와 무관하게 항상 자르고, 섹션 안에서는 semantic과 같은 임계값으로 자른다.

  최상위 섹션 수준 L = 문서에서 제목이 2개 이상 있는 가장 얕은 수준(# = 1). 문서 제목 하나만 #인 docx는 ##로 내려간다.
  수준이 L 이하인 제목이 새 섹션을 시작한다. 제목이 한 수준에 하나뿐이거나 없으면 구조 경계가 없다(semantic과 같아짐).
  단위는 "sentg"(제목 붙임, 표 한 덩어리, 캡션 붙임). 임계값 tau는 문서 전체 거리 기준(섹션마다 따로 구하지 않음).
  짧은 섹션 병합이나 크기 제어는 하지 않는다.
"""
from collections import Counter

from chunking.base import register
from chunking.semantic import Semantic


def section_level(head_level):
    """제목 수준 목록(시작 위치 -> 수준)에서 제목이 2개 이상인 가장 얕은 수준. 없으면 None."""
    counts = Counter(head_level.values())
    for lv in sorted(counts):
        if counts[lv] >= 2:
            return lv
    return None


def section_starts(ctx):
    """새 섹션이 시작하는 조각의 start 집합(문서 맨 앞 조각은 제외)."""
    lvl = section_level(ctx.head_level)
    if lvl is None or not ctx.pieces:
        return set()
    first = ctx.pieces[0].start
    return {s for s, lv in ctx.head_level.items() if lv <= lvl and s != first}


@register
class A(Semantic):
    name, units_kind = "A", "sentg"

    def forced(self, ctx, units):
        starts = section_starts(ctx)
        return [i for i in range(len(units) - 1) if units.spans[i + 1][0] in starts]
