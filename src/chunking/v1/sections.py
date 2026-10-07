"""최상위 섹션 분할 (구조 계층, smart 규칙 section_split). docs/restructure_plan.md 5절.

  최상위 섹션 수준 L = 문서에서 제목이 2개 이상 있는 가장 얕은 수준(# = 1). 문서 제목 하나만 #인 docx는 ##로 내려간다.
  수준이 L 이하인 제목이 새 섹션을 시작한다. 제목이 한 수준에 하나씩뿐이거나 없으면 분할하지 않는다(섹션 하나 = 문서 전체).
  섹션은 엔진이 서로 독립으로 처리한다(청크가 섹션을 가로지르지 않는다). 예외는 아래 병합뿐이다.
  짧은 섹션(min 토큰 미만)은 합쳐서 max 이하가 되는 앞뒤 이웃 섹션 중 임베딩 코사인 유사도가 높은 쪽과 합친다.
  둘 다 안 되면 그대로 두고(under_min으로 보고) 합친 횟수는 통계로 남긴다 - 합쳐진 섹션은 주제가 섞일 수 있다는 한계를 보고하기 위해서다.
"""
import numpy as np


def section_level(doc, pieces):
    """제목이 2개 이상 있는 가장 얕은 수준. 없으면 None."""
    blocks = {b["id"]: b for b in doc["blocks"]}
    levels = [blocks[p.block_ids[0]].get("level", 1) for p in pieces if p.kind == "heading" and p.block_ids and p.block_ids[0] in blocks]
    for lv in sorted(set(levels)):
        if levels.count(lv) >= 2:
            return lv
    return None


def split_sections(doc, pieces):
    """-> (섹션 목록 [(lo, hi)] (조각 번호 범위, hi 미포함), 수준 L)."""
    n = len(pieces)
    lvl = section_level(doc, pieces)
    if lvl is None or n == 0:
        return ([(0, n)] if n else []), lvl
    blocks = {b["id"]: b for b in doc["blocks"]}
    starts = [k for k, p in enumerate(pieces) if p.kind == "heading" and p.block_ids and blocks.get(p.block_ids[0], {}).get("level", 1) <= lvl]
    if not starts or starts[0] != 0:
        starts = [0] + starts                                         # 첫 제목 앞의 내용(머리말)은 별도 섹션
    return [(a, b) for a, b in zip(starts, starts[1:] + [n])], lvl


def merge_short_sections(text, pieces, sections, embed, count, min_tokens, max_tokens):
    """min 미만 섹션을 유사한 이웃과 합친다 -> (묶음 목록 [(lo, hi)], 합친 횟수). 합쳐서 max를 넘는 이웃은 대상이 아니다."""
    groups, stuck, merges = [list(s) for s in sections], set(), 0
    span = lambda g: text[pieces[g[0]].start:pieces[g[1] - 1].end]
    size = lambda g: count(span(g))

    def vec(g):
        v = np.asarray(embed.encode([span(g)], normalize_embeddings=True), dtype=float)[0]
        return v / max(np.linalg.norm(v), 1e-12)

    while True:
        for k, g in enumerate(groups):
            if tuple(g) in stuck or size(g) >= min_tokens:
                continue
            nbrs = []
            if k > 0 and size([groups[k - 1][0], g[1]]) <= max_tokens:
                nbrs.append(k - 1)
            if k + 1 < len(groups) and size([g[0], groups[k + 1][1]]) <= max_tokens:
                nbrs.append(k + 1)
            if not nbrs:
                stuck.add(tuple(g))
                continue
            if len(nbrs) == 2:
                me = vec(g)
                j = max(nbrs, key=lambda j: float(me @ vec(groups[j])))
            else:
                j = nbrs[0]
            lo, hi = sorted((k, j))
            groups[lo:hi + 1] = [[groups[lo][0], groups[hi][1]]]
            merges += 1
            break
        else:
            return [tuple(g) for g in groups], merges
