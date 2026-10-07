"""semantic: 이웃 문장 거리가 임계값을 넘는 곳에서 자르는 단순 의미 분할(LangChain SemanticChunker와 같은 계산, buffer_size 0).

  d_i = 1 - cos(e_i, e_{i+1})  (문장 i와 i+1의 임베딩 거리)
  tau = 문서 전체 d의 p 백분위수,  d_i > tau 인 경계를 자른다 (p가 클수록 청크가 커진다)
min/max 크기 제어는 하지 않는다(필요하면 chunking/sizectl 엔진). A(smart_a.py)는 이 청커에 "H1 섹션 경계는 항상 자른다"만 더한 것이다.
"""
import numpy as np

from chunking.base import Chunker, groups_from_cuts, register, spans_from_groups, whole_doc


def adjacent_distances(emb):
    return 1.0 - (emb[:-1] * emb[1:]).sum(1)


def semantic_groups(emb, p, forced=()):
    """거리 > p 백분위 임계값인 경계(+forced)에서 자른다 -> 그룹 [(a, b)]. 임계값은 문서 전체 거리 기준."""
    n = len(emb)
    if n <= 1:
        return [(0, n)] if n else []
    d = adjacent_distances(emb)
    tau = float(np.percentile(d, p))
    cuts = {i for i in range(n - 1) if d[i] > tau} | set(forced)
    return groups_from_cuts(n, cuts)


@register
class Semantic(Chunker):
    name, knob, units_kind = "semantic", "p", "sent"

    def forced(self, ctx, units):
        """항상 자를 경계 번호(단위 i와 i+1 사이). 서브클래스(A)가 덮어쓴다."""
        return []

    def chunk(self, ctx, p):
        u = ctx.units(self.units_kind)
        if not len(u):
            return whole_doc(ctx.text)
        return spans_from_groups(u.spans, semantic_groups(u.emb, p, self.forced(ctx, u)))
