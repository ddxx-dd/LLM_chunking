"""E (중심 병합): 문장 하나씩 따로 시작해서 가장 비슷한 이웃 덩어리 쌍부터 합치고, 가장 비슷한 쌍도 theta보다 덜 비슷하면 멈춘다.

  덩어리 중심 v_C = (sum t_i e_i) / ||sum t_i e_i||   (t_i = 문장의 토큰 수. 토큰 가중 평균을 정규화)
  실효 유사도 s_eff(L,R) = v_L . v_R - pen(R)          pen(R) = delta * decay^(수준-1)  (오른쪽 덩어리가 제목으로 시작할 때, 아니면 0)
  theta = 초기 이웃 코사인들의 (100-p) 백분위. p가 클수록 theta가 낮아져 더 오래 합치고 청크가 커진다.
  delta = 초기 이웃 코사인 표준편차의 delta_sigma배(기본 1). 절대값 delta를 직접 줄 수도 있다.
가장 큰 s_eff 쌍부터(동률은 왼쪽 먼저) 합치므로 결과가 문장 순서와 무관한 전역 탐욕이고, 중심으로 비교해서 사슬 병합이 늦게 끊긴다.
delta, decay와 theta의 백분위 정의는 논문 근거가 없는 자체 설계 값이다.
"""
import heapq

import numpy as np

from chunking.base import Chunker, register, spans_from_groups, whole_doc


def merge_blocks(emb, own, levels, theta, delta, decay=0.5):
    """-> 그룹 [(a, b)]. 가장 큰 실효 유사도가 theta 미만이면 멈춘다."""
    n = len(emb)
    if n <= 1:
        return [(0, n)] if n else []
    vs = emb.astype(np.float64) * np.asarray(own, dtype=np.float64)[:, None]
    cen = vs / np.maximum(np.linalg.norm(vs, axis=1, keepdims=True), 1e-12)
    nxt = list(range(1, n)) + [-1]
    prv = [-1] + list(range(n - 1))
    last = list(range(n))                       # 덩어리의 마지막 단위 번호
    alive = [True] * n
    ver = [0] * n

    def pen(r):
        return delta * decay ** (int(levels[r]) - 1) if levels[r] > 0 else 0.0

    def eff(l, r):
        return float(cen[l] @ cen[r]) - pen(r)

    heap = [(-eff(i, i + 1), i, 0, 0) for i in range(n - 1)]
    heapq.heapify(heap)
    while heap:
        neg, l, vl, vr = heapq.heappop(heap)
        r = nxt[l] if alive[l] else -1
        if r < 0 or ver[l] != vl or ver[r] != vr:
            continue                            # 낡은 항목(그 사이 병합으로 바뀜)
        if -neg < theta:
            break
        vs[l] += vs[r]
        cen[l] = vs[l] / max(np.linalg.norm(vs[l]), 1e-12)
        alive[r] = False
        nxt[l] = nxt[r]
        if nxt[r] >= 0:
            prv[nxt[r]] = l
        last[l] = last[r]
        ver[l] += 1
        if prv[l] >= 0:
            heapq.heappush(heap, (-eff(prv[l], l), prv[l], ver[prv[l]], ver[l]))
        if nxt[l] >= 0:
            heapq.heappush(heap, (-eff(l, nxt[l]), l, ver[l], ver[nxt[l]]))
    return [(i, last[i] + 1) for i in range(n) if alive[i]]


def e_groups(emb, own, levels, p, delta=None, delta_sigma=1.0, decay=0.5):
    n = len(emb)
    if n <= 1:
        return [(0, n)] if n else []
    sims = (emb[:-1] * emb[1:]).sum(1)
    theta = float(np.percentile(sims, 100 - p))
    if delta is None:
        delta = delta_sigma * float(sims.std())
    return merge_blocks(emb, own, levels, theta, delta, decay)


@register
class E(Chunker):
    name, knob, units_kind = "E", "p", "sentg"

    def chunk(self, ctx, p, delta=None, delta_sigma=1.0, decay=0.5):
        u = ctx.units(self.units_kind)
        if not len(u):
            return whole_doc(ctx.text)
        return spans_from_groups(u.spans, e_groups(u.emb, u.ntok, ctx.levels(u.spans), p, delta, delta_sigma, decay))
