"""B (응집도 DP): 청크 안이 얼마나 일관된지(응집도)와 청크당 벌점 lam을 합친 총비용이 가장 작은 나누기를 DP로 전역 최적화한다.

  응집도 비용 c(a,b) = m - ||S||^2 / m   (청크 [a,b)의 문장 m개, S = 임베딩 합. 평균에서의 제곱편차 합과 같은 값, Alemi & Ginsparg 2015 식 19)
  총비용 = sum_청크 [c(a,b) + lam] - sum_자른곳 bonus          f[j] = min_a ( f[a] + c(a,j) + lam - bonus[a] )
  lam   = 경계별 응집도 개선량 Delta_i(= 좌우 w문장 창의 Ward 거리)의 p 백분위. p가 클수록 lam이 커서 청크가 커진다.
  bonus = beta * lam * decay^(수준-1)  (문장이 제목으로 시작할 때, 그 앞에서 자르면 비용이 줄어든다. 하드 규칙이 아니라 선호)
구간 후보 길이에는 계산량 한도 cap(토큰)이 있다. 크기 제약이 아니라 O(n x 창) 한도이며, 단일 단위는 예외다.
beta, decay, w는 논문 근거가 없는 자체 설계 값이다(기본값 1, 0.5, 5).
"""
import numpy as np

from chunking.base import Chunker, register, spans_from_groups, whole_doc

CAP = 3000


def local_deltas(emb, w=5):
    """경계마다 좌우 w문장 창의 응집도 개선량 Delta = mL*mR/(mL+mR) * ||muL - muR||^2 (누적합으로 벡터화)."""
    n = len(emb)
    if n < 2:
        return np.zeros(0)
    P = np.vstack([np.zeros((1, emb.shape[1])), np.cumsum(emb.astype(np.float64), 0)])
    i = np.arange(n - 1)
    lo, hi = np.maximum(0, i - w + 1), np.minimum(n, i + 1 + w)
    mL, mR = (i + 1 - lo).astype(float), (hi - (i + 1)).astype(float)
    muL = (P[i + 1] - P[lo]) / mL[:, None]
    muR = (P[hi] - P[i + 1]) / mR[:, None]
    return mL * mR / (mL + mR) * ((muL - muR) ** 2).sum(1)


def bonus_array(levels, lam, beta, decay):
    lv = np.asarray(levels)
    return np.where(lv > 0, beta * lam * decay ** np.maximum(lv - 1, 0), 0.0)


def coherence_dp(emb, own, bonus, lam, cap=CAP):
    """-> 그룹 [(a, b)]. 비용 = sum[m - ||sum e||^2/m + lam] - sum bonus[a](a>0). cap: 한 청크의 후보 길이 상한(토큰, 단일 단위는 예외)."""
    n = len(emb)
    if n == 0:
        return []
    P = np.vstack([np.zeros((1, emb.shape[1])), np.cumsum(emb.astype(np.float64), 0)])
    pt = np.concatenate([[0], np.cumsum(np.asarray(own, dtype=np.float64))])
    f = np.full(n + 1, np.inf)
    f[0] = 0.0
    prev = np.zeros(n + 1, dtype=np.int64)
    lo = 0
    for j in range(1, n + 1):
        while lo < j - 1 and pt[j] - pt[lo] > cap:
            lo += 1
        a = np.arange(lo, j)
        m = (j - a).astype(np.float64)
        S = P[j] - P[a]
        cost = f[a] + m - (S * S).sum(1) / m + lam - np.where(a > 0, bonus[a], 0.0)
        k = int(np.argmin(cost))                       # 동점이면 가장 왼쪽 a(더 긴 마지막 청크)
        f[j], prev[j] = cost[k], a[k]
    groups, j = [], n
    while j > 0:
        groups.append((int(prev[j]), j))
        j = int(prev[j])
    return groups[::-1]


def b_groups(emb, own, levels, p, beta=1.0, decay=0.5, w=5, cap=CAP):
    n = len(emb)
    if n <= 1:
        return [(0, n)] if n else []
    lam = float(np.percentile(local_deltas(emb, w), p))
    return coherence_dp(emb, own, bonus_array(levels, lam, beta, decay), lam, cap)


@register
class B(Chunker):
    name, knob, units_kind = "B", "p", "sentg"

    def chunk(self, ctx, p, beta=1.0, decay=0.5, w=5, cap=CAP):
        u = ctx.units(self.units_kind)
        if not len(u):
            return whole_doc(ctx.text)
        return spans_from_groups(u.spans, b_groups(u.emb, u.ntok, ctx.levels(u.spans), p, beta, decay, w, cap))
