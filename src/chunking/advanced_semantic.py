"""advanced_semantic_chunker: LumberChunker(Duarte et al., 2024)의 틀에서 'LLM이 내용 변화 지점을 고르는 부분'만 코사인 유사도 판단으로 바꾼다.

틀(LumberChunker 3.1절 그대로)
  s = 문서 시작 단위
  반복: G = s부터 단위를 더해 토큰 합이 θ를 넘을 때까지(θ를 넘게 만든 단위 포함)
        G 안에서 첫 단위를 빼고 앞에서부터 보며, 처음으로 '분명한 변화'가 나타나는 단위 j에서 자른다(청크 = [s, j), s = j)
        변화가 없으면 θ를 넘게 만든 단위 앞에서 자른다(그 단위부터 다시 시작). θ보다 큰 단위 하나는 혼자 청크가 된다.
'분명한 변화' 판단(criterion)
  maxmin  Max–Min semantic chunking(Kiss, Nagy, Szilágyi, Discover Computing 28:117, 2025)의 넣기 조건을 그대로 쓴다.
          C = 현재 청크 [s, j), min_sim(C) = C 안 단위 쌍의 최소 코사인, max_sim(j, C) = 단위 j와 C 단위들의 최대 코사인
          |C| = 1:  init_const · cos(j-1, j) > hard_thr 이면 이어짐
          |C| ≥ 2:  max_sim(j, C) > max(c · min_sim(C) · sigmoid(|C|), hard_thr) 이면 이어짐
          이어지지 않으면 = 분명한 변화. 기본값 c=0.9, init_const=1.5, hard_thr=0.6은 논문 권장값(BGE-base-en v1.5 기준)으로
          bge-m3에 맞춘 값이 아니다(파일럿에서 정한다).
  centroid_mean / centroid_std  AGENTS.md의 원래 구상(2026-10-12): 현재 청크 C의 중심 벡터(토큰 가중 평균, 길이 1) c와 단위 j를 비교한다.
          sim_j = cos(c, e_j)  (= j와 앞 단위 각각의 유사도의 가중 평균 / ‖가중 평균 벡터‖)
          청크 응집도: C의 각 단위 i를 '자기를 뺀 나머지'의 중심과 비교한 코사인 l_i(j도 중심에 들어가지 않았으니 같은 조건),
          μ_C·σ_C = l_i의 토큰 가중 평균·표준편차
          centroid_mean(후보 1):  sim_j < μ_C 이면 분명한 변화(파라미터 없음)
          centroid_std(후보 2):   sim_j < μ_C − k·σ_C 이면 분명한 변화(k 기본 0.5 = TextTiling(Hearst 1997)의 μ − σ/2, Solbiati et al. 2021은 k=1)
          |C| = 1이면 l_i가 없으므로 같은 식을 문서 안 이웃 단위 코사인 분포(평균 μ_adj, 표준편차 σ_adj)에 적용한다:
          cos(e_s, e_j) < μ_adj − k·σ_adj 이면 분명한 변화(후보 1은 k=0). 중앙값을 쓰면 첫 단계의 절반이 무조건 잘려서 평균·표준편차로 맞췄다.
이름: advanced_semantic = maxmin, advanced_semantic1 = centroid_mean, advanced_semantic2 = centroid_std (VARIANTS).
단위(unit): 'sentence' = chunking.sentences.split_text(표·수식은 한 단위), 'para' = 빈 줄로 나눈 문단(θ보다 큰 문단은 그 문단의 문장들로).
크기 θ는 bge-m3 토큰(count_tokens), 임베딩은 encode(길이 1 벡터, 코사인 = 내적).
"""
import math

import numpy as np

from chunking.sentences import paragraphs, split_text

MAXMIN_DEFAULTS = {"c": 0.9, "init_const": 1.5, "hard_thr": 0.6}     # Kiss et al. 2025 권장값(확정 아님)
EPS = 1e-6                                                              # 부동소수 오차 여유
CENTROID_STD_K = 0.5                                                    # Hearst 1997 TextTiling의 μ − σ/2
VARIANTS = {"advanced_semantic": "maxmin", "advanced_semantic1": "centroid_mean", "advanced_semantic2": "centroid_std"}


def make_units(text, lang, unit, count_tokens, theta):
    if unit == "sentence":
        return split_text(text, lang)
    if unit != "para":
        raise ValueError(f"unit은 'sentence' 또는 'para': {unit!r}")
    out = []
    paras = paragraphs(text)
    for p, n in zip(paras, count_tokens(paras)):
        out.extend(split_text(p, lang) if n > theta else [p])
    return out


class MaxMin:
    """Max–Min 넣기 조건. 청크가 바뀔 때 reset, 단위를 넣을 때 add."""

    def __init__(self, emb, c=MAXMIN_DEFAULTS["c"], init_const=MAXMIN_DEFAULTS["init_const"], hard_thr=MAXMIN_DEFAULTS["hard_thr"]):
        self.emb, self.c, self.init_const, self.hard_thr = emb, c, init_const, hard_thr

    def reset(self, s):
        self.members, self.min_sim = [s], None

    def fits(self, j):
        sims = self.emb[self.members] @ self.emb[j]
        if len(self.members) == 1:
            return self.init_const * float(sims[0]) > self.hard_thr
        thr = max(self.c * self.min_sim * (1 / (1 + math.exp(-len(self.members)))), self.hard_thr)
        return float(sims.max()) > thr

    def add(self, j):
        sims = self.emb[self.members] @ self.emb[j]
        m = float(sims.min())
        self.min_sim = m if self.min_sim is None else min(self.min_sim, m)
        self.members.append(j)


class Centroid:
    """중심 벡터 판단(centroid_mean: k=0, centroid_std: k>0). 기존 단위는 자기를 뺀 중심과 비교해 응집도를 잰다."""

    def __init__(self, emb, tokens, k=0.0):
        self.emb, self.w, self.k = emb, np.asarray(tokens, dtype=np.float64), k
        adj = np.sum(emb[:-1] * emb[1:], axis=1).astype(np.float64) if len(emb) > 1 else np.ones(1)
        self.first_ref = float(adj.mean() - k * adj.std())     # |C| = 1일 때 기준: 문서 안 이웃 단위 코사인 분포에 같은 식

    def reset(self, s):
        self.members = [s]
        self.m = self.w[s] * self.emb[s].astype(np.float64)

    @staticmethod
    def _cos(v, e):
        n = np.linalg.norm(v)
        return float(v @ e / n) if n > 0 else 0.0

    def fits(self, j):
        e = self.emb[j].astype(np.float64)
        if len(self.members) == 1:
            return float(self.emb[self.members[0]] @ self.emb[j]) >= self.first_ref - EPS
        sim = self._cos(self.m, e)
        loo = np.array([self._cos(self.m - self.w[i] * self.emb[i], self.emb[i].astype(np.float64)) for i in self.members])
        w = self.w[self.members]
        mu = float(np.average(loo, weights=w))
        sd = float(np.sqrt(np.average((loo - mu) ** 2, weights=w)))
        return sim >= mu - self.k * sd - EPS

    def add(self, j):
        self.members.append(j)
        self.m = self.m + self.w[j] * self.emb[j].astype(np.float64)


def make_judge(criterion, tokens, emb, params):
    if criterion == "maxmin":
        return MaxMin(emb, **params)
    if criterion == "centroid_mean":
        return Centroid(emb, tokens, k=0.0)
    if criterion == "centroid_std":
        return Centroid(emb, tokens, k=params.get("k", CENTROID_STD_K))
    raise ValueError(f"criterion은 maxmin / centroid_mean / centroid_std: {criterion!r}")


def chunk_ranges(tokens, emb, theta, criterion="maxmin", **params):
    """단위별 토큰 수·임베딩 → 청크 단위 범위 [(i, j)] (j 미포함)."""
    criterion = VARIANTS.get(criterion, criterion)
    judge = make_judge(criterion, tokens, emb, params)
    n, s, out = len(tokens), 0, []
    while s < n:
        e, tot = s, 0
        while e < n and tot <= theta:              # G = [s, e): θ를 넘게 만든 단위 e-1까지
            tot += tokens[e]
            e += 1
        over = tot > theta
        judge.reset(s)
        cut = None
        for j in range(s + 1, e - 1 if over else e):   # θ를 넘게 만든 단위는 '변화 없음'일 때 어차피 다음 청크 시작
            if not judge.fits(j):
                cut = j
                break
            judge.add(j)
        if cut is None:
            cut = (e - 1 if e - 1 > s else e) if over else e
        out.append((s, cut))
        s = cut
    return out


def chunk(text, encode, count_tokens, theta, lang="en", unit="sentence", criterion="maxmin", return_ranges=False, **params):
    """텍스트 → 청크 문자열 목록(return_ranges면 (청크 목록, 단위 목록, 범위 목록)).
    encode: 글 목록 → 길이 1 벡터, count_tokens: 글 목록 → 토큰 수 목록, theta: 그룹 토큰 상한."""
    units = make_units(text, lang, unit, count_tokens, theta)
    if not units:
        return ([], [], []) if return_ranges else []
    tokens = count_tokens(units)
    emb = np.asarray(encode(units), dtype=np.float32)
    ranges = chunk_ranges(tokens, emb, theta, criterion, **params)
    sep = " " if unit == "sentence" else "\n\n"
    chunks = [sep.join(units[i:j]) for i, j in ranges]
    return (chunks, units, ranges) if return_ranges else chunks
