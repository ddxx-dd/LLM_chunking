"""의미 분할 공유 엔진 (docs/restructure_plan.md 5절). semantic과 smart가 이 엔진을 그대로 쓴다.

  semantic = 엔진(단위 = 평문 전체를 문장 정규식으로 나눈 것,          forced 없음)
  smart    = 엔진(단위 = 제약 규칙으로 만든 것(units.build_units),      forced = 제목 앞)

엔진은 구조(제목·표·캡션)를 모른다. 입력은 평문, 단위 구간 [(start, end)], 반드시 자를 경계 번호(forced)뿐이고,
제약은 단위를 어떻게 만들었는지(units.py)로만 표현된다. 그래서 두 청커의 차이는 입력 단위의 차이뿐이다.

알고리즘 (LangChain SemanticChunker와 같은 계산 + 우리 크기 정리)
  1 거리   단위마다 앞뒤 buffer_size개 단위를 " "로 이어 붙인 글을 bge-m3로 임베딩 -> 연속한 두 임베딩의 코사인 거리
           기본은 buffer_size=0(단위 하나씩 임베딩). 1이면 LangChain SemanticChunker 기본과 같다. 이웃을 결합하면 경계 양쪽 내용이
           두 임베딩에 모두 섞여 진짜 경계의 대비가 줄고(봉우리가 낮아짐) 양옆에 어깨가 생겨서 0을 기본으로 했다.
  2 임계값  모든 거리의 percentile
  3 컷     거리 > 임계값(초과)인 경계 + forced 경계
  4 쪼개기  max 토큰을 넘는 그룹은 그 안의 경계 중 거리가 가장 큰 곳에서 자른다(양쪽이 min 이상인 곳을 우선, 없으면 전체)
  5 합치기  min 토큰 미만 그룹은 다음 그룹과 합친다(합쳐서 max 이하일 때. 안 되면 앞 그룹). forced 경계도 min 미만이면 합친다.
  5' DP 크기 제어 (size_control="dp"일 때 2~5 대신. smart 설정에서만 쓴다 - semantic은 위의 그리디 2~5를 그대로 쓴다)
           경계 i를 자르면 r_i = 거리_i - 임계값 점수를 얻는 최적 분할을 동적계획법으로 한 번에 구한다.
           하드 제약: forced 경계는 반드시 자르고(청크가 포함할 수 없다), 청크는 max 이하(단위 하나가 max를 넘는 경우만 예외).
           소프트 제약: min 미만이면 min_weight x (min - 크기)/min 만큼 벌점(약하게). 제약이 안 걸리면 "거리 > 임계값"에서만 자르는 결과와 같다.
           그다음 min 미만으로 남은 청크는 forced가 아닌 앞뒤 이웃 중 청크 임베딩 코사인 유사도가 높은 쪽과 합친다(합쳐서 max 이하일 때만).
           둘 다 안 되면 그대로 두고 under_min으로 보고한다. forced 경계는 합치기로도 넘지 않는다.
청크는 평문의 리터럴 슬라이스(text[start:end])다 - LangChain처럼 문장을 다시 이어 붙이지 않으므로 위치 복원이 필요 없다.
"""
import re

import numpy as np

SENTENCE_SPLIT_REGEX = r"(?<=[.?!。])\s+|\n+"      # LangChain SemanticChunker 설정과 같다(줄바꿈에서도 나눔)
_SPLIT = re.compile(SENTENCE_SPLIT_REGEX)


def sentence_spans(text, start=0, end=None):
    """[start, end) 구간을 문장 (start, end)로 나눈다. 오프셋은 원문 기준이고, 각 구간은 앞뒤 공백을 뺀다."""
    end = len(text) if end is None else end
    cuts, cur = [], start
    for m in _SPLIT.finditer(text, start, end):
        cuts.append((cur, m.start()))
        cur = m.end()
    cuts.append((cur, end))
    out = []
    for s, e in cuts:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            out.append((s, e))
    return out


def combined_texts(texts, buffer_size=0):
    """단위마다 앞뒤 buffer_size개를 " "로 이어 붙인다 (LangChain combine_sentences와 같다)."""
    n = len(texts)
    return [" ".join(texts[max(0, i - buffer_size): i + buffer_size + 1]) for i in range(n)]


def consecutive_distances(embed_model, texts, buffer_size=0):
    """연속한 두 단위의 코사인 거리 n-1개. embed_model.encode(글 목록, normalize_embeddings=True) -> 행렬."""
    vecs = np.asarray(embed_model.encode(combined_texts(texts, buffer_size), normalize_embeddings=True), dtype=float)
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)     # sklearn cosine_similarity와 같은 계산
    return [float(1 - c) for c in (vecs[:-1] * vecs[1:]).sum(1)]


class SemanticEngine:
    def __init__(self, embed_model, count_tokens, percentile=90, min_tokens=100, max_tokens=400, buffer_size=0,
                 size_control="greedy", min_weight=0.5):
        self.embed, self.count = embed_model, count_tokens
        self.pct, self.min_t, self.max_t, self.buffer = percentile, min_tokens, max_tokens, buffer_size
        if size_control not in ("greedy", "dp"):
            raise ValueError(f"size_control must be 'greedy' or 'dp', got {size_control!r}")
        self.size_control, self.min_w = size_control, min_weight
        self.stats = {}

    # -- 크기 (합친 텍스트 기준) -----------------------------------------
    def _prepare_sizes(self, text, units):
        n = len(units)
        self._gap = [self.count(text[units[i][0]:units[i + 1][0]]) for i in range(n - 1)] + [0]   # 단위 + 뒤 공백
        self._own = [self.count(text[s:e]) for s, e in units]

    def _size(self, a, b):
        """단위 a..b-1을 이어 붙인 글의 토큰 수(근사: 마지막 단위 뒤 공백은 뺀다)."""
        return sum(self._gap[a:b - 1]) + self._own[b - 1]

    # -- 1~3: 경계 ------------------------------------------------------
    def breakpoint_groups(self, text, units, forced=()):
        """크기 정리 전 그룹 [(a, b)] (단위 번호 a..b-1), 거리, 임계값. 동치 테스트와 chunk_spans가 쓴다."""
        n = len(units)
        if n <= 1:
            return ([(0, n)] if n else []), [], float("inf")
        d = consecutive_distances(self.embed, [text[s:e] for s, e in units], self.buffer)
        tau = float(np.percentile(d, self.pct))
        forced = set(forced)
        groups, a = [], 0
        for i in range(n - 1):
            if i in forced or d[i] > tau:
                groups.append((a, i + 1))
                a = i + 1
        groups.append((a, n))
        return groups, d, tau

    # -- 4: 쪼개기 ------------------------------------------------------
    def _split_oversized(self, groups, d):
        out, stack = [], list(reversed(groups))
        while stack:
            a, b = stack.pop()
            if b - a == 1 or self._size(a, b) <= self.max_t:
                out.append((a, b))
                continue
            inner = range(a, b - 1)
            both = [i for i in inner if self._size(a, i + 1) >= self.min_t and self._size(i + 1, b) >= self.min_t]
            cut = max(both or inner, key=lambda i: d[i])          # min을 지킬 수 없으면 max를 우선한다
            stack.append((cut + 1, b))
            stack.append((a, cut + 1))
        return out

    # -- 5: 합치기 ------------------------------------------------------
    def _merge_small(self, groups, forced):
        groups, across, changed = list(groups), 0, True
        while changed:
            changed = False
            for k, (a, b) in enumerate(groups):
                if self._size(a, b) >= self.min_t:
                    continue
                if k + 1 < len(groups) and self._size(a, groups[k + 1][1]) <= self.max_t:
                    across += (b - 1) in forced
                    groups[k:k + 2] = [(a, groups[k + 1][1])]
                elif k > 0 and self._size(groups[k - 1][0], b) <= self.max_t:
                    across += (a - 1) in forced
                    groups[k - 1:k + 1] = [(groups[k - 1][0], b)]
                else:
                    continue
                changed = True
                break
        return groups, across

    # -- 5': DP 크기 제어 (smart 전용 옵션) -------------------------------------
    def _dp_groups(self, n, d, tau, forced):
        """단위 n개를 forced/max를 지키면서 점수(자른 경계의 거리-임계값 합 - min 벌점)가 최대가 되게 나눈다 -> [(a, b)]."""
        f, prev = [float("-inf")] * (n + 1), [0] * (n + 1)
        f[0] = 0.0
        for j in range(1, n + 1):
            for a in range(j - 1, -1, -1):
                size = self._size(a, j)
                if a < j - 1 and (a in forced or size > self.max_t):    # 더 왼쪽으로 늘려도 forced를 품거나 더 커질 뿐이다
                    break
                cut_gain = 0.0 if a == 0 or (a - 1) in forced else d[a - 1] - tau
                gain = f[a] + cut_gain - self.min_w * max(0, self.min_t - size) / self.min_t
                if gain > f[j]:
                    f[j], prev[j] = gain, a
        groups, j = [], n
        while j > 0:
            groups.append((prev[j], j))
            j = prev[j]
        return groups[::-1], f[n]

    def _chunk_vec(self, text, units, g):
        v = np.asarray(self.embed.encode([text[units[g[0]][0]:units[g[1] - 1][1]]], normalize_embeddings=True), dtype=float)[0]
        return v / max(np.linalg.norm(v), 1e-12)

    def _merge_similar(self, groups, forced, text, units):
        """min 미만 그룹을 forced가 아니고 합쳐서 max 이하인 이웃 중 청크 유사도가 높은 쪽과 합친다. -> (그룹, 합친 횟수)."""
        groups, stuck, merges = list(groups), set(), 0
        while True:
            for k, (a, b) in enumerate(groups):
                if self._size(a, b) >= self.min_t or (a, b) in stuck:
                    continue
                nbrs = []
                if k > 0 and (a - 1) not in forced and self._size(groups[k - 1][0], b) <= self.max_t:
                    nbrs.append(k - 1)
                if k + 1 < len(groups) and (b - 1) not in forced and self._size(a, groups[k + 1][1]) <= self.max_t:
                    nbrs.append(k + 1)
                if not nbrs:
                    stuck.add((a, b))
                    continue
                if len(nbrs) == 2:
                    me = self._chunk_vec(text, units, (a, b))
                    j = max(nbrs, key=lambda j: float(me @ self._chunk_vec(text, units, groups[j])))
                else:
                    j = nbrs[0]
                lo, hi = sorted((k, j))
                groups[lo:hi + 1] = [(groups[lo][0], groups[hi][1])]
                merges += 1
                break
            else:
                return groups, merges

    # -- 진입점 -----------------------------------------------------------
    def chunk_spans(self, text, units, forced=()):
        """units: [(start, end)], forced: 반드시 자를 경계 번호(i = 단위 i와 i+1 사이). -> [(start, end)] 청크 구간."""
        n = len(units)
        if n <= 1:
            self.stats = {"units": n, "chunks": n, "under_min": 0, "over_max": 0}
            return list(units)
        self._prepare_sizes(text, units)
        forced = set(forced)
        groups, d, tau = self.breakpoint_groups(text, units, forced)
        if self.size_control == "dp":
            dp, score = self._dp_groups(n, d, tau, forced)
            merged, merges = self._merge_similar(dp, forced, text, units)
            cuts = [a - 1 for a, _ in merged[1:]]
            self.stats = {
                "units": n, "chunks": len(merged),
                "forced_cuts": sum(1 for i in cuts if i in forced),
                "threshold_cuts": sum(1 for i in cuts if i not in forced and d[i] > tau),
                "size_cuts": sum(1 for i in cuts if i not in forced and d[i] <= tau),      # 임계값 아래인데 크기 때문에 자른 곳
                "merges": merges, "merges_across_forced": 0, "dp_score": round(score, 6),
                "under_min": sum(1 for a, b in merged if self._size(a, b) < self.min_t),
                "over_max": sum(1 for a, b in merged if self._size(a, b) > self.max_t and b - a > 1),
                "over_max_single_unit": sum(1 for a, b in merged if self._size(a, b) > self.max_t and b - a == 1)}
            return [(units[a][0], units[b - 1][1]) for a, b in merged]
        split = self._split_oversized(groups, d)
        merged, across = self._merge_small(split, forced)
        self.stats = {
            "units": n, "chunks": len(merged),
            "forced_cuts": sum(1 for a, _ in groups[1:] if (a - 1) in forced),
            "threshold_cuts": len(groups) - 1 - sum(1 for a, _ in groups[1:] if (a - 1) in forced),
            "oversize_splits": len(split) - len(groups), "merges": len(split) - len(merged),
            "merges_across_forced": across,
            "under_min": sum(1 for a, b in merged if self._size(a, b) < self.min_t),
            "over_max": sum(1 for a, b in merged if self._size(a, b) > self.max_t and b - a > 1),
            "over_max_single_unit": sum(1 for a, b in merged if self._size(a, b) > self.max_t and b - a == 1)}
        return [(units[a][0], units[b - 1][1]) for a, b in merged]
