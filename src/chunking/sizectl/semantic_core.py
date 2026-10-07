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
  5' DP 크기 제어 (size_control="dp"일 때 2~5 대신. semantic·smart 둘 다 기본) - 순수 함수 size_dp.partition을 호출한다
           경계 i를 자르면 gain_i = 거리_i - 임계값 (+ 구조 계층이 준 보정)을 얻는다. 목표는 사전식: max 초과 금지(하드, 단일 단위 예외)
           > min 부족분 합 최소화 > gain 합 최대화. min_weight 같은 튜닝 값이 없고, 후처리 병합도 필요 없다
           (최적이면 "합치면 위반이 줄어드는 이웃"이 남아 있을 수 없다). 제약이 안 걸리면 "거리 > 임계값"에서만 자르는 결과와 같다.
           forced(제목 앞 강제 절단 어블레이션)는 하드 제약이 아니라 gain 보정(FORCED_BONUS)이라서 min이 우선한다.
청크는 평문의 리터럴 슬라이스(text[start:end])다 - LangChain처럼 문장을 다시 이어 붙이지 않으므로 위치 복원이 필요 없다.
"""
import re

import numpy as np

from chunking.sizectl.size_dp import partition

FORCED_BONUS = 1000.0       # forced 경계의 gain 보정. 거리 점수(0~1)보다 훨씬 크지만 min 부족분(토큰 단위)보다 사전식에서 아래다

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
                 size_control="greedy"):
        self.embed, self.count = embed_model, count_tokens
        self.pct, self.min_t, self.max_t, self.buffer = percentile, min_tokens, max_tokens, buffer_size
        if size_control not in ("greedy", "dp"):
            raise ValueError(f"size_control must be 'greedy' or 'dp', got {size_control!r}")
        self.size_control = size_control
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

    # -- 진입점 -----------------------------------------------------------
    def chunk_groups(self, text, groups):
        """DP 전용. groups = [(units, forced)] 섹션 묶음 목록(문서 순서). 묶음끼리는 서로 독립이다(청크가 묶음을 가로지르지 않는다).
        임계값 τ는 모든 묶음 안의 이웃 거리 전체에서 구한다(문서 기준) - 묶음 사이 경계는 거리 계산에서 뺀다.
        묶음마다 size_dp.partition(순수 DP)을 부른다. forced 경계는 gain에 FORCED_BONUS를 더할 뿐이라 min 우선 목표를 깨지 않는다."""
        if self.size_control != "dp":
            raise ValueError("section groups need size_control='dp'")
        groups = [(u, set(f)) for u, f in groups if u]
        dists = [consecutive_distances(self.embed, [text[s:e] for s, e in u], self.buffer) if len(u) > 1 else [] for u, _ in groups]
        flat = [x for d in dists for x in d]
        tau = float(np.percentile(flat, self.pct)) if flat else float("inf")
        spans, tot = [], {"units": 0, "chunks": 0, "forced_cuts": 0, "threshold_cuts": 0, "size_cuts": 0,
                          "under_min": 0, "over_max": 0, "over_max_single_unit": 0}
        shortfall = gain = 0
        for (units, forced), d in zip(groups, dists):
            n = len(units)
            tot["units"] += n
            if n == 1:                                           # 단위 하나뿐인 묶음(짧은 섹션 등): 그대로 한 청크, 크기 위반은 여전히 센다
                spans.append(units[0])
                size = self.count(text[units[0][0]:units[0][1]])
                tot["chunks"] += 1
                tot["under_min"] += size < self.min_t
                tot["over_max_single_unit"] += size > self.max_t
                continue
            self._prepare_sizes(text, units)
            gains = [d[i] - tau + (FORCED_BONUS if i in forced else 0.0) for i in range(n - 1)]
            merged, (sf, gn) = partition(n, self._size, gains, self.min_t, self.max_t)
            shortfall += sf
            gain += gn
            cuts = [a - 1 for a, _ in merged[1:]]
            tot["chunks"] += len(merged)
            tot["forced_cuts"] += sum(1 for i in cuts if i in forced)
            tot["threshold_cuts"] += sum(1 for i in cuts if i not in forced and d[i] > tau)
            tot["size_cuts"] += sum(1 for i in cuts if i not in forced and d[i] <= tau)       # 임계값 아래인데 크기 때문에 자른 곳
            tot["under_min"] += sum(1 for a, b in merged if self._size(a, b) < self.min_t)
            tot["over_max"] += sum(1 for a, b in merged if self._size(a, b) > self.max_t and b - a > 1)
            tot["over_max_single_unit"] += sum(1 for a, b in merged if self._size(a, b) > self.max_t and b - a == 1)
            spans += [(units[a][0], units[b - 1][1]) for a, b in merged]
        tot.update(min_shortfall=int(shortfall), section_groups=len(groups))
        self.stats = tot
        return spans

    def chunk_spans(self, text, units, forced=()):
        """units: [(start, end)], forced: 반드시 자를 경계 번호(i = 단위 i와 i+1 사이). -> [(start, end)] 청크 구간."""
        n = len(units)
        if n <= 1:
            self.stats = {"units": n, "chunks": n, "under_min": 0, "over_max": 0}
            return list(units)
        forced = set(forced)
        if self.size_control == "dp":
            return self.chunk_groups(text, [(units, forced)])
        self._prepare_sizes(text, units)
        groups, d, tau = self.breakpoint_groups(text, units, forced)
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
