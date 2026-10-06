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
  단위 하나가 max를 넘으면 더 쪼갤 수 없으므로 그대로 둔다(표 등). 지킬 수 없는 min/max는 위반으로 세어 stats에 남긴다.
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
    def __init__(self, embed_model, count_tokens, percentile=90, min_tokens=100, max_tokens=400, buffer_size=0):
        self.embed, self.count = embed_model, count_tokens
        self.pct, self.min_t, self.max_t, self.buffer = percentile, min_tokens, max_tokens, buffer_size
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
    def chunk_spans(self, text, units, forced=()):
        """units: [(start, end)], forced: 반드시 자를 경계 번호(i = 단위 i와 i+1 사이). -> [(start, end)] 청크 구간."""
        n = len(units)
        if n <= 1:
            self.stats = {"units": n, "chunks": n, "under_min": 0, "over_max": 0}
            return list(units)
        self._prepare_sizes(text, units)
        forced = set(forced)
        groups, d, _tau = self.breakpoint_groups(text, units, forced)
        split = self._split_oversized(groups, d)
        merged, across = self._merge_small(split, forced)
        self.stats = {
            "units": n, "chunks": len(merged),
            "forced_cuts": sum(1 for a, _ in groups[1:] if (a - 1) in forced),
            "threshold_cuts": len(groups) - 1 - sum(1 for a, _ in groups[1:] if (a - 1) in forced),
            "oversize_splits": len(split) - len(groups), "merges": len(split) - len(merged),
            "merges_across_forced": across,
            "under_min": sum(1 for a, b in merged if self._size(a, b) < self.min_t),
            "over_max": sum(1 for a, b in merged if self._size(a, b) > self.max_t and b - a > 1)}
        return [(units[a][0], units[b - 1][1]) for a, b in merged]
