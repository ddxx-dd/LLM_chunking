"""예전 smart 청커 (legacy) - 기본 경로에서는 쓰지 않는다. docs/restructure_plan.md 5절 참고.

현재 smart는 의미 분할 공유 엔진 + 구조 제약 규칙이다(semantic_core.py, units.py). 이 파일은 두 가지를 보관한다.

  SmartChunker             제목으로 섹션을 먼저 나누고, 섹션이 max를 넘을 때만 의미(경계 점수)로 자른다.
                           srt 설정(algorithm="legacy")이 아직 이것을 쓴다(srt는 나중에 정리). pdf·docx에서는 대조군용.
  ConstrainedSmartChunker  문단 단위 조각 + 윈도우 평균 거리 + 문단 보너스 + 제약(제목 앞 강제, 제목 뒤·표-캡션 금지)을 거는
                           이전 구현(2026-10-06). 새 구조로 대체되어 숨겨 두었고, 문단 단위 엔진을 어블레이션으로 다시 볼 때 쓴다.

legacy는 입력 블록을 평문에서 정규식으로 다시 찾지 않고 common.docjson.block_spans(doc)가 준 (start, end, kind)를 받는다.
srt 모드는 영화 한 편이 섹션 하나이고, 문장이 다음 큐로 이어지는 경계는 경계 점수에 +1을 더해 사실상 자르지 않는다.
"""
import re
from bisect import bisect_right

import numpy as np

SENT_RE = re.compile(r"(?<=[.!?。])\s+")          # 한국어는 kss.split_sentences로 바꾸면 더 정확
EN_END_RE = re.compile(r"[.!?…\"')\]]$")
KO_CONT_RE = re.compile(r"(고|며|면서|는데|은데|지만|서|면|니까|다가|도록|를|을|의|,|…|\.\.\.)$")


def split_sentences(text, s, e):
    """[s, e) 구간을 문장 (start, end)로 자른다. 오프셋은 원문 기준."""
    out, cur = [], s
    for m in SENT_RE.finditer(text, s, e):
        if m.start() > cur:
            out.append((cur, m.start()))
        cur = m.end()
    if cur < e:
        out.append((cur, e))
    return out


def cue_continues(cur, nxt, lang):
    """자막 한 문장이 다음 큐로 이어지면 True -> 그 사이는 자르지 않도록."""
    cur, nxt = cur.strip(), nxt.strip()
    if cur.endswith(("...", "…", ",")):
        return True
    if lang == "en":
        return not EN_END_RE.search(cur) or nxt[:1].islower()
    return bool(KO_CONT_RE.search(cur))


class SmartChunker:
    def __init__(self, embed_model, count_tokens, mode, lang="ko",
                 min_tokens=100, max_tokens=500, window=2):
        self.embed = embed_model          # SentenceTransformer("BAAI/bge-m3")
        self.count = count_tokens         # 문자열 -> 토큰 수 (bge-m3 또는 Gemma 토크나이저)
        self.mode, self.lang = mode, lang
        self.min_t, self.max_t, self.w = min_tokens, max_tokens, window

    # 1단계: 구조로 섹션 나누기 ---------------------------------------------
    def sections(self, text, blocks):
        if self.mode == "srt":            # 자막은 섹션이 없다: 큐 전체가 한 섹션
            return [blocks]
        secs, cur = [], []
        for s, e, kind in blocks:
            if kind == "heading" and cur:
                secs.append(cur)
                cur = []
            if kind == "para":
                # 문단은 통째로 한 조각(문단 중간을 함부로 안 자름). max_tokens를
                # 넘는 문단만 문장으로 쪼개서 ②(경계점수 분할)가 그 안에서 자를 수
                # 있게 한다.
                if self.count(text[s:e]) > self.max_t:
                    cur += [(a, b, "sent") for a, b in split_sentences(text, s, e)]
                else:
                    cur.append((s, e, "para"))
            else:
                cur.append((s, e, kind))  # 헤더·표는 쪼개지 않는 한 조각
        if cur:
            secs.append(cur)
        return secs

    # 2단계: 경계 점수 (낮을수록 자르기 좋은 곳) ------------------------------
    def boundary_scores(self, text, pieces):
        vecs = np.asarray(self.embed.encode([text[s:e] for s, e, _ in pieces],
                                            normalize_embeddings=True))
        scores = []
        for i in range(len(pieces) - 1):
            left = vecs[max(0, i - self.w + 1): i + 1].mean(0)
            right = vecs[i + 1: i + 1 + self.w].mean(0)
            sim = float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9))
            if self.mode == "srt":
                a, b = pieces[i], pieces[i + 1]
                if cue_continues(text[a[0]:a[1]], text[b[0]:b[1]], self.lang):
                    sim += 1.0            # 문장이 이어지는 곳은 사실상 자르지 않음
            scores.append(sim)
        return scores

    def split_section(self, pieces, scores, toks):
        if sum(toks) <= self.max_t or len(pieces) == 1:
            return [pieces]
        # 양쪽이 모두 min_tokens 이상이 되는 경계 중 점수가 가장 낮은 곳에서 자른다
        cands = [i for i in range(len(pieces) - 1)
                 if sum(toks[:i + 1]) >= self.min_t and sum(toks[i + 1:]) >= self.min_t]
        cut = min(cands or range(len(pieces) - 1), key=lambda i: scores[i])
        return (self.split_section(pieces[:cut + 1], scores[:cut], toks[:cut + 1]) +
                self.split_section(pieces[cut + 1:], scores[cut + 1:], toks[cut + 1:]))

    # 3단계: 작은 청크 합치기 -----------------------------------------------
    def merge_small(self, text, groups):
        """앞쪽으로 훑으면서 직전 청크가 min_tokens 미만이면 현재 청크와 합친다.
        섹션 하나의 그룹 리스트에만 호출할 것 - 여러 섹션을 합쳐서 넘기면 헤더
        경계를 넘어 서로 다른 절의 내용이 한 청크로 섞인다(실측 발견, chunk_spans의
        섹션별 호출로 방지)."""
        out = []
        for g in groups:
            if out:
                prev_text = text[out[-1][0][0]:out[-1][-1][1]]
                joined = text[out[-1][0][0]:g[-1][1]]
                if self.count(prev_text) < self.min_t and self.count(joined) <= self.max_t:
                    out[-1] = out[-1] + g
                    continue
            out.append(g)
        # 마지막 청크는 위 루프에서 "다음 청크"가 없어 한 번도 검사되지 않는다 -
        # min_tokens 미만인 채로 남으면 이전 청크와 합쳐서 자투리로 방치되지 않게 한다.
        if len(out) >= 2 and self.count(text[out[-1][0][0]:out[-1][-1][1]]) < self.min_t:
            out[-2] = out[-2] + out[-1]
            out.pop()
        return out

    def chunk_spans(self, text, blocks):
        """blocks: common.docjson.block_spans(doc)의 [(start, end, kind)] -> [(start, end)] 청크 구간."""
        groups = []
        for pieces in self.sections(text, blocks):
            toks = [self.count(text[s:e]) for s, e, _ in pieces]
            scores = self.boundary_scores(text, pieces) if sum(toks) > self.max_t else []
            section_groups = self.split_section(pieces, scores, toks)
            groups += self.merge_small(text, section_groups)
        return [(g[0][0], g[-1][1]) for g in groups]


class ConstrainedSmartChunker:
    """제약 기반 의미 분할 - 설명은 모듈 위쪽. chunk_spans(text, blocks, glue)가 [(start, end)] 청크 구간을 돌려준다."""

    def __init__(self, embed_model, count_tokens, mode="pdf", lang="en", min_tokens=100, max_tokens=400,
                 window=2, percentile=90, para_bonus=0.05):
        self.embed, self.count, self.mode, self.lang = embed_model, count_tokens, mode, lang
        self.min_t, self.max_t, self.w = min_tokens, max_tokens, window
        self.pct, self.bonus = percentile, para_bonus
        self._sm = SmartChunker(embed_model, count_tokens, mode, lang, min_tokens, max_tokens, window)
        self.stats = {}

    # -- 크기 (합친 텍스트 기준) -----------------------------------------
    def _prepare_sizes(self, text, pieces):
        n = len(pieces)
        self._gap = [self.count(text[pieces[i][0]:pieces[i + 1][0]]) for i in range(n - 1)] + [0]   # 조각 + 뒤 공백
        self._own = [self.count(text[s:e]) for s, e, _ in pieces]

    def _size(self, a, b):
        """조각 a..b-1을 이어 붙인 텍스트의 토큰 수(근사: 마지막 조각 뒤 공백은 뺀다)."""
        return sum(self._gap[a:b - 1]) + self._own[b - 1]

    # -- 의미 거리와 제약 -------------------------------------------------
    def _distances(self, text, pieces, blocks):
        vecs = np.asarray(self.embed.encode([text[s:e] for s, e, _ in pieces], normalize_embeddings=True))
        starts = [b[0] for b in blocks]
        owner = lambda s: bisect_right(starts, s) - 1       # 조각이 속한 원래 블록(문단)
        out = []
        for i in range(len(pieces) - 1):
            left = vecs[max(0, i - self.w + 1): i + 1].mean(0)
            right = vecs[i + 1: i + 1 + self.w].mean(0)
            d = 1 - float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9))
            cur, nxt = pieces[i], pieces[i + 1]
            if self.mode == "srt":
                if cue_continues(text[cur[0]:cur[1]], text[nxt[0]:nxt[1]], self.lang):
                    d -= 1.0                                 # 문장이 이어지는 큐 사이는 사실상 자르지 않음
            elif owner(cur[0]) != owner(nxt[0]):
                d += self.bonus                              # 제약 5: 문단 경계 선호
            out.append(d)
        return out

    def _constraints(self, pieces, glue):
        n = len(pieces)
        forbid = [False] * (n - 1)
        force = [False] * (n - 1)
        for i in range(n - 1):
            cur, nxt = pieces[i], pieces[i + 1]
            forbid[i] = cur[2] == "heading" or nxt[0] in glue    # 제약 2(제목 뒤), 제약 4(표-캡션)
            force[i] = nxt[2] == "heading" and not forbid[i]     # 제약 1(제목 앞). 연속 제목은 금지가 이긴다
        return forbid, force                                     # 제약 3(표 안)은 표가 한 조각이라 자동

    # -- 크기 정리 --------------------------------------------------------
    def _split_oversized(self, a, b, d, forbid):
        if b - a == 1 or self._size(a, b) <= self.max_t:
            return [(a, b)]
        allowed = [i for i in range(a, b - 1) if not forbid[i]]
        if not allowed:
            return [(a, b)]                                      # 자를 수 있는 곳이 없다(예: 제목 + 표)
        both = [i for i in allowed if self._size(a, i + 1) >= self.min_t and self._size(i + 1, b) >= self.min_t]
        cut = max(both or allowed, key=lambda i: d[i])           # min을 지킬 수 없으면 max를 우선한다
        return self._split_oversized(a, cut + 1, d, forbid) + self._split_oversized(cut + 1, b, d, forbid)

    def _merge_small(self, groups):
        groups = list(groups)
        changed = True
        while changed:
            changed = False
            for k, (a, b) in enumerate(groups):
                if self._size(a, b) >= self.min_t:
                    continue
                if k + 1 < len(groups) and self._size(a, groups[k + 1][1]) <= self.max_t:
                    groups[k:k + 2] = [(a, groups[k + 1][1])]
                elif k > 0 and self._size(groups[k - 1][0], b) <= self.max_t:
                    groups[k - 1:k + 1] = [(groups[k - 1][0], b)]
                else:
                    continue
                changed = True
                break
        return groups

    # -- 진입점 -----------------------------------------------------------
    def chunk_spans(self, text, blocks, glue=()):
        """blocks: common.docjson.block_spans(doc)의 [(start, end, kind)]. glue: caption_glue(doc, pieces)."""
        pieces = [p for sec in self._sm.sections(text, blocks) for p in sec]
        if len(pieces) <= 1:
            self.stats = {"pieces": len(pieces), "chunks": len(pieces), "under_min": 0, "over_max": 0}
            return [(pieces[0][0], pieces[0][1])] if pieces else []
        n = len(pieces)
        self._prepare_sizes(text, pieces)
        d = self._distances(text, pieces, blocks)
        forbid, force = self._constraints(pieces, set(glue))
        cand = [i for i in range(n - 1) if not forbid[i] and not force[i]]
        tau = float(np.percentile([d[i] for i in cand], self.pct)) if cand else float("inf")
        cut = [force[i] or (not forbid[i] and d[i] > tau) for i in range(n - 1)]
        groups, a = [], 0
        for i in range(n - 1):
            if cut[i]:
                groups.append((a, i + 1))
                a = i + 1
        groups.append((a, n))
        n_forced = sum(force)
        split = [g for a0, b0 in groups for g in self._split_oversized(a0, b0, d, forbid)]
        merged = self._merge_small(split)
        self.stats = {"pieces": n, "chunks": len(merged), "forced_cuts": n_forced,
                      "semantic_cuts": sum(1 for i in range(n - 1) if cut[i]) - n_forced,
                      "oversize_splits": len(split) - len(groups), "merges": len(split) - len(merged),
                      "under_min": sum(1 for a0, b0 in merged if self._size(a0, b0) < self.min_t),
                      "over_max": sum(1 for a0, b0 in merged if self._size(a0, b0) > self.max_t and b0 - a0 > 1)}
        return [(pieces[a0][0], pieces[b0 - 1][1]) for a0, b0 in merged]
