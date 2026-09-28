"""스마트 청커 (구조 우선 + 의미 분할 + 토큰 크기 제어)

아이디어는 3단계뿐이다.
  1) 구조로 먼저 자른다   : 헤더(#, 논문 섹션 번호, "Passage N:")마다 새 섹션.
                             표는 통째로 한 조각, 자막은 큐 하나가 한 조각.
  2) 너무 큰 섹션만 의미로 자른다 : 조각 사이 유사도(좌우 창 평균)가 가장 낮은 곳에서
                             반으로 자르기를 max_tokens 이하가 될 때까지 반복.
  3) 너무 작은 청크는 이웃과 합친다 : min_tokens 미만이면 다음 청크와 합침.

반환하는 청크는 항상 원문의 리터럴 슬라이스(text[start:end])라서, 기존 코드의
add_start_index / fragments() / merge_to_units()가 그대로 동작한다.
"""
import re

import numpy as np
from langchain_text_splitters.base import TextSplitter

SENT_RE = re.compile(r"(?<=[.!?。])\s+")          # 한국어는 kss.split_sentences로 바꾸면 더 정확
MD_HEADING_RE = re.compile(r"^#{1,6} ")
PASSAGE_RE = re.compile(r"^Passage \d+:")                                            # LongBench 다중 문서
EN_END_RE = re.compile(r"[.!?…\"')\]]$")
KO_CONT_RE = re.compile(r"(고|며|면서|는데|은데|지만|서|면|니까|다가|도록|를|을|의|,|…|\.\.\.)$")


# ---------------------------------------------------------------------------
# 0. 텍스트 -> 블록 (start, end, kind)   kind: heading / table / para / cue
# ---------------------------------------------------------------------------
def detect_blocks(text, mode):
    if mode == "srt":                     # srt 로더는 큐 하나를 한 줄로 저장한다
        return [(m.start(), m.end(), "cue") for m in re.finditer(r"[^\n]+", text)]

    if mode == "docx":                    # Docling 마크다운: 빈 줄로 구분된 덩어리
        pattern = r"[^\n]+(?:\n(?!\n)[^\n]+)*"
    elif mode == "pdf":                   # Docling 마크다운(docx와 동일 형식): 빈 줄로 구분된 덩어리
        pattern = r"[^\n]+(?:\n(?!\n)[^\n]+)*"
    else:                                 # longbench: 한 줄 = 한 문단
        pattern = r"[^\n]+"

    blocks = []
    for m in re.finditer(pattern, text):
        body = m.group().strip()
        if MD_HEADING_RE.match(body) or PASSAGE_RE.match(body):
            kind = "heading"
        elif body.startswith(("|", "<table")):             # 마크다운 표 / HTML 표
            kind = "table"
        else:
            kind = "para"
        # 한 줄=한 블록 모드(longbench)에서 표 행이 여러 줄로 쪼개져 들어오면 하나로 묶는다 -
        # docx/pdf는 이미 멀티라인 패턴이라 표 전체가 한 블록으로 잡히므로 이 병합이 필요 없음
        # (지금은 실제로 이 분기를 타는 경로가 없음 - longbench에 표가 나오면 대비용으로 남겨둠).
        if kind == "table" and blocks and blocks[-1][2] == "table" and mode not in ("docx", "pdf"):
            blocks[-1] = (blocks[-1][0], m.end(), "table")
        else:
            blocks.append((m.start(), m.end(), kind))
    return blocks


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


# ---------------------------------------------------------------------------
# 스마트 청커
# ---------------------------------------------------------------------------
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
                # ★ 2026-09-28 변경 - 문단은 통째로 한 조각(문단 중간을 함부로 안 자름).
                # max_tokens를 넘는 문단만 문장으로 쪼개서 ②(경계점수 분할)가 그 안에서
                # 자를 수 있게 한다 - 이전엔 문단을 항상 문장으로 쪼개서, 짧은 문단끼리도
                # 문장 단위로 잘게 나뉘어 ②가 문단 중간에서 자르는 경우가 있었음.
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

    def chunk_spans(self, text):
        groups = []
        for pieces in self.sections(text, detect_blocks(text, self.mode)):
            toks = [self.count(text[s:e]) for s, e, _ in pieces]
            scores = self.boundary_scores(text, pieces) if sum(toks) > self.max_t else []
            section_groups = self.split_section(pieces, scores, toks)
            groups += self.merge_small(text, section_groups)
        return [(g[0][0], g[-1][1]) for g in groups]


class SmartTextSplitter(TextSplitter):
    """LangChain TextSplitter로 감싸서 기존 파이프라인(split_documents,
    add_start_index=True)에 그대로 꽂는다."""

    def __init__(self, chunker, **kwargs):
        kwargs.setdefault("chunk_overlap", 0)
        kwargs.setdefault("add_start_index", True)
        super().__init__(**kwargs)
        self._chunker = chunker

    def split_text(self, text):
        return [text[s:e] for s, e in self._chunker.chunk_spans(text)]


# ---------------------------------------------------------------------------
# semantic(docx/pdf) - LangChain SemanticChunker + 설정 보정 + max_tokens 후처리
# ---------------------------------------------------------------------------
# ★ 실측 확인(2026-09-28) - 기본 sentence_split_regex(마침표류 뒤 공백만 문장 경계)는
# 마크다운 제목·표 행·목록처럼 줄바꿈으로만 구분되는 텍스트를 전부 "한 문장"으로
# 뭉쳐버려서 비정상적으로 큰 청크가 나온다 - 줄바꿈도 경계 후보로 추가.
SEMANTIC_SENTENCE_SPLIT_REGEX = r"(?<=[.?!。])\s+|\n+"
# bge-m3 토크나이저로 코퍼스 표본 실측한 글자/토큰 비율 - smart의 min_tokens(글자
# 수가 아니라 토큰 수)에 대응하는 min_chunk_size(글자 수)를 여기서 환산한다.
CHARS_PER_TOKEN = 2.448


def recover_spans(flat, chunks):
    """공백 무시하고 청크 텍스트를 flat_text에서 순서대로 찾아 (start, end) 복원 -
    SemanticChunker가 " ".join()으로 문장을 재조합해서(개행이 공백으로 바뀜)
    start_index가 실제 위치와 어긋나는 오프셋 오류를 우회한다(실측 확인된 문제)."""
    pos = [i for i, ch in enumerate(flat) if not ch.isspace()]
    squeezed = "".join(flat[i] for i in pos)
    spans, cur = [], 0
    for c in chunks:
        key = "".join(c.split())
        k = squeezed.find(key, cur)
        if k < 0 or not key:
            spans.append(None)  # 못 찾으면 None - 호출하는 쪽에서 개수로 집계
            continue
        spans.append((pos[k], pos[k + len(key) - 1] + 1))
        cur = k + len(key)
    return spans


def _sentence_spans(text, base):
    spans, cur = [], base
    for m in re.finditer(SEMANTIC_SENTENCE_SPLIT_REGEX, text):
        end = base + m.start()
        if end > cur:
            spans.append((cur, end))
        cur = base + m.end()
    if cur < base + len(text):
        spans.append((cur, base + len(text)))
    return spans or [(base, base + len(text))]


def _split_oversized(flat_text, span, embed_model, count_tokens, max_tokens):
    """max_tokens를 넘는 semantic 청크를 그 안의 인접 문장 사이 거리(=유사도 반대)가
    가장 큰 지점에서 반복 분할한다 - 문장 중간은 안 자름(문장이 하나뿐이면 그대로 둠)."""
    s, e = span
    if count_tokens(flat_text[s:e]) <= max_tokens:
        return [span]
    sents = _sentence_spans(flat_text[s:e], s)
    if len(sents) == 1:
        return [span]
    vecs = np.asarray(embed_model.encode([flat_text[a:b] for a, b in sents], normalize_embeddings=True))
    cut, best_dist = 0, -1.0
    for i in range(len(sents) - 1):
        dist = 1 - float(vecs[i] @ vecs[i + 1])
        if dist > best_dist:
            best_dist, cut = dist, i
    left, right = (sents[0][0], sents[cut][1]), (sents[cut + 1][0], sents[-1][1])
    return (_split_oversized(flat_text, left, embed_model, count_tokens, max_tokens) +
            _split_oversized(flat_text, right, embed_model, count_tokens, max_tokens))


class SemanticMaxSplitter(TextSplitter):
    """SemanticChunker(경계 보정) + max_tokens 후처리 + recover_spans 위치복원을
    하나로 묶는다 - max_tokens=None이면 후처리 없이 그대로(어블레이션 옵션)."""

    def __init__(self, embeddings, count_tokens, percentile=90, min_tokens=100, max_tokens=400, **kwargs):
        from langchain_experimental.text_splitter import SemanticChunker
        kwargs.setdefault("chunk_overlap", 0)
        kwargs.setdefault("add_start_index", True)
        super().__init__(**kwargs)
        self.embeddings, self.count, self.max_t = embeddings, count_tokens, max_tokens
        self._inner = SemanticChunker(embeddings, breakpoint_threshold_type="percentile",
                                       breakpoint_threshold_amount=percentile,
                                       sentence_split_regex=SEMANTIC_SENTENCE_SPLIT_REGEX,
                                       min_chunk_size=round(min_tokens * CHARS_PER_TOKEN))

    def split_text(self, text):
        texts = self._inner.split_text(text)
        spans = [sp for sp in recover_spans(text, texts) if sp is not None]
        if self.max_t is None:
            return [text[s:e] for s, e in spans]
        final = []
        for sp in spans:
            final += _split_oversized(text, sp, self.embeddings._client, self.count, self.max_t)
        return [text[s:e] for s, e in final]
