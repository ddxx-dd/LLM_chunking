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
                cur += [(a, b, "sent") for a, b in split_sentences(text, s, e)]
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
