"""공통 청커 부품 - fixed/semantic/smart 구현을 여기 하나씩만 둔다. 데이터셋마다
다른 값(청크 크기, min/max 토큰, mode 등)은 전부 함수 인자로 받는다 - 실제 값은
각 트랙 pipeline.py의 CHUNKERS 정의에서 넘긴다.

세 청커 모두 LangChain TextSplitter를 반환해서 기존 파이프라인(split_documents,
create_documents, add_start_index=True)에 그대로 꽂힌다. 그리고 세 청커 모두
청크가 항상 원문의 리터럴 슬라이스(text[start:end])라서 add_start_index가 정확한
위치를 계산해준다 - 이게 srt/mapper.py의 fragments()(오프셋 기반 타임스탬프 매핑)와
translate.py의 블록 매핑이 그대로 동작하는 전제 조건이다.
"""
import re

import numpy as np
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters.base import TextSplitter

SENT_RE = re.compile(r"(?<=[.!?。])\s+")          # 한국어는 kss.split_sentences로 바꾸면 더 정확
MD_HEADING_RE = re.compile(r"^#{1,6} ")
PASSAGE_RE = re.compile(r"^Passage \d+:")                                            # LongBench 다중 문서
EN_END_RE = re.compile(r"[.!?…\"')\]]$")
KO_CONT_RE = re.compile(r"(고|며|면서|는데|은데|지만|서|면|니까|다가|도록|를|을|의|,|…|\.\.\.)$")
SEMANTIC_SENTENCE_SPLIT_REGEX = r"(?<=[.?!。])\s+|\n+"


# ---------------------------------------------------------------------------
# fixed - LangChain CharacterTextSplitter 그대로
# ---------------------------------------------------------------------------
def make_fixed(chunk_size, strip_whitespace=True, chunk_overlap=0):
    return CharacterTextSplitter(separator="", chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                  strip_whitespace=strip_whitespace, add_start_index=True)


# ---------------------------------------------------------------------------
# 텍스트 -> 블록 (start, end, kind)   kind: heading / table / para / cue
# smart의 구조 인식과 translate.py의 번역 단위(unit) 나누기가 같이 쓴다.
# ---------------------------------------------------------------------------
def detect_blocks(text, mode):
    if mode == "srt":                     # srt 로더는 큐 하나를 한 줄로 저장한다
        return [(m.start(), m.end(), "cue") for m in re.finditer(r"[^\n]+", text)]

    if mode in ("docx", "pdf"):           # Docling 마크다운: 빈 줄로 구분된 덩어리
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
# smart - ① 구조로 자르기 ② 너무 큰 섹션만 의미로 자르기 ③ 너무 작은 청크는 합치기
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


def make_smart(embed_model, count_tokens, mode, lang="ko", min_tokens=100, max_tokens=500, window=2):
    return SmartTextSplitter(SmartChunker(embed_model, count_tokens, mode=mode, lang=lang,
                                           min_tokens=min_tokens, max_tokens=max_tokens, window=window))


# ---------------------------------------------------------------------------
# semantic - LangChain SemanticChunker로 경계만 정하고, recover_spans로 원문 위치를
# 복원한 뒤 min/max_tokens 후처리를 적용한다.
# ---------------------------------------------------------------------------
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


def _merge_small_spans(text, spans, count_tokens, min_tokens, max_tokens):
    """smart의 merge_small()과 같은 규칙 - min_tokens 미만 청크는 이웃과 합친다
    (합친 크기가 max_tokens 이하일 때만). 글자/토큰 비율 환산 대신 토큰을 직접
    세서 판단한다(영어 pdf에서 글자/토큰 비율로 환산한 min_chunk_size가 너무
    작게 잡히는 문제를 실측으로 확인해서, semantic도 smart와 같은 토큰 기준
    후처리로 통일)."""
    out = []
    for sp in spans:
        if out:
            prev_s, prev_e = out[-1]
            joined_tokens = count_tokens(text[prev_s:sp[1]])
            if count_tokens(text[prev_s:prev_e]) < min_tokens and joined_tokens <= max_tokens:
                out[-1] = (prev_s, sp[1])
                continue
        out.append(sp)
    if len(out) >= 2 and count_tokens(text[out[-1][0]:out[-1][1]]) < min_tokens:
        merged = (out[-2][0], out[-1][1])
        if count_tokens(text[merged[0]:merged[1]]) <= max_tokens:
            out[-2] = merged
            out.pop()
    return out


class _SemanticSplitter(TextSplitter):
    """make_semantic()이 실제로 만드는 클래스 - SemanticChunker(경계 후보) +
    recover_spans(위치 복원) + min/max_tokens 후처리."""

    def __init__(self, inner, embed_model, count_tokens, min_tokens, max_tokens, **kwargs):
        super().__init__(**kwargs)
        self._inner = inner
        self._embed_model = embed_model
        self._count = count_tokens
        self._min_t, self._max_t = min_tokens, max_tokens

    def split_text(self, text):
        texts = self._inner.split_text(text)
        spans = [sp for sp in recover_spans(text, texts) if sp is not None]
        spans = _merge_small_spans(text, spans, self._count, self._min_t, self._max_t)
        if self._max_t is not None:
            oversized_fixed = []
            for sp in spans:
                oversized_fixed += _split_oversized(text, sp, self._embed_model, self._count, self._max_t)
            spans = oversized_fixed
        return [text[s:e] for s, e in spans]


def make_semantic(embeddings, count_tokens, percentile=90, min_tokens=100, max_tokens=400, **kwargs):
    """embeddings: LangChain Embeddings(예: HuggingFaceEmbeddings) - SemanticChunker가
    요구하는 인터페이스. max_tokens 후처리(_split_oversized)의 문장 임베딩은
    embeddings._client(내부 SentenceTransformer)로 직접 encode()한다."""
    from langchain_experimental.text_splitter import SemanticChunker
    kwargs.setdefault("chunk_overlap", 0)
    kwargs.setdefault("add_start_index", True)
    inner = SemanticChunker(embeddings, breakpoint_threshold_type="percentile",
                             breakpoint_threshold_amount=percentile,
                             sentence_split_regex=SEMANTIC_SENTENCE_SPLIT_REGEX)
    return _SemanticSplitter(inner, embeddings._client, count_tokens, min_tokens, max_tokens, **kwargs)
