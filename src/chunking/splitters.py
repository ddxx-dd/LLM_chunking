"""세 청커를 같은 모양으로 쓰게 하는 모듈 (docs/restructure_plan.md 5.1절).

  chunk_spans(doc, text, pieces, chunker, cfg, models) -> ([(start, end)], info)
  chunk(doc, chunker, cfg, models)                     -> ([청크 dict], info)

fixed / semantic 알고리즘은 legacy/src/chunkers.py를 그대로 옮겼다. 바뀐 곳:
  - semantic은 LangChain TextSplitter 래퍼 대신 구간 (start, end)를 직접 돌려준다.
    (LangChain의 start_index는 청크 길이를 누적해서 위치를 정하는데, SemanticChunker는 문장을 " "로 이어
    붙여 글이 원문과 달라지므로 맞지 않는다 - 그래서 recover_spans로 위치를 복원한다.)
  - recover_spans가 못 찾은 청크는 예전에는 조용히 버려졌다. 알고리즘은 그대로 두고, 버려진 청크 수를
    info["semantic_dropped_chunks"]로 돌려준다(내용 보존 검사가 이를 확인한다).
"""
import re

import numpy as np
from langchain_text_splitters import CharacterTextSplitter

from chunking.meta import attach_meta
from chunking.smart import ConstrainedSmartChunker, SmartChunker, caption_glue
from common.docjson import render

SEMANTIC_SENTENCE_SPLIT_REGEX = r"(?<=[.?!。])\s+|\n+"


# ---------------------------------------------------------------------------
# fixed - LangChain CharacterTextSplitter 그대로
# ---------------------------------------------------------------------------
def make_fixed(chunk_size, strip_whitespace=True, chunk_overlap=0):
    return CharacterTextSplitter(separator="", chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                 strip_whitespace=strip_whitespace, add_start_index=True)


def fixed_spans(text, chunk_size, strip_whitespace=True, chunk_overlap=0):
    spans = []
    for d in make_fixed(chunk_size, strip_whitespace, chunk_overlap).create_documents([text]):
        s = d.metadata["start_index"]
        if s < 0:
            raise ValueError("fixed 청크의 위치를 원문에서 찾지 못했다")
        spans.append((s, s + len(d.page_content)))
    return spans


def resolve_fixed_size(doc, cfg):
    """(chunk_size_chars, overlap_chars). cfg에 chunk_size_chars가 있으면 그대로(테스트·수동 지정),
    없으면 목표 토큰 × 측정한 글자/토큰 비율(manifests/<dataset>.ratio.json, 5.4.1절)로 계산한다."""
    if cfg.get("chunk_size_chars"):
        size = int(cfg["chunk_size_chars"])
        return size, round(cfg.get("overlap_ratio", 0.0) * size)
    from chunking.ratio import fixed_chunk_size
    return fixed_chunk_size(doc["dataset"], doc["lang"], cfg["target_tokens"], cfg.get("overlap_ratio", 0.0))


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


class SemanticSpanner:
    """SemanticChunker(경계 후보) + recover_spans(위치 복원) + min/max_tokens 후처리(병합 → 재분할).
    embeddings: LangChain Embeddings. 후처리의 문장 임베딩은 embeddings._client(SentenceTransformer)로
    직접 encode()한다. inner는 테스트에서 SemanticChunker 대신 가짜를 넣기 위한 자리다."""

    def __init__(self, embeddings, count_tokens, percentile=90, min_tokens=100, max_tokens=400, inner=None):
        if inner is None:
            from langchain_experimental.text_splitter import SemanticChunker
            inner = SemanticChunker(embeddings, breakpoint_threshold_type="percentile",
                                    breakpoint_threshold_amount=percentile,
                                    sentence_split_regex=SEMANTIC_SENTENCE_SPLIT_REGEX)
        self._inner = inner
        self._embed_model = embeddings._client
        self._count = count_tokens
        self._min_t, self._max_t = min_tokens, max_tokens

    def spans(self, text):
        """-> ([(start, end)], {"semantic_dropped_chunks": 위치를 못 찾아 버려진 청크 수})."""
        if not text.strip():
            return [], {"semantic_dropped_chunks": 0}
        recovered = recover_spans(text, self._inner.split_text(text))
        dropped = sum(1 for sp in recovered if sp is None)
        spans = [sp for sp in recovered if sp is not None]
        spans = _merge_small_spans(text, spans, self._count, self._min_t, self._max_t)
        if self._max_t is not None:
            resized = []
            for sp in spans:
                resized += _split_oversized(text, sp, self._embed_model, self._count, self._max_t)
            spans = resized
        return spans, {"semantic_dropped_chunks": dropped}


# ---------------------------------------------------------------------------
# 통합 진입점
# ---------------------------------------------------------------------------
def chunk_spans(doc, text, pieces, chunker, cfg, models):
    """-> ([(start, end)], info, params). params는 이 문서에 실제로 쓴 값(청크 파일에 그대로 기록)."""
    if chunker == "fixed":
        size, overlap = resolve_fixed_size(doc, cfg)
        strip = cfg.get("strip_whitespace", True)
        params = {**cfg, "chunk_size_chars": size, "chunk_overlap_chars": overlap}
        return fixed_spans(text, size, strip, overlap), {}, params
    if chunker == "semantic":
        spans, info = SemanticSpanner(models["embeddings"], models["count"], cfg["percentile"],
                                      cfg["min_tokens"], cfg["max_tokens"]).spans(text)
        return spans, info, dict(cfg)
    if chunker == "smart" and cfg.get("algorithm", "legacy") == "constrained":
        sm = ConstrainedSmartChunker(models["embed_model"], models["count"], mode=cfg["mode"], lang=doc["lang"],
                                     min_tokens=cfg["min_tokens"], max_tokens=cfg["max_tokens"], window=cfg.get("window", 2),
                                     percentile=cfg.get("percentile", 90), para_bonus=cfg.get("para_bonus", 0.05))
        blocks = [(p.start, p.end, p.kind) for p in pieces]
        spans = sm.chunk_spans(text, blocks, caption_glue(doc, pieces))
        return spans, {f"smart_{k}": v for k, v in sm.stats.items()}, {**cfg, "lang": doc["lang"]}
    if chunker == "smart":
        sm = SmartChunker(models["embed_model"], models["count"], mode=cfg["mode"], lang=doc["lang"],
                          min_tokens=cfg["min_tokens"], max_tokens=cfg["max_tokens"], window=cfg.get("window", 2))
        blocks = [(p.start, p.end, p.kind) for p in pieces]            # = common.docjson.block_spans(doc)
        return sm.chunk_spans(text, blocks), {}, {**cfg, "lang": doc["lang"]}
    raise KeyError(f"unknown chunker: {chunker!r}")


def chunk(doc, chunker, cfg, models):
    """공통 JSON 문서 하나 -> ([청크 dict], info). 청크 텍스트는 항상 to_text(doc)[start:end]."""
    text, pieces = render(doc)
    spans, info, params = chunk_spans(doc, text, pieces, chunker, cfg, models)
    return attach_meta(doc, text, pieces, spans, models["count"], chunker, params), info
