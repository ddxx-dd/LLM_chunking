"""세 청커를 같은 모양으로 쓰게 하는 모듈 (docs/restructure_plan.md 5.1절).

  chunk_spans(doc, text, pieces, chunker, cfg, models) -> ([(start, end)], info, params)
  chunk(doc, chunker, cfg, models)                     -> ([청크 dict], info)

  fixed     LangChain CharacterTextSplitter (글자 수). 구조·의미를 보지 않는다.
  semantic  의미 분할 공유 엔진(semantic_core) + 규칙 전부 off. LangChain SemanticChunker와 같은 계산이다.
  smart     같은 엔진 + 구조 제약 규칙 전부 on (units.py). semantic과의 차이는 엔진에 넘기는 단위뿐이다.
  (srt 설정의 smart는 아직 예전 방식 legacy_smart.SmartChunker를 쓴다 - cfg["algorithm"] == "legacy")
청크 텍스트는 항상 평문의 리터럴 슬라이스(text[start:end])다.
"""
from langchain_text_splitters import CharacterTextSplitter

from srt.smart import SmartChunker
from chunking.v1.meta import attach_meta
from chunking.v1.sections import merge_short_sections, split_sections
from chunking.sizectl.semantic_core import SemanticEngine
from chunking.v1.units import build_units, caption_glue, rules_for, size_control_for
from common.docjson import render


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
    from chunking.v1.ratio import fixed_chunk_size
    return fixed_chunk_size(doc["dataset"], doc["lang"], cfg["target_tokens"], cfg.get("overlap_ratio", 0.0))


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
    if chunker == "smart" and cfg.get("algorithm") == "legacy":
        sm = SmartChunker(models["embed_model"], models["count"], mode=cfg["mode"], lang=doc["lang"],
                          min_tokens=cfg["min_tokens"], max_tokens=cfg["max_tokens"], window=cfg.get("window", 2))
        blocks = [(p.start, p.end, p.kind) for p in pieces]            # = common.docjson.block_spans(doc)
        return sm.chunk_spans(text, blocks), {}, {**cfg, "lang": doc["lang"]}
    if chunker in ("semantic", "smart"):
        rules = rules_for(chunker, cfg)
        glue = caption_glue(doc, pieces)
        engine = SemanticEngine(models["embed_model"], models["count"], percentile=cfg["percentile"],
                                min_tokens=cfg["min_tokens"], max_tokens=cfg["max_tokens"], buffer_size=cfg.get("buffer_size", 0),
                                size_control=size_control_for(chunker, cfg))
        extra = {}
        if rules.section_split:
            sections, _lvl = split_sections(doc, pieces)
            groups, n_merges = merge_short_sections(text, pieces, sections, models["embed_model"], models["count"], cfg["min_tokens"], cfg["max_tokens"])
            spans = engine.chunk_groups(text, [build_units(text, pieces[lo:hi], glue, rules, count=models["count"], max_tokens=cfg.get("max_tokens"))
                                               for lo, hi in groups])
            bounds = [pieces[lo].start for lo, _ in sections[1:]]                      # 원래 섹션 경계(합치기 전)
            extra = {"orig_sections": len(sections), "section_merges": n_merges,
                     "cross_section_chunks": sum(1 for s, e in spans if any(s < b < e for b in bounds))}
        else:
            units, forced = build_units(text, pieces, glue, rules, count=models["count"], max_tokens=cfg.get("max_tokens"))
            spans = engine.chunk_spans(text, units, forced)
        info = {f"{chunker}_{k}": v for k, v in {**engine.stats, **extra}.items()}
        return spans, info, {**cfg, "rules": rules.as_dict(), "size_control": size_control_for(chunker, cfg), "lang": doc["lang"]}
    raise KeyError(f"unknown chunker: {chunker!r}")


def chunk(doc, chunker, cfg, models):
    """공통 JSON 문서 하나 -> ([청크 dict], info). 청크 텍스트는 항상 to_text(doc)[start:end]."""
    text, pieces = render(doc)
    spans, info, params = chunk_spans(doc, text, pieces, chunker, cfg, models)
    return attach_meta(doc, text, pieces, spans, models["count"], chunker, params), info
