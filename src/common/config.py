"""경로·모델·데이터셋 설정 - 단일 진실 공급원 (docs/restructure_plan.md 11절).

청커 파라미터는 튜닝 전 "시작값"이다(8절 프로토콜로 바꾼다). fixed의 chunk_size(글자 수)는
여기에 없다: target_tokens × 측정한 chars_per_token(manifests/<dataset>.json의
fixed_char_per_token, 5.4.1절)에서 계산한다.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
INDEX_DIR = DATA_DIR / "index"
QA_DIR = DATA_DIR / "qa"
MANIFEST_DIR = DATA_DIR / "manifests"
RESULTS_DIR = ROOT / "results"

EMBED_MODEL = "BAAI/bge-m3"
TOKENIZER = "google/gemma-4-12B-it-qat-q4_0-unquantized"


def _doc_chunkers():
    """pdf/docx 공통 시작값: min 100 / max 400 토큰(bge-m3 카운터). 튜닝 전 값이다.
    semantic과 smart는 같은 엔진(chunking/semantic_core.py)이고 차이는 구조 제약 규칙(rules)뿐이다(chunking/units.py)."""
    none = dict(heading_forced=False, heading_glue=False, table_atomic=False, caption_glue=False)
    allr = dict(heading_forced=True, heading_glue=True, table_atomic=True, caption_glue=True)
    return dict(
        fixed=dict(target_tokens=400, overlap_ratio=0.0),
        semantic=dict(percentile=90, min_tokens=100, max_tokens=400, buffer_size=0, rules=none),
        smart=dict(percentile=90, min_tokens=100, max_tokens=400, buffer_size=0, rules=allr, rebalance=True),   # rebalance는 smart만(semantic은 기본 합치기)
    )


DATASETS = {
    "vectara": dict(
        fmt="pdf", lang="en", raw="vectara_pdf", counter="bge-m3",
        tasks=["retrieval", "qa"], chunkers=_doc_chunkers(),
    ),
    "allganize": dict(
        fmt="pdf", lang="ko", raw="allganize_pdf", counter="bge-m3",
        tasks=["retrieval", "qa"], chunkers=_doc_chunkers(),
    ),
    "docx_corpus": dict(
        fmt="docx", lang="en", raw="docx_corpus", counter="bge-m3",
        tasks=["retrieval", "qa"], chunkers=_doc_chunkers(),
    ),
    # srt: 영화 쌍은 manifests/srt.json. smart의 lang은 번역 방향(원문 언어)마다 정한다.
    "srt": dict(
        fmt="srt", lang=("en", "ko"), raw=("srt_eng", "srt_kor"), counter="gemma",
        tasks=["translate"],
        chunkers=dict(
            fixed=dict(target_tokens=400, overlap_ratio=0.0, strip_whitespace=False),
            # semantic max는 기존 1024에서 512로 낮췄다(smart와 같은 값, 5.4절의 공정성 조정)
            semantic=dict(percentile=90, min_tokens=128, max_tokens=512, buffer_size=0),
            # srt smart는 아직 예전 방식(legacy_smart.SmartChunker). pdf·docx smart와 달리 srt 전략은 나중에 정리한다
            smart=dict(mode="srt", algorithm="legacy", min_tokens=128, max_tokens=512, window=2),
        ),
    ),
}


def dataset_cfg(name):
    if name not in DATASETS:
        raise KeyError(f"unknown dataset: {name} (choices: {sorted(DATASETS)})")
    return DATASETS[name]
