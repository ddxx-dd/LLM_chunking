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


def _doc_chunkers(mode):
    """pdf/docx 공통 시작값: min 100 / max 400 토큰(bge-m3 카운터)."""
    return dict(
        fixed=dict(target_tokens=400, overlap_ratio=0.0),
        semantic=dict(percentile=90, min_tokens=100, max_tokens=400),
        # smart: 제약 기반 의미 분할 (chunking/smart.py의 ConstrainedSmartChunker). "legacy"가 예전 방식(대조군)
        smart=dict(mode=mode, algorithm="constrained", min_tokens=100, max_tokens=400, window=2, percentile=90, para_bonus=0.05),
    )


DATASETS = {
    "vectara": dict(
        fmt="pdf", lang="en", raw="vectara_pdf", counter="bge-m3",
        tasks=["retrieval", "qa"], chunkers=_doc_chunkers("pdf"),
    ),
    "allganize": dict(
        fmt="pdf", lang="ko", raw="allganize_pdf", counter="bge-m3",
        tasks=["retrieval", "qa"], chunkers=_doc_chunkers("pdf"),
    ),
    "docx_corpus": dict(
        fmt="docx", lang="en", raw="docx_corpus", counter="bge-m3",
        tasks=["retrieval", "qa"], chunkers=_doc_chunkers("docx"),
    ),
    # srt: 영화 쌍은 manifests/srt.json. smart의 lang은 번역 방향(원문 언어)마다 정한다.
    "srt": dict(
        fmt="srt", lang=("en", "ko"), raw=("srt_eng", "srt_kor"), counter="gemma",
        tasks=["translate"],
        chunkers=dict(
            fixed=dict(target_tokens=400, overlap_ratio=0.0, strip_whitespace=False),
            # semantic max는 기존 1024에서 512로 낮췄다(smart와 같은 값, 5.4절의 공정성 조정)
            semantic=dict(percentile=90, min_tokens=128, max_tokens=512),
            smart=dict(mode="srt", min_tokens=128, max_tokens=512, window=2),
        ),
    ),
}


def dataset_cfg(name):
    if name not in DATASETS:
        raise KeyError(f"unknown dataset: {name} (choices: {sorted(DATASETS)})")
    return DATASETS[name]
