"""경로·모델·데이터셋 설정 - 단일 진실 공급원 (docs/restructure_plan.md 11절).

pdf·docx 데이터셋의 `chunkers` 항목(_doc_chunkers)은 **옛 v1 파이프라인(chunking/v1) 전용 시작값**이다. 새 청커(chunking/)는
이 값을 읽지 않고 scripts/chunk.py에서 p·T를 명시해서 받는다. srt 설정은 src/srt/config.py에 있고 여기서는 등록만 한다.
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
    semantic과 smart는 같은 엔진(chunking/semantic_core.py)과 같은 크기 제어(DP)를 쓰고 차이는 구조 제약 규칙(rules)뿐이다(chunking/units.py)."""
    none = dict(heading_forced=False, heading_glue=False, table_atomic=False, caption_glue=False, para_atomic=False, section_split=False)
    allr = dict(heading_forced=False, heading_glue=True, table_atomic=True, caption_glue=True, para_atomic=False, section_split=True)   # smart 기본: 문장 단위(semantic과 같음) + 구조 묶음 + 최상위 섹션 분할. 문단 단위·제목 앞 강제 절단은 어블레이션
    return dict(
        fixed=dict(target_tokens=400, overlap_ratio=0.0),
        semantic=dict(percentile=90, min_tokens=100, max_tokens=400, buffer_size=0, rules=none, size_control="dp"),
        smart=dict(percentile=90, min_tokens=100, max_tokens=400, buffer_size=0, rules=allr, size_control="dp"),   # 크기 제어는 semantic과 같은 DP(차이는 규칙뿐). 그리디는 size_control로 어블레이션
    )


from srt.config import SRT_DATASET  # noqa: E402

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
    "srt": SRT_DATASET,          # srt 설정은 src/srt/config.py
}


def dataset_cfg(name):
    if name not in DATASETS:
        raise KeyError(f"unknown dataset: {name} (choices: {sorted(DATASETS)})")
    return DATASETS[name]
