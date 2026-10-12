"""경로, 모델 이름, 데이터셋 목록. 다른 모듈은 이것들을 여기서만 가져온다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                # 원본 (git 추적 안 함, 사본 없음 - 삭제 금지)
META_DIR = DATA_DIR / "meta_text"         # Docling 전처리 결과: 블록 + 메타데이터 (문서 1개 = JSON 1개, 수정하지 않음)
PARA_DIR = DATA_DIR / "processed_para"    # 깨끗한 텍스트, 문단 단위 (scripts/build_units.py로 생성)
SENT_DIR = DATA_DIR / "processed_sen"     # 깨끗한 텍스트, 문장 단위 (scripts/build_units.py로 생성)
QA_DIR = DATA_DIR / "qa"                  # 질문·정답·근거 (git 추적)
MANIFEST_DIR = DATA_DIR / "manifests"     # 문서 목록 (git 추적)
CHUNKS_DIR = DATA_DIR / "chunks"          # 청크 파일 (A3)
INDEX_DIR = DATA_DIR / "index"            # FAISS 인덱스 캐시 (A4)
RESULTS_DIR = ROOT / "results"

EMBED_MODEL = "BAAI/bge-m3"               # 모든 청커·검색에 같은 임베딩
EMBED_DIM = 1024
EMBED_MAX_TOKENS = 8192                   # bge-m3 입력 상한. 넘는 글은 임베딩할 때 뒷부분이 잘린다
LLM_MODEL = "google/gemma-4-12B-it-qat-q4_0-unquantized"   # 생성(질의응답·요약)·판정·번역용, 4비트로 올린다

# 트랙 A 데이터셋. 폴더마다 <이름>/<doc_id>.json (raw는 data/raw/<raw>)
DATASETS = {
    "vectara": {"fmt": "pdf", "lang": "en", "raw": "vectara_pdf"},
    "allganize": {"fmt": "pdf", "lang": "ko", "raw": "allganize_pdf"},
    "docx_corpus": {"fmt": "docx", "lang": "en", "raw": "docx_corpus"},
}
REPORT_FILE = "_report.json"              # 전처리 폴더 안의 실행 보고서(문서가 아님). doc_id가 '_'로 시작하는 문서도 있으니 이름 전체로 거른다
