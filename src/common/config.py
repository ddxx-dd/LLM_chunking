"""경로, 모델 이름, 데이터셋 목록. 다른 모듈은 이것들을 여기서만 가져온다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                # 원본 (git 추적 안 함, 사본 없음 - 삭제 금지)
META_DIR = DATA_DIR / "meta_text"         # 옛 Docling 전처리 결과(v2 확정 후 지울 예정)
DOCLING_DIR = DATA_DIR / "docling_json"   # 1차: Docling 변환 결과(<id>.docling.json)와 변환 기록(<id>.convert.json). src/preprocess/stage1.py
META2_DIR = DATA_DIR / "meta_text_v2"     # 1차: 블록 + 위치·줄 정보(문서 1개 = JSON 1개). src/preprocess/stage1.py
CLEAN_DIR = DATA_DIR / "clean_text"       # 2차: 모든 청커의 입력(문서 하나 = 긴 텍스트 하나, 문단은 빈 줄). src/preprocess/stage2.py
PARA_DIR = DATA_DIR / "processed_para"    # LongBench 단위(src/preprocess/longbench.py). pdf·docx는 clean_text로 바뀜(LongBench도 나중에 바꿀 예정)
SENT_DIR = DATA_DIR / "processed_sen"     # 깨끗한 텍스트, 문장 단위. pdf·docx는 src/preprocess/sentences.py, LongBench는 src/preprocess/longbench.py
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
# 트랙 A LongBench 데이터셋(원본 data/raw/longbench/<이름>.jsonl). pdf·docx와 따로 src/preprocess/longbench.py가 단위를 만든다
LONGBENCH = {
    "hotpotqa": {"task": "qa", "lang": "en"},
    "narrativeqa": {"task": "qa", "lang": "en"},
    "gov_report": {"task": "summary", "lang": "en"},
}
REPORT_FILE = "_report.json"             # 전처리 폴더 안의 실행 보고서(문서가 아님). doc_id가 '_'로 시작하는 문서도 있으니 이름 전체로 거른다
