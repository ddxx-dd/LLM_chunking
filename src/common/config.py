"""경로와 모델 이름. 다른 모듈은 경로·모델 이름을 여기서만 가져온다.

데이터셋 설정은 A2(전처리) 단계에서 추가한다.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                # 원본 (git 추적 안 함, 사본 없음 - 삭제 금지)
PROCESSED_DIR = DATA_DIR / "processed"    # 전처리 결과(문서 1개 = JSON 1개)
QA_DIR = DATA_DIR / "qa"                  # 질문·정답·근거 (git 추적)
MANIFEST_DIR = DATA_DIR / "manifests"     # 문서 목록 (git 추적)
CHUNKS_DIR = DATA_DIR / "chunks"          # 청크 파일 (A3)
INDEX_DIR = DATA_DIR / "index"            # FAISS 인덱스 캐시 (A4)
RESULTS_DIR = ROOT / "results"

EMBED_MODEL = "BAAI/bge-m3"               # 모든 청커·검색에 같은 임베딩
EMBED_DIM = 1024
EMBED_MAX_TOKENS = 8192                   # bge-m3 입력 상한. 넘는 글은 임베딩할 때 뒷부분이 잘린다
LLM_MODEL = "google/gemma-4-12B-it-qat-q4_0-unquantized"   # 생성(질의응답·요약)·판정·번역용, 4비트로 올린다
