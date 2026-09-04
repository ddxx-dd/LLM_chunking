import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from config import DOCX_KOR_DIR,DOCX_ENG_DIR
from preprocessing.loader import load_file
from chunking.fixed_chunker import fixed_chunking
from chunking.semantic_chunker import semantic_chunking
from analysis.analyzer import print_chunks



#설정
FILE_PATH = DOCX_ENG_DIR / "4-1.pointer and array.docx"
CHUNK_SIZE = 512
OVERLAP = 0
METHOD = "percentile"
AMOUNT = 10


# ---- 로드 ----
print("파일:", FILE_PATH.name)
text = load_file(FILE_PATH)
print("텍스트 길이:", len(text), "자\n")

print("모델 로드중..")
model = SentenceTransformer("BAAI/bge-m3")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
print("로드 완료\n")


# ---- 청킹 ----
fixed = fixed_chunking(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
semantic = semantic_chunking(text, model, method=METHOD, amount=AMOUNT)

# ---- 청크 전체 나열  ----
print("=" * 70)
print("단순 분할 — 전체 청크")
print("=" * 70)
for i, chunk in enumerate(fixed):
    print("\n[청크 " + str(i + 1) + "]  " + str(len(chunk)) + "자")
    print(chunk)
print("\n총 " + str(len(fixed)) + "개 청크\n")

print("=" * 70)
print("의미 기반 분할 — 전체 청크")
print("=" * 70)
for i, chunk in enumerate(semantic):
    print("\n[청크 " + str(i + 1) + "]  " + str(len(chunk)) + "자")
    print(chunk)
print("\n총 " + str(len(semantic)) + "개 청크\n")

# ---- 요약 통계 (평균 토큰 등) ----
print_chunks(fixed, tokenizer, "단순 분할 요약")
print_chunks(semantic, tokenizer, "의미 기반 분할 요약")





    

    

    



