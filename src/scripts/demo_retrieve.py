
"""
top-k 검색 동작 확인용 데모.

혼합 코퍼스에서 단순 분할과 의미 분할의 검색 결과를 비교한다.
결과는 docs/baseline_report.md 참조.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer

from config import DOCX_KOR_DIR
from preprocessing.loader import load_file
from chunking.fixed_chunker import fixed_chunking
from chunking.semantic_chunker import semantic_chunking
from retrieval.retriever import retrieve_top_k, print_retrieved


FILE_PATH = DOCX_KOR_DIR / "혼합코퍼스_A.docx"
K = 3
QUERY = "과거제도는 어떻게 나뉘었나?"


model = SentenceTransformer("BAAI/bge-m3")
text = load_file(FILE_PATH)

fixed_chunks = fixed_chunking(text, 512, 0)
semantic_chunks = semantic_chunking(text, model, method="percentile", amount=10)

print("단순 분할: " + str(len(fixed_chunks)) + "개")
print("의미 분할: " + str(len(semantic_chunks)) + "개\n")

print("=" * 60)
print("단순 분할에서 검색")
print("=" * 60)
print_retrieved(QUERY, retrieve_top_k(QUERY, fixed_chunks, model, K))

print("=" * 60)
print("의미 분할에서 검색")
print("=" * 60)
print_retrieved(QUERY, retrieve_top_k(QUERY, semantic_chunks, model, K))