import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DOCX_KOR_DIR
from preprocessing.loader import load_file
from chunking.fixed_chunker import fixed_chunking
from chunking.semantic_chunker import semantic_chunking
from retrieval.retriever import retrieve_top_k, print_retrieved

FILE_PATH = DOCX_KOR_DIR / "혼합코퍼스_A.docx"
K = 3
QUERY = "과거제도는 어떻게 나뉘었나?"

if FILE_PATH.exists():
    model = SentenceTransformer("BAAI/bge-m3")
    doc = load_file(FILE_PATH)
    text = doc.text

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
else:
    print("파일을 찾을 수 없습니다:", FILE_PATH)