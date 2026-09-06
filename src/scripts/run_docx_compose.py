"""DOCX 요약 실험: 자료구조 강의노트 묶음에서 질문 관련 내용 검색해 요약 문서 생성."""
import sys
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DOCX_KOR_DIR, RESULTS_DIR, EMBED_MODEL, TOKENIZER
from preprocessing.loader import load_docx_bundle
from chunking.fixed_chunker import fixed_chunking
from chunking.semantic_chunker import semantic_chunking
from llm.client import load_llm
from eval.docx_compose import run_compose

RESULTS_DIR.mkdir(exist_ok=True)

BUNDLE_FILES = [
    DOCX_KOR_DIR / "1-1.배열.docx",
    DOCX_KOR_DIR / "2-1.스택.docx",
    DOCX_KOR_DIR / "3-1.큐.docx",
    DOCX_KOR_DIR / "4-1.이진트리.docx",
    DOCX_KOR_DIR / "5-1.탐색트리.docx",
]
QUERY = "큐(Queue)란 무엇이고 어떻게 동작하는지, 특징과 활용을 설명해줘"
TOP_K = 8
FIXED_CHUNK_SIZE = 600

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"실행 디바이스: {device.upper()}")
bundle_docs = load_docx_bundle([p for p in BUNDLE_FILES if p.exists()])
print("문서 묶음 로드 완료:", ", ".join(d.name for d in bundle_docs))

embed_model = SentenceTransformer(EMBED_MODEL, device=device)
tokenizer, llm_model, device = load_llm(TOKENIZER, device)
print("모델 준비 완료")

CHUNKERS = {
    "fixed": lambda text: fixed_chunking(text, chunk_size=FIXED_CHUNK_SIZE),
    "semantic": lambda text: semantic_chunking(text, embed_model, method="percentile", amount=15, max_chunk_chars=2000),
}

for label, chunker_fn in CHUNKERS.items():
    run_compose(label, chunker_fn, bundle_docs, QUERY, TOP_K, embed_model, tokenizer, llm_model, device, RESULTS_DIR)
