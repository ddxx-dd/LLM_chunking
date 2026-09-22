"""DOCX 요약 실험: 관련 정책문서 묶음에서 질문 관련 내용 검색해 요약 문서 생성."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import ALLGANIZE_DIR, RESULTS_DIR
from docx_track.loader import load_docx_bundle
from splitters import make_fixed_splitter, SemanticTextSplitter
from docx_track.docx_compose import run_compose
from llm import to_lc_pipeline
from common import setup_models

# TODO: 코퍼스가 allganize로 교체됨에 따라 실제 실험 설계(묶을 문서/질문)를 다시
# 정해야 함 - 폴더 재편 시점엔 손대지 않고, 파이프라인을 실제로 돌릴 때 같이 고치기로
# 함(이전엔 RAG-Multi-Corpus의 CloudWay-24 항공사 수하물 정책 5개 문서를 썼음).
BUNDLE_DIR = ALLGANIZE_DIR / "docx" / "public"
BUNDLE_FILES = sorted(BUNDLE_DIR.glob("*.docx"))
QUERY = "TODO: allganize 코퍼스에 맞는 질문으로 교체"
TOP_K = 8
FIXED_CHUNK_SIZE = 600


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    device, embed_model, tokenizer, llm_model = setup_models()
    lc_llm = to_lc_pipeline(tokenizer, llm_model)

    bundle_docs = load_docx_bundle([p for p in BUNDLE_FILES if p.exists()])
    print("문서 묶음 로드 완료:", ", ".join(d.metadata["name"] for d in bundle_docs))

    chunkers = {
        "fixed": make_fixed_splitter(chunk_size=FIXED_CHUNK_SIZE),
        "semantic": SemanticTextSplitter(embed_model, method="percentile", amount=15,
                                          max_chunk_tokens=500, min_chunk_tokens=128),
    }

    for label, splitter in chunkers.items():
        run_compose(label, splitter, bundle_docs, QUERY, TOP_K, embed_model, tokenizer, lc_llm, RESULTS_DIR)


if __name__ == "__main__":
    main()
