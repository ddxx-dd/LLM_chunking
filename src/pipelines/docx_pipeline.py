"""DOCX 요약 실험: 관련 정책문서 묶음에서 질문 관련 내용 검색해 요약 문서 생성."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DOCX_ENG_DIR, RESULTS_DIR
from loaders import load_docx_bundle
from splitters import make_fixed_splitter, make_semantic_splitter
from eval.docx_compose import run_compose
from llm import to_lc_pipeline
from pipelines._common import setup_models

# CloudWay-24(가상 항공사)의 수하물 관련 정책 5개 - 서로 다른 문서에 흩어진 내용을
# 하나의 질문으로 가로질러 찾아야 하는 시나리오(원래 이 스크립트가 참조하던 "자료구조
# 강의노트 묶음"은 docx_kor 코퍼스가 대용량문서로 개편되며 삭제됐음 - 현재 코퍼스로 교체).
BUNDLE_DIR = DOCX_ENG_DIR / "RAG-Multi-Corpus" / "CloudWay-24"
BUNDLE_FILES = [
    BUNDLE_DIR / "Baggage Allowance.docx",
    BUNDLE_DIR / "Damaged Baggage Policy.docx",
    BUNDLE_DIR / "Dangerous Goods Policy.docx",
    BUNDLE_DIR / "Delayed Baggage.docx",
    BUNDLE_DIR / "Special Baggage Policy.docx",
]
QUERY = "What are CloudWay-24's baggage policies, including allowance, damage, and delays?"
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
        "semantic": make_semantic_splitter(embed_model, method="percentile", amount=15,
                                            max_chunk_tokens=500, min_chunk_tokens=128),
    }

    for label, splitter in chunkers.items():
        run_compose(label, splitter, bundle_docs, QUERY, TOP_K, embed_model, tokenizer, lc_llm, RESULTS_DIR)


if __name__ == "__main__":
    main()
