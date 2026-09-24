"""allganize 검색 품질 벤치마크: docx 로드 -> fixed/semantic 청킹 -> 검색 -> QA 채점.
LLM 생성 없음(순수 검색 단계만) - doc_name + target_answer 내용겹침으로 직접 채점,
LLM-judge는 안 씀(project 정책). *** 초안 - 아직 실행/검증 안 됨 ***"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from config import ALLGANIZE_DIR, RESULTS_DIR
from docx_track.loader import load_docx
from indexing import build_retriever

TOP_K = 5
# 문헌/공식 가이드에 이 문서 유형(표 섞인 리포트)에 맞는 정해진 값이 없어서
# (Pinecone 가이드·SemanticChunker 원 출처인 Kamradt 노트북 둘 다 "직접 실험해서
# 정하라"는 입장) 소규모 그리드로 직접 비교해서 정한다.
FIXED_CHUNK_SIZES = []  # 임시: 이미 재현 확인된 fixed_256/500/1000은 재실행 생략
SEMANTIC_BREAKPOINT_AMOUNTS = [90, 95]  # 임시: OOM으로 못 끝낸 90/95만 재실행


def load_qa_and_docs():
    """documents.csv로 QA의 target_file_name(원본 파일명) -> 실제 보유 docx 경로를
    매핑한다. 파일명이 전부 "{pdf 파일명 stem}.docx"로 정리되어 있어(실측 확인,
    45/45 일치) 확장자만 바꾸면 된다 - QA도 전부 이 45개 문서에 매칭되는 것만
    남아있어서(실측 확인) 별도 필터링이 필요 없다."""
    with open(ALLGANIZE_DIR / "documents.csv", encoding="utf-8") as f:
        doc_rows = list(csv.DictReader(f))
    docx_path_by_orig_name = {
        row["orig_file_name"]: ALLGANIZE_DIR / "docx" / row["domain"] / (Path(row["file_name"]).stem + ".docx")
        for row in doc_rows
    }

    with open(ALLGANIZE_DIR / "rag_evaluation_result.csv", encoding="utf-8") as f:
        qa_rows = list(csv.DictReader(f))

    return qa_rows, docx_path_by_orig_name


def _normalize(text):
    """SQuAD 방식 정규화: 소문자화 + 구두점 제거 + 공백 정리."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def f1_score(target_answer, chunk_text):
    """SQuAD 방식 word-level F1 - 페이지 번호가 없어서(docx는 고정 페이지 개념
    자체가 없음, 실측 확인됨) 쓰는 대체 채점 방식. recall만 보는 단순 겹침 비율과
    달리, precision도 같이 봐서 "쓸데없이 큰 청크가 우연히 단어 몇 개 포함하는"
    경우에 벌점을 준다(실측 예시로 확인: recall만이면 0.6인데 F1은 0.029까지 떨어짐).
    Counter 교집합을 써서 중복 단어도 정확히 처리한다(SQuAD 공식 구현과 동일)."""
    gold_tokens = _normalize(target_answer).split()
    pred_tokens = _normalize(chunk_text).split()
    if not gold_tokens or not pred_tokens:
        return 0.0
    num_common = sum((Counter(gold_tokens) & Counter(pred_tokens)).values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_chunker(label, splitter, docs_by_orig_name, qa_rows, embeddings):
    print(f"\n[{label}] 청킹 중...")
    chunks = splitter.split_documents(list(docs_by_orig_name.values()))
    print(f"  총 청크 {len(chunks)}개")

    retriever = build_retriever(chunks, embeddings, k=TOP_K)

    results = []
    for i, qa in enumerate(qa_rows, 1):
        retrieved = retriever.invoke(qa["question"])
        target_docx_name = docs_by_orig_name[qa["target_file_name"]].metadata["name"]

        rank = next((r + 1 for r, c in enumerate(retrieved) if c.metadata["name"] == target_docx_name), None)
        best_f1 = max(
            (f1_score(qa["target_answer"], c.page_content)
             for c in retrieved if c.metadata["name"] == target_docx_name),
            default=0.0,
        )
        results.append({
            "context_type": qa["context_type"],
            "hit": rank is not None,
            "mrr": 1 / rank if rank else 0.0,
            "f1": best_f1,
        })
        if i % 50 == 0:
            print(f"  {i}/{len(qa_rows)} 질문 처리", flush=True)
    return results


def _agg(rows):
    n = len(rows)
    if n == 0:
        return None
    return dict(
        n=n,
        hit_rate=sum(r["hit"] for r in rows) / n,
        mrr=sum(r["mrr"] for r in rows) / n,
        f1=sum(r["f1"] for r in rows) / n,
    )


def summarize(label, results):
    overall = _agg(results)
    print(f"\n=== [{label}] 전체 ({len(results)}문항) ===")
    print(overall)
    table_rows = [r for r in results if r["context_type"] == "table"]
    print(f"=== [{label}] context_type=table만 ({len(table_rows)}문항) ===")
    print(_agg(table_rows))
    return overall


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    qa_rows, docx_path_by_orig_name = load_qa_and_docs()
    print(f"QA {len(qa_rows)}개, 대상 문서 {len(docx_path_by_orig_name)}개")

    print("문서 로드 중...")
    docs_by_orig_name = {}
    for i, (name, path) in enumerate(docx_path_by_orig_name.items(), 1):
        docs_by_orig_name[name] = load_docx(str(path))
        print(f"  [{i}/{len(docx_path_by_orig_name)}] {path.name}", flush=True)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings": True})

    configs = [(f"fixed_{size}", CharacterTextSplitter(separator="", chunk_size=size, chunk_overlap=0))
               for size in FIXED_CHUNK_SIZES]
    configs += [(f"semantic_{amount}", SemanticChunker(
                    embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=amount))
                for amount in SEMANTIC_BREAKPOINT_AMOUNTS]

    overall_by_label = {}
    for i, (label, splitter) in enumerate(configs, 1):
        print(f"\n### 설정 {i}/{len(configs)}: {label} ###", flush=True)
        results = score_chunker(label, splitter, docs_by_orig_name, qa_rows, embeddings)
        overall_by_label[label] = summarize(label, results)

    fixed_labels = [f"fixed_{size}" for size in FIXED_CHUNK_SIZES]
    semantic_labels = [f"semantic_{amount}" for amount in SEMANTIC_BREAKPOINT_AMOUNTS]
    print("\n=== 그리드 서치 요약 (hit_rate 기준) ===")
    for label in fixed_labels + semantic_labels:
        print(f"  {label}: {overall_by_label[label]}")
    if fixed_labels:
        best_fixed = max(fixed_labels, key=lambda l: overall_by_label[l]["hit_rate"])
        print(f"최적 fixed: {best_fixed} {overall_by_label[best_fixed]}")
    if semantic_labels:
        best_semantic = max(semantic_labels, key=lambda l: overall_by_label[l]["hit_rate"])
        print(f"최적 semantic: {best_semantic} {overall_by_label[best_semantic]}")


if __name__ == "__main__":
    main()
