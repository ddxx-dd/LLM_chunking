"""검색(코퍼스 단위, docx/pdf 공통) - 데이터셋 전체 문서를 청커 하나로 청킹해서
InMemoryVectorStore 인덱스 하나에 넣고 QA 질문으로 top-5 검색, hit@5/MRR/F1 채점.
LLM 생성 없음(순수 검색 단계만) - doc_name + target_answer 내용겹침으로 직접 채점.

청커 자체는 안 만든다 - 트랙 pipeline.py가 chunkers.py로 이미 만들어둔 splitter를
넘겨받아서 쓴다(데이터셋마다 다른 설정값은 트랙 pipeline.py의 CHUNKERS에 있음)."""
import csv
import json
import re
from collections import Counter
from pathlib import Path

from langchain_core.vectorstores import InMemoryVectorStore

from config import ALLGANIZE_DIR, VECTARA_DIR
from docx_track.loader import load_docx_bundle
from pdf_track.loader import load_pdf_bundle

TOP_K = 5


def build_retriever(documents, embeddings, k=5):
    store = InMemoryVectorStore(embeddings)
    store.add_documents(documents)
    return store.as_retriever(search_kwargs={"k": k})


def _normalize(text):
    """SQuAD 방식 정규화: 소문자화 + 구두점 제거 + 공백 정리."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def f1_score(target_answer, chunk_text):
    """SQuAD 방식 word-level F1 - 페이지 번호가 없어서 쓰는 대체 채점 방식."""
    gold = _normalize(target_answer).split()
    pred = _normalize(chunk_text).split()
    if not gold or not pred:
        return 0.0
    common = sum((Counter(gold) & Counter(pred)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def load_docx_qa():
    """documents.csv로 QA의 target_file_name -> 실제 docx 경로를 매핑(실측 확인:
    45/45 일치, 별도 필터링 불필요)."""
    with open(ALLGANIZE_DIR / "documents.csv", encoding="utf-8") as f:
        doc_rows = list(csv.DictReader(f))
    path_by_orig = {row["orig_file_name"]: ALLGANIZE_DIR / "docx" / row["domain"] / (Path(row["file_name"]).stem + ".docx")
                     for row in doc_rows}
    with open(ALLGANIZE_DIR / "rag_evaluation_result.csv", encoding="utf-8") as f:
        qa_rows = list(csv.DictReader(f))
    qas = [{"question": r["question"], "target_answer": r["target_answer"],
            "target_file": path_by_orig[r["target_file_name"]], "context_type": r["context_type"]}
           for r in qa_rows if r["target_file_name"] in path_by_orig]
    return qas, sorted(set(path_by_orig.values()))


def load_pdf_qa():
    """vectara open_ragbench의 queries/answers/qrels.json으로 질문 -> 정답 문서를 매핑."""
    queries = json.loads((VECTARA_DIR / "queries.json").read_text(encoding="utf-8"))
    answers = json.loads((VECTARA_DIR / "answers.json").read_text(encoding="utf-8"))
    qrels = json.loads((VECTARA_DIR / "qrels.json").read_text(encoding="utf-8"))
    files = sorted(VECTARA_DIR.glob("*.pdf"))
    stem_to_path = {p.stem: p for p in files}
    qas = []
    for qid, q in queries.items():
        gold_stem = qrels.get(qid, {}).get("doc_id")
        if gold_stem not in stem_to_path:
            continue
        qas.append({"question": q["query"], "target_answer": answers.get(qid, ""),
                     "target_file": stem_to_path[gold_stem], "context_type": None})
    return qas, files


def load_qa(dataset):
    return load_docx_qa() if dataset == "docx" else load_pdf_qa()


def load_docs(dataset, files):
    return load_docx_bundle([str(f) for f in files]) if dataset == "docx" else load_pdf_bundle([str(f) for f in files])


def score_chunker(label, splitter, docs, qas, embeddings):
    print(f"\n[{label}] 청킹 중...", flush=True)
    chunks = splitter.split_documents(docs)
    print(f"  총 청크 {len(chunks)}개", flush=True)
    retriever = build_retriever(chunks, embeddings, k=TOP_K)

    results = []
    for i, qa in enumerate(qas, 1):
        retrieved = retriever.invoke(qa["question"])
        target_name = qa["target_file"].name
        rank = next((r + 1 for r, c in enumerate(retrieved) if c.metadata["name"] == target_name), None)
        best_f1 = max((f1_score(qa["target_answer"], c.page_content)
                        for c in retrieved if c.metadata["name"] == target_name), default=0.0)
        results.append({"context_type": qa["context_type"], "hit": rank is not None,
                         "mrr": 1 / rank if rank else 0.0, "f1": best_f1})
        if i % 50 == 0:
            print(f"  {i}/{len(qas)} 질문 처리", flush=True)
    return results, len(chunks), retriever


def aggregate(rows):
    n = len(rows)
    if n == 0:
        return None
    return {"n": n, "hit_rate": sum(r["hit"] for r in rows) / n,
            "mrr": sum(r["mrr"] for r in rows) / n, "f1": sum(r["f1"] for r in rows) / n}


def run(dataset, chunker, splitter, embeddings, limit=None):
    """run.py/트랙 pipeline.py가 부르는 단일 설정 실행 - QA limit개, 문서는 전부
    인덱싱(검색이 여러 문서 중에서 정답을 찾는 문제이려면 코퍼스 전체가 후보
    풀이어야 함). retriever도 같이 반환 - summary.run()이 같은 인덱스를 재사용
    (청킹을 두 번 안 함). splitter는 트랙 pipeline.py의 CHUNKERS[chunker]."""
    qas, files = load_qa(dataset)
    if limit:
        qas = qas[:limit]
    docs = load_docs(dataset, files)
    results, n_chunks, retriever = score_chunker(f"{dataset}/{chunker}", splitter, docs, qas, embeddings)
    overall = aggregate(results)
    table_rows = [r for r in results if r["context_type"] == "table"]
    return {"overall": overall, "table": aggregate(table_rows), "n_chunks": n_chunks, "n_docs": len(files),
            "retriever": retriever, "qas": qas}
