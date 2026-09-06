"""LongBench 긴 문맥 QA로 fixed vs semantic 청킹이 top-k 검색 기반 QA 정확도(F1)에
미치는 영향을 비교한다. 사용법: python run_longbench.py [multifieldqa_en|hotpotqa]"""
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import ROOT, TOKENIZER, EMBED_MODEL
from chunking.fixed_chunker import fixed_chunking
from chunking.semantic_chunker import semantic_chunking
from retrieval.retriever import retrieve_top_k
from llm.client import load_llm, generate

LONGBENCH_DIR = ROOT / "data" / "longbench"
DATASETS = {
    "multifieldqa_en": dict(jsonl="multifieldqa_en.jsonl", n_samples=50, top_k=3),
    "hotpotqa": dict(jsonl="hotpotqa.jsonl", n_samples=15, top_k=5),
}
FIXED_CHUNK_SIZE = 600

dataset_name = sys.argv[1] if len(sys.argv) > 1 else "multifieldqa_en"
cfg = DATASETS[dataset_name]

device = "cuda" if torch.cuda.is_available() else "cpu"
with open(LONGBENCH_DIR / cfg["jsonl"]) as f:
    samples = [json.loads(l) for l in f][:cfg["n_samples"]]
print(f"[{dataset_name}] 평가 샘플 수: {len(samples)}개")

embed_model = SentenceTransformer(EMBED_MODEL, device=device)
tokenizer, llm_model, device = load_llm(TOKENIZER, device)


def normalize_answer(s):
    """SQuAD 스타일 정규화: 소문자화 + 구두점/관사 제거."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred, gold):
    """단어 단위 F1."""
    pred_tokens, gold_tokens = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision, recall = num_same / len(pred_tokens), num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_f1(pred, golds):
    """정답 후보 중 가장 높은 F1."""
    return max(f1_score(pred, g) for g in golds)


def build_qa_prompt(question, retrieved_texts):
    """검색된 조각으로 QA 프롬프트 구성."""
    body = "\n\n".join(f"[Excerpt {i + 1}]\n{t}" for i, t in enumerate(retrieved_texts))
    return (
        "Answer the question using ONLY the excerpts below. Be concise (a short phrase, not a full sentence). "
        "If the answer is not in the excerpts, say \"unknown\".\n\n"
        f"{body}\n\nQuestion: {question}\nAnswer:"
    )


def run_method(label, chunker_fn):
    """청킹 -> top-k 검색 -> QA 답변 -> F1 채점, 전체 샘플 평균."""
    print(f"\n{'=' * 70}\n[{label}]\n{'=' * 70}")
    f1s = []
    for i, ex in enumerate(samples, 1):
        chunks = chunker_fn(ex["context"])
        if not chunks:
            continue
        retrieved = retrieve_top_k(ex["input"], chunks, embed_model, k=cfg["top_k"])
        retrieved_texts = [r["chunk"].text for r in retrieved]
        prompt = build_qa_prompt(ex["input"], retrieved_texts)
        pred = generate(tokenizer, llm_model, device, [{"role": "user", "content": prompt}], max_new_tokens=80)
        score = best_f1(pred, ex["answers"])
        f1s.append(score)
        print(f"  [{i}/{len(samples)}] 청크수={len(chunks)} F1={score:.3f} | 예측='{pred[:50]}'")
    avg = sum(f1s) / len(f1s) if f1s else 0.0
    print(f"\n[{label}] 평균 F1: {avg:.4f} (n={len(f1s)})")
    return avg


fixed_avg = run_method("FIXED (글자수 고정분할)", lambda text: fixed_chunking(text, chunk_size=FIXED_CHUNK_SIZE))
semantic_avg = run_method(
    "SEMANTIC (BGE-M3 의미기반 분할)",
    lambda text: semantic_chunking(text, embed_model, method="percentile", amount=15, max_sentence_length=200, max_chunk_chars=2000),
)

print("\n" + "=" * 70)
print(f"최종 비교 ({dataset_name}, n={len(samples)})")
print("=" * 70)
print(f"Fixed    평균 F1: {fixed_avg:.4f}")
print(f"Semantic 평균 F1: {semantic_avg:.4f}")
print(f"차이(semantic - fixed): {semantic_avg - fixed_avg:+.4f}")
