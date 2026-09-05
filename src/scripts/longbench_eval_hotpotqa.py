"""LongBench(hotpotqa, 멀티홉 QA/평균 문맥 5만자+)로,
글자수 분할 vs 의미기반 분할이 top-k 검색 기반 QA 정확도(F1)에 미치는 영향을 비교한다."""
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("src")
from config import TOKENIZER, EMBED_MODEL
from chunking.fixed_chunker import fixed_chunking_tokens
from chunking.semantic_chunker import semantic_chunking
from retrieval.retriever import retrieve_top_k

LONGBENCH_JSONL = Path(
    "/tmp/claude-0/-root-jupyter-LLM-chunking/9a24a7b2-903a-4e97-81f3-4032fdcdabeb/scratchpad/longbench/data/hotpotqa.jsonl"
)
N_SAMPLES = 15
TOP_K = 5
MAX_TOKENS_PER_CHUNK = 150

device = "cuda" if torch.cuda.is_available() else "cpu"

print("데이터 로드 중...")
with open(LONGBENCH_JSONL) as f:
    samples = [json.loads(l) for l in f][:N_SAMPLES]
print(f"평가 샘플 수: {len(samples)}개")

print("모델 로드 중...")
embed_model = SentenceTransformer(EMBED_MODEL, device=device)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
llm_model = AutoModelForCausalLM.from_pretrained(TOKENIZER, torch_dtype=torch.float16, device_map="auto")

def call_llm(prompt, max_new_tokens=80):
    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    model_inputs = tokenizer([text_input], return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = llm_model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)
        generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
        raw = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>")[0].strip()
    return cleaned.strip()

# ---- SQuAD 스타일 정규화 + F1 (LongBench 등에서 쓰는 표준 QA 평가 방식) ----
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def f1_score(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def best_f1(pred, golds):
    return max(f1_score(pred, g) for g in golds)

def build_qa_prompt(question, retrieved_texts):
    body = "\n\n".join(f"[Excerpt {i+1}]\n{t}" for i, t in enumerate(retrieved_texts))
    return (
        "Answer the question using ONLY the excerpts below. Be concise (a short phrase, not a full sentence). "
        "If the answer is not in the excerpts, say \"unknown\".\n\n"
        f"{body}\n\nQuestion: {question}\nAnswer:"
    )

def run_method(label, chunker_fn):
    print(f"\n{'='*70}\n[{label}]\n{'='*70}")
    f1s = []
    for i, ex in enumerate(samples, 1):
        chunks = chunker_fn(ex["context"])
        if not chunks:
            continue
        retrieved = retrieve_top_k(ex["input"], chunks, embed_model, k=TOP_K)
        retrieved_texts = [r["chunk"].text for r in retrieved]
        prompt = build_qa_prompt(ex["input"], retrieved_texts)
        pred = call_llm(prompt)
        score = best_f1(pred, ex["answers"])
        f1s.append(score)
        print(f"  [{i}/{len(samples)}] 청크수={len(chunks)} F1={score:.3f} | 예측='{pred[:50]}' | 정답={ex['answers']}")
    avg = sum(f1s) / len(f1s) if f1s else 0.0
    print(f"\n[{label}] 평균 F1: {avg:.4f} (n={len(f1s)})")
    return avg, f1s

fixed_avg, fixed_f1s = run_method(
    "FIXED (토큰 기반 고정분할, 150토큰)",
    lambda text: fixed_chunking_tokens(text, tokenizer, max_tokens=MAX_TOKENS_PER_CHUNK)
)
semantic_avg, semantic_f1s = run_method(
    "SEMANTIC (BGE-M3 의미기반 분할, 완화된 상한 2000자)",
    lambda text: semantic_chunking(text, embed_model, method="percentile", amount=15, max_sentence_length=200, max_chunk_chars=2000)
)

print("\n" + "=" * 70)
print("최종 비교 (LongBench multifieldqa_en, n=%d)" % len(samples))
print("=" * 70)
print(f"Fixed    평균 F1: {fixed_avg:.4f}")
print(f"Semantic 평균 F1: {semantic_avg:.4f}")
print(f"차이(semantic - fixed): {semantic_avg - fixed_avg:+.4f}")
