"""여러 docx 문서 묶음(자료구조 소주제 5개 + 성격이 다른 혼합코퍼스 1개)에 대해
글자수 분할 vs 의미기반 분할이 top-k 검색 및 요약 품질에 미치는 영향을 비교한다."""
import re
import sys
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("src")
from config import DOCX_KOR_DIR, TOKENIZER, EMBED_MODEL
from preprocessing.loader import load_docx_bundle
from chunking.fixed_chunker import fixed_chunking_tokens
from chunking.semantic_chunker import semantic_chunking
from pipeline.bundle import chunk_bundle, build_summary_prompt
from retrieval.retriever import retrieve_top_k_bundle

BUNDLE_FILES = [
    DOCX_KOR_DIR / "1-1.배열.docx",
    DOCX_KOR_DIR / "2-1.스택.docx",
    DOCX_KOR_DIR / "3-1.큐.docx",
    DOCX_KOR_DIR / "4-1.이진트리.docx",
    DOCX_KOR_DIR / "5-1.탐색트리.docx",
    DOCX_KOR_DIR / "혼합코퍼스_A.docx",   # 성격이 다른 문서(관련없는 주제 포함) -> 관련성 필터링 테스트용
]
QUERY = "스택과 큐의 차이점을 요약해줘"
TOP_K = 5
MAX_TOKENS_PER_CHUNK = 150

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"실행 디바이스: {device.upper()}")

print("문서 묶음 로드 중...")
docs = load_docx_bundle([p for p in BUNDLE_FILES if p.exists()])
for d in docs:
    print(f"  - {d.name} (유닛 {len(d.units)}개, 글자수 {len(d.text)})")

print("\n임베딩/LLM 모델 로드 중...")
embed_model = SentenceTransformer(EMBED_MODEL, device=device)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
llm_model = AutoModelForCausalLM.from_pretrained(TOKENIZER, torch_dtype=torch.float16, device_map="auto")

def call_llm(prompt, max_new_tokens=512):
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
    return cleaned

def run_pipeline(label, chunker_fn):
    print("\n" + "=" * 70)
    print(f"[{label}]")
    print("=" * 70)
    bundle_chunks = chunk_bundle(docs, chunker_fn)
    print(f"총 청크 수: {len(bundle_chunks)}개 (문서별: " +
          ", ".join(f"{d.name}={sum(1 for b in bundle_chunks if b['doc_name']==d.name)}" for d in docs) + ")")

    retrieved = retrieve_top_k_bundle(QUERY, bundle_chunks, embed_model, k=TOP_K)
    print(f"\nTop-{TOP_K} 검색 결과:")
    for i, item in enumerate(retrieved, 1):
        preview = item["chunk"].text.strip().replace("\n", " / ")[:60]
        print(f"  [{i}위] {item['doc_name']} | 유사도={item['score']:.3f} | {preview}")

    prompt = build_summary_prompt(QUERY, retrieved)
    summary = call_llm(prompt)
    print(f"\n[요약 결과]\n{summary}")

    doc_names_hit = set(item["doc_name"] for item in retrieved)
    return {"n_chunks": len(bundle_chunks), "retrieved_docs": doc_names_hit, "summary": summary}

fixed_result = run_pipeline(
    "FIXED (토큰 기반 고정분할, 150토큰)",
    lambda text: fixed_chunking_tokens(text, tokenizer, max_tokens=MAX_TOKENS_PER_CHUNK)
)
semantic_result = run_pipeline(
    "SEMANTIC (BGE-M3 의미기반 분할, 완화된 상한 2000자)",
    lambda text: semantic_chunking(text, embed_model, method="percentile", amount=15, max_sentence_length=200, max_chunk_chars=2000)
)

print("\n" + "=" * 70)
print("비교 요약")
print("=" * 70)
print(f"질문: {QUERY}")
print(f"Fixed    : 청크 {fixed_result['n_chunks']}개, 검색된 문서 = {fixed_result['retrieved_docs']}")
print(f"Semantic : 청크 {semantic_result['n_chunks']}개, 검색된 문서 = {semantic_result['retrieved_docs']}")
