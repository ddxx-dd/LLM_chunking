"""모델 로딩 점검 (A1). 실제 모델로 임베딩·토큰 세기·LLM 생성이 되는지와 속도·GPU 메모리를 출력한다.

  source env.sh
  .venv311/bin/python scripts/check_models.py [--no-llm] [--sample-doc data/processed/docx_corpus/<doc>.json]

결과는 화면에 출력하고 results/check_models.json 에도 저장한다.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from common.config import LLM_MODEL, PROCESSED_DIR, RESULTS_DIR  # noqa: E402
from common.models import generate, get_tokenizer, load_embedder, load_llm, make_encode, make_token_counter  # noqa: E402

SENTENCES = [
    "The cat sat on the warm windowsill all afternoon.",
    "A kitten was sleeping in the sun by the window.",
    "The central bank raised interest rates by 0.25 percentage points.",
    "고양이가 오후 내내 따뜻한 창가에 앉아 있었다.",
]


def say(*a):
    print(*a, flush=True)


def gib():
    import torch

    return torch.cuda.max_memory_allocated() / 2 ** 30


def sample_texts(path, limit=400):
    """처리된 문서 JSON 하나에서 블록 글을 꺼낸다(속도 측정용)."""
    if path is None:
        path = sorted((PROCESSED_DIR / "docx_corpus").glob("*.json"))[0]
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    texts = [b["text"] for b in doc.get("blocks", []) if b.get("text", "").strip()]
    return Path(path).name, texts[:limit]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-llm", action="store_true", help="Gemma 점검을 건너뛴다")
    ap.add_argument("--sample-doc", help="속도 측정에 쓸 처리된 문서 JSON")
    ns = ap.parse_args(argv)
    import torch

    rep = {"gpu": torch.cuda.get_device_name(0), "torch_cuda": torch.version.cuda}
    say(f"GPU: {rep['gpu']} (torch CUDA {rep['torch_cuda']})")

    # 1. 임베딩
    t = time.time()
    model = load_embedder("fp16")
    encode = make_encode(model)
    rep["embed_load_s"] = round(time.time() - t, 1)
    v = encode(SENTENCES)
    sims = (v.astype(np.float64) @ v.T.astype(np.float64)).round(3)
    rep["embed_shape"] = list(v.shape)
    rep["embed_norms"] = [round(float(x), 4) for x in np.linalg.norm(v, axis=1)]
    rep["cosine_matrix"] = sims.tolist()
    say(f"\n[1] bge-m3 fp16 로드 {rep['embed_load_s']}s, 벡터 {tuple(v.shape)}, 노름 {rep['embed_norms']}")
    say("    코사인 (문장: 0 고양이 창가 / 1 새끼고양이 창가 / 2 금리 인상 / 3 고양이 창가(한국어))")
    for i, row in enumerate(sims):
        say(f"    {i}: {row.tolist()}")

    # 2. 속도 (실제 문서 블록)
    count_many = make_token_counter()
    name, texts = sample_texts(ns.sample_doc)
    ntok = sum(count_many(texts))
    encode(texts[:8])                                   # 예열
    torch.cuda.synchronize()
    t = time.time()
    encode(texts)
    torch.cuda.synchronize()
    sec = time.time() - t
    rep["throughput"] = {"doc": name, "n_texts": len(texts), "tokens": ntok, "seconds": round(sec, 2), "tokens_per_s": round(ntok / sec)}
    say(f"\n[2] 속도: {name}의 블록 {len(texts)}개, {ntok}토큰을 {sec:.2f}s에 임베딩 → {ntok / sec:,.0f}토큰/s, GPU 최대 {gib():.2f}GiB")

    # 3. fp16 vs fp32
    model32 = load_embedder("fp32")
    v32 = make_encode(model32)(SENTENCES + texts[:50])
    v16 = encode(SENTENCES + texts[:50])
    agree = (v16 * v32).sum(1)
    rep["fp16_vs_fp32_cos"] = {"min": round(float(agree.min()), 5), "mean": round(float(agree.mean()), 5)}
    say(f"\n[3] 같은 글의 fp16·fp32 벡터 코사인: 최소 {agree.min():.5f}, 평균 {agree.mean():.5f} ({len(agree)}개)")
    del model32
    torch.cuda.empty_cache()

    # 4. 토큰 세기
    gemma_count = make_token_counter(get_tokenizer(LLM_MODEL))
    rep["token_counts"] = [{"text": s, "bge_m3": a, "gemma": b} for s, a, b in zip(SENTENCES, count_many(SENTENCES), gemma_count(SENTENCES))]
    say("\n[4] 토큰 수 (특수 토큰 제외): bge-m3 / Gemma")
    for r in rep["token_counts"]:
        say(f"    {r['bge_m3']:3d} / {r['gemma']:3d}  {r['text']}")

    # 5. LLM
    if not ns.no_llm:
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        llm = load_llm()
        rep["llm_load_s"] = round(time.time() - t, 1)
        say(f"\n[5] Gemma 4비트 로드 {rep['llm_load_s']}s, GPU 최대 {gib():.2f}GiB (bge-m3 포함)")
        for prompt in ["In one sentence, what is retrieval-augmented generation?", "검색 증강 생성(RAG)을 한 문장으로 설명해 줘."]:
            t = time.time()
            out = generate(llm, prompt, max_new_tokens=80)
            sec = time.time() - t
            rep.setdefault("llm_samples", []).append({"prompt": prompt, "output": out, "seconds": round(sec, 1)})
            say(f"    Q: {prompt}\n    A({sec:.1f}s): {out}")
        rep["gpu_peak_gib"] = round(gib(), 2)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "check_models.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    say(f"\n저장: {RESULTS_DIR / 'check_models.json'}")
    sys.stdout.flush()
    os._exit(0)                     # CUDA 정리 단계에서 프로세스가 오래 매달린 적이 있어 바로 끝낸다


if __name__ == "__main__":
    main()
