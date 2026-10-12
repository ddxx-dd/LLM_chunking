"""모델 로딩: 임베딩(bge-m3), 토큰 세기, LLM(Gemma 4비트).

  load_embedder()          bge-m3 SentenceTransformer (기본 fp16, GPU)
  make_encode(model)       글 목록 -> float32 (n, 1024), 행마다 길이 1 (코사인 = 내적)
  make_token_counter()     글 목록 -> 토큰 수 목록 (bge-m3 토크나이저, 특수 토큰 제외)
  load_llm()               Gemma 4비트 (토크나이저, 모델)
  generate(llm, prompt)    대화 형식 한 번 생성 (탐욕 디코딩)

실행 전에 `source env.sh`(HF_HOME 프로젝트 캐시, HF_HUB_OFFLINE=1)가 필요하다.
무거운 라이브러리는 함수 안에서 import한다(테스트는 가짜 모델로 모델 없이 돈다).
"""
import numpy as np

from common.config import EMBED_DIM, EMBED_MODEL, LLM_MODEL

EMBED_BATCH = 32


def load_embedder(precision="fp16", device="cuda"):
    """bge-m3. fp16은 fp32와 결과가 거의 같고(옛 실험: top-3 99% 일치) 약 4배 빠르다."""
    if precision not in ("fp16", "fp32"):
        raise ValueError(f"precision은 fp16 또는 fp32: {precision!r}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device=device)
    if precision == "fp16":
        model.half()
    return model


def make_encode(model, batch_size=EMBED_BATCH):
    """model.encode를 감싼다. 결과는 항상 float32, 행마다 길이 1.
    fp16 모델은 fp16 상태에서 정규화한 값이라 float32로 바꾸면 길이가 1에서 조금 벗어난다(실측 1.0002) → float32에서 한 번 더 정규화한다."""
    def encode(texts):
        texts = list(texts)
        if not texts:
            return np.zeros((0, EMBED_DIM), np.float32)
        vecs = np.asarray(model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    return encode


def get_tokenizer(name=EMBED_MODEL):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def make_token_counter(tokenizer=None):
    """글 목록 -> 토큰 수 목록. 특수 토큰은 세지 않는다. 기본은 bge-m3 토크나이저(모든 청커의 크기 기준)."""
    tok = tokenizer if tokenizer is not None else get_tokenizer()
    if hasattr(tok, "model_max_length"):
        tok.model_max_length = 10 ** 9            # 토큰 수를 세는 용도라 모델 입력 길이 제한(8192) 경고를 끈다. 글을 자르지 않으므로 긴 글도 끝까지 센다

    def count_many(texts):
        texts = list(texts)
        out = []
        for i in range(0, len(texts), 4096):
            out.extend(len(ids) for ids in tok(texts[i:i + 4096], add_special_tokens=False)["input_ids"])
        return out
    return count_many


def load_llm(name=LLM_MODEL):
    """Gemma를 4비트(nf4, 계산은 bf16)로 GPU 0에 올린다. 반환: (토크나이저, 모델)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(name)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16)
    return tok, model.eval()


def generate(llm, prompt, max_new_tokens=256):
    """사용자 메시지 하나로 한 번 생성한다(탐욕 디코딩이라 같은 입력이면 같은 출력). 반환: 생성된 글."""
    import torch

    tok, model = llm
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
