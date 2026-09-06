"""LLM 모델 로드 + 호출 공용 함수."""
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_llm(model_id, device=None):
    """토크나이저 + 모델 로드."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model, device


def generate(tokenizer, model, device, messages, max_new_tokens=512, do_sample=False,
             temperature=None, repetition_penalty=1.1, **extra_gen_kwargs):
    """채팅 메시지로 생성, <think> 블록 제거 후 텍스트만 반환."""
    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    model_inputs = tokenizer([text_input], return_tensors="pt").to(device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, repetition_penalty=repetition_penalty)
    if do_sample:
        gen_kwargs.update(temperature=temperature or 0.8, top_p=0.95)
    gen_kwargs.update(extra_gen_kwargs)

    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, **gen_kwargs)
        generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
        raw = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>")[0].strip()
    return cleaned
