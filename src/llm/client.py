"""LLM 모델 로드 + 호출 공용 함수."""
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_llm(model_id, device=None):
    """토크나이저 + 모델 로드. Gemma4 12B 계열은 통합 멀티모달 아키텍처(전용 클래스
    필요) + bf16으로는 24GB를 넘어서 4bit 양자화 필수(QAT 학습된 체크포인트라 4bit로도
    품질 손실이 적음 - 실측으로 일반 12B-4bit보다 chrF/BLEU 전반에서 더 나음을 확인)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if "gemma-4" in model_id.lower():
        from transformers import Gemma4UnifiedForConditionalGeneration, BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map={"": 0}, quantization_config=quant_config
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model, device


def _clean(raw):
    """<think> 블록 제거 (thinking 모델이 실수로 새어나온 경우 대비)."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>")[0].strip()
    return cleaned


def generate(tokenizer, model, device, messages, max_new_tokens=512, do_sample=False,
             temperature=None, repetition_penalty=1.1, **extra_gen_kwargs):
    """채팅 메시지 1개로 생성 - generate_batch()에 크기 1짜리 배치로 위임(중복 제거).
    배치=1이면 왼쪽 패딩이 실질적으로 패딩 없음과 동일해서 동작 차이 없음."""
    return generate_batch(tokenizer, model, device, [messages], max_new_tokens=max_new_tokens,
                           do_sample=do_sample, temperature=temperature,
                           repetition_penalty=repetition_penalty, **extra_gen_kwargs)[0]


def generate_batch(tokenizer, model, device, list_of_messages, max_new_tokens=512,
                    do_sample=False, temperature=None, repetition_penalty=1.1, **extra_gen_kwargs):
    """여러 messages를 한 번의 generate() 호출로 배치 처리 (순서 그대로 텍스트 리스트 반환).
    causal LM 배치 생성은 왼쪽 패딩이 표준 - 모든 행의 입력 길이가 패딩 후 동일해져서,
    생성분 슬라이싱이 단건 generate()와 똑같이 input_ids.shape[1] 기준으로 가능해진다."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_inputs = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for messages in list_of_messages
    ]
    model_inputs = tokenizer(text_inputs, return_tensors="pt", padding=True).to(device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, repetition_penalty=repetition_penalty)
    if do_sample:
        gen_kwargs.update(temperature=temperature or 0.8, top_p=0.95)
    gen_kwargs.update(extra_gen_kwargs)

    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, **gen_kwargs)
        input_len = model_inputs.input_ids.shape[1]
        new_ids = generated_ids[:, input_len:]
        raw_texts = tokenizer.batch_decode(new_ids, skip_special_tokens=True)

    return [_clean(raw) for raw in raw_texts]
