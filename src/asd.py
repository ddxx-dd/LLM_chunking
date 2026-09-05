"""
run_llm_subtitle_pipeline.py - 추론형 LLM의 <think> 생각 블록을 완벽히 도려내고 순수 번역만 추출하는 스크립트
"""
import os
import sys
import re
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# src 폴더 모듈 임포트
sys.path.append("src")
from config import SRT_ENG_DIR, SRT_KOR_DIR, RESULTS_DIR, EMBED_MODEL, TOKENIZER
from core import Chunk
from preprocessing.loader import load_srt
from chunking.fixed_chunker import fixed_chunking
from chunking.semantic_chunker import semantic_chunking
from pipeline.mapper import build_prompt, parse_marked, merge_to_units, write_srt

# 0. 번역 방향 설정: "en2ko"(영->한) 또는 "ko2en"(한->영)
DIRECTION = "en2ko"
SUB_UNITS = 70

DIRECTION_CONFIG = {
    "en2ko": dict(
        input_srt=SRT_ENG_DIR / "The_Truman_Show_Eng.srt",
        fixed_out_path=RESULTS_DIR / "트루먼쇼_fixed_kor_llm.srt",
        semantic_out_path=RESULTS_DIR / "트루먼쇼_semantic_kor_llm.srt",
        system_msg=(
            "You are a subtitle translator. Translate English subtitles into natural Korean.\n"
            "STRICT RULES:\n"
            "1. Output ONLY the numbered Korean translations like [1] 안녕하세요.\n"
            "2. Do NOT explain words, grammar, or your thought process.\n"
            "3. Do NOT output English."
        ),
        ex_user="[1] What is that?\n[2] Where are you going?",
        ex_assistant="[1] 저게 뭐야?\n[2] 어디 가세요?",
        label="영→한",
    ),
    "ko2en": dict(
        input_srt=SRT_KOR_DIR / "트루먼쇼.srt",
        fixed_out_path=RESULTS_DIR / "트루먼쇼_fixed_eng_llm.srt",
        semantic_out_path=RESULTS_DIR / "트루먼쇼_semantic_eng_llm.srt",
        system_msg=(
            "You are a subtitle translator. Translate Korean subtitles into natural English.\n"
            "STRICT RULES:\n"
            "1. Output ONLY the numbered English translations like [1] Hello.\n"
            "2. Do NOT explain words, grammar, or your thought process.\n"
            "3. Do NOT output Korean."
        ),
        ex_user="[1] 아~뭐여~?\n[2] 어디 가세요?",
        ex_assistant="[1] What is that?\n[2] Where are you going?",
        label="한→영",
    ),
}

cfg = DIRECTION_CONFIG[DIRECTION]

# 1. 파일 및 모델 경로 설정
input_srt = cfg["input_srt"]
if not input_srt.exists():
    input_srt = Path("data/samples/sample_en.srt" if DIRECTION == "en2ko" else "data/samples/sample_ko.srt")

fixed_out_path = cfg["fixed_out_path"]
semantic_out_path = cfg["semantic_out_path"]
os.makedirs(RESULTS_DIR, exist_ok=True)

LLM_MODEL_ID = TOKENIZER       # "Qwen/Qwen3-4B"
EMBED_MODEL_ID = EMBED_MODEL   # "BAAI/bge-m3"

print("=" * 70)
print(f"🚀 [트루먼쇼 {cfg['label']} 자막 번역 및 병합 파이프라인 시작]")
print("=" * 70)

# 2. 모델 GPU 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"• 실행 디바이스: {device.upper()}")

print("• [1/2] BGE-M3 임베딩 모델 로드 중...")
embed_model = SentenceTransformer(EMBED_MODEL_ID, device=device)

print(f"• [2/2] {LLM_MODEL_ID} LLM 로드 중 (torch_dtype=float16)...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("✅ 모델 준비 완료!\n")

def _is_hangul_token(clean):
    return any("가" <= ch <= "힣" for ch in clean)

def _is_foreign_script_token(clean):
    """한중일 중 한글이 아닌 문자(중국어 한자, 일본어 가나 등)가 섞인 토큰인지 확인.
    영어를 막았을 때 모델이 한국어 대신 다른 외국어로 새는 것(off-target)까지 막기 위함."""
    return any(
        "一" <= ch <= "鿿"   # 한자(중국어/한자어 공용 영역)
        or "぀" <= ch <= "ヿ"  # 히라가나/가타카나
        for ch in clean
    )

def has_foreign_contamination(s):
    return bool(re.search(r"[一-鿿぀-ヿ]", s))

def _compute_bad_ids(ban):
    """생성 후보에서 배제할 토큰 id 목록 계산.
    ban="english" -> 영어 단어 토큰 배제 (한글 강제용, en2ko 방향)
    ban="hangul"  -> 한글 토큰 배제 (영어 강제용, ko2en 방향)
    어느 쪽이든 한자/가나 토큰은 함께 배제한다 (엉뚱한 제3언어로 새는 것 방지)."""
    bad_ids = []
    for tok_str, tok_id in tokenizer.get_vocab().items():
        clean = tok_str.lstrip("Ġ▁").strip()
        if not clean:
            continue
        if _is_foreign_script_token(clean):
            bad_ids.append([tok_id])
        elif ban == "english" and len(clean) >= 3 and clean.isascii() and clean.isalpha():
            bad_ids.append([tok_id])
        elif ban == "hangul" and _is_hangul_token(clean):
            bad_ids.append([tok_id])
    return bad_ids

# 최종 강제 수단: 재시도로도 안 뚫리면 반대쪽 언어 토큰을 생성 후보에서 배제해
# 목표 언어가 나오도록 기계적으로 강제한다 (en2ko는 영어 배제, ko2en은 한글 배제)
_FORCE_BAD_IDS = _compute_bad_ids("english" if DIRECTION == "en2ko" else "hangul")

# 3. [핵심] <think> 생각 블록 제거 및 순수 번역문 추출 함수
def call_llm_translate(prompt_body, do_sample=False, temperature=None, force_target=False):
    """모델의 <think> 생각 블록을 완전히 도려내고 순수 번역 [1], [2]만 추출합니다."""
    system_msg = cfg["system_msg"]
    ex_user = cfg["ex_user"]
    ex_assistant = cfg["ex_assistant"]

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": ex_user},
        {"role": "assistant", "content": ex_assistant},
        {"role": "user", "content": f"Translate these lines. Output ONLY numbered translations:\n{prompt_body}"}
    ]

    # 단순 번역 작업이라 추론(thinking)이 불필요 -> 꺼서 답변 토큰 예산을 온전히 확보
    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    model_inputs = tokenizer([text_input], return_tensors="pt").to(device)
    max_new_tokens = 1024

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, repetition_penalty=1.1)
    if do_sample:
        gen_kwargs.update(temperature=temperature or 0.8, top_p=0.95)
    if force_target:
        gen_kwargs.update(bad_words_ids=_FORCE_BAD_IDS)

    with torch.no_grad():
        generated_ids = llm_model.generate(
            **model_inputs,
            **gen_kwargs
        )
        generated_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        if len(generated_ids[0]) >= max_new_tokens:
            print(f"    ⚠️ 생성이 max_new_tokens({max_new_tokens})에 도달해 잘렸을 수 있음")
        raw_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 🌟 [치료제 1] <think> ... </think> 생각 블록을 정규식으로 완전히 도려냄!
    cleaned_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.S).strip()

    # 만약 </think>가 닫히기 전에 잘린 경우 <think> 뒷부분을 통째로 날림
    if "<think>" in cleaned_response:
        cleaned_response = cleaned_response.split("<think>")[0].strip()

    # 🌟 [치료제 2] [1], [2] 번호가 붙은 줄만 쏙 골라냄
    final_lines = []
    for line in cleaned_response.split("\n"):
        line = line.strip()
        if re.match(r"^\[\d+\]", line):
            # 대사 뒤에 붙은 " - 해설" 제거
            if " – " in line:
                tag = line.split("]")[0] + "]"
                text = line.split(" – ")[-1]
                line = f"{tag} {text}"
            elif " - " in line and not line.split("]")[1].strip().startswith("-"):
                tag = line.split("]")[0] + "]"
                text = line.split(" - ")[-1]
                line = f"{tag} {text}"
            final_lines.append(line)

    return "\n".join(final_lines)

def has_hangul(s):
    return bool(re.search(r"[가-힣]", s))

def is_translated_ok(s):
    """목표 언어로 제대로 번역됐는지 판단. en2ko는 한글 포함 여부,
    ko2en은 한글이 안 남아있는지(원문 잔존 여부)로 판단.
    빈 문자열(아예 생성 안 됨)은 방향과 무관하게 항상 실패로 취급.
    한자/가나 등 제3언어가 섞여 있어도 실패로 취급(off-target 오염 방지)."""
    if not s.strip():
        return False
    if has_foreign_contamination(s):
        return False
    if DIRECTION == "en2ko":
        return has_hangul(s)
    return not has_hangul(s)

def translate_chunk_with_retry(doc, chunk):
    """청크를 번역하되, 배치 안에서 '복사 모드'(원문 그대로 반환)로 새버린 줄만
    골라 그 줄 하나만 단독으로 재요청해 복구한다. (배치로 묶었을 때만 발생하고
    단독으로 물으면 정상 번역되는 현상을 확인해서 만든 대응책.)"""
    frs, prompt = build_prompt(doc, chunk)
    llm_out = call_llm_translate(prompt)
    pieces = parse_marked(llm_out, frs)

    MAX_RETRIES = 3

    fixed_pieces = []
    for (j, txt), (jj, s, e, w) in zip(pieces, frs):
        if not is_translated_ok(txt):
            single_chunk = Chunk(doc.text[s:e], s, e)
            sub_frs, sub_prompt = build_prompt(doc, single_chunk)
            recovered = False
            for attempt in range(1, MAX_RETRIES + 1):
                # 그리디는 GPU 부동소수점 비결정성 때문에 매번 결과가 갈릴 수 있어서
                # 1차 시도만 greedy, 그 다음부턴 온도를 올려가며 샘플링으로 재시도
                if attempt == 1:
                    sub_out = call_llm_translate(sub_prompt, do_sample=False)
                else:
                    temp = min(1.4, 0.5 + 0.15 * attempt)
                    sub_out = call_llm_translate(sub_prompt, do_sample=True, temperature=temp)
                sub_pieces = parse_marked(sub_out, sub_frs)
                if sub_pieces and sub_pieces[0][1] and is_translated_ok(sub_pieces[0][1]):
                    print(f"    ↻ unit{j} 복사 모드 감지 -> 단독 재번역으로 복구 ({attempt}번째 시도)")
                    txt = sub_pieces[0][1]
                    recovered = True
                    break
            if not recovered:
                # 최종 안전장치: 원문 언어 토큰 자체를 생성 후보에서 배제해
                # 목표 언어가 나오도록 기계적으로 강제 (확률적 재시도가 아니라 보장됨)
                sub_out = call_llm_translate(sub_prompt, force_target=True)
                sub_pieces = parse_marked(sub_out, sub_frs)
                if sub_pieces and sub_pieces[0][1] and is_translated_ok(sub_pieces[0][1]):
                    print(f"    🔒 unit{j} 원문 언어 토큰 강제 배제로 최종 복구")
                    txt = sub_pieces[0][1]
                else:
                    print(f"    ⚠️ unit{j} 강제 배제로도 실패 (원문 유지)")
        fixed_pieces.append((j, txt))
    return fixed_pieces

# 4. 자막 로드 및 앞부분 SUB_UNITS개 테스트
doc = load_srt(str(input_srt))
sub_text = doc.text[:doc.units[SUB_UNITS - 1].end]

class SubDoc:
    def __init__(self, name, text, units, fmt, log):
        self.name, self.text, self.units, self.fmt, self.log = name, text, units, fmt, log
doc = SubDoc(doc.name, sub_text, doc.units[:SUB_UNITS], doc.fmt, {"truncated": True})
print(f"• [실행] {doc.name} 앞부분 {SUB_UNITS}개 대사로 번역 및 병합을 시작합니다.\n")

# ══════════════════════════════════════════════════════════════════════════════
# [파이프라인 1] 단순 글자수 분할 (Fixed) -> Qwen3 번역 -> fixed_llm.srt 병합
# ══════════════════════════════════════════════════════════════════════════════
print("-" * 70)
print("✂️ 1. 단순 글자 수 분할(Fixed, 120자) 기반 자막 생성 중...")
fixed_chunks = fixed_chunking(doc.text, chunk_size=120)
print(f"  - 생성된 청크 수: {len(fixed_chunks)}개")

fixed_pieces = []
for c_idx, c in enumerate(fixed_chunks, 1):
    fixed_pieces.extend(translate_chunk_with_retry(doc, c))
    print(f"    [Fixed] 청크 {c_idx}/{len(fixed_chunks)} 번역 완료")

fixed_unit_texts = merge_to_units(doc, fixed_pieces)
write_srt(doc, fixed_unit_texts, str(fixed_out_path))
print(f"  ✅ [Fixed 완료] -> {fixed_out_path}\n")

# ══════════════════════════════════════════════════════════════════════════════
# [파이프라인 2] BGE-M3 의미 기반 분할 (Semantic) -> Qwen3 번역 -> semantic_llm.srt 병합
# ══════════════════════════════════════════════════════════════════════════════
print("-" * 70)
print("🧠 2. BGE-M3 의미 기반 분할(Semantic) 기반 자막 생성 중...")
semantic_chunks = semantic_chunking(doc.text, embed_model, amount=15)
print(f"  - 생성된 청크 수: {len(semantic_chunks)}개")

semantic_pieces = []
for c_idx, c in enumerate(semantic_chunks, 1):
    semantic_pieces.extend(translate_chunk_with_retry(doc, c))
    print(f"    [Semantic] 청크 {c_idx}/{len(semantic_chunks)} 번역 완료")

semantic_unit_texts = merge_to_units(doc, semantic_pieces)
write_srt(doc, semantic_unit_texts, str(semantic_out_path))
print(f"  ✅ [Semantic 완료] -> {semantic_out_path}\n")

print("=" * 70)
print(f"🎉 {cfg['label']} 자막 생성이 완료되었습니다!")
print(f"1. 글자수 분할 결과: {fixed_out_path}")
print(f"2. 의미 분할 결과  : {semantic_out_path}")
print("=" * 70)
