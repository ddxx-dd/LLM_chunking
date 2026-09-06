"""
asd.py - 베이스라인 파이프라인 검증 스크립트

PART A: 트루먼쇼 자막 한→영 / 영→한 양방향 번역 + 병합(Fixed·Semantic 둘 다) 후
        results/에 저장하고, 실제 반대쪽 언어 원본 자막과 타임스탬프로 정렬해 F1로 비교한다.
PART B: DOCX 강의자료(1-1~5-1) 묶음에서 '큐' 관련 내용만 검색해 모아
        설명 문서를 새로 합성한 뒤 results/에 .docx로 저장한다.
"""
import json
import os
import re
import string
import sys
from collections import Counter
from pathlib import Path
import torch
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("src")
from config import SRT_ENG_DIR, SRT_KOR_DIR, DOCX_KOR_DIR, RESULTS_DIR, EMBED_MODEL, TOKENIZER
from core import Chunk
from preprocessing.loader import load_srt, load_docx_bundle
from chunking.fixed_chunker import fixed_chunking, fixed_chunking_tokens
from chunking.semantic_chunker import semantic_chunking
from pipeline.mapper import build_prompt, parse_marked, merge_to_units, write_srt
from pipeline.bundle import chunk_bundle
from retrieval.retriever import retrieve_top_k_bundle
from analysis.analyzer import split_tables

os.makedirs(RESULTS_DIR, exist_ok=True)

DIRECTION_CONFIG = {
    "en2ko": dict(
        system_msg=(
            "You are a subtitle translator. Translate English subtitles into natural Korean.\n"
            "STRICT RULES:\n"
            "1. Output ONLY the numbered Korean translations like [1] 안녕하세요.\n"
            "2. Do NOT explain words, grammar, or your thought process.\n"
            "3. Do NOT output English."
        ),
        ex_user="[1] What is that?\n[2] Where are you going?",
        ex_assistant="[1] 저게 뭐야?\n[2] 어디 가세요?",
        instruction="Translate each line to Korean. Keep the [n] numbers exactly:\n{body}",
        label="영→한",
    ),
    "ko2en": dict(
        system_msg=(
            "You are a subtitle translator. Translate Korean subtitles into natural English.\n"
            "STRICT RULES:\n"
            "1. Output ONLY the numbered English translations like [1] Hello.\n"
            "2. Do NOT explain words, grammar, or your thought process.\n"
            "3. Do NOT output Korean."
        ),
        ex_user="[1] 아~뭐여~?\n[2] 어디 가세요?",
        ex_assistant="[1] What is that?\n[2] Where are you going?",
        instruction="Translate each line to English. Keep the [n] numbers exactly:\n{body}",
        label="한→영",
    ),
}

def build_prompt_for(doc, chunk, direction):
    """build_prompt()의 기본 지시문이 방향과 무관하게 'to English'로 고정돼 있던
    버그를 우회 - 방향에 맞는 지시문을 명시적으로 넘긴다. 특정 데이터셋이 아니라
    direction 하나로만 갈리므로 다른 자막 세트에도 그대로 적용된다."""
    return build_prompt(doc, chunk, instruction=DIRECTION_CONFIG[direction]["instruction"])

# ══════════════════════════════════════════════════════════════════════════════
# 0. 모델 로드 (PART A / PART B 공용)
# ══════════════════════════════════════════════════════════════════════════════
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"• 실행 디바이스: {device.upper()}")
print("• BGE-M3 임베딩 모델 로드 중...")
embed_model = SentenceTransformer(EMBED_MODEL, device=device)
print(f"• {TOKENIZER} LLM 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
llm_model = AutoModelForCausalLM.from_pretrained(TOKENIZER, torch_dtype=torch.float16, device_map="auto")
print("✅ 모델 준비 완료!\n")

# 방향별 "목표 스크립트" 하나만 정의한다. 어떤 스크립트가 "오염"인지 하드코딩된
# 목록(한자/가나 등)을 따로 관리하지 않고, "원본 자막 파일은 그 언어 하나로만
# 되어 있다"고 가정한 뒤 "목표 스크립트가 아닌 알파벳 문자가 하나라도 있으면
# 실패"로 통일한다 - 목표 스크립트 정규식 하나만 등록하면 되므로 새 언어쌍이
# 추가돼도 이 로직 자체는 손댈 필요가 없다.
SCRIPT_REGEX = {
    "hangul": re.compile(r"[가-힣]"),
    "latin": re.compile(r"[A-Za-z]"),
}
DIRECTION_CONFIG["en2ko"]["target_script"] = "hangul"
DIRECTION_CONFIG["ko2en"]["target_script"] = "latin"

def has_target_script(s, target_script):
    return bool(SCRIPT_REGEX[target_script].search(s))

def has_off_target_letters(s, target_script):
    """s의 알파벳 문자 중 목표 스크립트가 아닌 게 하나라도 있으면 True.
    한자/가나/키릴 등 "어떤 스크립트들을 따로 조사할지" 목록을 유지할 필요가
    없다 - 목표 스크립트 하나만 알면 그 외 전부(무엇이든)를 자동으로 걸러낸다."""
    target_re = SCRIPT_REGEX[target_script]
    return any(ch.isalpha() and not target_re.match(ch) for ch in s)

def _build_off_target_ids(target_script):
    """vocab 전체를 실제로 디코딩해서, 목표 스크립트가 아닌 알파벳 문자를 포함한
    토큰 id 목록을 만든다. Qwen 등 byte-level BPE는 vocab의 원문 토큰 문자열
    자체가 바이트 치환 표현이라(예: 한글 '안'이 'ìķĪ'처럼 저장됨) 문자 범위를
    직접 비교하면 안 되고, 반드시 tokenizer.decode()로 실제 텍스트를 복원한 뒤
    판단해야 한다."""
    vocab = tokenizer.get_vocab()
    ids = list(vocab.values())
    decoded = tokenizer.batch_decode([[i] for i in ids])
    off_ids = []
    for tok_id, text in zip(ids, decoded):
        if text and has_off_target_letters(text, target_script):
            off_ids.append(tok_id)
    return off_ids

print("• vocab 전체를 디코딩해 방향별 '목표 아닌 스크립트' 토큰 목록 계산 중...")
_OFF_TARGET_IDS = {
    direction: _build_off_target_ids(cfg["target_script"])
    for direction, cfg in DIRECTION_CONFIG.items()
}
for direction, ids in _OFF_TARGET_IDS.items():
    print(f"  {direction}: 목표({DIRECTION_CONFIG[direction]['target_script']}) 아닌 토큰 {len(ids)}개")

_BAD_IDS = {d: [[i] for i in ids] for d, ids in _OFF_TARGET_IDS.items()}
# 하드 배제(-inf) 대신 로짓에 강한 음의 편향만 주는 소프트 제약.
# 완전 차단은 모델이 남은 후보들 중 확률이 아주 낮은(=종종 의미 없는) 토큰을
# 고르게 만들어 오히려 문장이 깨지는 부작용이 있어서(문헌상 "hard constraint
# disrupts the underlying probability distribution"), 방향을 강하게 유도하되
# 분포 자체는 무너뜨리지 않는 soft-constrained decoding으로 대체.
_SOFT_BIAS = {d: {(i,): -8.0 for i in ids} for d, ids in _OFF_TARGET_IDS.items()}

_PROMPT_ECHO_MARKERS = (
    "self-check", "previous attempt", "rewrite it completely",
    "translating every single word", "keep the [n] numbers",
)

def looks_like_prompt_echo(s):
    """모델이 실제 번역 대신, 우리가 넣어준 지시문(특히 self-refine 교정 프롬프트)을
    그대로 되풀이했는지 확인. 이 마커들은 우리 코드 자신의 고정 템플릿 문구라서
    특정 데이터셋 내용과 무관하게 항상 같은 방식으로 걸러낼 수 있다."""
    low = s.lower()
    return any(m in low for m in _PROMPT_ECHO_MARKERS)

def is_translated_ok(s, direction):
    """목표 언어로 온전히 번역됐는지 판단 - 언어 방향(direction)만 받고 특정
    문서/데이터셋의 내용에는 의존하지 않아 어떤 자막 세트에도 그대로 쓸 수 있다.
    "원본 자막 파일은 그 언어 하나로만 되어 있다"고 가정하고, 목표 스크립트가
    있고 그 외의 알파벳 문자는 하나도 없어야 통과시킨다."""
    if not s.strip():
        return False
    if looks_like_prompt_echo(s):
        return False
    if not any(ch.isalpha() for ch in s):
        # 알파벳 문자가 하나도 없는 조각(예: 고정 글자수 분할이 유닛 중간을 끊어
        # "!" 하나만 남긴 경우) - 번역할 대상 자체가 없으므로 스크립트 검사를
        # 적용할 수 없다. 이런 조각은 몇 번을 재시도해도 목표 스크립트가 나올 수
        # 없어 항상 전체 재시도 단계(최대 10회 LLM 호출)를 낭비하고 결국 원래
        # 텍스트를 그대로 쓰게 되므로, 애초에 통과시켜 낭비를 없앤다.
        return True
    target_script = DIRECTION_CONFIG[direction]["target_script"]
    return has_target_script(s, target_script) and not has_off_target_letters(s, target_script)

def call_llm_translate(prompt_body, direction, do_sample=False, temperature=None,
                        soft_force=False, hard_force=False):
    cfg = DIRECTION_CONFIG[direction]
    messages = [
        {"role": "system", "content": cfg["system_msg"]},
        {"role": "user", "content": cfg["ex_user"]},
        {"role": "assistant", "content": cfg["ex_assistant"]},
        {"role": "user", "content": f"Translate these lines. Output ONLY numbered translations:\n{prompt_body}"}
    ]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    model_inputs = tokenizer([text_input], return_tensors="pt").to(device)
    max_new_tokens = 1024

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, repetition_penalty=1.1)
    if do_sample:
        gen_kwargs.update(temperature=temperature or 0.8, top_p=0.95)
    if soft_force:
        # 하드 배제 대신 로짓에 음의 편향만 줘서 유창성을 지키며 언어를 유도 (soft-constrained decoding)
        gen_kwargs.update(sequence_bias=_SOFT_BIAS[direction])
    if hard_force:
        # 정말 최후의 수단: 토큰 자체를 완전 배제 (유창성을 희생해서라도 언어만 강제)
        gen_kwargs.update(bad_words_ids=_BAD_IDS[direction])

    with torch.no_grad():
        generated_ids = llm_model.generate(**model_inputs, **gen_kwargs)
        generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
        raw_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.S).strip()
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>")[0].strip()

    final_lines = []
    for line in cleaned.split("\n"):
        line = line.strip()
        if re.match(r"^\[\d+\]", line):
            if " – " in line:
                tag = line.split("]")[0] + "]"; text = line.split(" – ")[-1]; line = f"{tag} {text}"
            elif " - " in line and not line.split("]")[1].strip().startswith("-"):
                tag = line.split("]")[0] + "]"; text = line.split(" - ")[-1]; line = f"{tag} {text}"
            final_lines.append(line)
    return "\n".join(final_lines)

def translate_chunk_with_retry(doc, chunk, direction):
    """청크를 번역하되, 목표 언어로 온전히 안 나온 조각만 골라 단독으로 재요청한다.
    아래 3단계는 전부 '방향(direction)'과 텍스트 패턴에만 의존해서 특정 파일/데이터셋에
    묶이지 않고, 어떤 자막 세트를 넣어도 그대로 동작한다.
      1단계: greedy 1회 + 온도를 올려가며 샘플링 재시도 (GPU 비결정성 활용)
      2단계: soft-constrained decoding - 목표 언어가 아닌 토큰에 음의 편향만 줘서
             유창성을 지키며 언어를 유도 (완전 배제는 문헌상 확률분포를 무너뜨려
             오히려 의미 없는 문자열을 만들어내는 부작용이 있어 그 전에 시도)
      3단계: 정말 최후의 수단 - 목표가 아닌 언어 토큰을 완전 배제 (유창성보다
             "어쨌든 목표 언어로 나오게"를 우선)"""
    frs, prompt = build_prompt_for(doc, chunk, direction)
    llm_out = call_llm_translate(prompt, direction)
    pieces = parse_marked(llm_out, frs)

    MAX_TEMP_RETRIES = 5
    SOFT_BIAS_RETRIES = 2
    CORRECTION_RETRIES = 2

    fixed_pieces = []
    for (j, txt), (jj, s, e, w) in zip(pieces, frs):
        if not is_translated_ok(txt, direction):
            single_chunk = Chunk(doc.text[s:e], s, e)
            sub_frs, sub_prompt = build_prompt_for(doc, single_chunk, direction)
            recovered = False

            for attempt in range(1, MAX_TEMP_RETRIES + 1):
                if attempt == 1:
                    sub_out = call_llm_translate(sub_prompt, direction, do_sample=False)
                else:
                    temp = min(1.4, 0.5 + 0.15 * attempt)
                    sub_out = call_llm_translate(sub_prompt, direction, do_sample=True, temperature=temp)
                sub_pieces = parse_marked(sub_out, sub_frs)
                if sub_pieces and sub_pieces[0][1] and is_translated_ok(sub_pieces[0][1], direction):
                    txt = sub_pieces[0][1]
                    recovered = True
                    print(f"    ↻ unit{j} 재시도 {attempt}회차로 복구")
                    break

            if not recovered:
                for attempt in range(1, SOFT_BIAS_RETRIES + 1):
                    sub_out = call_llm_translate(sub_prompt, direction, do_sample=True, temperature=0.7, soft_force=True)
                    sub_pieces = parse_marked(sub_out, sub_frs)
                    if sub_pieces and sub_pieces[0][1] and is_translated_ok(sub_pieces[0][1], direction):
                        txt = sub_pieces[0][1]
                        recovered = True
                        print(f"    ↻ unit{j} soft-bias {attempt}회차로 복구")
                        break

            if not recovered:
                # self-refine 스타일 교정 (Madaan et al., Self-Refine 2023).
                # 주의: 실패한 이전 답을 그대로 인용해서 보여주면 모델이 그 인용문
                # 자체를 "번역할 내용"으로 착각해 되풀이하는 사고가 있었다(echo 버그).
                # 그래서 이전 답을 인용하지 않고, "그런 시도가 있었으니 무시하고
                # 새로 번역하라"는 지시만 짧게 덧붙인다 - 인용할 텍스트 자체가 없어야
                # 되풀이할 거리도 없어진다.
                other_lang = "English" if direction == "en2ko" else "Korean"
                target_lang = "Korean" if direction == "en2ko" else "English"
                correction_body = (
                    f"{sub_prompt}\n\n"
                    f"(Note: a prior attempt at this exact line failed - it left some {other_lang} "
                    f"words untranslated. Do NOT mention or repeat that failed attempt. "
                    f"Simply output the numbered line above translated completely into natural "
                    f"{target_lang}, with every word translated.)"
                )
                for attempt in range(1, CORRECTION_RETRIES + 1):
                    sub_out = call_llm_translate(correction_body, direction, do_sample=True, temperature=0.6)
                    sub_pieces = parse_marked(sub_out, sub_frs)
                    if sub_pieces and sub_pieces[0][1] and is_translated_ok(sub_pieces[0][1], direction):
                        txt = sub_pieces[0][1]
                        recovered = True
                        print(f"    ↻ unit{j} 자기수정 {attempt}회차로 복구")
                        break

            if not recovered:
                sub_out = call_llm_translate(sub_prompt, direction, hard_force=True)
                sub_pieces = parse_marked(sub_out, sub_frs)
                if sub_pieces and sub_pieces[0][1] and is_translated_ok(sub_pieces[0][1], direction):
                    txt = sub_pieces[0][1]
                    print(f"    🔒 unit{j} 완전 배제로 최종 복구")
                else:
                    print(f"    ⚠️ unit{j} 모든 단계 실패 (원문 유지)")
        fixed_pieces.append((j, txt))
    return fixed_pieces

# ---- 참조 자막과 비교용 SQuAD 스타일 F1 ----
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())

def f1_score(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec, rec = num_same / len(p), num_same / len(g)
    return 2 * prec * rec / (prec + rec)

def compare_full_text(unit_texts, ref_doc):
    """큐 단위로 정밀하게 대응시키지 않고, 번역 결과 전체와 참조 자막 전체를
    통째로(단어 단위 F1) 비교한다. 번역은 청크 단위로 이뤄지므로(글자수 분할이
    문맥 없이 잘라 번역을 망치는 것도 청크 단계에서 벌어지는 일) 큐 단위로
    정밀하게 안 맞춰도 분할 방식에 따른 번역 품질 차이는 그대로 드러난다.
    두 파일의 타임스탬프가 서로 몇 초씩 어긋나 있어도(실측 확인됨) 이 비교는
    타임스탬프를 아예 안 쓰므로 영향받지 않는다."""
    src_full = " ".join(t for t in unit_texts if t.strip())
    ref_full = " ".join(ref_doc.text[u.start:u.end] for u in ref_doc.units)
    return f1_score(src_full, ref_full)

def check_timestamp_integrity(out_srt_path, src_doc):
    """번역 품질과는 별개로, merge_to_units + write_srt가 원본(같은 파일에서 나온)
    유닛들의 타임스탬프를 그대로 보존했는지 확인한다. 서로 다른 두 파일(예: 한국어
    파일과 영어 파일) 사이의 타임스탬프 비교가 아니라, 출력 파일과 그 출력이
    파생된 원본 소스 파일 사이의 비교라서 파일 간 드리프트 문제와 무관하다."""
    out_doc = load_srt(str(out_srt_path))
    if len(out_doc.units) != len(src_doc.units):
        return False, [f"유닛 개수 불일치: 출력 {len(out_doc.units)} vs 원본 {len(src_doc.units)}"]
    bad = [
        i for i, (o, s) in enumerate(zip(out_doc.units, src_doc.units))
        if abs(o.meta["t_start"] - s.meta["t_start"]) > 1e-6 or abs(o.meta["t_end"] - s.meta["t_end"]) > 1e-6
    ]
    return (len(bad) == 0), bad

def run_translation(direction, src_doc, ref_doc):
    cfg = DIRECTION_CONFIG[direction]
    print("=" * 70)
    print(f"🚀 [{cfg['label']}] {src_doc.name} 전체 {len(src_doc.units)}줄")
    print("=" * 70)

    results = {}
    for method_label, chunks in [
        ("fixed", fixed_chunking(src_doc.text, chunk_size=120)),
        ("semantic", semantic_chunking(src_doc.text, embed_model, method="percentile", amount=15)),
    ]:
        print(f"  [{method_label}] 청크 {len(chunks)}개 번역 중...")
        pieces = []
        for c_idx, c in enumerate(chunks, 1):
            pieces.extend(translate_chunk_with_retry(src_doc, c, direction))
            print(f"    {method_label} 청크 {c_idx}/{len(chunks)} 완료")
        unit_texts = merge_to_units(src_doc, pieces)

        out_path = RESULTS_DIR / f"트루먼쇼_{direction}_{method_label}.srt"
        write_srt(src_doc, unit_texts, str(out_path))
        print(f"  ✅ [{method_label}] -> {out_path}")

        ts_ok, ts_bad = check_timestamp_integrity(out_path, src_doc)
        print(f"  🕒 [{method_label}] 타임스탬프 보존(병합 무결성) 확인: {'OK' if ts_ok else f'문제 {len(ts_bad)}건 {ts_bad[:5]}'}")

        f1 = compare_full_text(unit_texts, ref_doc)
        print(f"  📊 [{method_label}] 참조 자막 전체 대비 텍스트 F1(통짜 비교): {f1:.4f}")
        results[method_label] = dict(out_path=out_path, f1=f1, ts_ok=ts_ok, ts_bad=ts_bad,
                                      unit_texts=unit_texts)
    return results

# ══════════════════════════════════════════════════════════════════════════════
# PART A : 트루먼쇼 자막 양방향 번역 + 병합 + 원본 비교
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "#" * 70)
print("# PART A - 트루먼쇼 자막 한→영 / 영→한 번역 + 병합 + 원본 비교")
print("#" * 70)

ko_full = load_srt(str(SRT_KOR_DIR / "트루먼쇼.srt"))
en_full = load_srt(str(SRT_ENG_DIR / "The_Truman_Show_Eng.srt"))

report_lines = []
all_results = {}
for direction, src_full, ref_full in [("ko2en", ko_full, en_full), ("en2ko", en_full, ko_full)]:
    all_results[direction] = run_translation(direction, src_full, ref_full)

report_lines.append("트루먼쇼 자막 번역 파이프라인 검증 결과")
report_lines.append("=" * 60)
for direction, methods in all_results.items():
    label = DIRECTION_CONFIG[direction]["label"]
    for method_label, r in methods.items():
        ts_str = "OK" if r["ts_ok"] else f"문제 {len(r['ts_bad'])}건"
        report_lines.append(f"[{label}] {method_label:9s} 텍스트 F1={r['f1']:.4f}  타임스탬프보존={ts_str}  -> {r['out_path'].name}")
report_path = RESULTS_DIR / "트루먼쇼_번역_비교결과.txt"
report_path.write_text("\n".join(report_lines), encoding="utf-8")
print(f"\n📄 비교 리포트 저장: {report_path}")

# ══════════════════════════════════════════════════════════════════════════════
# PART B : DOCX 강의자료(1-1~5-1) 묶음에서 '큐' 설명 문서 합성
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "#" * 70)
print("# PART B - DOCX 강의자료(1-1~5-1)에서 '큐' 설명 문서 합성")
print("#" * 70)

BUNDLE_FILES = [
    DOCX_KOR_DIR / "1-1.배열.docx",
    DOCX_KOR_DIR / "2-1.스택.docx",
    DOCX_KOR_DIR / "3-1.큐.docx",
    DOCX_KOR_DIR / "4-1.이진트리.docx",
    DOCX_KOR_DIR / "5-1.탐색트리.docx",
]
QUERY = "큐(Queue)란 무엇이고 어떻게 동작하는지, 특징과 활용을 설명해줘"
TOP_K = 8

bundle_docs = load_docx_bundle([p for p in BUNDLE_FILES if p.exists()])
print("문서 묶음 로드 완료:", ", ".join(d.name for d in bundle_docs))

def call_llm_compose(prompt, max_new_tokens=1400):
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

def extract_row_dicts(text):
    """청크 텍스트 안에 섞여 있는 표 행(JSON 한 줄짜리) 그대로 파싱해서 꺼낸다."""
    rows = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and obj:
                    rows.append(obj)
            except Exception:
                pass
    return rows

def strip_row_json(text):
    """표 행(JSON) 줄을 제거한 나머지(순수 문단)만 남긴다 - 표 데이터는 LLM 손을
    거치지 않고 원본 그대로 별도 렌더링할 것이므로, LLM에게는 안 보여줘서
    "누락된 행을 지식으로 채워 넣는" 걸 원천 차단한다."""
    kept = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                json.loads(s)
                continue
            except Exception:
                pass
        kept.append(line)
    return "\n".join(kept)

def build_compose_prompt(query, retrieved):
    # 이미 top-k로 관련성 필터링된 조각만 들어오고 그 내용만 근거로 합성하므로
    # 출처 라벨을 LLM에서 굳이 숨기지 않는다.
    body = "\n\n".join(f"[출처: {r['doc_name']}]\n{r['chunk'].text.strip()}" for r in retrieved if r["chunk"].text.strip())
    return (
        "다음은 여러 강의자료에서 검색된, 아래 질문과 관련된 내용입니다. "
        "이 내용만 근거로 삼아 체계적으로 정리된 설명 문서를 작성하세요.\n"
        "형식 규칙: 소제목은 반드시 '## 소제목' 형식으로 쓰고, 일반 서술은 문단으로 쓰세요. "
        "표는 별도로 원본 그대로 첨부되니 여기서는 표를 새로 만들지 마세요. "
        "자료에 없는 내용은 지어내지 마세요.\n\n"
        f"질문: {query}\n\n검색된 내용:\n{body}\n\n작성할 문서:"
    )

def _md_table_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]

def _is_md_table_separator(line):
    stripped = line.strip().strip("|")
    return bool(stripped) and "-" in stripped and all(c in " -:|" for c in stripped)

def _add_raw_table(d, doc_name, rows):
    p = d.add_paragraph()
    run = p.add_run(f"[출처: {doc_name} · 검색된 조각에서 그대로 추출된 {len(rows)}행]")
    run.italic = True
    headers = list(rows[0].keys())
    table = d.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    for c, h in enumerate(headers):
        cell = table.rows[0].cells[c]
        cell.text = h
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for r, row in enumerate(rows, start=1):
        for c, h in enumerate(headers):
            table.rows[r].cells[c].text = str(row.get(h, ""))

def write_docx_from_markdown(title, body_md, out_path, raw_tables=None):
    """LLM이 마크다운으로 써낸 문서를 실제 .docx로 변환.
    '## 소제목'은 제목 스타일로, '| a | b |' 형태의 마크다운 표는
    워드에서 열어도 진짜 표로 보이도록 add_table()로 변환한다.
    raw_tables: [(doc_name, [row_dict, ...]), ...] - 검색된 청크에 실제로 담겨
    있던 표 행을 LLM을 거치지 않고 그대로 렌더링한다. 청킹이 표의 일부 행을
    다른 청크로 떼어놓거나(top-k에서 탈락) 아예 빠뜨렸다면, 그 결손이 보정 없이
    그대로 이 표에 드러난다 - 분할 기법의 한계를 감추지 않기 위함."""
    d = DocxDocument()
    d.add_heading(title, level=1)
    if raw_tables:
        d.add_heading("검색된 원본 표 데이터 (청킹·검색 결과 그대로, 보정 없음)", level=2)
        for doc_name, rows in raw_tables:
            _add_raw_table(d, doc_name, rows)
        d.add_heading("종합 설명", level=2)
    lines = body_md.split("\n")
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("## "):
            d.add_heading(line[3:].strip(), level=2)
            i += 1
            continue
        if line.startswith("# "):
            d.add_heading(line[2:].strip(), level=2)
            i += 1
            continue
        if line.startswith("|") and i + 1 < n and _is_md_table_separator(lines[i + 1]):
            header = _md_table_row(line)
            i += 2  # 헤더 줄 + 구분선(---) 줄 건너뜀
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                row = _md_table_row(lines[i])
                i += 1
                # 생성이 표 중간에서 잘리면 셀 수가 안 맞거나 깨진 문자(�)가 남는데,
                # 이런 불완전한 줄은 표에 넣지 않고 건너뛴다.
                if len(row) == len(header) and "�" not in "".join(row):
                    rows.append(row)
            table = d.add_table(rows=1 + len(rows), cols=len(header))
            table.style = "Table Grid"
            for c, val in enumerate(header):
                cell = table.rows[0].cells[c]
                cell.text = val
                for run in cell.paragraphs[0].runs:
                    run.bold = True
            for r, row in enumerate(rows, start=1):
                for c, val in enumerate(row):
                    if c < len(header):
                        table.rows[r].cells[c].text = val
            continue
        d.add_paragraph(line)
        i += 1
    d.save(str(out_path))

def run_compose(label, chunker_fn):
    print(f"\n[{label}] 청킹 및 검색 중...")
    bundle_chunks = chunk_bundle(bundle_docs, chunker_fn)
    print(f"  총 청크 {len(bundle_chunks)}개")

    # 청킹이 표를 몇 개나 여러 청크로 쪼갰는지 보고만 한다 - 잘린 표를 다시 합치거나
    # 보정하지 않는다. 이것 자체가 분할 기법의 한계를 보여주는 관측값이기 때문이다.
    for doc_idx, doc in enumerate(bundle_docs):
        doc_chunks = [b["chunk"] for b in bundle_chunks if b["doc_idx"] == doc_idx]
        broken = split_tables(doc, doc_chunks)
        if broken:
            print(f"  ⚠️ {doc.name}: 표 {list(broken.keys())}이(가) 서로 다른 청크로 쪼개짐 {broken}")

    retrieved = retrieve_top_k_bundle(QUERY, bundle_chunks, embed_model, k=TOP_K)
    doc_hits = Counter(r["doc_name"] for r in retrieved)
    print(f"  top-{TOP_K} 출처: {dict(doc_hits)}")

    # 표 행(JSON)은 top-k에 실제로 들어온 것만, LLM 손을 거치지 않고 그대로 표로 만든다.
    # 행이 일부만 들어왔거나(청크가 표를 잘라서) 아예 안 들어왔다면(그 행만 담긴 청크가
    # top-k에서 탈락) 그 결손이 그대로 드러난다 - 절대 보정하지 않는다.
    raw_tables = []
    prose_retrieved = []
    for r in retrieved:
        rows = extract_row_dicts(r["chunk"].text)
        if rows:
            raw_tables.append((r["doc_name"], rows))
            print(f"    표 행 {len(rows)}개 검색됨 (출처: {r['doc_name']})")
        prose_text = strip_row_json(r["chunk"].text)
        if prose_text.strip():
            prose_retrieved.append({**r, "chunk": r["chunk"]._replace(text=prose_text)})

    prompt = build_compose_prompt(QUERY, prose_retrieved)
    body_md = call_llm_compose(prompt)

    out_path = RESULTS_DIR / f"큐_설명_{label}.docx"
    write_docx_from_markdown(f"큐(Queue) 설명 — {label} 청킹 기반", body_md, out_path, raw_tables=raw_tables)
    print(f"  ✅ 저장 완료 -> {out_path}")
    return out_path, doc_hits

fixed_docx_path, fixed_hits = run_compose("fixed", lambda text: fixed_chunking_tokens(text, tokenizer, max_tokens=150))
semantic_docx_path, semantic_hits = run_compose(
    "semantic", lambda text: semantic_chunking(text, embed_model, method="percentile", amount=15, max_chunk_chars=2000)
)

print("\n" + "=" * 70)
print("🎉 PART A + PART B 전체 완료")
print("=" * 70)
print(f"PART A 결과: {RESULTS_DIR}/트루먼쇼_{{ko2en,en2ko}}_{{fixed,semantic}}.srt")
print(f"PART A 비교 리포트: {report_path}")
print(f"PART B 결과: {fixed_docx_path}, {semantic_docx_path}")
