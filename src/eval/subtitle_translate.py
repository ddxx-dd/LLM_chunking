"""자막 파이프라인: 청킹 -> 청크 단위 번역 -> 원래 유닛(번호/타임스탬프)에
재병합 -> SRT 저장 -> 참조 자막과 텍스트 비교."""
import re
import string
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core import Chunk
from preprocessing.loader import load_srt
from pipeline.mapper import build_prompt, parse_marked, merge_to_units, write_srt
from llm.client import generate

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
        target_script="hangul",
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
        target_script="latin",
    ),
}

# 방향별 목표 스크립트 하나만 등록 - 그 외 알파벳은 뭐든 오염으로 취급
SCRIPT_REGEX = {
    "hangul": re.compile(r"[가-힣]"),
    "latin": re.compile(r"[A-Za-z]"),
}


def has_target_script(s, target_script):
    """목표 스크립트 문자가 하나라도 있는지."""
    return bool(SCRIPT_REGEX[target_script].search(s))


def has_off_target_letters(s, target_script):
    """목표 스크립트가 아닌 알파벳이 섞여있는지."""
    target_re = SCRIPT_REGEX[target_script]
    return any(ch.isalpha() and not target_re.match(ch) for ch in s)


def is_translated_ok(s, direction):
    """목표 언어로 온전히 번역됐는지 판단."""
    if not s.strip():
        return False
    if not any(ch.isalpha() for ch in s):
        return True  # 구두점뿐인 조각 - 번역 대상 자체가 없으니 통과
    target_script = DIRECTION_CONFIG[direction]["target_script"]
    return has_target_script(s, target_script) and not has_off_target_letters(s, target_script)


def build_prompt_for(doc, chunk, direction):
    """방향에 맞는 지시문으로 번역 프롬프트 생성."""
    return build_prompt(doc, chunk, instruction=DIRECTION_CONFIG[direction]["instruction"])


def call_llm_translate(tokenizer, model, device, prompt_body, direction, do_sample=False, temperature=None):
    """번호 태그 프롬프트로 LLM 호출, 번호 붙은 줄만 추려서 반환."""
    cfg = DIRECTION_CONFIG[direction]
    messages = [
        {"role": "system", "content": cfg["system_msg"]},
        {"role": "user", "content": cfg["ex_user"]},
        {"role": "assistant", "content": cfg["ex_assistant"]},
        {"role": "user", "content": f"Translate these lines. Output ONLY numbered translations:\n{prompt_body}"},
    ]
    raw = generate(tokenizer, model, device, messages, max_new_tokens=1024,
                    do_sample=do_sample, temperature=temperature)

    final_lines = []
    for line in raw.split("\n"):
        line = line.strip()
        if re.match(r"^\[\d+\]", line):
            final_lines.append(line)
    return "\n".join(final_lines)


def translate_chunk(doc, chunk, direction, tokenizer, model, device):
    """청크 번역, 실패한 조각만 단독 재시도 1회. (j, txt, ok) 목록 반환."""
    frs, prompt = build_prompt_for(doc, chunk, direction)
    llm_out = call_llm_translate(tokenizer, model, device, prompt, direction)
    pieces = parse_marked(llm_out, frs)

    results = []
    for (j, txt), (jj, s, e, w) in zip(pieces, frs):
        ok = is_translated_ok(txt, direction)
        if not ok:
            single_chunk = Chunk(doc.text[s:e], s, e)
            sub_frs, sub_prompt = build_prompt_for(doc, single_chunk, direction)
            sub_out = call_llm_translate(tokenizer, model, device, sub_prompt, direction,
                                          do_sample=True, temperature=0.7)
            sub_pieces = parse_marked(sub_out, sub_frs)
            if sub_pieces and sub_pieces[0][1] and is_translated_ok(sub_pieces[0][1], direction):
                txt, ok = sub_pieces[0][1], True
        results.append((j, txt, ok))
    return results


def normalize_answer(s):
    """소문자화 + 구두점 제거 (SQuAD 스타일 F1 전처리)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def f1_score(pred, gold):
    """단어 단위 F1."""
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
    """큐 단위 정렬 없이, 번역 전체 vs 참조 전체를 통짜로 F1 비교."""
    src_full = " ".join(t for t in unit_texts if t.strip())
    ref_full = " ".join(ref_doc.text[u.start:u.end] for u in ref_doc.units)
    return f1_score(src_full, ref_full)


def check_timestamp_integrity(out_srt_path, src_doc):
    """출력 SRT 타임스탬프가 원본 유닛과 그대로 일치하는지 확인 (병합 무결성)."""
    out_doc = load_srt(str(out_srt_path))
    if len(out_doc.units) != len(src_doc.units):
        return False, [f"유닛 개수 불일치: 출력 {len(out_doc.units)} vs 원본 {len(src_doc.units)}"]
    bad = [
        i for i, (o, s) in enumerate(zip(out_doc.units, src_doc.units))
        if abs(o.meta["t_start"] - s.meta["t_start"]) > 1e-6 or abs(o.meta["t_end"] - s.meta["t_end"]) > 1e-6
    ]
    return (len(bad) == 0), bad


def run_translation(direction, src_doc, ref_doc, chunkers, tokenizer, model, device, out_dir):
    """chunkers: {method_label: chunker_fn(text) -> list[Chunk]}"""
    cfg = DIRECTION_CONFIG[direction]
    print("=" * 70)
    print(f"[{cfg['label']}] {src_doc.name} 전체 {len(src_doc.units)}줄")
    print("=" * 70)

    results = {}
    for method_label, chunker_fn in chunkers.items():
        chunks = chunker_fn(src_doc.text)
        print(f"  [{method_label}] 청크 {len(chunks)}개 번역 중...")
        pieces = []
        fail_count = 0
        for c_idx, c in enumerate(chunks, 1):
            for j, txt, ok in translate_chunk(src_doc, c, direction, tokenizer, model, device):
                pieces.append((j, txt))
                if not ok:
                    fail_count += 1
            print(f"    {method_label} 청크 {c_idx}/{len(chunks)} 완료")
        unit_texts = merge_to_units(src_doc, pieces)

        out_path = Path(out_dir) / f"{src_doc.name.rsplit('.', 1)[0]}_{direction}_{method_label}.srt"
        write_srt(src_doc, unit_texts, str(out_path))

        ts_ok, ts_bad = check_timestamp_integrity(out_path, src_doc)
        f1 = compare_full_text(unit_texts, ref_doc)
        fail_rate = fail_count / max(1, len(src_doc.units))
        print(f"  ✅ [{method_label}] -> {out_path}")
        print(f"     텍스트 F1(통짜 비교)={f1:.4f}  번역 실패율={fail_rate:.3%}({fail_count}건)  타임스탬프보존={'OK' if ts_ok else f'문제 {len(ts_bad)}건'}")

        results[method_label] = dict(out_path=out_path, f1=f1, fail_count=fail_count,
                                      fail_rate=fail_rate, ts_ok=ts_ok, ts_bad=ts_bad)
    return results
