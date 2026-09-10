"""자막 파이프라인: 청킹 -> 청크 단위 번역 -> 원래 유닛(번호/타임스탬프)에
재병합 -> SRT 저장 -> 참조 자막과 텍스트 비교."""
import re
import string
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from preprocessing.loader import _to_sec
from pipeline.mapper import build_prompt, parse_marked, merge_to_units, write_srt
from llm.client import generate
from eval.timestamp_align import align_by_overlap

_TS_RE = re.compile(r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})")

# 자막 대사는 격식체 글이 아니라 구어체 대화라는 걸 알려주면 번역 톤이 훨씬
# 자연스러워진다는 게 실측으로 확인됨(직역투/누락 감소). 특정 영화 정보 없이
# 어떤 자막 데이터셋에도 그대로 쓸 수 있는 일반적인 문구만 사용한다.
_DIALOGUE_CONTEXT = (
    "This text is dialogue from a movie or TV show's subtitles - natural spoken "
    "conversation between characters, not formal writing. Translate in a natural, "
    "colloquial spoken tone that fits the flow of conversation, not a literal "
    "word-by-word translation."
)

DIRECTION_CONFIG = {
    "en2ko": dict(
        system_msg=(
            "You are a subtitle translator. Translate English subtitles into natural Korean.\n"
            f"{_DIALOGUE_CONTEXT}\n"
            "STRICT RULES:\n"
            "1. Output ONLY the numbered Korean translations like [1] 안녕하세요.\n"
            "2. Do NOT explain words, grammar, or your thought process.\n"
            "3. Do NOT output English."
        ),
        # 무관한 문장 나열 대신 실제 대화 흐름을 few-shot으로 보여줘야 구어체 톤이
        # 더 자연스러워진다는 게 실측으로 확인됨(F1 상승 + 참조 자막과의 말투(반말/
        # 존댓말) 정합성 개선, 실패율/타임스탬프엔 부작용 없음).
        ex_user=(
            "[1] Hey, you got a minute?\n[2] Yeah, what's up?\n"
            "[3] Nothing, forget it.\n[4] Come on, tell me."
        ),
        ex_assistant=(
            "[1] 야, 시간 좀 있어?\n[2] 어, 왜?\n"
            "[3] 아니야, 됐어.\n[4] 아 왜 그래, 말해봐."
        ),
        instruction="Translate each line to Korean. Keep the [n] numbers exactly:\n{body}",
        label="영→한",
    ),
    "ko2en": dict(
        system_msg=(
            "You are a subtitle translator. Translate Korean subtitles into natural English.\n"
            f"{_DIALOGUE_CONTEXT}\n"
            "STRICT RULES:\n"
            "1. Output ONLY the numbered English translations like [1] Hello.\n"
            "2. Do NOT explain words, grammar, or your thought process.\n"
            "3. Do NOT output Korean."
        ),
        # en2ko와 같은 대화를 방향만 뒤집어 재사용.
        ex_user=(
            "[1] 야, 시간 좀 있어?\n[2] 어, 왜?\n"
            "[3] 아니야, 됐어.\n[4] 아 왜 그래, 말해봐."
        ),
        ex_assistant=(
            "[1] Hey, you got a minute?\n[2] Yeah, what's up?\n"
            "[3] Nothing, forget it.\n[4] Come on, tell me."
        ),
        instruction="Translate each line to English. Keep the [n] numbers exactly:\n{body}",
        label="한→영",
    ),
}


def call_llm_translate(tokenizer, model, device, prompt_body, direction):
    """번호 태그 프롬프트로 LLM 호출, 번호 붙은 줄만 추려서 반환."""
    cfg = DIRECTION_CONFIG[direction]
    messages = [
        {"role": "system", "content": cfg["system_msg"]},
        {"role": "user", "content": cfg["ex_user"]},
        {"role": "assistant", "content": cfg["ex_assistant"]},
        {"role": "user", "content": f"Translate these lines. Output ONLY numbered translations:\n{prompt_body}"},
    ]
    raw = generate(tokenizer, model, device, messages, max_new_tokens=1024)

    final_lines = []
    for line in raw.split("\n"):
        line = line.strip()
        if re.match(r"^\[\d+\]", line):
            final_lines.append(line)
    return "\n".join(final_lines)


def translate_chunk(doc, chunk, direction, tokenizer, model, device):
    """청크 번역. (j, txt) 목록 반환."""
    frs, prompt = build_prompt(doc, chunk, instruction=DIRECTION_CONFIG[direction]["instruction"])
    llm_out = call_llm_translate(tokenizer, model, device, prompt, direction)
    return parse_marked(llm_out, frs)


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


def compare_per_cue(unit_texts, aligned_reference):
    """큐 단위 F1 평균 - 시간으로 정렬된 참조와 유닛끼리 하나씩 비교."""
    scores = [f1_score(p, r) for p, r in zip(unit_texts, aligned_reference) if p.strip() and r.strip()]
    return sum(scores) / len(scores) if scores else 0.0


def check_timestamp_integrity(out_srt_path, src_doc):
    """출력 SRT 타임스탬프가 원본 유닛과 그대로 일치하는지 확인 (병합 무결성).
    load_srt()는 본문이 빈 큐를 노이즈로 보고 버리는데, 우리 출력은 번역이 정직하게
    실패해 본문이 빈 큐가 있을 수 있어(내용과 무관하게 셀 수 있어야 함) 그대로 쓰면
    안 된다 - 그래서 본문 파싱 없이 타임스탬프만 순서대로 직접 뽑는다."""
    raw = Path(out_srt_path).read_text(encoding="utf-8")
    out_ts = [(_to_sec(a), _to_sec(b)) for a, b in _TS_RE.findall(raw)]
    if len(out_ts) != len(src_doc.units):
        return False, [f"유닛 개수 불일치: 출력 {len(out_ts)} vs 원본 {len(src_doc.units)}"]
    bad = [
        i for i, ((ot0, ot1), s) in enumerate(zip(out_ts, src_doc.units))
        if abs(ot0 - s.meta["t_start"]) > 1e-6 or abs(ot1 - s.meta["t_end"]) > 1e-6
    ]
    return (len(bad) == 0), bad


def run_translation(direction, src_doc, ref_doc, chunkers, tokenizer, model, device, out_dir):
    """chunkers: {method_label: chunker_fn(text) -> list[Chunk]}
    ref_doc가 None이면(참조 자막이 없는 데이터셋) 큐 단위 F1 비교를 건너뛴다."""
    cfg = DIRECTION_CONFIG[direction]
    print("=" * 70)
    print(f"[{cfg['label']}] {src_doc.name} 전체 {len(src_doc.units)}줄")
    print("=" * 70)

    # 방향당 한 번만 계산 - fixed/semantic 양쪽에 동일하게 재사용(공정성 유지).
    # 시간 정렬도 청킹 방식과 무관하게 문서당 한 번만 계산해서 그대로 공유한다.
    aligned_reference = align_by_overlap(src_doc, ref_doc) if ref_doc is not None else None

    results = {}
    for method_label, chunker_fn in chunkers.items():
        chunks = chunker_fn(src_doc.text)
        print(f"  [{method_label}] 청크 {len(chunks)}개 번역 중...")
        pieces = []
        for c_idx, c in enumerate(chunks, 1):
            pieces.extend(translate_chunk(src_doc, c, direction, tokenizer, model, device))
            if c_idx % 50 == 0 or c_idx == len(chunks):
                print(f"    {method_label} 청크 {c_idx}/{len(chunks)} 완료")
        unit_texts = merge_to_units(src_doc, pieces)

        out_path = Path(out_dir) / f"{src_doc.name.rsplit('.', 1)[0]}_{direction}_{method_label}.srt"
        write_srt(src_doc, unit_texts, str(out_path))

        ts_ok, ts_bad = check_timestamp_integrity(out_path, src_doc)
        f1 = compare_per_cue(unit_texts, aligned_reference) if aligned_reference is not None else None
        f1_display = f"{f1:.4f}" if f1 is not None else "N/A(참조 없음)"
        print(f"  ✅ [{method_label}] -> {out_path}")
        print(f"     텍스트 F1(큐 단위)={f1_display}  타임스탬프보존={'OK' if ts_ok else f'문제 {len(ts_bad)}건'}")

        results[method_label] = dict(out_path=out_path, f1=f1, ts_ok=ts_ok, ts_bad=ts_bad)
    return results
