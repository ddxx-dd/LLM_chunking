"""번역(마크다운 블록 단위, 세 청커 공정 비교) - 자막 트랙(srt/mapper.py)과 같은 원리:
청크 하나당 LLM 1회, 청크와 겹친 블록 조각에 [[n]] 번호를 붙여 보내고 [[n]]으로
파싱해서 블록별로 이어붙인다. 이중 괄호인 이유: 논문 인용번호 [8]과 충돌 방지
(실측 확인 - 단일 괄호는 본문 인용을 다음 조각 번호로 오인해 내용이 잘림)."""
import re

from smart_chunker import detect_blocks

FORMULA_RE = re.compile(r"^<!--\s*formula-not-decoded\s*-->$", re.I)
NUM_ONLY_RE = re.compile(r"^[\d\s.,%+\-−×/:()$€₩~]*$")
NUM_TAG_RE = re.compile(r"\[\[(\d+)\]\]\s*(.*?)(?=\s*\[\[\d+\]\]|\Z)", re.S)
WS_COLLAPSE_RE = re.compile(r"[ \t]{3,}")


def _collapse_ws(text):
    """3칸 이상 연속 공백을 하나로 줄인다 - ★ 실측 확인(2026-09-28, pdf 번역 샘플에서
    발견) - Docling이 마크다운 표로 내보낼 때 목차(TOC) 같은 셀을 열 정렬용 공백으로
    수천 자까지 채운다(예: "Introduction . . . 2" 뒤에 공백 약 3800자, 표 전체 블록이
    합쳐지며 더 커짐). 이 공백을 그대로 LLM에 보내면 모델이 내용 없는 공백에 압도돼
    "[[n]] 없이 마침표만 반복"하는 퇴화 생성에 빠진다(실측 확인 - 이 패턴이 pdf 샘플의
    [[n]] 번호 누락 대부분의 원인이었음, 해당 청크는 truncated=1까지 겹침)."""
    return WS_COLLAPSE_RE.sub(" ", text)

PROMPT_TEMPLATES = {
    "docx": ("Translate each numbered fragment into natural English. Rules:\n"
             "- Keep the [[n]] numbers exactly, one per line.\n"
             "- Copy formulas, symbols, variables, and numbers exactly as-is - do not convert to LaTeX.\n"
             "- Keep proper nouns (names, organizations, products) in the original language.\n\n{body}"),
    "pdf": ("다음 번호가 매겨진 조각들을 자연스러운 한국어로 번역하세요. 규칙:\n"
            "- [[n]] 번호를 그대로 유지하고, 한 줄에 하나씩 쓰세요.\n"
            "- 수식·기호·변수·숫자는 원문 그대로 복사하세요 - LaTeX로 바꾸지 마세요.\n"
            "- 인명·기관명·제품명 등 고유명사는 원문 언어 그대로 두세요.\n\n{body}"),
}


def build_units(flat_text, fmt):
    """마크다운을 빈 줄 단위 블록(문단/제목/표)으로 나눈다 - smart_chunker.detect_blocks
    재사용(같은 구조 인식 기준 - 표는 여러 줄이어도 한 블록). 수식 주석과 숫자·기호만
    있는 블록은 번역 대상에서 제외."""
    units = []
    for s, e, kind in detect_blocks(flat_text, fmt):
        text = flat_text[s:e]
        stripped = text.strip()
        skip = "formula" if FORMULA_RE.match(stripped) else ("num_only" if NUM_ONLY_RE.match(stripped) else None)
        units.append({"start": s, "end": e, "text": text, "skip": skip})
    return units


def _units_for_chunk(units, chunk_start, chunk_end):
    """청크와 겹친 블록 "조각"만(겹친 부분 텍스트) - skip 대상은 뺀다."""
    out = []
    for u in units:
        if u["skip"]:
            continue
        s, e = max(u["start"], chunk_start), min(u["end"], chunk_end)
        if s < e:
            out.append((u, u["text"][s - u["start"]:e - u["start"]]))
    return out


def fake_translate_batch(prompts):
    """--fake-llm용 - [[n]] 조각마다 «»로 감싸서 그대로 돌려준다(끊긴 호출 0건)."""
    responses = ["\n".join(f"[[{m.group(1)}]] «{m.group(2).strip()}»" for m in NUM_TAG_RE.finditer(p))
                 for p in prompts]
    return responses, 0


def translate_document(flat_text, fmt, chunk_spans, batch_translate_fn):
    """chunk_spans: [(start,end), ...](청커가 만든 청크 경계). batch_translate_fn(prompts)
    -> (responses, truncated) - llm.translate_batch를 감싼 함수(진짜) 또는
    fake_translate_batch(가짜) 둘 다 됨. 반환: (pairs, stats). pairs는 CometKiwi 채점용
    (원문 블록, 번역) 쌍 - {"start","end","src","mt"}."""
    units = build_units(flat_text, fmt)
    chunk_frags = []  # [(tagged, prompt), ...] - 조각이 하나도 없는 청크는 호출 안 함
    for s, e in chunk_spans:
        tagged = _units_for_chunk(units, s, e)
        if not tagged:
            continue
        body = "\n".join(f"[[{i + 1}]] {_collapse_ws(t)}" for i, (_, t) in enumerate(tagged))
        chunk_frags.append((tagged, PROMPT_TEMPLATES[fmt].format(body=body)))

    prompts = [p for _, p in chunk_frags]
    responses, truncated = batch_translate_fn(prompts)

    frags_by_unit = {}
    missing = 0
    for (tagged, _), response in zip(chunk_frags, responses):
        parsed = {int(m.group(1)): m.group(2).strip() for m in NUM_TAG_RE.finditer(response)}
        for i, (u, _) in enumerate(tagged):
            key = (u["start"], u["end"])
            if (i + 1) not in parsed:
                missing += 1
            frags_by_unit.setdefault(key, []).append(parsed.get(i + 1, ""))

    pairs = []
    for (start, end), frags in frags_by_unit.items():
        mt = " ".join(f for f in frags if f).strip()
        if mt:
            pairs.append({"start": start, "end": end, "src": _collapse_ws(flat_text[start:end]), "mt": mt})

    stats = {"units_total": sum(1 for u in units if not u["skip"]), "units_translated": len(pairs),
              "llm_calls": len(chunk_frags), "missing_numbers": missing, "truncated": truncated}
    return pairs, stats
