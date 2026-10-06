"""docx_corpus QA의 근거 구간 샘플링 (docs/restructure_plan.md 7.3절 1단계). LLM은 쓰지 않는다.

문서마다 근거 구간 후보 N개(기본 5)를 뽑아 data/raw/docx_corpus/_meta/candidates.json에 저장한다. 이 구간을 읽고
질문·정답을 만드는 것은 다음 단계(Claude)이고, 여기서는 "어느 글 조각을 보여 줄지"만 정한다.

공정성: 구간은 청커·제목 구조와 무관하게 고른다. 무작위 앵커 블록 하나에서 시작해 앞뒤로 조각을 붙여 나가므로
제목 섹션 경계를 따르지 않는다(제목을 넘나들 수 있다). 목표 길이는 구간마다 [MIN_TOKENS, MAX_TOKENS]에서 무작위로 정한다.

  - 구간은 to_text(doc)의 리터럴 슬라이스 [start, end)이고 block_ids를 함께 저장한다(근거 스팬).
  - 시드는 (SEED, doc_id)로 정해서 문서마다 독립적이고 문서 목록이 바뀌어도 다른 문서의 구간이 변하지 않는다.
  - 표가 있는 문서는 후보 최대 TABLE_PER_DOC개를 표 블록에서 시작한다(표 질문용). 구간이 서로 겹치지 않는다.
    구간마다 table_ratio(표 글자 비중)를 남긴다 - 생성 단계에서 질문 유형(text / table / text-table)을 정하는 데 쓴다.
  - 구간이 중간에서 끊기지 않게 한다: 문장 중간(소문자)이나 번호 항목 "(a)"로 시작하면 앞쪽으로, 콜론·세미콜론·접속어로 끝나거나
    다음 조각이 이어지는 문장이면 뒤쪽으로 구간을 넓히고, 길이 상한(MAX_TOKENS)을 넘으면 그 후보를 버린다. 표는 통째로만 넣고
    헤더만 있거나 열 수가 어긋난 표는 쓰지 않는다.
  - 목차·참고문헌·색인 제목 아래, 글자 비율이 낮은 구간(숫자·기호 위주), 점선 목차 모양, 빈칸 위주의 표는 뺀다.
"""
import json
import os
import random
import re

from chunking.semantic_core import sentence_spans
from common.config import MANIFEST_DIR, PROCESSED_DIR, RAW_DIR
from common.docjson import load_doc, render

SEED = 42
MIN_TOKENS = 150
MAX_TOKENS = 600
PER_DOC = 5
MAX_TABLE_TRIES = 20  # 표에서 시작하는 시도 횟수 상한(표 구간이 조건을 못 채우는 문서에서 헛돌지 않게)
TABLE_PER_DOC = 2                          # 표가 있는 문서에서 표 블록에서 시작하는 후보 수(최대)
MAX_ATTEMPTS = 300                         # 후보 하나를 만들려고 시도하는 최대 횟수(문서 전체 기준)
MIN_ALPHA_RATIO = 0.55                     # 공백 아닌 글자 중 알파벳 비율의 하한
MAX_EMPTY_CELL_RATIO = 0.5                 # 표의 빈 셀 비율 상한
MAX_TOC_LINE_RATIO = 0.3                   # 비어 있지 않은 줄 중 목차 모양 줄의 비율 상한(그 이상이면 목차 구간)
MIN_TOC_LINES = 4

META_DIR = RAW_DIR / "docx_corpus" / "_meta"
CANDIDATES_PATH = META_DIR / "candidates.json"

_BAD_HEADING = re.compile(r"\b(table of contents|contents|references|bibliography|index|list of (tables|figures)|acknowledg\w*)\b", re.I)
_BAD_ANCESTOR = re.compile(r"\b(references|bibliography)\b", re.I)      # 참고문헌은 하위 제목 아래도 참고문헌이다
_DOT_LEADER = re.compile(r"(\.\s?){5,}")
_TOC_LINE = re.compile(r"\S\t+[\w\-]*\d[\w\-]*\s*$")          # "제목<탭>쪽번호" 모양의 줄(목차·색인)
_ENUM_START = re.compile(r"^\(?([a-z]|[ivx]+|\d+)\)(\s|$)", re.I)    # "(a)", "(ii)", "3)" 같은 번호 항목 시작
_OPEN_END = re.compile(r"([:;,\u2014\u2013]|\b(and|or))\s*$", re.I)       # 문장·목록이 이어질 것 같은 끝
_TOC_TITLE = re.compile(r"^\s*(table of (contents|figures)|contents)\s*$", re.I | re.M)
_CELL_SPLIT = re.compile(r"\s*\|\s*")


def _atoms(text, pieces, count, max_tokens):
    """조각 -> 자를 수 없는 단위 목록. 너무 긴 문단은 문장으로 쪼개고, 너무 긴 표는 못 쓰는 단위(usable=False)로 둔다.
    각 단위 = dict(start, end, kind, block_ids, tokens, usable)."""
    atoms = []
    for p in pieces:
        n = count(text[p.start:p.end])
        if n <= max_tokens:
            atoms.append(dict(start=p.start, end=p.end, kind=p.kind, block_ids=p.block_ids, tokens=n, usable=True))
        elif p.kind == "table":
            atoms.append(dict(start=p.start, end=p.end, kind=p.kind, block_ids=p.block_ids, tokens=n, usable=False))
        else:                                                           # 긴 문단/제목 -> 문장 단위
            for s, e in sentence_spans(text, p.start, p.end):
                atoms.append(dict(start=s, end=e, kind=p.kind, block_ids=p.block_ids, tokens=count(text[s:e]), usable=True))
    return atoms


def _table_ok(table_text):
    rows = [l for l in table_text.split("\n") if l.startswith("|")]
    if len(rows) < 3 or len({r.count("|") for r in rows}) > 1:               # 헤더 + 구분선 + 행 1개 이상, 열 수가 일정
        return False
    cells = [c for c in _CELL_SPLIT.split(table_text) if c != ""]
    if not cells:
        return False
    empty = sum(1 for c in cells if not re.search(r"\w", c))
    return empty / len(cells) <= MAX_EMPTY_CELL_RATIO


def _passes_filters(text, pieces_in_span, heading_path):
    """-> 거절 이유(문자열) 또는 None."""
    # 가장 가까운 제목만 본다: Docling이 "Table of Contents"를 닫지 않으면 본문 전체가 그 아래 경로가 되기 때문
    if heading_path and (_BAD_HEADING.search(heading_path[-1]) or any(_BAD_ANCESTOR.search(h) for h in heading_path)):
        return "heading"
    body = re.sub(r"\s+", "", text)
    if not body or sum(ch.isalpha() for ch in body) / len(body) < MIN_ALPHA_RATIO:
        return "alpha"
    if _DOT_LEADER.search(text):
        return "dotleader"
    lines = [l for l in text.splitlines() if l.strip()]
    toc = sum(bool(_TOC_LINE.search(l)) for l in lines)
    if _TOC_TITLE.search(text) or (toc >= MIN_TOC_LINES and toc / len(lines) >= MAX_TOC_LINE_RATIO):
        return "toc"
    for kind, t in pieces_in_span:
        if kind == "table" and not _table_ok(t):
            return "table_empty"
    return None


def _grow(atoms, anchor, target, count, text, rng, max_tokens, allowed=None):
    """앵커 단위에서 앞뒤로 무작위로 단위를 붙여 target 토큰에 이를 때까지 늘린다. -> (i, j) 또는 None."""
    i = j = anchor
    total = atoms[anchor]["tokens"]
    while total < target:
        options = []
        if i > 0 and atoms[i - 1]["usable"] and (allowed is None or allowed(i - 1)):
            options.append("L")
        if j + 1 < len(atoms) and atoms[j + 1]["usable"] and (allowed is None or allowed(j + 1)):
            options.append("R")
        rng.shuffle(options)
        for side in options:
            k = i - 1 if side == "L" else j + 1
            span_tokens = count(text[atoms[k]["start"]:atoms[j]["end"]] if side == "L" else text[atoms[i]["start"]:atoms[k]["end"]])
            if span_tokens <= max_tokens:
                i, j, total = (k, j, span_tokens) if side == "L" else (i, k, span_tokens)
                break
        else:
            break
    return i, j


def _start_open(atoms, i, text):
    """atoms[i]가 앞 내용에 이어지는 조각처럼 보이는가(문장 중간 / 상위 문장이 있는 번호 항목)."""
    t = text[atoms[i]["start"]:atoms[i]["end"]].lstrip()
    if not t or atoms[i]["kind"] == "table":
        return False
    if t[0].islower():
        return True
    return bool(_ENUM_START.match(t)) and i > 0 and atoms[i - 1]["kind"] != "heading"


def _end_open(atoms, j, text):
    """atoms[j]가 뒤에 이어지는 조각이 있어야 완결되는 모양인가(콜론·세미콜론·접속어 끝 / 다음 조각이 소문자로 이어짐)."""
    if atoms[j]["kind"] in ("table", "heading"):
        return False
    t = text[atoms[j]["start"]:atoms[j]["end"]].rstrip()
    if _OPEN_END.search(t):
        return True
    nxt = j + 1
    return nxt < len(atoms) and atoms[nxt]["kind"] not in ("table", "heading") and text[atoms[nxt]["start"]:atoms[nxt]["end"]].lstrip()[:1].islower()


def _complete(atoms, i, j, text, count, max_tokens, allowed=None):
    """구간 [i, j]가 끊긴 데서 시작하거나 끝나면 앞뒤로 단위를 더 붙여 완결시킨다. 길이 상한을 넘거나 붙일 수 없으면 None."""
    ok = lambda k: 0 <= k < len(atoms) and atoms[k]["usable"] and (allowed is None or allowed(k))
    while atoms[i]["kind"] != "heading" and _start_open(atoms, i, text):
        if not ok(i - 1):
            return None
        i -= 1
        if count(text[atoms[i]["start"]:atoms[j]["end"]]) > max_tokens:
            return None
    while _end_open(atoms, j, text):
        if not ok(j + 1):
            return None
        j += 1
        if count(text[atoms[i]["start"]:atoms[j]["end"]]) > max_tokens:
            return None
    return i, j


def _trim_headings(atoms, i, j):
    """구간이 제목으로 시작하거나 끝나지 않게 다듬는다(제목이 가운데에 있는 것은 허용)."""
    while i <= j and atoms[i]["kind"] == "heading":
        i += 1
    while j >= i and atoms[j]["kind"] == "heading":
        j -= 1
    return (i, j) if i <= j else None


def sample_doc(doc, count, topic=None, doc_type=None, per_doc=PER_DOC, seed=SEED,
               min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS):
    """공통 JSON 문서 하나 -> (후보 목록, 거절 사유 집계). 같은 입력이면 결과가 항상 같다."""
    text, pieces = render(doc)
    atoms = _atoms(text, pieces, count, max_tokens)
    blocks = {b["id"]: b for b in doc["blocks"]}
    usable = [k for k, a in enumerate(atoms) if a["usable"] and a["kind"] != "heading"]
    table_anchors = [k for k in usable if atoms[k]["kind"] == "table" and _table_ok(text[atoms[k]["start"]:atoms[k]["end"]])]
    rng = random.Random(f"{seed}:{doc['doc_id']}")
    out, rejected, taken = [], {}, []
    attempts = table_tries = 0
    is_free = lambda k: not any(atoms[k]["start"] < b and a < atoms[k]["end"] for a, b in taken)
    while len(out) < per_doc and attempts < MAX_ATTEMPTS and usable:
        attempts += 1
        free_tables = [k for k in table_anchors if is_free(k)]
        want_table = bool(free_tables) and sum(c["anchor"] == "table" for c in out) < TABLE_PER_DOC and table_tries < MAX_TABLE_TRIES
        table_tries += want_table
        pool = free_tables if want_table else [k for k in usable if is_free(k)]
        if not pool:
            break
        anchor = rng.choice(pool)
        target = rng.randint(min_tokens, max_tokens)
        i, j = _grow(atoms, anchor, target, count, text, rng, max_tokens, allowed=is_free)
        done = _complete(atoms, i, j, text, count, max_tokens, allowed=is_free)
        if done is None:
            rejected["cut"] = rejected.get("cut", 0) + 1
            continue
        trimmed = _trim_headings(atoms, *done)
        if trimmed is None:
            rejected["heading_only"] = rejected.get("heading_only", 0) + 1
            continue
        i, j = trimmed
        if _start_open(atoms, i, text) or _end_open(atoms, j, text):          # 제목을 떼고 나니 끊긴 조각이 드러난 경우
            rejected["cut"] = rejected.get("cut", 0) + 1
            continue
        s, e = atoms[i]["start"], atoms[j]["end"]
        n = count(text[s:e])
        if n < min_tokens or n > max_tokens:
            rejected["length"] = rejected.get("length", 0) + 1
            continue
        if any(not (e <= a or s >= b) for a, b in taken):
            rejected["overlap"] = rejected.get("overlap", 0) + 1
            continue
        in_span = [p for p in pieces if p.start >= s and p.end <= e]
        block_ids = sorted({b for p in in_span for b in p.block_ids}, key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
        first = blocks.get(block_ids[0], {}) if block_ids else {}
        heading_path = list(first.get("heading_path") or [])
        reason = _passes_filters(text[s:e], [(p.kind, text[p.start:p.end]) for p in in_span], heading_path)
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        taken.append((s, e))
        out.append(dict(doc_id=doc["doc_id"], topic=topic, type=doc_type, start=s, end=e, block_ids=block_ids,
                        n_tokens=n, has_table=any(p.kind == "table" for p in in_span),
                        table_ratio=round(sum(p.end - p.start for p in in_span if p.kind == "table") / (e - s), 3),
                        anchor="table" if want_table else "random", heading_path=heading_path, text=text[s:e]))
    out.sort(key=lambda c: c["start"])
    for k, c in enumerate(out):
        c["cid"] = f"{doc['doc_id']}#{k}"
    return out, rejected


def sample_corpus(count, doc_ids=None, per_doc=PER_DOC):
    """data/processed/docx_corpus 전체 -> ({cid: 후보}, {doc_id: 후보 수 부족 등 문제}, 거절 사유 총계)."""
    meta = {os.path.splitext(m["file"])[0]: m for m in json.loads((MANIFEST_DIR / "docx_corpus.json").read_text(encoding="utf-8"))}
    cands, problems, rejected_total = {}, {}, {}
    for path in sorted((PROCESSED_DIR / "docx_corpus").glob("*.json")):
        if path.name.startswith("_"):
            continue
        doc = load_doc(path)
        if doc_ids is not None and doc["doc_id"] not in doc_ids:
            continue
        m = meta[doc["doc_id"]]
        got, rej = sample_doc(doc, count, topic=m["topic"], doc_type=m["type"], per_doc=per_doc)
        for c in got:
            cands[c["cid"]] = c
        for k, v in rej.items():
            rejected_total[k] = rejected_total.get(k, 0) + v
        if len(got) < per_doc:
            problems[doc["doc_id"]] = f"후보 {len(got)}/{per_doc}개"
    return cands, problems, rejected_total


def locate(text, span_text):
    """span_text가 text 안에 정확히 한 번 나오면 (start, end), 없거나 여러 번이면 None. 정확히 일치하지 않으면 공백 차이를 무시하고 찾는다."""
    i = text.find(span_text)
    if i >= 0:
        return (i, i + len(span_text)) if text.find(span_text, i + 1) < 0 else None
    words = span_text.split()
    if not words:
        return None
    hits = list(re.finditer(r"\s+".join(re.escape(w) for w in words), text))
    return (hits[0].start(), hits[0].end()) if len(hits) == 1 else None


def stale_report(cands, processed_dir=None):
    """저장된 구간 텍스트가 지금의 문서(to_text)와 맞는지 검사한다(전처리를 다시 하면 위치가 틀어진다, 계획서 3.7절).
    -> (ok, moved, lost): ok = 위치가 그대로, moved = {cid: (start, end)} 텍스트는 있고 위치만 바뀜(새 위치), lost = 텍스트가 없거나 여러 번."""
    processed_dir = processed_dir or PROCESSED_DIR / "docx_corpus"
    texts, ok, moved, lost = {}, [], {}, []
    for cid, c in cands.items():
        if c["doc_id"] not in texts:
            texts[c["doc_id"]] = render(load_doc(processed_dir / f"{c['doc_id']}.json"))[0]
        t = texts[c["doc_id"]]
        if t[c["start"]:c["end"]] == c["text"]:
            ok.append(cid)
            continue
        pos = locate(t, c["text"])
        if pos is None:
            lost.append(cid)
        else:
            moved[cid] = pos
    return ok, moved, lost


def relocate(cands, moved, processed_dir=None):
    """stale_report의 moved를 반영한 새 후보 사본 - start/end와 block_ids를 새 문서 기준으로 다시 계산한다."""
    processed_dir = processed_dir or PROCESSED_DIR / "docx_corpus"
    out, cache = {}, {}
    for cid, c in cands.items():
        if cid in moved:
            if c["doc_id"] not in cache:
                cache[c["doc_id"]] = render(load_doc(processed_dir / f"{c['doc_id']}.json"))[1]
            s, e = moved[cid]
            ids = sorted({b for p in cache[c["doc_id"]] if p.start >= s and p.end <= e for b in p.block_ids},
                         key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
            c = {**c, "start": s, "end": e, "block_ids": ids}
        out[cid] = c
    return out


def write_candidates(cands, path=CANDIDATES_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cands, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    return path
