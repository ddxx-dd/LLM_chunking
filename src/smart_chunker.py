"""스마트 청커 (구조 우선 + 의미 분할 + 토큰 크기 제어)

아이디어는 3단계뿐이다.
  1) 구조로 먼저 자른다   : 헤더(#, 논문 섹션 번호, "Passage N:")마다 새 섹션.
                             표는 통째로 한 조각, 자막은 큐 하나가 한 조각.
  2) 너무 큰 섹션만 의미로 자른다 : 조각 사이 유사도(좌우 창 평균)가 가장 낮은 곳에서
                             반으로 자르기를 max_tokens 이하가 될 때까지 반복.
  3) 너무 작은 청크는 이웃과 합친다 : min_tokens 미만이면 다음 청크와 합침.

반환하는 청크는 항상 원문의 리터럴 슬라이스(text[start:end])라서, 기존 코드의
add_start_index / fragments() / merge_to_units()가 그대로 동작한다.
"""
import math
import re

import numpy as np
from langchain_text_splitters.base import TextSplitter

SENT_RE = re.compile(r"(?<=[.!?。])\s+")          # 한국어는 kss.split_sentences로 바꾸면 더 정확
MD_HEADING_RE = re.compile(r"^#{1,6} ")
PASSAGE_RE = re.compile(r"^Passage \d+:")                                            # LongBench 다중 문서
EN_END_RE = re.compile(r"[.!?…\"')\]]$")
KO_CONT_RE = re.compile(r"(고|며|면서|는데|은데|지만|서|면|니까|다가|도록|를|을|의|,|…|\.\.\.)$")


# ---------------------------------------------------------------------------
# 0. 텍스트 -> 블록 (start, end, kind)   kind: heading / table / para / cue
# ---------------------------------------------------------------------------
def detect_blocks(text, mode):
    if mode == "srt":                     # srt 로더는 큐 하나를 한 줄로 저장한다
        return [(m.start(), m.end(), "cue") for m in re.finditer(r"[^\n]+", text)]

    if mode == "docx":                    # Docling 마크다운: 빈 줄로 구분된 덩어리
        pattern = r"[^\n]+(?:\n(?!\n)[^\n]+)*"
    elif mode == "pdf":                   # Docling 마크다운(docx와 동일 형식): 빈 줄로 구분된 덩어리
        pattern = r"[^\n]+(?:\n(?!\n)[^\n]+)*"
    else:                                 # longbench: 한 줄 = 한 문단
        pattern = r"[^\n]+"

    blocks = []
    for m in re.finditer(pattern, text):
        body = m.group().strip()
        if MD_HEADING_RE.match(body) or PASSAGE_RE.match(body):
            kind = "heading"
        elif body.startswith(("|", "<table")):             # 마크다운 표 / HTML 표
            kind = "table"
        else:
            kind = "para"
        # 한 줄=한 블록 모드(longbench)에서 표 행이 여러 줄로 쪼개져 들어오면 하나로 묶는다 -
        # docx/pdf는 이미 멀티라인 패턴이라 표 전체가 한 블록으로 잡히므로 이 병합이 필요 없음
        # (지금은 실제로 이 분기를 타는 경로가 없음 - longbench에 표가 나오면 대비용으로 남겨둠).
        if kind == "table" and blocks and blocks[-1][2] == "table" and mode not in ("docx", "pdf"):
            blocks[-1] = (blocks[-1][0], m.end(), "table")
        else:
            blocks.append((m.start(), m.end(), kind))
    return blocks


def split_sentences(text, s, e):
    """[s, e) 구간을 문장 (start, end)로 자른다. 오프셋은 원문 기준."""
    out, cur = [], s
    for m in SENT_RE.finditer(text, s, e):
        if m.start() > cur:
            out.append((cur, m.start()))
        cur = m.end()
    if cur < e:
        out.append((cur, e))
    return out


def cue_continues(cur, nxt, lang):
    """자막 한 문장이 다음 큐로 이어지면 True -> 그 사이는 자르지 않도록."""
    cur, nxt = cur.strip(), nxt.strip()
    if cur.endswith(("...", "…", ",")):
        return True
    if lang == "en":
        return not EN_END_RE.search(cur) or nxt[:1].islower()
    return bool(KO_CONT_RE.search(cur))


# ---------------------------------------------------------------------------
# 스마트 청커
# ---------------------------------------------------------------------------
class SmartChunker:
    def __init__(self, embed_model, count_tokens, mode, lang="ko",
                 min_tokens=100, max_tokens=500, window=2):
        self.embed = embed_model          # SentenceTransformer("BAAI/bge-m3")
        self.count = count_tokens         # 문자열 -> 토큰 수 (bge-m3 또는 Gemma 토크나이저)
        self.mode, self.lang = mode, lang
        self.min_t, self.max_t, self.w = min_tokens, max_tokens, window

    # 1단계: 구조로 섹션 나누기 ---------------------------------------------
    def sections(self, text, blocks):
        if self.mode == "srt":            # 자막은 섹션이 없다: 큐 전체가 한 섹션
            return [blocks]
        secs, cur = [], []
        for s, e, kind in blocks:
            if kind == "heading" and cur:
                secs.append(cur)
                cur = []
            if kind == "para":
                cur += [(a, b, "sent") for a, b in split_sentences(text, s, e)]
            else:
                cur.append((s, e, kind))  # 헤더·표는 쪼개지 않는 한 조각
        if cur:
            secs.append(cur)
        return secs

    # 2단계: 경계 점수 (낮을수록 자르기 좋은 곳) ------------------------------
    def boundary_scores(self, text, pieces):
        vecs = np.asarray(self.embed.encode([text[s:e] for s, e, _ in pieces],
                                            normalize_embeddings=True))
        scores = []
        for i in range(len(pieces) - 1):
            left = vecs[max(0, i - self.w + 1): i + 1].mean(0)
            right = vecs[i + 1: i + 1 + self.w].mean(0)
            sim = float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9))
            if self.mode == "srt":
                a, b = pieces[i], pieces[i + 1]
                if cue_continues(text[a[0]:a[1]], text[b[0]:b[1]], self.lang):
                    sim += 1.0            # 문장이 이어지는 곳은 사실상 자르지 않음
            scores.append(sim)
        return scores

    def split_section(self, pieces, scores, toks):
        if sum(toks) <= self.max_t or len(pieces) == 1:
            return [pieces]
        # 양쪽이 모두 min_tokens 이상이 되는 경계 중 점수가 가장 낮은 곳에서 자른다
        cands = [i for i in range(len(pieces) - 1)
                 if sum(toks[:i + 1]) >= self.min_t and sum(toks[i + 1:]) >= self.min_t]
        cut = min(cands or range(len(pieces) - 1), key=lambda i: scores[i])
        return (self.split_section(pieces[:cut + 1], scores[:cut], toks[:cut + 1]) +
                self.split_section(pieces[cut + 1:], scores[cut + 1:], toks[cut + 1:]))

    # 3단계: 작은 청크 합치기 -----------------------------------------------
    def merge_small(self, text, groups):
        """앞쪽으로 훑으면서 직전 청크가 min_tokens 미만이면 현재 청크와 합친다.
        섹션 하나의 그룹 리스트에만 호출할 것 - 여러 섹션을 합쳐서 넘기면 헤더
        경계를 넘어 서로 다른 절의 내용이 한 청크로 섞인다(실측 발견, chunk_spans의
        섹션별 호출로 방지)."""
        out = []
        for g in groups:
            if out:
                prev_text = text[out[-1][0][0]:out[-1][-1][1]]
                joined = text[out[-1][0][0]:g[-1][1]]
                if self.count(prev_text) < self.min_t and self.count(joined) <= self.max_t:
                    out[-1] = out[-1] + g
                    continue
            out.append(g)
        # 마지막 청크는 위 루프에서 "다음 청크"가 없어 한 번도 검사되지 않는다 -
        # min_tokens 미만인 채로 남으면 이전 청크와 합쳐서 자투리로 방치되지 않게 한다.
        if len(out) >= 2 and self.count(text[out[-1][0][0]:out[-1][-1][1]]) < self.min_t:
            out[-2] = out[-2] + out[-1]
            out.pop()
        return out

    def chunk_spans(self, text):
        groups = []
        for pieces in self.sections(text, detect_blocks(text, self.mode)):
            toks = [self.count(text[s:e]) for s, e, _ in pieces]
            scores = self.boundary_scores(text, pieces) if sum(toks) > self.max_t else []
            section_groups = self.split_section(pieces, scores, toks)
            groups += self.merge_small(text, section_groups)
        return [(g[0][0], g[-1][1]) for g in groups]


# ---------------------------------------------------------------------------
# 4단계: elements 기반 smart 청킹(docx/pdf 트랙, smart_chunk.md 5절)
# 위 SmartChunker(srt/longbench용, 정규식으로 블록을 찾는 flat-text 방식)와는 별개 -
# elements는 이미 파싱 단계에서 label(section_header/table/...)이 정해져 있어서
# 블록 탐지가 필요 없다. 알고리즘(구조 우선 -> 의미 분할 -> 작은 그룹 병합)은 같다.
# ---------------------------------------------------------------------------
def _join_texts(id_to_el, ids):
    return "\n".join(id_to_el[i]["text"] for i in ids)


def _boundary_scores(embed_model, texts, window):
    vecs = np.asarray(embed_model.encode(texts, normalize_embeddings=True))
    scores = []
    for i in range(len(texts) - 1):
        left = vecs[max(0, i - window + 1): i + 1].mean(0)
        right = vecs[i + 1: i + 1 + window].mean(0)
        sim = float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9))
        scores.append(sim)
    return scores


def _split_by_score(ids, scores, toks, min_tokens, max_tokens):
    """SmartChunker.split_section과 같은 재귀 이분할 - 요소 경계에서만 자르고(요소를
    쪼개지 않음), 양쪽 다 min_tokens 이상 되는 경계 중 유사도가 가장 낮은 곳을 고른다."""
    if sum(toks) <= max_tokens or len(ids) == 1:
        return [ids]
    cands = [i for i in range(len(ids) - 1)
             if sum(toks[:i + 1]) >= min_tokens and sum(toks[i + 1:]) >= min_tokens]
    cut = min(cands or range(len(ids) - 1), key=lambda i: scores[i])
    return (_split_by_score(ids[:cut + 1], scores[:cut], toks[:cut + 1], min_tokens, max_tokens) +
            _split_by_score(ids[cut + 1:], scores[cut + 1:], toks[cut + 1:], min_tokens, max_tokens))


def _merge_small_groups_global(groups, count_tokens, min_tokens, max_tokens):
    """★ 2026-09-27 수정(전역 병합 규칙 보완) - min_tokens 미만 그룹은 **합친 크기가
    max_tokens 이하일 때만** 다음 그룹 앞에 합친다(순서 유지). 넘으면 이전 그룹과
    합치되, 그것도 max_tokens를 넘으면 그대로 둔다(작아도 무리하게 합치지 않음).
    표 그룹(_kind="table")은 병합 대상으로 안 씀(캡션까지 붙은 표 자체 마크다운에
    남의 텍스트가 섞이면 안 되므로 - 작은 표는 `_merge_small_tables`가 별도로 처리).
    제목 뒤에서 자르지 않는 원칙은 섹션 구성 단계(section_header가 새 섹션을 열되
    그 뒤 본문과 같은 그룹에 들어감)에서 이미 지켜짐 - 여기선 안 건드림."""
    out = list(groups)
    i = 0
    while i < len(out):
        g = out[i]
        if g.get("_kind") == "table" or count_tokens(g["text"]) >= min_tokens:
            i += 1
            continue
        if i + 1 < len(out) and out[i + 1].get("_kind") != "table":
            nxt = out[i + 1]
            combined = g["text"] + "\n" + nxt["text"]
            if count_tokens(combined) <= max_tokens:
                out[i + 1] = {"owners": g["owners"] + nxt["owners"], "text": combined,
                              "_kind": nxt.get("_kind", "text")}
                out.pop(i)
                continue  # 합친 결과가 이제 i번 자리 - 여전히 작을 수 있으니 다시 검사
        if i > 0 and out[i - 1].get("_kind") != "table":
            prev = out[i - 1]
            combined = prev["text"] + "\n" + g["text"]
            if count_tokens(combined) <= max_tokens:
                out[i - 1] = {"owners": prev["owners"] + g["owners"], "text": combined,
                              "_kind": prev.get("_kind", "text")}
                out.pop(i)
                i -= 1  # 합친 결과가 이제 i-1 자리 - 다시 검사
                continue
        i += 1  # 앞뒤 다 안 되면(표이거나 합치면 max_tokens 초과) 작아도 그대로 둠
    return out


def _merge_small_tables(groups, count_tokens, min_tokens, max_tokens):
    """★ 2026-09-27 추가 - 작은 표(min_tokens 미만)는 쪼개지 않고 통째로 앞 그룹(없으면
    뒤 그룹)과 합친다. 합친 크기가 max_tokens 이하일 때만 - 표를 쪼개거나 표 마크다운
    "안에" 다른 텍스트를 끼워 넣는 게 아니라, 표 마크다운 블록 전체를 다른 그룹의
    텍스트 뒤/앞에 그대로 붙이는 것뿐이라 원칙(표는 안 쪼갬, 안 섞음)에 안 어긋남.
    합쳐진 결과는 표 콘텐츠를 포함하므로 계속 "table"로 표시해 이후 병합에서 보호한다."""
    out = list(groups)
    i = 0
    while i < len(out):
        g = out[i]
        if g.get("_kind") != "table" or count_tokens(g["text"]) >= min_tokens:
            i += 1
            continue
        merged = False
        if i > 0:
            prev = out[i - 1]
            combined = prev["text"] + "\n" + g["text"]
            if count_tokens(combined) <= max_tokens:
                out[i - 1] = {"owners": prev["owners"] + g["owners"], "text": combined, "_kind": "table"}
                out.pop(i)
                merged = True
        if not merged and i + 1 < len(out):
            nxt = out[i + 1]
            combined = g["text"] + "\n" + nxt["text"]
            if count_tokens(combined) <= max_tokens:
                out[i + 1] = {"owners": g["owners"] + nxt["owners"], "text": combined, "_kind": "table"}
                out.pop(i)
                merged = True
        if not merged:
            i += 1  # 앞뒤 다 안 되면(합치면 max_tokens 초과) 작아도 그대로 둠
    return out


def _split_big_table(table_el, count_tokens, max_tokens):
    """표 하나가 max_tokens를 넘으면 행 묶음으로 나누고, 나뉜 조각마다 헤더 행을 반복해서
    앞에 붙인다(smart_chunk.md 5절) - 표 자체를 통째로 한 그룹으로 못 넣을 때만 호출됨.
    ★ 두 번째 조각부터 반복되는 헤더 행은 owners에 안 넣는다(헤더 셀은 첫 조각에서만
    번역 대상 - 텍스트에는 매 조각 앞에 계속 나오지만 "참고 문맥"일 뿐, [n] 번호가
    안 붙어 중복 번역을 안 함, 6절 원리와 동일)."""
    n_cols = table_el["n_cols"]
    rows = {}
    for c in table_el["cells"]:
        rows.setdefault(c["row"], []).append(c)
    header_row = min(rows)
    header_cells = sorted(rows[header_row], key=lambda c: c["col"])

    def row_md(row_cells):
        by_col = {c["col"]: c["text"].replace("\n", " ").replace("|", "\\|") for c in row_cells}
        return "| " + " | ".join(by_col.get(c, "") for c in range(n_cols)) + " |"

    prefix = row_md(header_cells) + "\n" + "|" + "|".join([" --- "] * n_cols) + "|"

    fragments = []  # ([row_no, ...], text)
    body_rows = sorted(r for r in rows if r != header_row)
    cur_rows, cur_text = [], prefix
    for r in body_rows:
        line = row_md(sorted(rows[r], key=lambda c: c["col"]))
        candidate = cur_text + "\n" + line
        if cur_rows and count_tokens(candidate) > max_tokens:
            fragments.append((cur_rows, cur_text))
            cur_rows, cur_text = [r], prefix + "\n" + line
        else:
            cur_rows.append(r)
            cur_text = candidate
    if cur_rows:
        fragments.append((cur_rows, cur_text))

    out = []
    for i, (row_nums, text) in enumerate(fragments):
        owners = []
        if i == 0:  # 헤더 셀·캡션은 첫 조각에서만 번역 대상(위 docstring 참고)
            owners += [("cell", table_el["id"], c["cell_id"]) for c in header_cells]
            owners += [("el", cid) for cid in table_el.get("caption_ids", [])]
        for r in row_nums:
            owners += [("cell", table_el["id"], c["cell_id"]) for c in rows[r]]
        out.append({"owners": owners, "text": text, "_kind": "table"})
    return out


def group_elements(elements, embed_model, count_tokens, min_tokens=100, max_tokens=400, window=2):
    """smart 청커 본체(smart_chunk.md 5절) - elements를 구조 우선(섹션 헤더 경계, 표+캡션은
    한 그룹) + 의미 분할(너무 큰 섹션만, 요소 경계에서) + 크기 조절(작은 그룹은 문서 전체
    기준으로 다음/이전 그룹과 병합)로 그룹핑한다. 단일 요소가 max_tokens를 넘어도 안
    쪼갠다(요소 경계는 항상 지킴). 반환: [{"owners": [...], "text": "..."}] - owners는
    번역 매핑에 바로 쓸 수 있는 owner 키 목록(5절과 동일 형식: ("el", id) 또는
    ("cell", table_id, cell_id)) - 표는 항상 셀 단위, 일반 요소는 통째로 하나."""
    id_to_el = {e["id"]: e for e in elements}

    sections = []  # ("table", table_element) 또는 ("text", [element_id, ...])
    cur, caption_ids_used = [], set()
    for e in elements:
        if e.get("skip") == "picture" or e["id"] in caption_ids_used:
            continue
        if e["label"] == "table":
            if cur:
                sections.append(("text", cur)); cur = []
            sections.append(("table", e))
            caption_ids_used.update(e.get("caption_ids", []))
            continue
        if e["label"] == "section_header" and cur:
            sections.append(("text", cur)); cur = []
        cur.append(e["id"])
    if cur:
        sections.append(("text", cur))

    groups = []
    for kind, payload in sections:
        if kind == "table":
            table_el = payload
            caption_text = _join_texts(id_to_el, table_el.get("caption_ids", []))
            full_text = table_el["text"] + ("\n" + caption_text if caption_text else "")
            if count_tokens(full_text) <= max_tokens:
                owners = [("cell", table_el["id"], c["cell_id"]) for c in table_el["cells"]]
                owners += [("el", cid) for cid in table_el.get("caption_ids", [])]
                groups.append({"owners": owners, "text": full_text, "_kind": "table"})
            else:
                groups += _split_big_table(table_el, count_tokens, max_tokens)
            continue

        ids = payload
        toks = [count_tokens(id_to_el[i]["text"]) for i in ids]
        if sum(toks) > max_tokens and len(ids) > 1:
            texts = [id_to_el[i]["text"] for i in ids]
            scores = _boundary_scores(embed_model, texts, window)
            id_groups = _split_by_score(ids, scores, toks, min_tokens, max_tokens)
        else:
            id_groups = [ids]
        for g in id_groups:
            groups.append({"owners": [("el", i) for i in g], "text": _join_texts(id_to_el, g), "_kind": "text"})

    groups = _merge_small_tables(groups, count_tokens, min_tokens, max_tokens)  # 작은 표 먼저(통째로 이웃과 병합)
    groups = _merge_small_groups_global(groups, count_tokens, min_tokens, max_tokens)  # 그다음 일반 텍스트 그룹
    for g in groups:
        g.pop("_kind", None)
    return groups


# ---------------------------------------------------------------------------
# fixed/semantic 개수 맞추기(smart_chunk.md 5절) - smart가 N개로 나눴을 때 fixed/semantic도
# 같은 N개로 맞춰서 세 청커를 공정하게 비교한다.
# ---------------------------------------------------------------------------
def recover_spans(flat, chunks):
    """공백 무시하고 청크 텍스트를 flat_text에서 순서대로 찾아 (start, end) 복원 -
    LangChain SemanticChunker가 " ".join()으로 문장을 재조합해서(개행이 공백으로 바뀜)
    start_index가 실제 위치와 어긋나는 오프셋 오류를 우회한다(실측 확인된 문제)."""
    pos = [i for i, ch in enumerate(flat) if not ch.isspace()]
    squeezed = "".join(flat[i] for i in pos)
    spans, cur = [], 0
    for c in chunks:
        key = "".join(c.split())
        k = squeezed.find(key, cur)
        if k < 0 or not key:
            spans.append(None); continue  # 못 찾으면 None - 호출하는 쪽에서 span_missing으로 집계
        spans.append((pos[k], pos[k + len(key) - 1] + 1))
        cur = k + len(key)
    return spans


def make_fixed_chunks(flat_text, n):
    """smart가 만든 N개에 맞춰 fixed 청크를 만든다 - chunk_overlap=0 필수(기본값 200을
    두면 겹친 부분이 두 번 번역돼서 요소별 번역문이 중복됨). add_start_index는 fixed에서
    정확하다(문제는 SemanticChunker만)."""
    from langchain_text_splitters import CharacterTextSplitter
    size = math.ceil(len(flat_text) / n) if n > 0 else max(len(flat_text), 1)
    splitter = CharacterTextSplitter(separator="", chunk_size=size, chunk_overlap=0, add_start_index=True)
    docs = splitter.create_documents([flat_text])
    return [(d.metadata["start_index"], d.metadata["start_index"] + len(d.page_content)) for d in docs]


def make_semantic_chunks(flat_text, embeddings, n):
    """smart가 만든 N개에 맞춰 semantic 청크를 만든다 - SemanticChunker(number_of_chunks=N)로
    경계 기준값이 N개에 맞게 자동으로 정해지고, recover_spans로 위치를 복원한다."""
    from langchain_experimental.text_splitter import SemanticChunker
    splitter = SemanticChunker(embeddings, number_of_chunks=n)
    chunks = splitter.split_text(flat_text)
    return recover_spans(flat_text, chunks)


def section_for_span(elements, start):
    """청크 시작 위치(flat_text 절대 좌표) 이전의 가장 가까운 section_header 요소 id -
    section_hit 채점을 fixed/semantic도 smart와 같은 기준으로 하기 위함(4절). 못 찾으면
    None(첫 섹션 헤더보다 앞에 있는 청크 - 문서 서두)."""
    best_id, best_start = None, -1
    for e in elements:
        if e["label"] == "section_header" and e["span"][0] <= start and e["span"][0] > best_start:
            best_id, best_start = e["id"], e["span"][0]
    return best_id


def section_for_owners(elements, owners):
    """smart 청크는 flat_text 절대 위치를 따로 안 들고 있어서(owners만 있음) - 첫
    owner가 속한 최상위 요소의 시작 위치로 section_for_span을 그대로 재사용한다."""
    if not owners:
        return None
    return section_for_span(elements, elements[owners[0][1]]["span"][0])


class SmartTextSplitter(TextSplitter):
    """LangChain TextSplitter로 감싸서 기존 파이프라인(split_documents,
    add_start_index=True)에 그대로 꽂는다."""

    def __init__(self, chunker, **kwargs):
        kwargs.setdefault("chunk_overlap", 0)
        kwargs.setdefault("add_start_index", True)
        super().__init__(**kwargs)
        self._chunker = chunker

    def split_text(self, text):
        return [text[s:e] for s, e in self._chunker.chunk_spans(text)]
