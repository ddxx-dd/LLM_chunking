"""2차 전처리의 마지막 정리: 2차가 만든 단위 목록 → clean_text(모든 청커의 입력: 문서 하나 = 긴 텍스트 하나). stage2.process가 부른다.

형식(2026-10-17 v4, Vectara 공식 코퍼스와 비교해 정함; 비교 도구는 tests/clean_text/ref_compare.py)
  단위(문단) 사이               빈 줄('\\n\\n')
  제목과 본문 사이              줄바꿈('\\n'). 단위의 heading이 글 안 어디에 있든 제목 줄 끝 줄바꿈을 유지, 마크다운 표시 없음
  짧은 조각을 붙인 자리          공백(2차 '조각 붙이기'가 넣은 줄바꿈)
  원문 블록에 있던 줄바꿈        유지(표 행, Docling 블록 글 안의 줄바꿈)
  단위 안 연속 빈 줄             줄바꿈 하나로
  참고문헌(2차 ref 단위)         연속된 것을 줄바꿈으로 이은 한 문단, 항목 앞에 Docling 목록 표지('[12]', '3.')를 붙임
  디스플레이 수식('$$ … $$')     자기 문단(앞 글 / 수식 / 이어지는 글)
  참고문헌 모양 표               참고문헌 영역 안(앞·뒤 문단이 참고문헌이거나 표 제목이 References)이고, 데이터 행 3줄 이상 중 80% 이상이
                               '짧은 열쇠 칸(10자 이하) + 연도가 든 긴 칸(30자 이상)'이면 '열쇠 본문' 줄로 펴서 참고문헌 문단에 합침
  그 밖(2차 그대로)             표는 캡션과 한 블록의 마크다운, 그림 캡션 남김, 각주 문서 끝, 논문 제목·저자는 본문 앞
결과의 paras는 text를 빈 줄로 나눈 문단과 순서대로 대응한다: {kind(para/list/caption/footnote/table/code/heading/formula/references), pages, block_ids}.
청커는 text만 쓰고, paras는 평가·분석·추적용이다.
"""
import json
import re
from collections import Counter

NS = lambda t: "".join(t.split())
YEAR = re.compile(r"\b(1[89]|20)\d\d[a-z]?\b")
KEY = re.compile(r"^\[?[\w+\-. ]{1,10}\]?$")
REFS_HEAD = re.compile(r"^\s*(\d+\.?\s*)?(references?|bibliography)\s*$", re.I)


def original_breaks(blocks, block_ids):
    """원문 블록 글 안의 줄바꿈을 앞뒤 글(각 12자)로 기록한 집합."""
    out = set()
    for i in block_ids:
        t = blocks.get(i, "")
        for m in re.finditer(r"\n", t):
            out.add((t[max(0, m.start() - 12):m.start()].strip(), t[m.end():m.end() + 12].strip()))
    return out


def unit_text(u, keep, log):
    """단위 하나 → 문단 글: 제목 줄바꿈·원문 줄바꿈은 유지, 조각 붙이기 줄바꿈은 공백."""
    t = re.sub(r"\n\s*\n+", "\n", u["text"].strip())
    if u["kind"] == "table":
        return t
    head = re.sub(r"\n\s*\n+", "\n", u["heading"]) if u["heading"] else ""
    hs = t.find(head + "\n") if head else -1
    protect = set(range(hs, hs + len(head) + 1)) if hs >= 0 else set()
    out = []
    for i, ch in enumerate(t):
        if ch != "\n":
            out.append(ch)
        elif i in protect:
            out.append("\n")
            log["제목 줄바꿈 유지"] += 1
        elif (t[max(0, i - 12):i].strip(), t[i + 1:i + 13].strip()) in keep:
            out.append("\n")
            log["원문 줄바꿈 유지"] += 1
        else:
            out.append(" ")
            log["조각 붙이기 → 공백"] += 1
    return "".join(out)


def list_markers(path):
    """Docling 원본 목록 항목 글 → 표지('[12]', '12.')."""
    if not path.exists():
        return {}
    d = json.loads(path.read_text(encoding="utf-8"))
    return {" ".join(t["text"].split()): t["marker"].strip() for t in d.get("texts", []) if t.get("label") == "list_item" and (t.get("marker") or "").strip()}


def ref_table_lines(t):
    """참고문헌 모양 표 → (제목 줄, 항목 줄 목록) 또는 None."""
    lines = t.split("\n")
    head = [ln for ln in lines if not ln.lstrip().startswith("|")]
    rows = [ln for ln in lines if ln.lstrip().startswith("|") and not re.fullmatch(r"\|[\s\-:|]*\|?", ln.strip())]
    if len(rows) < 3:
        return None
    items, good = [], 0
    for r in rows:
        cells = [x.strip() for x in r.strip().strip("|").split("|")]
        cells = [x for x in cells if x]
        if len(cells) == 2 and KEY.match(cells[0]) and len(cells[1]) >= 30 and YEAR.search(cells[1]):
            good += 1
        items.append(" ".join(cells))
    if good < 0.8 * len(rows):
        return None
    return "\n".join(head).strip(), items


def _meta(u, kind):
    return {"kind": kind, "pages": list(u.get("pages") or []), "block_ids": list(u.get("block_ids") or [])}


def _merge(m, u):
    return {**m, "pages": m["pages"] + [p for p in (u.get("pages") or []) if p not in m["pages"]], "block_ids": m["block_ids"] + list(u.get("block_ids") or [])}


def build(units, blocks, markers):
    """2차 단위 목록 → (text, paras, 규칙별 적용 수, 검사 결과). blocks: meta_text_v2 블록 id → 글, markers: list_markers 결과."""
    log, paras, raw = Counter(), [], []
    for u in units:
        t = unit_text(u, original_breaks(blocks, u["block_ids"]), log)
        raw.append(t)
        if u.get("ref"):
            head = u["heading"] if u["heading"] and t.startswith(u["heading"] + "\n") else ""
            item = t[len(head) + 1:] if head else t
            m = markers.get(" ".join(item.split()))
            if m and not item.startswith(m):
                item = m + " " + item
                log["참고문헌 항목 번호 붙임"] += 1
            body = (head + "\n" + item) if head else item
            if paras and paras[-1][1]["kind"] == "references":
                paras[-1] = (paras[-1][0] + "\n" + body, _merge(paras[-1][1], u))
            else:
                paras.append((body, _meta(u, "references")))
            continue
        pos, pieces = 0, []                       # 디스플레이 수식을 자기 문단으로
        for a, b in u.get("formulas", []):
            f = u["text"][a:b].strip()
            j = t.find(f, pos) if f else -1
            if j < 0:
                continue
            pieces += [(t[pos:j], u["kind"]), (f, "formula")]
            pos = j + len(f)
        pieces.append((t[pos:], u["kind"]))
        pieces = [(x.strip(), kd) for x, kd in pieces if x.strip()]
        if len(pieces) > 1:
            log["수식으로 나눈 단위"] += 1
        for x, kd in pieces:
            paras.append((x, _meta(u, kd)))
    out = []                                      # 참고문헌 모양 표 펴기
    for i, (t, m) in enumerate(paras):
        if m["kind"] == "table":
            near = (out and out[-1][1]["kind"] == "references") or (i + 1 < len(paras) and paras[i + 1][1]["kind"] == "references") \
                or bool(REFS_HEAD.match(t.split("\n", 1)[0]))
            r = ref_table_lines(t) if near else None
            if r:
                head, items = r
                body = (head + "\n" if head else "") + "\n".join(items)
                log["참고문헌 표 펴기"] += 1
                if out and out[-1][1]["kind"] == "references":
                    out[-1] = (out[-1][0] + "\n" + body, _merge(out[-1][1], m))
                else:
                    out.append((body, {**m, "kind": "references"}))
                continue
        if m["kind"] == "references" and out and out[-1][1]["kind"] == "references":
            out[-1] = (out[-1][0] + "\n" + t, _merge(out[-1][1], m))
            continue
        out.append((t, m))
    text = "\n\n".join(t for t, _ in out)
    ok = {"번호 붙이기 전 글자 = 단위 글": NS("".join(raw)) == NS("".join(u["text"] for u in units)),
          "문단 수 = paras": len(text.split("\n\n")) == len(out)}
    return text, [m for _, m in out], log, ok
