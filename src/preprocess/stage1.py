"""1차 전처리: 원본 PDF·docx → Docling 변환 → 블록 + 위치·줄 정보(meta_text_v2).

  source env.sh
  PYTHONPATH=src .venv311/bin/python -m preprocess.stage1 --dataset all [--worker 0 --workers 3]     # 변환(없는 문서만) + 블록
  PYTHONPATH=src .venv311/bin/python -m preprocess.stage1 --dataset all --build-only                 # 저장된 Docling 결과로 블록만 다시

출력(--out 기본 data/)
  docling_json/<데이터셋>/<doc_id>.docling.json   Docling 결과(구석 항목을 지운 뒤)
  docling_json/<데이터셋>/<doc_id>.convert.json   변환 기록(쪽 수, 지운 구석 항목 수, PDF 정제, 걸린 시간)
  meta_text_v2/<데이터셋>/<doc_id>.json           블록 + 위치·줄 정보(2차 입력)
변환 오류는 docling_json/errors.jsonl에 남기고 다음 문서로 넘어간다. 이미 변환한 문서는 건너뛴다(중단 후 이어서).

PDF 변환: OCR 끔, 표 FAST, cuDNN 끔(서버 충돌 우회), PyMuPDF 정제 후 변환, 쪽 위아래 5% 구석 항목 제거, 제목 번호로 수준 복원,
  수식 인식(별행 수식을 LaTeX로). docx: Docling 기본 변환기, Word 제목 스타일로 수준(쪽·위치 정보 없음).
공통 블록 규칙(docling_blocks): 그림·머리말·바닥글 제외, 각주는 문서 끝, 표는 캡션 없는 마크다운, 표 안 문단 중복 제거, inline 그룹 잇기.
PDF 블록에 더하는 것
  읽기 순서  목차(DOCUMENT_INDEX)로 분류된 본문 살리기 / 쪽 끝으로 밀려난 항목을 원래 높이로 / 위아래가 뒤바뀐 이웃 두 항목 맞바꿈 /
             2단 쪽: 두 단을 걸치는 항목을 띠 구분자로, 띠 안에서 왼쪽 단 → 오른쪽 단 / 같은 단 안에서는 위에서 아래로 /
             Docling이 쪽을 넘는 문단을 잘못된 앞 문단에 붙인 경우 쪽 경계에서 나눔 / 그림 영역 안에 잡힌 본문 살리기
  위치·줄    boxes(쪽마다 [쪽, 왼, 위, 오른, 아래]), page_size, geom(줄 수·줄 높이·마지막 줄이 찼나·들여쓰기·글자 크기),
             para_breaks(블록 안 숨은 문단 경계), 수식 영역의 원본 글자 층(text_layer), 그림 영역(pictures),
             한글 가짜 띄어쓰기 위치(fake_spaces)
"""
import argparse
import json
import os
import re
import sys
import tempfile
import time
import traceback
import unicodedata
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from common.config import DATA_DIR, DATASETS, ROOT
from preprocess import docling_blocks as dlb
from preprocess.sources import documents

PAGE_NO = re.compile(r"^\s*[\divxlc]{1,4}\s*$", re.I)
LEADER = re.compile(r"(\.\s?){4,}|(·\s?){4,}|…{2,}")
ENDS_NO = re.compile(r"\s[A-Z]?\d{1,3}\s*$")
PROSE_END = re.compile(r"[.?!:;\"”’)\]]\s*$")
PARA_END = re.compile(r"[.?!:;。다요함음임됨□■∎]\s*[\"'”’)\]]?$")
_NORM = re.compile(r"[^0-9a-z가-힣]")


# ---------------------------------------------------------------- 목차로 분류된 본문
def index_as_body(t):
    """목차 항목이 본문처럼 보이면 행 순서로 칸 글을 이은 글, 진짜 목차면 None."""
    cells = [c.text.strip() for c in t.data.table_cells if c.text.strip()]
    if not cells:
        return None
    if sum(bool(PAGE_NO.match(c)) for c in cells) / len(cells) > 0.15 or sum(len(c.split()) for c in cells) / len(cells) < 3:
        return None
    if LEADER.search(" ".join(cells)) or sum(bool(ENDS_NO.search(c)) for c in cells) / len(cells) > 0.3:
        return None                              # 점선 이음표, 쪽 번호로 끝나는 칸이 많으면 진짜 목차
    lines = []
    for row in t.data.grid:
        seen, parts = set(), []
        for c in row:
            s = c.text.strip()
            if s and s not in seen:
                seen.add(s)
                parts.append(s)
        if parts:
            lines.append(" ".join(parts))
    text = ""
    for ln in lines:
        text = (text[:-1] + ln) if text.endswith("-") and ln[:1].islower() else (text + " " + ln if text else ln)
    return text


# ---------------------------------------------------------------- 읽기 순서 보정
def _box(it, doc):
    p = it.prov[0]
    b = p.bbox.to_top_left_origin(doc.pages[p.page_no].size.height)
    return b.t, b.l, b.r


def fix_trailing(items, doc, log):
    """Docling 읽기 순서가 짧은 글('Therefore')·수식을 쪽 맨 끝에 붙인 경우만 바로잡는다.
    쪽의 끝 묶음 = 모든 항목이 앞부분의 같은 단 항목보다 위에 있는 가장 긴 꼬리(쪽 항목의 절반 이하).
    각각을 같은 단에서 자기보다 아래에 있는 첫 항목 바로 앞으로 옮긴다. 2단 쪽이면 가로로 겹쳐야 같은 단."""
    pages = {}
    for k, (it, _) in enumerate(items):
        if getattr(it, "prov", None):
            pages.setdefault(it.prov[0].page_no, []).append(k)
    order = list(range(len(items)))
    for pg, ks in pages.items():
        if len(ks) < 3 or ks != list(range(ks[0], ks[-1] + 1)):
            continue
        tops = [_box(items[k][0], doc) for k in ks]
        w = doc.pages[pg].size.width
        two_col = (sum(r <= w * 0.52 and r - l < w * 0.48 for _, l, r in tops) >= 2
                   and sum(l >= w * 0.48 and r - l < w * 0.48 for _, l, r in tops) >= 2)

        def same_col(a, b):
            return not two_col or min(a[2], b[2]) - max(a[1], b[1]) > 0

        def above_earlier(i, j):
            return any(tops[m][0] > tops[i][0] + 2 and same_col(tops[i], tops[m]) for m in range(j))

        j = len(ks)
        while j > 1 and all(above_earlier(i, j - 1) for i in range(j - 1, len(ks))):
            j -= 1
        tail = ks[j:]
        if not tail or len(tail) > len(ks) // 2:
            continue
        main = [k for k in ks if k not in tail]
        for k in tail:
            me = _box(items[k][0], doc)
            dest = next((m for m in main if _box(items[m][0], doc)[0] > me[0] and same_col(me, _box(items[m][0], doc))), None)
            if dest is None:
                continue
            main.insert(main.index(dest), k)
            log["trailing_moved"] = log.get("trailing_moved", 0) + 1
        if len(main) == len(ks):
            for pos, k in zip(range(ks[0], ks[-1] + 1), main):
                order[pos] = k
    return [items[k] for k in order]


def fix_two_columns(items, doc, log):
    """2단 쪽의 읽기 순서를 다시 맞춘다(Docling이 오른쪽 단을 먼저 읽거나, 위쪽 가운데의 제목·저자를 단 중간에 끼운 경우).
    2단 쪽 = 폭이 쪽의 48% 미만인 본문 글 블록(80자 이상, 글·목록 항목만. 수식 옆 짧은 'Given …'으로 1단 쪽을 2단으로 오판한 실측)이
    왼쪽·오른쪽에 나란히(세로로 겹치게) 놓임. 단 사이 홈 = 그런 쌍의 (왼쪽 블록 오른쪽 끝 + 오른쪽 블록 왼쪽 끝)/2 중앙값.
    그런 쪽에서: 홈을 걸치는 항목(제목, 저자, 두 단에 걸친 그림·표·초록)은 '띠 구분자'로 위→아래 순서,
    구분자 사이의 띠 안에서는 왼쪽 단(위→아래) 다음 오른쪽 단(위→아래). 순서가 실제로 바뀐 쪽만 센다."""
    pages = {}
    for k, (it, _) in enumerate(items):
        if getattr(it, "prov", None):
            pages.setdefault(it.prov[0].page_no, []).append(k)
    items = list(items)
    for pg, ks in pages.items():
        if len(ks) < 3 or ks != list(range(ks[0], ks[-1] + 1)):
            continue
        h, w = doc.pages[pg].size.height, doc.pages[pg].size.width
        box = {k: items[k][0].prov[0].bbox.to_top_left_origin(h) for k in ks}
        txt = {k: (getattr(items[k][0], "text", "") or "") for k in ks}
        narrow = [k for k in ks if box[k].r - box[k].l < 0.48 * w and len(txt[k]) >= 80
                  and str(items[k][0].label) in ("text", "list_item")]
        left = [k for k in narrow if (box[k].l + box[k].r) / 2 < w / 2]
        right = [k for k in narrow if (box[k].l + box[k].r) / 2 >= w / 2]
        gaps = sorted((box[a].r + box[b].l) / 2 for a in left for b in right
                      if min(box[a].b, box[b].b) - max(box[a].t, box[b].t) > 0 and box[a].r < box[b].l)
        if not gaps:
            continue
        gutter, tol = gaps[len(gaps) // 2], 0.02 * w     # 쪽 가운데가 아니라 실제 단 사이(오른쪽 단이 가운데보다 왼쪽에서 시작하는 실측)
        side = {k: ("L" if box[k].r <= gutter + tol else "R" if box[k].l >= gutter - tol else "W") for k in ks}
        walls = sorted((k for k in ks if side[k] == "W"), key=lambda k: box[k].t)
        order, prev_y = [], -1.0
        for wall in walls + [None]:
            y = box[wall].t if wall is not None else 1e9
            band = [k for k in ks if side[k] != "W" and prev_y <= (box[k].t + box[k].b) / 2 < y and k not in order]
            order += sorted((k for k in band if side[k] == "L"), key=lambda k: box[k].t)
            order += sorted((k for k in band if side[k] == "R"), key=lambda k: box[k].t)
            if wall is not None:
                order.append(wall)
                prev_y = box[wall].t             # 띠 경계 = 구분자 위쪽 끝(구분자 옆에 나란히 붙은 저자 칸이 쪽 끝으로 밀리던 결함)
        order += [k for k in ks if k not in order]
        if order != ks:
            snapshot = [items[k] for k in order]
            for pos, it in zip(ks, snapshot):
                items[pos] = it
            log["two_column_pages"] = log.get("two_column_pages", 0) + 1
    return items


def _union_box(it, h, pg=None):
    """항목의 위치 정보(같은 쪽만, pg를 주면) 전부를 합친 영역. Docling이 떨어진 두 줄을 한 항목으로 합친 경우 첫 위치만 보면
    엉뚱한 자리로 옮긴다(실측 2411.03299v2 10쪽 맨 위 문장이 같은 항목의 'and'(위 403) 때문에 아래로)."""
    bs = [p.bbox.to_top_left_origin(h) for p in it.prov if pg is None or p.page_no == pg] or [it.prov[0].bbox.to_top_left_origin(h)]
    return SimpleNamespace(l=min(x.l for x in bs), t=min(x.t for x in bs), r=max(x.r for x in bs), b=max(x.b for x in bs))


def fix_local_swaps(items, doc, log):
    """Docling 읽기 순서에서 바로 이웃한 두 항목이 위아래가 뒤바뀐 경우만 바로잡는다(실측: 'Since…'가 그 위의 'Furthermore…'보다 먼저).
    같은 쪽·같은 단에서 뒤 항목이 앞 항목보다 '완전히 위'(뒤 항목 아래 끝 ≤ 앞 항목 위 끝)일 때 맞바꾼다(쪽마다 한 번만 훑음).
    2단 쪽에서 단을 건너는 이동(가로로 안 겹침)은 건드리지 않는다."""
    pages = {}
    for k, (it, _) in enumerate(items):
        if getattr(it, "prov", None):
            pages.setdefault(it.prov[0].page_no, []).append(k)
    items = list(items)
    for pg, ks in pages.items():
        if len(ks) < 2 or ks != list(range(ks[0], ks[-1] + 1)):
            continue
        h, w = doc.pages[pg].size.height, doc.pages[pg].size.width
        bs = [_union_box(items[k][0], h, pg) for k in ks]
        two_col = (sum(b.r <= w * 0.52 and b.r - b.l < w * 0.48 for b in bs) >= 2
                   and sum(b.l >= w * 0.48 and b.r - b.l < w * 0.48 for b in bs) >= 2)
        i = ks[0]
        while i < ks[-1]:                          # 한 번만 훑는다: 맞바꾼 쌍은 다시 보지 않는다(전체 정렬이 되지 않게)
            a, b = _union_box(items[i][0], h, pg), _union_box(items[i + 1][0], h, pg)
            same = not two_col or min(a.r, b.r) - max(a.l, b.l) > 0
            if same and b.b <= a.t + 2:
                items[i], items[i + 1] = items[i + 1], items[i]
                log["local_swaps"] = log.get("local_swaps", 0) + 1
                i += 2
            else:
                i += 1
    return items


def fix_column_order(items, doc, log):
    """같은 단 안에서는 위에서 아래로: Docling 읽기 순서가 같은 단 안에서 위아래가 뒤바뀐 곳을 바로잡는다
    (실측 2406.11917v2 1단 쪽: '† Meanwhile…'(위 117)이 '† For the first…'(위 68)보다 먼저).
    쪽마다 지금 순서에서 같은 단 항목이 연달아 있는 구간(가로로 좁은 쪽 폭의 절반 이상 겹침; 2단 쪽에서는 쪽 폭의 60% 넘는 항목과
    단 폭 항목은 다른 단)을 찾고, 구간 안에서 어떤 항목이 앞 항목보다 위(위 끝이 더 위, 세로 겹침 ≤ 작은 쪽 높이 35%)면 그 앞으로
    옮긴다(삽입 정렬). 세로로 겹치는(나란한) 항목의 순서와, 여러 쪽에 걸친 항목의 자리는 그대로 둔다."""
    items = list(items)
    pages = {}
    for k, (it, _) in enumerate(items):
        prov = getattr(it, "prov", None)
        if prov and len({p.page_no for p in prov}) == 1:
            pages.setdefault(prov[0].page_no, []).append(k)
    for pg, ks in pages.items():
        if len(ks) < 2:
            continue
        h, w = doc.pages[pg].size.height, doc.pages[pg].size.width
        bx = {k: _union_box(items[k][0], h) for k in ks}
        narrow = [k for k in ks if bx[k].r - bx[k].l < w * 0.48]
        two_col = sum(bx[k].r <= w * 0.52 for k in narrow) >= 2 and sum(bx[k].l >= w * 0.48 for k in narrow) >= 2

        def same_col(a, b):
            ov = min(a.r, b.r) - max(a.l, b.l)
            if ov < 0.5 * min(a.r - a.l, b.r - b.l):
                return False
            return not two_col or ((a.r - a.l > w * 0.6) == (b.r - b.l > w * 0.6))

        def above(x, y):                           # Docling 영역 상자는 줄 간격만큼 6~7pt 겹친다(실측)
            X, Y = bx[x], bx[y]
            return X.t < Y.t - 2 and X.b - Y.t <= max(2, 0.35 * min(X.b - X.t, Y.b - Y.t))
        runs, cur = [], [ks[0]]
        for k in ks[1:]:
            if same_col(bx[cur[-1]], bx[k]):
                cur.append(k)
            else:
                runs.append(cur)
                cur = [k]
        runs.append(cur)
        for run in runs:
            order = []
            for k in run:                          # 삽입 정렬: 앞 항목들 중 k보다 아래인 것들 앞으로
                j = len(order)
                while j > 0 and above(k, order[j - 1]):
                    j -= 1
                order.insert(j, k)
            if order != run:
                log["column_order_moves"] = log.get("column_order_moves", 0) + sum(a != b for a, b in zip(order, run))
                vals = [items[k] for k in order]
                for slot, v in zip(run, vals):
                    items[slot] = v
    return items


def with_picture_prose(doc, extra):
    """Docling 항목 순서 + 그림 안으로 들어간 본문. Docling은 그림 영역 안 글을 그림의 자식으로 두고 기본 순회에서 빼는데,
    본문·초록이 통째로 그림 영역에 잡힌 문서가 있다(실측 2412.11399v2 초록, 2412.03606v1 본문 등 33편).
    기본 순회에서 빠진 항목 중 글·목록·제목 항목이 25낱말 이상이고 문장부호로 끝나면 살린다(그림 글자는 이 길이에 못 미친다).
    본문 크기 글자인지는 2차(G4)가 가린다."""
    from docling_core.types.doc import DocItemLabel

    default = {it.self_ref for it, _ in doc.iterate_items()}   # 기본 순회(그림 캡션은 여기 나온다)에 있는 항목은 그대로 둔다
    out = []
    for it, lvl in doc.iterate_items(traverse_pictures=True):
        if it.self_ref not in default:
            t = (getattr(it, "text", None) or "").strip()
            if not (it.label in (DocItemLabel.TEXT, DocItemLabel.PARAGRAPH, DocItemLabel.LIST_ITEM, DocItemLabel.SECTION_HEADER)
                    and len(t.split()) >= 25 and PROSE_END.search(t)):
                continue
            extra.setdefault("picture_prose", set()).add(it.self_ref)
        out.append((it, lvl))
    return out


# ---------------------------------------------------------------- PDF 블록
def split_wrong_page_merges(blocks, log):
    """Docling이 쪽을 넘는 문단을 한 항목으로 합칠 때, 앞 쪽의 '마지막 문단'이 아닌 그 앞 문단에 다음 쪽 첫 글을 붙이는 경우가 있다
    (실측 9%: 'Returning back …' + 다음 쪽 'copulas in both …'). 합친 블록 뒤에 같은 쪽 본문 블록이 더 있으면 잘못 합친 것으로 보고,
    Docling charspan으로 쪽 경계에서 나눠 뒤 조각을 다음 쪽 맨 앞(앞 쪽의 마지막 블록 바로 뒤)으로 옮긴다.
    다시 잇는 것은 2차의 문단 잇기 규칙이 맡는다. 바르게 합친 것(뒤에 같은 쪽 본문 없음)은 그대로 둔다."""
    BODY = ("paragraph", "list_item", "formula")
    out = list(blocks)
    i = 0
    while i < len(out):
        b = out[i]
        pieces = b.pop("_pieces", None)
        if not pieces or b["type"] not in BODY or not b.get("boxes"):
            i += 1
            continue
        p0 = pieces[0][0]
        if not any(x.get("page") == p0 and x["type"] in BODY for x in out[i + 1:]):
            i += 1
            continue
        first = " ".join(pieces[0][1].split())
        rest = " ".join(" ".join(t for p, t in pieces[1:]).split())
        if not first or not rest:
            i += 1
            continue
        a = {**b, "text": first, "boxes": [bx for bx in b["boxes"] if bx[0] == p0] or b["boxes"][:1]}
        nb = {**b, "id": b["id"] + "p", "text": rest, "page": pieces[1][0], "boxes": [bx for bx in b["boxes"] if bx[0] != p0] or b["boxes"][-1:]}
        out[i] = a
        j = i + 1                                   # 앞 쪽의 마지막 블록 바로 뒤
        while j < len(out) and (out[j].get("page") is None or out[j]["page"] <= p0):
            j += 1
        out.insert(j, nb)
        log["page_merge_split"] = log.get("page_merge_split", 0) + 1
        i += 1
    for b in out:
        b.pop("_pieces", None)
    return out


class _PageNo(int):
    """쪽 번호(int) + 블록 영역. build_blocks가 page_of 결과를 블록의 page에 그대로 넣는 점을 이용한다."""
    box = None


class _Text:
    """살린 목차·그림 안 제목 항목을 글 항목으로 다루게 하는 껍데기."""

    def __init__(self, item, text):
        from docling_core.types.doc import DocItemLabel
        self._item, self.text, self.label = item, text, DocItemLabel.TEXT
        self.prov, self.parent, self.self_ref = item.prov, None, item.self_ref


def build_blocks_pdf(doc):
    from docling_core.types.doc import DocItemLabel

    rescued = {t.self_ref: body for t in doc.tables if t.label == DocItemLabel.DOCUMENT_INDEX for body in [index_as_body(t)] if body}
    extra = {}
    items0 = with_picture_prose(doc, extra)
    for ref in extra.get("picture_prose", ()):   # 그림 안에서 살린 긴 '제목' 라벨 글은 본문 글로
        it = doc.texts[int(ref.split("/")[-1])]
        if it.label == DocItemLabel.SECTION_HEADER:
            rescued[ref] = it.text
    items = fix_column_order(fix_two_columns(fix_local_swaps(fix_trailing(items0, doc, extra), doc, extra), doc, extra), doc, extra)
    items = [(_Text(it, rescued[it.self_ref]), lvl) if getattr(it, "self_ref", None) in rescued else (it, lvl) for it, lvl in items]

    def page_with_box(item):
        """블록이 차지하는 영역들 = 단·쪽마다 하나. inline 그룹이면 그룹 구성원 전부, 쪽·단을 넘는 블록이면 따로따로(읽는 순서)."""
        pg = dlb.page_of(item)
        if pg is None:
            return None
        src = getattr(item, "_item", item)
        members = [src]
        if dlb.inline_group_ref(src, doc) is not None:
            members = [c.resolve(doc) for c in src.parent.resolve(doc).children]
        boxes = []                                # 읽는 순서대로, 같은 쪽·같은 단(가로로 겹침)이면 한 영역으로 합치고 아니면 새 영역
        for m in members:
            for p in getattr(m, "prov", None) or []:
                b = p.bbox.to_top_left_origin(doc.pages[p.page_no].size.height)
                last = boxes[-1] if boxes else None
                if last and last[0] == p.page_no and min(last[3], b.r) - max(last[1], b.l) > 0:
                    last[1:] = [min(last[1], b.l), min(last[2], b.t), max(last[3], b.r), max(last[4], b.b)]
                else:
                    boxes.append([p.page_no, b.l, b.t, b.r, b.b])
        out = _PageNo(pg)
        out.box = boxes or None
        if len(members) == 1 and len({p.page_no for p in (src.prov or [])}) > 1 and getattr(src, "text", None):
            out.pieces = [(p.page_no, src.text[p.charspan[0]:p.charspan[1]]) for p in src.prov]   # 쪽마다 글 조각(Docling charspan)
        return out

    blocks, log = dlb.build_blocks(doc, dlb.PdfHeadingLevel(), page_of=page_with_box, items=items)
    for b in blocks:
        if getattr(b.get("page"), "pieces", None):
            b["_pieces"] = b["page"].pieces
        boxes = getattr(b.get("page"), "box", None)
        if b.get("page") is not None:
            b["page"] = int(b["page"])
        if boxes:                                 # boxes = [[쪽, 왼쪽, 위, 오른쪽, 아래], ...] 쪽 순서. 첫 영역 = 시작, 마지막 영역 = 끝
            b["boxes"] = [[p] + [round(v, 1) for v in bx] for p, *bx in boxes]
            size = doc.pages[b["page"]].size
            b["page_size"] = [round(size.width, 1), round(size.height, 1)]
    blocks = split_wrong_page_merges(blocks, extra)
    heads = [" ".join(it.text.split())[:60] for ref in extra.get("picture_prose", ()) for it in [doc.texts[int(ref.split("/")[-1])]]]
    for bl in blocks:                             # 그림 안에서 살린 본문 표시(2차 그림 글자 규칙에서 따로 다룸)
        if heads and any(h and h in " ".join(bl["text"].split()) for h in heads):
            bl["from_picture"] = True
    log["picture_prose"] = sum(bool(bl.get("from_picture")) for bl in blocks)
    log["index_as_body"] = len(rescued)
    for k in ("trailing_moved", "page_merge_split", "two_column_pages", "local_swaps", "column_order_moves"):
        log[k] = extra.get(k, 0)
    return blocks, log


# ---------------------------------------------------------------- 줄 정보
def _find_offset(text, row_text, start):
    """row_text(PyMuPDF 행 글)의 첫 글자 12개(영숫자·한글만, 소문자)가 text의 start 이후 어디서 시작하는지 → text 안 글자 위치
    (낱말 시작으로 맞춤). 못 찾거나 가까이에 같은 글이 또 있으면(애매) None."""
    key = _NORM.sub("", unicodedata.normalize("NFKC", row_text).lower())[:12]
    if len(key) < 4:
        return None
    idx, norm = [], []
    for i, ch in enumerate(text):
        for c in _NORM.sub("", unicodedata.normalize("NFKC", ch).lower()):
            idx.append(i)
            norm.append(c)
    norm = "".join(norm)
    k0 = next((k for k, i in enumerate(idx) if i >= start), len(idx))
    k = norm.find(key, k0)
    if k < 0 or norm.find(key, k + 1) >= 0 and norm.find(key, k + 1) < k + 200:
        return None
    pos = idx[k]
    while pos > 0 and not text[pos - 1].isspace():
        pos -= 1
    return pos


def add_line_geometry(blocks, pdf_path):
    """블록 영역 안의 원본 PDF 줄(PyMuPDF line을 같은 높이끼리 묶은 행)로 줄 정보와 숨은 문단 경계를 계산한다.
    쪽·단을 넘는 블록은 영역이 여러 개: 첫 줄은 첫 영역에서, 마지막 줄은 마지막 영역에서 본다.
      geom.rows         행 수(모든 영역 합)       geom.line_h  행 높이 중앙값
      geom.last_full    |마지막 영역 오른쪽 - 마지막 행 오른쪽| < 줄 높이 (줄이 끝까지 찬 채 끝남)
      geom.first_indent 첫 영역의 첫 행이 같은 영역 나머지 행보다 줄 높이 절반 넘게 들어감(새 문단 시작)
      geom.font         글자 크기 중앙값(줄마다 가장 큰 글자)
      para_breaks       본문 블록(paragraph, list_item) 안에서 새 문단이 시작하는 글자 위치 목록.
                        같은 영역 안 행 i가 소문자로 시작하지 않고, 다음 중 하나: 빈 줄(줄 사이 빈 공간 > 문서의 보통 빈 공간 + 0.6 lh) /
                        들여쓰기(0.5~4 lh 들어가고 앞 행이 1.5 lh 넘게 짧게 끝남) / 짧은 끝 줄(앞 행이 폭 75% 미만에서 마침표류로 끝남).
      para_break_pages  경계마다 그 줄이 있던 쪽
      text_layer        수식 블록 영역의 원본 글자 층(수식 인식이 틀렸을 때 2차가 대신 쓸 재료)"""
    import pymupdf

    lines_by_page = {}
    sizes_seen = []

    def rows_in(d, pg, l, t, r, bt):
        if pg not in lines_by_page:
            lines_by_page[pg] = [(ln["bbox"], "".join(s["text"] for s in ln["spans"]),
                                  max((s["size"] for s in ln["spans"] if s["text"].strip()), default=0.0))
                                 for blk in d[pg - 1].get_text("dict")["blocks"]
                                 for ln in blk.get("lines", []) if any(s["text"].strip() for s in ln["spans"])]
        inside = [x for x in lines_by_page[pg] if l - 2 <= (x[0][0] + x[0][2]) / 2 <= r + 2 and t - 2 <= (x[0][1] + x[0][3]) / 2 <= bt + 2]
        sizes_seen.extend(x[2] for x in inside)
        rows = []
        for bb, tx, _ in sorted(inside, key=lambda x: (x[0][1], x[0][0])):
            if rows and min(rows[-1][3], bb[3]) - max(rows[-1][1], bb[1]) > 0.5 * min(rows[-1][3] - rows[-1][1], bb[3] - bb[1]):
                rw = rows[-1]
                rows[-1] = [min(rw[0], bb[0]), min(rw[1], bb[1]), max(rw[2], bb[2]), max(rw[3], bb[3]),
                            (rw[4] + " " + tx) if bb[0] > rw[0] else (tx + " " + rw[4])]
            else:
                rows.append([bb[0], bb[1], bb[2], bb[3], tx])
        return rows

    with pymupdf.open(str(pdf_path)) as d:
        parts_of, sizes_of = {}, {}
        for b in blocks:
            if "boxes" in b:
                sizes_seen = []
                parts_of[id(b)] = [rows_in(d, *bx) for bx in b["boxes"]]
                sizes_of[id(b)] = sizes_seen
        body = [rs for b in blocks if b["type"] in ("paragraph", "list_item") and id(b) in parts_of for rs in parts_of[id(b)]]
        leads = [rs[i + 1][1] - rs[i][3] for rs in body for i in range(len(rs) - 1)]
        lead = sorted(leads)[len(leads) // 2] if leads else 0.0
        for b in blocks:
            parts = parts_of.get(id(b))
            if not parts:
                continue
            sz = sizes_of.get(id(b), [])
            first, last = next((p for p in parts if p), None), next((p for p in reversed(parts) if p), None)
            if not first:
                continue
            allrows = [rw for p in parts for rw in p]
            h = sorted(rw[3] - rw[1] for rw in allrows)[len(allrows) // 2]
            r_last = b["boxes"][[i for i, p in enumerate(parts) if p][-1]][3]
            left = min(rw[0] for rw in first[1:]) if len(first) > 1 else first[0][0]
            b["geom"] = {"rows": len(allrows), "line_h": round(h, 1), "last_full": bool(r_last - last[-1][2] < h),
                         "first_indent": bool(len(first) > 1 and first[0][0] - left > 0.5 * h),
                         "font": round(sorted(sz)[len(sz) // 2], 1) if sz else None}
            if b["type"] == "formula":
                b["text_layer"] = " ".join(" ".join(d[p - 1].get_text("text", clip=pymupdf.Rect(l, t, r, bt)).split())
                                           for p, l, t, r, bt in b["boxes"]).strip()
            if b["type"] not in ("paragraph", "list_item"):
                continue
            breaks, cursor = [], 1
            for (bpage, *_), rs in zip(b["boxes"], parts):
                if len(rs) < 2:
                    continue
                bl, br = min(rw[0] for rw in rs), max(rw[2] for rw in rs)
                for i in range(1, len(rs)):
                    p, c = rs[i - 1], rs[i]
                    gap = c[1] - p[3] > lead + 0.6 * h
                    if re.match(r"^\s*[a-z]", c[4]) and not (gap and PARA_END.search(p[4].strip())):
                        continue                  # 소문자 시작 줄은 경계 아님. 단 빈 줄 + 앞 줄이 마침표로 끝나면 경계
                    indent = 0.5 * h < c[0] - bl < 4 * h and p[2] < br - 1.5 * h
                    short = p[2] < bl + 0.75 * (br - bl) and PARA_END.search(p[4].strip()) and c[0] - bl < 0.5 * h
                    if gap or indent or short:
                        off = _find_offset(b["text"], c[4], cursor)
                        if off and off > 0:
                            breaks.append((off, bpage))
                            cursor = off + 1
            if breaks:
                uniq = sorted(set(breaks))
                b["para_breaks"] = [o for o, _ in uniq]
                b["para_break_pages"] = [p for _, p in uniq]


def picture_boxes(doc):
    """Docling이 찾은 그림 영역 [[쪽, 왼, 위, 오른, 아래], ...](위쪽 원점). 2차 G4(그림 영역 안 글 지우기)가 쓴다."""
    out = []
    for p in doc.pictures:
        for pr in p.prov or []:
            b = pr.bbox.to_top_left_origin(doc.pages[pr.page_no].size.height)
            out.append([pr.page_no] + [round(v, 1) for v in (b.l, b.t, b.r, b.b)])
    return out


def add_fake_spaces(blocks, pdf_path):
    """한글이 있는 블록에서, Docling 글의 공백 중 원본 PDF 글자 좌표상 틈이 없는 것(가짜 띄어쓰기)의 글자 위치를 fake_spaces에 넣는다.
    Docling은 서식(굵게 등)이 바뀌는 곳에서 조각을 공백으로 잇는다('감소 로'). 블록 영역 안 PDF 글자를 읽는 순서로 모아,
    공백을 뺀 글이 Docling 글(공백 뺀 것)과 정확히 같을 때만 판정한다. 같은 줄이고 두 글자 사이 틈이 글자 크기의 15% 이하이며
    PDF에 공백 문자도 없으면 가짜. 줄이 바뀌는 곳의 공백은 진짜로 본다."""
    import pymupdf

    with pymupdf.open(str(pdf_path)) as d:
        for b in blocks:
            if not b.get("boxes") or not re.search("[가-힣]", b["text"]) or b["type"] not in ("paragraph", "list_item", "heading"):
                continue
            chars = []
            for pg, l, t, r, bt in b["boxes"]:
                for blk in d[pg - 1].get_text("rawdict", clip=pymupdf.Rect(l - 1, t - 1, r + 1, bt + 1))["blocks"]:
                    for ln in blk.get("lines", []):
                        for sp in ln["spans"]:
                            for ch in sp["chars"]:
                                chars.append((ch["c"], ch["bbox"], sp["size"]))
            solid = [x for x in chars if not x[0].isspace()]
            explicit, k = set(), 0
            for ch in chars:
                if ch[0].isspace():
                    explicit.add(k)
                else:
                    k += 1
            text = b["text"]
            if "".join(c for c in text if not c.isspace()) != "".join(x[0] for x in solid):
                continue
            fake, k = [], 0
            for i, ch in enumerate(text):
                if not ch.isspace():
                    k += 1
                    continue
                if not (0 < k < len(solid)) or k in explicit:
                    continue
                a, nb = solid[k - 1], solid[k]
                same_line = abs(a[1][1] - nb[1][1]) < 0.5 * a[2]
                if same_line and nb[1][0] - a[1][2] <= 0.15 * a[2]:
                    fake.append(i)
            if fake:
                b["fake_spaces"] = fake


# ---------------------------------------------------------------- 문서 하나
def paths(out, ds, doc_id):
    return (out / "docling_json" / ds / f"{doc_id}.docling.json", out / "docling_json" / ds / f"{doc_id}.convert.json",
            out / "meta_text_v2" / ds / f"{doc_id}.json")


def convert(doc, out, conv):
    """Docling 변환(느림) → docling_json에 Docling 결과와 변환 기록. PDF는 정제본으로 변환하고 구석 항목을 지운 뒤 저장한다."""
    dl_f, cv_f, _ = paths(out, doc["dataset"], doc["doc_id"])
    src = ROOT / doc["source_file"]
    t = time.time()
    rec = {"parser": f"docling-{version('docling')}"}
    if doc["fmt"] == "pdf":
        with tempfile.TemporaryDirectory() as tmp:
            path, clean_info = dlb.prepare_pdf(src, tmp)
            dl = conv.convert(str(path)).document
        rec.update(corner_dropped=dlb.drop_corner_items(dl), pdf_clean=clean_info, n_pages=len(dl.pages))
    else:
        dl = conv.convert(str(src)).document
        rec["n_pages"] = None
    rec["seconds"] = round(time.time() - t, 1)
    dl_f.parent.mkdir(parents=True, exist_ok=True)
    dl.save_as_json(dl_f)
    cv_f.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return dl


def build_meta(doc, out, dl=None):
    """docling_json → meta_text_v2 (Docling을 다시 돌리지 않는다. 규칙을 고친 뒤에는 이것만 다시)."""
    from docling_core.types.doc import DoclingDocument

    dl_f, cv_f, meta_f = paths(out, doc["dataset"], doc["doc_id"])
    dl = dl or DoclingDocument.load_from_json(dl_f)
    rec = json.loads(cv_f.read_text(encoding="utf-8"))
    src = ROOT / doc["source_file"]
    if doc["fmt"] == "pdf":
        blocks, log = build_blocks_pdf(dl)
        add_line_geometry(blocks, src)
        add_fake_spaces(blocks, src)
        log.update(corner_dropped=rec["corner_dropped"], pdf_clean=rec["pdf_clean"], seconds=rec["seconds"])
        opts, pics = dlb.pdf_options(rec["pdf_clean"].get("applied")), picture_boxes(dl)
    else:
        blocks, log = dlb.build_blocks(dl, dlb.docx_heading_level(dl))
        log["seconds"] = rec["seconds"]
        opts, pics = {"converter": "DocumentConverter(default)"}, []
    title = doc["title"] or next((b["text"] for b in blocks if b["type"] == "heading"), None) or doc["doc_id"]
    meta = {"doc_id": doc["doc_id"], "dataset": doc["dataset"], "fmt": doc["fmt"], "lang": doc["lang"], "title": title,
            "source_file": doc["source_file"], "parser": rec["parser"], "parser_options": opts, "n_pages": rec["n_pages"],
            "log": log, "pictures": pics, "blocks": blocks}
    meta_f.parent.mkdir(parents=True, exist_ok=True)
    meta_f.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="all", help="all 또는 " + ", ".join(DATASETS))
    ap.add_argument("--doc", action="append", help="이 doc_id만(여러 번 줄 수 있음)")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--build-only", action="store_true", help="저장된 Docling 결과로 meta_text_v2만 다시 만들기")
    ap.add_argument("--out", default=str(DATA_DIR), help="출력 루트(기본 data/)")
    ns = ap.parse_args()
    out = Path(ns.out)
    names = list(DATASETS) if ns.dataset == "all" else [ns.dataset]
    docs = [d for ds in names for d in documents(ds) if not ns.doc or d["doc_id"] in ns.doc]
    docs = [d for k, d in enumerate(docs) if k % ns.workers == ns.worker]
    convs = {}
    for i, doc in enumerate(docs, 1):
        dl_f, cv_f, _ = paths(out, doc["dataset"], doc["doc_id"])
        try:
            dl = None
            if not (dl_f.exists() and cv_f.exists()):
                if ns.build_only:
                    continue
                if doc["fmt"] not in convs:
                    convs[doc["fmt"]] = dlb.pdf_converter() if doc["fmt"] == "pdf" else dlb.docx_converter()
                dl = convert(doc, out, convs[doc["fmt"]])
            meta = build_meta(doc, out, dl)
            print(f"[w{ns.worker}] {i}/{len(docs)} {doc['dataset']}:{doc['doc_id'][:40]} 블록 {len(meta['blocks'])}", flush=True)
        except Exception as e:
            (out / "docling_json").mkdir(parents=True, exist_ok=True)
            with open(out / "docling_json" / "errors.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"doc": f"{doc['dataset']}:{doc['doc_id']}", "error": repr(e), "trace": traceback.format_exc()[-1500:]},
                                    ensure_ascii=False) + "\n")
            print(f"[w{ns.worker}] 오류 {doc['dataset']}:{doc['doc_id']}: {e!r}", flush=True)
    sys.stdout.flush()
    os._exit(0)                                     # Docling·torch 스레드가 끝나길 기다리지 않는다


if __name__ == "__main__":
    main()
