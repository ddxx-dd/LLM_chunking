"""2차 전처리: meta_text_v2(1차 결과) → processed_para(청커 입력, 문단 단위의 깨끗한 텍스트).

  source env.sh
  PYTHONPATH=src .venv311/bin/python -m preprocess.stage2 --dataset all [--worker 0 --workers 4]

입력: <out>/meta_text_v2/<데이터셋>/<doc_id>.json (--out 기본 data/)
출력: <out>/processed_para/<데이터셋>/<doc_id>.json          문서 필드 + units[{text, kind, heading, pages, block_ids, n_tokens}] + log(규칙별 적용 수)
      <out>/processed_para/<데이터셋>/<doc_id>.dropped.json  지운 블록과 규칙 이름(확인용)

적용 순서
  K1 한국어 가짜 띄어쓰기 지움(1차 fake_spaces) / C3 글꼴 글리프 이름 쓰레기 / 1 문자 정리(제어·PUA·대체 문자, NBSP, 폭 없는 문자,
     소프트 하이픈, 합자·수학 문자 NFKC, 연속 공백) / T7 칸이 하나뿐인 표 → 본문 / G1 깨진 글 블록이 30% 이상이면 문서 제외
  B1 1차가 찾은 숨은 문단 경계(para_breaks)에서 블록 나눔 / G3 블록 앞 그림 낱글자 떼기
  블록 제거  R1 참고문헌 영역(제목 또는 'References' 한 줄부터, 본문 단락 2개 연속·수식·다른 제목에서 끝) / N1 줄 번호 사슬 /
             D3 여러 쪽 반복 짧은 블록 / H2 반복 머리말·바닥글(Marker 기본값 + 2쪽 간격) / T1·T2 목차 /
             F1 그림 캡션(캡션 모양, 축 숫자 떼고 판정) / G4 그림 영역 안 글(그림 안에서 살린 글은 본문보다 작은 글자일 때만) /
             G2 그림 속 글자 조각 / G5 글자가 흩어진 그림 글 / N2 본문 사이 쪽 번호 / N3 arXiv 세로 도장·저작권·라이선스·
             첫 쪽 접수·교신·소속·연구비 문구·이메일/URL 줄 / R2 제목 없는 끝 참고문헌
  D1 단어가 두 번씩 반복된 블록 반으로 / F2 본문으로 분류된 각주를 문서 끝으로
  표         T6 표 캡션을 표 앞에 / T8 표 바로 뒤 주석('Note:', '*** p <')을 표 뒤에 / T9 짝 없는 표 캡션·번호 줄은 뒤쪽 표에, 없으면 제거
  문단 잇기  Q1 수식을 앞 문단 안으로(읽기 쉬운 글자, Q3 고장 난 수식은 원본 글자 층, Q4 식 번호 뒤 덧붙은 글 자르기,
             Q5 \text{} 낱글자 붙이기, Q6 수식 앞뒤에 반복된 이웃 문장 자르기) / Q2 수식 뒤 소문자 문장 이어 붙임 /
             P1·P2 영어 소문자 이어짐 / P5 하이픈 / P3 한국어 쪽 바뀜 / L1 줄 위치
  제목 붙이기(제목을 바로 아래 단위 앞에) / S 글자 없거나 2토큰 이하 조각을 다음 단위 앞에 / S2 양식 항목('…:')을 다음 단위 앞에
"""
import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from common.config import DATA_DIR, DATASETS
from preprocess.sources import documents

LETTER = re.compile(r"[A-Za-z0-9가-힣\u3131-\u318E\u4E00-\u9FFF]")
WORD = re.compile(r"[0-9a-z가-힣]+")
NUM = re.compile(r"^\d{1,4}$")
END = re.compile(r"[.?!:;)\]\"'”’…。다요까함음임됨□■∎]$")
REFS = re.compile(r"^\s*(\d+\.?\s*)?((bibliographical\s+)?references?(\s+list)?|bibliography|literature\s+cited|참고\s*문헌|참고자료)\s*:?\s*$", re.I)
TOC = re.compile(r"(\.{4,}|…{2,}|·{4,})\s*\d+\s*$|\t\s*\d+\s*$")
TOC_HEAD = re.compile(r"^\s*((table of )?contents|목\s*차|차\s*례)\s*$", re.I)
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PUA = re.compile(r"[\ue000-\uf8ff\ufffd]")
ZERO = re.compile(r"[\u200b-\u200f\ufeff\u00ad]")
NBSP = re.compile(r"[\u00a0\u2028\u2029]")
SPACES = re.compile(r"[^\S\n]{2,}|\t")
HANGUL = re.compile(r"[가-힣]")
AUTHORS = re.compile(r"[A-Z][A-Za-z'´`\-]+,\s*(?:[A-Z]\.\s*-?\s*)+")
REF_LIKE = re.compile(r"\b(19|20)\d{2}[a-z]?\b|et al\.|\[\d+\]|^\s*\(?\d+[.)]\s|doi|arxiv|pp\.|vol\.|proc\.|journal|conference|"
                      r"press|ed(s)?\.|in:|isbn|https?://|학회|학술|논문|연구원|출판", re.I)
FIG_CAP = re.compile(r"^\s*((sup(p|plementary)?\.?\s*)?fig(ure)?s?\.?|그림|<그림|사진|도표|chart)\s*[-–]?\s*[\dA-Za-z]", re.I)
CAP_SHAPE = re.compile(r"^\s*((extended\s+data\s+|sup(p|plementary)?\.?\s*)?fig(ure)?s?\.?|그림)\s*[-–]?\s*([A-Z]\.?)?\d+(\.\d+)*[a-z]?\s*[:.|](?!\d)", re.I)
# 'Fig. 4 :', 'Figure 2.', 'Figure A.1.', 'Figure S.1.', 'fig. S2.', 'Extended Data Fig. 1:'(캡션 모양). 'Figure 4.5 illustrates'(본문)는 번호 뒤 구두점이 없어 제외
AXIS_LEAD = re.compile(r"^\s*(?:[-−]?\d+(?:\.\d+)?\s+){2,}(?=(fig(ure)?s?\.?|그림)\s*[A-Z]?\.?\d)", re.I)   # 캡션 앞에 붙은 그래프 축 숫자('10 0 10 1 FIG. 4.')
TAB_CAP = re.compile(r"^\s*(tab(le)?s?\.?|표|<표)\s*[\dA-Za-z]", re.I)
BULLET = re.compile(r"^[\d\W_]|^[ㅇ○●◦□■▪◆◇※ㆍ·①-⑳]")
REPEAT_MIN_PAGES, REPEAT_MAX_WORDS = 5, 5


def nwords(t):
    return len(WORD.findall(unicodedata.normalize("NFKC", t).lower()))


def garbled(t):
    chars = [c for c in t if not c.isspace()]
    if len(chars) < 5:
        return 0.0
    bad = 0
    for c in chars:
        o = ord(c)
        ok = (c.isascii() or 0xAC00 <= o <= 0xD7A3 or 0x3130 <= o <= 0x318F or 0x4E00 <= o <= 0x9FFF or 0x3000 <= o <= 0x30FF
              or 0x00C0 <= o <= 0x024F or 0x0370 <= o <= 0x03FF or 0x1D400 <= o <= 0x1D7FF or 0x2000 <= o <= 0x206F
              or 0x2100 <= o <= 0x22FF or 0x2460 <= o <= 0x24FF or 0x25A0 <= o <= 0x25FF or 0xFF00 <= o <= 0xFFEF
              or unicodedata.category(c).startswith(("P", "N", "M")))
        bad += not ok
    return bad / len(chars)


# ---------------------------------------------------------------- 1 문자 정리
def clean_chars(t, c):
    for name, f in (("W3 제어 문자 제거", lambda x: CONTROL.sub("", x)),
                    ("C1 사용자 정의 영역·대체 문자 제거", lambda x: PUA.sub("", x)),
                    ("W2 보이지 않는 문자 정리", lambda x: ZERO.sub("", NBSP.sub(" ", x))),
                    ("C2 합자·수학 문자 정규화", lambda x: "".join(unicodedata.normalize("NFKC", ch) if (0x1D400 <= ord(ch) <= 0x1D7FF or 0xFB00 <= ord(ch) <= 0xFB06) else ch for ch in x)),
                    ("W1 연속 공백·탭 정리", lambda x: "\n".join(SPACES.sub(" ", line).strip() for line in x.split("\n")).strip())):
        t2 = f(t)
        if t2 != t:
            c[name] += 1
        t = t2
    return t


# ---------------------------------------------------------------- 2+ 숨은 문단 경계에서 나누기
def split_blocks(blocks, c):
    """para_breaks 위치에서 블록을 나눈다. 나뉜 조각: 첫 조각은 원래 시작 정보, 마지막 조각은 원래 끝 정보를 물려받고,
    안쪽 경계는 진짜 문단 경계이므로 앞 조각 last_full=False, 뒤 조각 first_indent=True로 둔다(2차 L1이 다시 잇지 않게)."""
    out = []
    for b in blocks:
        offs = [o for o in b.get("para_breaks", []) if 0 < o < len(b["text"])]
        if not offs:
            out.append(b)
            continue
        pmap = dict(zip(b.get("para_breaks", []), b.get("para_break_pages", [])))
        cuts = [0] + offs + [len(b["text"])]
        starts = [b.get("page")] + [pmap.get(o, b.get("page")) for o in offs]   # 조각마다 시작 쪽(1차가 경계마다 기록)
        pieces = [(b["text"][x:y].strip(), starts[k]) for k, (x, y) in enumerate(zip(cuts, cuts[1:]))]
        pieces = [(t, pg) for t, pg in pieces if t]
        for k, (t, pg) in enumerate(pieces):
            g = dict(b.get("geom", {}))
            if g:
                if k < len(pieces) - 1:
                    g["last_full"] = False
                if k > 0:
                    g["first_indent"] = True
            nxt = pieces[k + 1][1] if k + 1 < len(pieces) else None
            boxes = [bx for bx in b.get("boxes", []) if pg is None or (bx[0] >= pg and (nxt is None or bx[0] <= nxt))] or b.get("boxes")
            span = sorted({bx[0] for bx in boxes}) if boxes else ([pg] if pg is not None else [])
            out.append({**b, "id": f"{b['id']}.{k}", "text": t, "page": pg, "boxes": boxes, "geom": g or b.get("geom"), "para_breaks": [],
                        "page_span": span})
        c["B1 숨은 문단 경계에서 나눔"] += len(pieces) - 1
    return out


# ---------------------------------------------------------------- 3 블록 제거
def line_number_ids(blocks):
    nums = [(i, int(b["text"].strip())) for i, b in enumerate(blocks) if NUM.match(b["text"].strip())]
    used, out = set(), set()
    for k, (i, v) in enumerate(nums):
        if i in used:
            continue
        chain, last_i, last_v = [i], i, v
        for j, w in nums[k + 1:]:
            if j - last_i > 6:
                break
            if w == last_v + 1:
                chain.append(j)
                last_i, last_v = j, w
        if len(chain) >= 5:
            out.update(chain)
            used.update(chain)
    return out


def repeated_short_ids(blocks):
    pages, ids = defaultdict(set), defaultdict(list)
    for i, b in enumerate(blocks):
        s = " ".join(b["text"].split())
        if b["type"] == "table" or not s or nwords(s) > REPEAT_MAX_WORDS or b.get("page") is None:
            continue
        pages[s].add(b["page"])
        ids[s].append(i)
    return {i for s, p in pages.items() if len(p) >= REPEAT_MIN_PAGES for i in ids[s]}


ARXIV_STAMP = re.compile(r"arXiv:\d{4}\.\d{4,5}(v\d+)?\s*\[[^\]]+\]\s*\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4}")   # 세로 도장(날짜까지)
BOILER = re.compile(r"Permission to make digital or hard copies|^\s*(copyright\s*)?©|copyright\s+(held|\d{4})|all rights reserved|"
                    r"licensed under (a |the )?(creative commons|cc[- ]?by|[^.]{0,40}licen[cs]e\b)|this work is licensed|creative commons", re.I)   # 'licensed under the … Act'(법 조문)는 제외
FIRST_PAGE_NOTE = re.compile(r"Manuscript\s+received|Accepted\s+(for\s+publication|in\s+final)|Address\s+(for\s+)?(correspondence|reprint)|"
                             r"Corresponding\s+author|접수\s*일|게재\s*(승인|확정)|교신\s*저자|"
                             r"\b(is|are)\s+(now\s+)?with\s+the\s+([\w-]+\s+){0,4}(School|Department|Dept\.|Faculty|College|Institute|University|Laborator(y|ies)|Labs?|Center|Centre|Division|Group)\b|"
                             r"recommended\s+for\s+publication|accepted\s+for\s+publication\s+in|Digital\s+Object\s+Identifier|Color\s+versions\s+of\s+one\s+or\s+more|"
                             r"^\s*(This|The)\s+(work|research|study|project|paper)\s+(was|is|has\s+been)\s+(partially\s+|partly\s+|in\s+part\s+|also\s+)?(supported|funded|sponsored)\s+by|"
                             r"produced\s+by\s+the\s+IEEE\s+Publication", re.I)   # 학술지 첫 쪽 접수·교신·소속(IEEE 'X is with the School of …')·연구비 문구
                             # 연구비·소속 문구가 1쪽 본문 문장 사이에 끼던 실측(2411.12363v3, 2407.16566v5, 2412.14369v2 등)
NOTE_START = re.compile(r"^\s*(Received|Accepted|Revised|Correspondence(\s+to)?|Corresponding\s+author|Address\s+(for\s+)?correspondence|"
                        r"접수|게재\s*(승인|확정)|교신\s*저자|책임\s*저자)\b\s*:?", re.I)   # 학술지 첫 쪽 접수·교신 묶음의 시작 줄
GLYPH = re.compile(r"(?<![A-Za-z])[a-z]{30,}(?![A-Za-z])")     # 글꼴 글리프 이름이 글자 층에 샌 쓰레기('twooldstyleproportionalzero…')
TAB_NOTE = re.compile(r"^\s*(notes?\s*[:.]|source\s*[:.]|standard\s+errors|\*{1,3}\s*p\s*[<≤]|[*†]+\s*(denotes|indicates|significant))", re.I)
TAB_LABEL = re.compile(r"^\s*(table|tab\.|표)\s*[A-Z]?\.?\d+[a-z]?\s*[.:]?\s*$", re.I)   # 단독 표 번호 줄('TABLE 9')
TAB_CAP_SHAPE = re.compile(r"^\s*(tab(le)?\.?|표|<표)\s*[A-Z]?\d+[a-z]?\s*[.:>]", re.I)   # 'Table 1. …', '표 3.' (캡션 모양)
CONTACT = re.compile(r"^\s*(e-?mail:?\s*)?(\S+@\S+\.\S+|https?://\S+)(\s*[,;]\s*(\S+@\S+\.\S+|https?://\S+))*\s*$", re.I)
TOC_LINE = re.compile(r"\S\s+\d{1,4}\s*$")
REF_STRICT = re.compile(r"et al\.|\bvol\.|\bpp\.|\bdoi\b|arxiv|journal|proc\.|conference|press|\[\d+\]|[A-Z][a-z]+,\s*[A-Z]\.|"
                        r"\b(phys|rev|lett|nat|sci|chem|math|astrophys|mon\. not|j)\.\s", re.I)   # 학술지 약어('Phys. Rev. Lett.', 'J. Chem.'): 물리·화학 참고문헌 실측
YEAR = re.compile(r"\b(19|20)\d{2}[a-z]?\b")
REF_START = re.compile(r"^\s*(\[\d+\]\s*|\d+\.\s+)?([A-Z][A-Za-z'´`\-]+,\s*([A-Z]\.|[A-Z][a-z])|[A-Z][A-Za-z'\-]+\s[A-Z]{1,2}[,.]|"
                       r"[A-Z]\.(\s*-?[A-Z]\.)*\s+[A-Z][A-Za-z'´`\-]+(\s[A-Z][a-z]+)?\s*(,|and\b|&))")   # 이니셜이 앞('S. Popescu, A. J. Short, and …')  # 'Meyer, A.,', '[1] Adak, A.,', 'Chirumbolo S,'로 시작


def _inside(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


def running_ids(blocks, n_pages):
    """H2 반복 머리말·바닥글(Marker ignoretext 기본값): 쪽마다 첫·마지막 본문 블록 중 쪽 위·아래 12% 안에 있는 것을 후보로,
    숫자를 지운 글끼리 퍼지 일치 90 이상이면 같은 것. 3쪽 넘게 나오고 (전체 쪽의 20% 이상 또는 3쪽 연속)이면 지움."""
    from rapidfuzz import fuzz

    by_page = defaultdict(list)
    for i, b in enumerate(blocks):
        if b["type"] in ("paragraph", "heading", "list_item") and b.get("boxes") and b.get("page_size") and not b.get("role"):
            by_page[b["page"]].append(i)
    cands = []
    for pg, ids in by_page.items():
        ph = blocks[ids[0]]["page_size"][1]
        f, l = ids[0], ids[-1]
        if blocks[f]["boxes"][0][2] < 0.12 * ph:
            cands.append((f, pg))
        if l != f and blocks[l]["boxes"][-1][4] > 0.88 * ph:
            cands.append((l, pg))
    norm = lambda t: " ".join(re.sub(r"\d+", "", t.lower()).split())
    clusters = []                                  # [대표 글, {쪽}, [블록 번호]]
    for i, pg in cands:
        t = norm(blocks[i]["text"])
        if len(t) < 3:
            continue
        for cl in clusters:
            if fuzz.ratio(t, cl[0]) >= 90:
                cl[1].add(pg)
                cl[2].append(i)
                break
        else:
            clusters.append([t, {pg}, [i]])
    out = set()
    for t, pages, ids in clusters:
        ps = sorted(pages)
        streak = max((sum(1 for _ in g) for _, g in __import__("itertools").groupby(enumerate(ps), lambda x: x[1] - x[0])), default=0)
        alt = max((sum(1 for _ in g) for _, g in __import__("itertools").groupby(enumerate(ps), lambda x: x[1] - 2 * x[0])), default=0)
        if len(ps) > 3 and (len(ps) / max(1, n_pages) >= 0.2 or streak >= 3):
            out.update(ids)
        elif len(ps) >= 3 and alt >= 3 and len(t.split()) <= 25 and len({" ".join(blocks[i]["text"].split()) for i in ids}) == 1:   # 숫자까지 같은 글만
            # (숫자를 빼고 묶으면 22·24·26쪽의 'Proof of Lemma 2/4/7' 제목을 머리말로 지운 실측)   # 짝수·홀수 쪽에 번갈아 나오는 머리말(2쪽 간격 3번 이상, 25낱말 이하: 실측 2408.02322v2 짝수 쪽 논문 제목)
            out.update(ids)
    return out


def toc_run_ids(blocks):
    """T2 번호로 끝나는 짧은 줄(12낱말 이하, 'A2.1 Context 18')이 3개 이상 이어지는 구간 = 점선 없는 목차(docx)."""
    out, run = set(), []
    for i, b in enumerate(blocks + [{"type": "end", "text": ""}]):
        s = b["text"].strip()
        if (b["type"] in ("paragraph", "list_item") and len(s.split()) <= 12 and TOC_LINE.search(s) and not END.search(s[:-1].rstrip())
                and re.search(r"[A-Za-z가-힣]{3,}", s)):   # 진짜 낱말이 있어야 목차 줄('V 21' 같은 달력 칸은 아님)
            run.append(i)
            continue
        if len(run) >= 3:
            out.update(run)
        run = []
    return out


def remove_blocks(blocks, c, doc, dropped=None):
    """3 블록 제거(잡음 지우기). doc은 그림 영역(pictures)과 쪽 수를 준다. dropped(목록)를 주면 지운 블록을 기록한다."""
    dropped = dropped if dropped is not None else []
    lines, repeated = line_number_ids(blocks), repeated_short_ids(blocks)
    running, tocs = running_ids(blocks, doc.get("n_pages") or 0), toc_run_ids(blocks)
    pics = defaultdict(list)
    for pg, *box in doc.get("pictures") or []:
        pics[pg].append(box)
    is_refs_head = lambda b: b["type"] in ("heading", "paragraph") and REFS.match(b["text"].strip())   # 'References' 한 줄이 문단으로 분류된 문서도(실측 2403.15810v1, 2404.11221v4)
    has_refs_heading = any(is_refs_head(b) for b in blocks)
    keep, in_refs, nonref, noted, note_run = [], False, 0, set(), False
    fonts = sorted((b.get("geom") or {}).get("font") for b in blocks if b["type"] == "paragraph" and not b.get("from_picture") and (b.get("geom") or {}).get("font"))
    body_font = fonts[len(fonts) // 2] if fonts else None   # 본문 글자 크기(문단 블록 글자 크기의 중앙값)

    def drop(name, b):
        c[name] += 1
        c["words:" + name] += nwords(b["text"])
        dropped.append({"rule": name, "page": b.get("page"), "type": b["type"], "text": b["text"]})   # 검토·삭제 검사용 기록(글 전체)

    for i, b in enumerate(blocks):
        s = b["text"].strip()
        if is_refs_head(b):
            in_refs, nonref = True, 0
            drop("R1 참고문헌 제거", b)
            continue
        if b["type"] == "heading":
            in_refs = False
        elif in_refs and b["type"] == "footnote":
            # 참고문헌 영역의 각주: 참고문헌 모양이 뚜렷할 때만 지우고('저자, 이니셜'로 시작 + 연도 + 표지 2개 이상), 아니면 남기되 영역은 끝내지 않는다.
            # (코드로 분류된 참고문헌 묶음은 아래 일반 판정: 여러 항목이 한 블록이라 길다)
            # 옛: 각주가 나오면 영역 끝 → '[1] Adak, …' 각주가 남음. 모두 지우면 문서 끝으로 옮긴 본문 각주까지 지움(실측 2403.15810v1)
            if REF_START.match(s) and YEAR.search(s) and len(REF_STRICT.findall(s)) >= 2 and len(s.split()) <= 120:
                drop("R1 참고문헌 제거", b)
                continue
        elif in_refs:
            if b["type"] == "paragraph" and CAP_SHAPE.match(s):
                drop("F1 그림 캡션 제거", b)
                continue
            if b["type"] == "table" or b.get("role") == "caption":
                pass                              # 참고문헌 쪽에 떠 온 표·캡션: 영역을 끝내지 않고 아래 규칙으로 넘김
            elif b["type"] == "formula":
                in_refs = False                   # 수식 = 참고문헌 뒤에 제목 없이 붙은 부록(실측 2410.02927v1 보충 자료)
                c["R1 참고문헌 영역 끝(수식)"] += 1
            elif not (len(s.split()) >= 30 and not YEAR.search(s) and len(REF_STRICT.findall(s)) <= 1
                      and not REF_START.match(s) and not re.match(r"^\s*(\[\d+\]|\d+\.\s)", s)):
                nonref = 0                        # 참고문헌 영역 안에서 본문 단락(30낱말 이상, 연도 없음, 인용 표지 1개 이하('… in [26].'), 항목 번호·저자로 시작 안 함)이
                                                  # 아닌 블록 = 항목·잘린 조각. (연도 1개까지 허용하면 항목 하나가 본문으로 판정되어 영역이 일찍 끝나는 실측)
                drop("R1 참고문헌 제거", b)
                continue
            else:
                nonref += 1                       # 본문 단락이 2개 연속이면 영역 끝(제목 없이 이어지는 부록 등).
                if nonref >= 2:                   # 옛: '참고문헌답지 않은 블록 3개'로 끝내 쪽·단 경계에서 잘린 항목 조각 3개에 영역이 일찍 끝남(실측 약 2만 낱말이 남음)
                    in_refs = False
                    c["R1 참고문헌 영역 끝(본문 단락 2개 연속)"] += 1
        if i in lines:
            drop("N1 줄 번호 제거", b)
            continue
        if i in repeated:
            drop("D3 여러 쪽에 반복되는 짧은 블록 제거", b)
            continue
        if i in running:
            drop("H2 반복 머리말·바닥글 제거", b)
            continue
        if b["type"] != "table" and TOC.search(s):
            drop("T1 목차 제거", b)
            continue
        if i in tocs:
            drop("T2 번호로 끝나는 목차 줄 제거", b)
            continue
        s_cap = AXIS_LEAD.sub("", s)
        if ((b.get("role") == "caption" and not b.get("for") and FIG_CAP.match(s_cap))
                or (b["type"] in ("paragraph", "list_item", "footnote", "code") and CAP_SHAPE.match(s_cap))
                or (b["type"] == "heading" and CAP_SHAPE.match(s_cap) and len(s.split()) <= 40)):   # 목록·각주·코드·제목으로 분류된 그림 캡션도(실측 'fig. S2.', 'Figure 8:' 각주)
            drop("F1 그림 캡션 제거", b)
            continue
        if b.get("from_picture"):                 # 1차가 그림 영역에서 살린 글(25낱말 이상 문장)
            f = (b.get("geom") or {}).get("font")
            if body_font and f and f < 0.9 * body_font:   # 본문보다 작은 글자 = 그림 속 예시(대화·프롬프트, 본문의 40~80% 크기 실측) → 그림 글로 지움
                drop("G4 그림 영역 안 글 제거", b)
                continue
            keep.append(b)                        # 본문 크기 = Docling이 그림 영역으로 잘못 잡은 본문(초록·단락): 그림 글자 규칙을 건너뜀
            continue
        if b["type"] in ("paragraph", "list_item") and not b.get("role") and len(s) <= 200 and b.get("boxes"):
            pg, *box = b["boxes"][0]
            if any(_inside(box, pb) > 0.5 for pb in pics.get(pg, [])):
                drop("G4 그림 영역 안 글 제거", b)
                continue
        if fig_fragment(b):
            drop("G2 그림 속 글자 조각 제거", b)
            continue
        if spread_letters(b):
            drop("G5 글자가 흩어진 그림 글 제거", b)
            continue
        edge = bool(b.get("boxes") and b.get("page_size") and (b["boxes"][0][2] < 0.1 * b["page_size"][1] or b["boxes"][-1][4] > 0.9 * b["page_size"][1]))
        if b["type"] == "paragraph" and NUM.match(s) and b.get("page") is not None and (abs(int(s) - b["page"]) <= 2 or edge):   # 쪽 위·아래 10% 안 숫자만 = 인쇄 쪽 번호(두 쪽 펼침 문서)
            drop("N2 본문 사이 쪽 번호 제거", b)
            continue
        m = ARXIV_STAMP.search(s)
        if m:
            if len(s) <= len(m.group(0)) + 10:
                drop("N3 arXiv 세로 도장 제거", b)
                continue
            b = {**b, "text": (s[:m.start()] + s[m.end():]).strip()}
            c["N3 arXiv 세로 도장 제거"] += 1
        if b["type"] != "table" and (len(s.split()) <= 80 or (len(s.split()) <= 250 and BOILER.match(s))) and BOILER.search(s):   # 블록이 문구로 시작하면 긴 것도(ACM 'Permission to make …' 블록 실측 100낱말 안팎)
            drop("N3 저작권·라이선스 문구 제거", b)
            continue
        base = str(b["id"]).split(".")[0]
        if (base in noted and b.get("page") == 1 and len(s) <= 150
                and not re.search(r"(다|요)\.\s*$", s)):   # 숨은 경계로 나뉜 같은 블록의 나머지 조각(교신 저자 이름·주소 줄)도 함께. 첫 쪽의 짧은 줄만:
            # Docling이 여러 쪽을 한 블록으로 합친 문서에서 같은 블록의 초록~5쪽 본문까지 지운 실측(2412.11355v2)
            drop("N3 첫 쪽 접수·교신 정보 제거", b)
            continue
        if b.get("page") == 1 and b["type"] != "table" and len(s.split()) <= 80 and (FIRST_PAGE_NOTE.search(s) or NOTE_START.match(s)):
            noted.add(base)
            note_run = True                       # 이어지는 짧은 줄(날짜, 이름, 소속, Tel, E-mail)도 같은 묶음
            drop("N3 첫 쪽 접수·교신 정보 제거", b)
            continue
        if note_run:
            if (b.get("page") == 1 and b["type"] in ("paragraph", "list_item") and len(s) <= 150
                    and not re.search(r"(다|요)\.\s*$", s)):   # 짧은 줄만(글자 수로: 띄어쓰기 없는 한국어 본문을 낱말 수로 짧다고 오판한 실측), 문장(~다.)이면 본문
                drop("N3 첫 쪽 접수·교신 정보 제거", b)
                continue
            note_run = False                      # 긴 본문·제목·다음 쪽이 나오면 묶음 끝
        if b.get("page") == 1 and CONTACT.match(s):
            drop("N3 첫 쪽 이메일·URL 줄 제거", b)
            continue
        keep.append(b)
    out = []
    for b in keep:
        if b["type"] == "heading" and TOC_HEAD.match(b["text"].strip()):
            drop("T1 목차 제거", b)
            continue
        out.append(b)
    if not has_refs_heading:                       # R2 제목 없는 끝 참고문헌: 문서 뒤 절반에서 참고문헌 같은 블록 5개 이상 연속
        def refy(b):                                # 참고문헌 항목(들): 짧은 항목, 또는 여러 항목이 뭉친 긴 블록(연도가 50낱말에 1번 이상)
            t = b["text"]
            n = len(t.split())
            if not (b["type"] in ("paragraph", "list_item", "code", "footnote") and YEAR.search(t) and len(REF_STRICT.findall(t)) >= 2):
                return False
            if n <= 120:                            # 짧은 항목은 항목 번호·저자로 시작해야(인용 2개 있는 본문 문장 'Smith et al. (2020) and …' 제외)
                return bool(REF_START.match(t) or re.match(r"^\s*(\[\d+\]|\d+\.\s)", t))
            return len(YEAR.findall(t)) >= n / 50

        def piece(b):                               # 쪽 넘김 등으로 잘린 항목 조각(짧거나 소문자 시작): 연속을 끊지 않는다
            t = b["text"].strip()
            return bool(b["type"] in ("paragraph", "list_item", "code") and (len(t.split()) < 25 or re.match(r"^[a-z(]", t)))
        run, kill, gap = [], set(), []
        for k, b in enumerate(out + [{"type": "end", "text": ""}]):
            if k >= len(out) // 2 and k < len(out) and refy(b):
                run += gap + [k]
                gap = []
                continue
            if run and k < len(out) and piece(b) and len(gap) < 2:
                gap.append(k)
                continue
            if sum(refy(out[x]) for x in run) >= 5:
                kill.update(run)
            run, gap = [], []
        for k in kill:
            drop("R2 제목 없는 끝 참고문헌 제거", out[k])
        out = [b for k, b in enumerate(out) if k not in kill]
    return out


LATIN_TOK = re.compile(r"^[A-Za-z0-9_()]{1,2}$")       # 1~2글자 라틴 문자·숫자 조각('f', '1v', 'E', '(')
LEAD_LETTERS = re.compile(r"^(?:[A-Za-z]\s+){3,}(?=[A-Za-z]{2,})")


SPREAD_BAD = re.compile(r"[=<>≤≥+∑∫×/^가-힣]")


def spread_letters(b):
    """G5 글자가 흩어진 그림 쓰레기(라틴 문자만): 6낱말 이상, 70% 이상이 1~2글자, 4글자 넘는 낱말 없음, 수식 기호·한글 없음
    ('f 1v f o g e T r a i n i n g …'). 한글은 제외('특 허 법 원'처럼 일부러 띄운 제목이 많다)."""
    if b["type"] not in ("paragraph", "list_item") or b.get("role"):
        return False
    t = b["text"].split()
    words = [re.sub(r"[^A-Za-z]", "", x) for x in t]
    return (len(t) >= 6 and sum(len(x) <= 2 for x in t) / len(t) >= 0.7 and not any(len(w) >= 4 for w in words)
            and sum(bool(re.fullmatch(r"[A-Za-z]", x)) for x in t) / len(t) >= 0.6     # 대부분 한 글자 알파벳('u e T e t i o n'), 'for k = 1 , …, K do'는 아님
            and b["text"].isascii() and not SPREAD_BAD.search(b["text"]))   # 그리스 문자·수학 기호가 있으면 수식 문장('For all λ ∈ [0, 1]')


SHORT_WORDS = {"a", "i", "an", "as", "at", "be", "by", "if", "in", "is", "it", "no", "of", "on", "or", "so", "to", "up", "we",
               "and", "the", "for", "are", "not", "but", "all", "any", "let", "then", "with", "where", "since", "thus", "hence"}


def fig_fragment(b):
    """G2 Docling이 그림 영역을 못 찾아 그림 속 글자(축 이름, 범례)가 글 블록으로 나온 것. PDF 블록(위치 있음)만.
    ① 12자 이하이고 낱말이 모두 1~2글자 라틴 문자·숫자('D i', 'E W (', 'f 1v')이며 흔한 짧은 낱말('or', 'of')만으로 된 것은 아님,
    ② 또는 폭이 쪽 폭의 5% 미만인 12자 이하 글 중 한글·날짜/숫자 꼴·4글자 넘는 낱말이 없는 것(연표의 '1984.7', '관계', 'Domingos'는 남김)."""
    if b["type"] not in ("paragraph", "list_item") or b.get("role") or not b.get("boxes"):
        return False
    s = b["text"].strip()
    toks = s.split()
    if (len(s) <= 12 and toks and all(LATIN_TOK.match(t) for t in toks) and not NUM.match(s)
            and not all(t.lower() in SHORT_WORDS for t in toks)):
        return True
    if (b.get("page_size") and len(s) <= 12 and not HANGUL.search(s) and not re.fullmatch(r"[\d.,\-~/ ]+", s)
            and not re.search(r"[A-Za-z]{3,}", s) and "=" not in s):   # 낱말(3글자+)·등호가 있으면 수식 줄 조각('FFT(', '= 399.')이라 남김   # 수식 사이 'and'는 남김        # 4글자 넘는 낱말('Domingos')이 있으면 사람 이름·낱말일 수 있어 남김
        bx = b["boxes"][0]
        return bx[3] - bx[1] < 0.05 * b["page_size"][0]
    return False


def strip_lead_letters(b, c):
    """블록 앞에 붙은 그림 낱글자 줄('E W r a g e d … though the 125')을 떼어 낸다: 한 글자 라틴 낱말 3개 이상이 연달아 오고 그다음 진짜 낱말이 올 때."""
    m = LEAD_LETTERS.match(b["text"])
    if m and b["type"] in ("paragraph", "list_item"):
        c["G3 블록 앞 그림 낱글자 떼기"] += 1
        return {**b, "text": b["text"][m.end():]}
    return b


FOOT_MARK = re.compile(r"^\s*(\d{1,2}|[*†‡§¶⋆])\s*\S")


def move_footnote_like(blocks, c):
    """F2 본문으로 분류된 각주를 문서 끝으로(Docling 각주와 같은 처리). 쪽 아래 20% 안, 각주 표시(숫자·*·†)로 시작,
    글자가 본문보다 작음(1차가 기록한 글자 크기 < 본문 글자 크기 중앙값의 90%; 줄 높이는 위첨자·URL 때문에 못 씀), 80낱말 이하인 본문 블록. 그대로 두면 단 끝 문장과 다음 단 첫 문장 사이에 끼어
    잘못 이어진다(실측: '1 For SALMONN: https://…' + 'which is NISQA [41] …')."""
    fs = sorted(b["geom"]["font"] for b in blocks if b["type"] == "paragraph" and b.get("geom", {}).get("rows", 0) >= 3 and b["geom"].get("font"))
    if not fs:
        return blocks
    body_f = fs[len(fs) // 2]
    keep, moved = [], []
    for b in blocks:
        g, bx = b.get("geom"), (b.get("boxes") or [None])[-1]
        if (b["type"] == "paragraph" and g and bx and b.get("page_size") and bx[4] > 0.8 * b["page_size"][1]
                and FOOT_MARK.match(b["text"]) and g.get("font") and g["font"] < 0.9 * body_f and len(b["text"].split()) <= 80):
            moved.append({**b, "type": "footnote"})
            c["F2 본문으로 분류된 각주를 문서 끝으로"] += 1
        else:
            keep.append(b)
    return keep + moved


def halve_duplicates(b, c):
    ws = b["text"].split()
    if len(ws) >= 4 and len(ws) % 2 == 0 and all(ws[k] == ws[k + 1] for k in range(0, len(ws), 2)):
        c["D1 단어 반복 정리"] += 1
        return {**b, "text": " ".join(ws[0::2])}
    return b


# ---------------------------------------------------------------- 수식
MATH_OP = re.compile(r"\s*([=<>≤≥≠≈∼∈∉⊂⊆⊃⊇→←↦⇒⇔±×∪∩∞∑∏∫√])\s*")
MATH_FUNC = re.compile(r"\\(deg|det|dim|ker|max|min|sup|inf|log|ln|exp|sin|cos|tan|arg|lim|Pr|gcd|mod|tr|var|Var|Tr)\b")
MATH_NAME = re.compile(r"\\(operatorname|mathrm|text|textrm|mathit|mathbf|boldsymbol|mathsf)\s*\*?\s*\{([^{}]*)\}")
REPEAT_RUN = re.compile(r"((?:\S+\s+){1,6}?)\1{4,}")
BROKEN_RUN = re.compile(r"((?:\S+\s+){3,40}?)\1{3,}")
_L2T = None


def readable_formula(latex):
    """LaTeX → 원문 PDF 글자 층과 비슷한 읽기 쉬운 글자(∑, ∞, ≤; 첨자는 띄어쓰기). 실패하면 LaTeX 그대로."""
    global _L2T
    if _L2T is None:
        from pylatexenc.latex2text import LatexNodes2Text
        _L2T = LatexNodes2Text(math_mode="text")
    s = re.sub(r"\\\\|&", " ", latex)
    s = MATH_NAME.sub(lambda m: " " + re.sub(r"(?<=\b\w) (?=\w\b)", "", " ".join(m.group(2).split())) + " ", s)   # Q5 \text{w h e r e}→where
    s = MATH_FUNC.sub(lambda m: " " + m.group(1) + " ", s)
    try:
        s = _L2T.latex_to_text(s)
    except Exception:
        pass
    s = re.sub(r"\s*[_^]\s*", " ", s)
    s = " ".join(s.split()) + " "
    s = REPEAT_RUN.sub(r"\1", s)
    s = MATH_OP.sub(r" \1 ", s)
    return " ".join(s.split())


def formula_text(b, c):
    """수식 블록의 글: 인식이 고장 난 것으로 보이면 원본 글자 층, 아니면 읽기 쉬운 글자로 바꾼 LaTeX."""
    latex, layer = b["text"], b.get("text_layer", "")
    if layer and (BROKEN_RUN.search(latex + " ") or len(latex) > 6 * len(layer) + 80):
        c["Q3 고장 난 수식을 원본 글자 층으로"] += 1
        return layer
    f = readable_formula(latex)
    tag = re.search(r"\((\d{1,3}(?:\.\d{1,3})?[a-z]?)\)\s*$", layer)     # 원본이 식 번호 '(1)'로 끝나면, 변환 글에서 첫 '( 1 )' 뒤는 모델이 덧붙인 것
    if tag:
        m = re.search(r"\(\s*" + r"\s*".join(map(re.escape, tag.group(1))) + r"\s*\)", f)
        if m and re.search(r"[A-Za-z]", f[m.end():]):
            c["Q4 식 번호 뒤 덧붙은 글 자르기"] += 1
            f = f[:m.end()]
    return f


_WT = re.compile(r"[A-Za-z]{2,}")
_MATH_WORDS = {"sin", "cos", "tan", "exp", "log", "ln", "max", "min", "sup", "inf", "lim", "det", "arg", "mod", "for", "and", "if", "or"}


def trim_neighbor_dup(f, prev_text, next_text, c):
    """Q6 수식 글 앞뒤의 이웃 본문 반복 자르기: 수식 인식이 수식 영역에 걸친 앞·뒤 문장까지 \text{…}로 받아 적어,
    수식을 문장 안에 넣으면 같은 문장이 두 번 나오던 실측(2412.20294v2 '…so if l is even then the x integral…' 등, vectara 310편 중복의 원인).
    수식 글의 뒤쪽 낱말(2글자 이상 영문)이 4개 이상이고 그 80% 이상이 다음 블록 첫 60낱말에 있으면 그 앞에서 자르고, 앞쪽도 앞 블록 끝 60낱말과 같은 방식."""
    toks = list(_WT.finditer(f))
    if len(toks) < 4:
        return f
    low = [m.group(0).lower() for m in toks]
    nxt = {w.lower() for w in _WT.findall(next_text or "")[:60]}
    prv = {w.lower() for w in _WT.findall(prev_text or "")[-60:]}

    def textual(ws):
        return sum(1 for w in ws if len(w) >= 3 and w not in _MATH_WORDS) >= 3
    if nxt:
        for k in range(len(low) - 3):
            tail = low[k:]
            if textual(tail) and sum(w in nxt for w in tail) / len(tail) >= 0.8:
                c["Q6 수식 글 앞뒤의 이웃 본문 반복 자르기"] += 1
                f = f[:toks[k].start()].rstrip()
                low, toks = low[:k], toks[:k]
                break
    if prv and len(low) >= 4:
        for k in range(len(low), 3, -1):
            head = low[:k]
            if textual(head) and sum(w in prv for w in head) / len(head) >= 0.8:
                c["Q6 수식 글 앞뒤의 이웃 본문 반복 자르기"] += 1
                f = f[toks[k - 1].end():].lstrip(" ,.;:")
                break
    return f


# ---------------------------------------------------------------- 5 문단 잇기
def _kind(b):
    if b.get("role") == "caption":
        return "caption"
    return {"paragraph": "para", "list_item": "list", "table": "table", "code": "code", "formula": "formula",
            "footnote": "footnote", "heading": "heading"}.get(b["type"], "para")


def _unit(b):
    pages = b.get("page_span") or sorted({bx[0] for bx in b.get("boxes", [])}) or ([b["page"]] if b.get("page") is not None else [])
    return {"text": b["text"], "kind": _kind(b), "block_ids": [b["id"]], "pages": pages,
            "heading": "", "_last": b}


def geo_continues(pb, b, col_w):
    """줄 위치로 본 '앞 블록이 문단 중간에서 끊겼나'. pb = 앞 단위의 마지막 블록. 앞 블록은 끝 영역(boxes[-1]), 뒤 블록은 시작 영역(boxes[0])."""
    g, h = pb.get("geom"), b.get("geom")
    if not g or not h or not pb.get("boxes") or not b.get("boxes"):
        return False
    pa, l, t, r, bt = pb["boxes"][-1]
    pn, nl, nt, nr, nb = b["boxes"][0]
    lh = g["line_h"]
    full = g["last_full"] and r >= pb.get("_margin", r) - lh
    if not full or h["first_indent"] or BULLET.match(b["text"].lstrip()):
        return False
    pw, ph = b["page_size"]
    if pn == pa:
        if abs(nl - l) < lh and -2 <= nt - bt <= 2 * lh:
            return True
        wide = col_w and r - l >= 0.8 * col_w and nr - nl >= 0.8 * col_w
        return bool(wide and nl >= r - 2 and nt < t and bt > ph * 0.6)
    return pn == pa + 1 and bt > ph * 0.5 and nt < ph * 0.5 and nl < pw * 0.5


def continues(prev, b, c, col_w):
    if prev["kind"] != "para" or _kind(b) != "para":
        return None
    p, n = prev["text"].rstrip(), b["text"].lstrip()
    if prev.get("after_formula") and re.match(r"^[a-z]", n):
        c["Q2 수식 뒤 문장 이어 붙임"] += 1
        return "space"
    if not p or not n or END.search(p):
        return None
    if p.endswith("-") and re.match(r"^[a-z]", n):
        c["P5 하이픈 분리 합침"] += 1
        return "hyphen"
    if re.match(r"^[a-z]", n):
        c["P1·P2 영어 끊긴 문단 합침"] += 1
        return "space"
    if HANGUL.search(p[-1]) and HANGUL.match(n[0]) and prev["pages"] and b.get("page") is not None and b["page"] > max(prev["pages"]):
        c["P3 한국어 쪽 바뀜 합침"] += 1
        return "space"
    if len(prev["_last"]["text"]) >= 20 and (HANGUL.match(n[0]) or p.endswith(",")) and geo_continues(prev["_last"], b, col_w):
        c["L1 줄 위치로 합침"] += 1
        return "space"
    return None


# ---------------------------------------------------------------- 문서 하나
def build_units(doc, count_many, dropped=None):
    c = Counter()
    blocks = []
    for b in doc["blocks"]:
        raw = b["text"] or ""
        if b.get("fake_spaces"):                   # K1 한국어 가짜 띄어쓰기(1차가 PDF 글자 좌표로 찾은 위치) 지우기
            fs = set(b["fake_spaces"])
            raw = "".join(ch for i, ch in enumerate(raw) if not (i in fs and ch.isspace()))
            c["K1 한국어 가짜 띄어쓰기 지움"] += len(fs)
        if GLYPH.search(raw):
            c["C3 글꼴 글리프 이름 쓰레기 제거"] += len(GLYPH.findall(raw))
            raw = GLYPH.sub("", raw)
        t = clean_chars(raw, c)
        if t:
            blocks.append({**b, "text": t})
        else:
            c["빈 블록 제외"] += 1
    for i, b in enumerate(blocks):                 # T7 칸이 하나뿐인 표 = 테두리 친 본문·알고리즘 상자를 Docling이 표로 잡은 것(실측 2403.15810v1 본문 542낱말이 '| … |' 줄로)
        if b["type"] == "table":
            rows = [l.strip().strip("|").strip() for l in b["text"].split("\n") if l.lstrip().startswith("|")]
            rows = [r for r in rows if r and not set(r) <= set("-: |")]
            if rows and all("|" not in r for r in rows) and len(" ".join(rows).split()) >= 25:
                blocks[i] = {**b, "type": "paragraph", "text": " ".join(rows)}
                c["T7 한 칸짜리 표를 본문으로"] += 1
    long_blocks = [b for b in blocks if len(b["text"]) >= 5]
    if long_blocks and sum(garbled(b["text"]) >= 0.3 for b in long_blocks) / len(long_blocks) >= 0.3:
        c["G1 깨진 문서 제외"] += 1
        return None, c
    blocks = [strip_lead_letters(b, c) for b in split_blocks(blocks, c)]       # 그림 낱글자를 먼저 떼야 참고문헌·이어짐 판단이 맞다
    blocks = [halve_duplicates(b, c) for b in remove_blocks(blocks, c, doc, dropped)]
    blocks = move_footnote_like(blocks, c)

    tables = {b["id"] for b in blocks if b["type"] == "table"}
    caps = defaultdict(list)
    for i, b in enumerate(blocks):
        if b.get("role") != "caption" and b["type"] in ("heading", "paragraph") and TAB_CAP_SHAPE.match(b["text"]) and len(b["text"]) <= 300:
            near = next((blocks[j]["id"] for j in list(range(i + 1, min(len(blocks), i + 4))) + [i - 1]
                         if 0 <= j < len(blocks) and blocks[j]["type"] == "table" and blocks[j].get("page") == b.get("page")), None)
            if near is not None:                  # 제목·본문으로 분류된 표 캡션('Table 1. Primary Cancer') → 가까운 표(뒤 3개, 앞 1개)에 붙임
                caps[near].append(b)
                c["T6 표 캡션 붙이기(캡션 모양 제목·본문)"] += 1
            continue
        if b.get("role") != "caption":
            continue
        t = b.get("for") if b.get("for") in tables else None
        if t is None and TAB_CAP.match(b["text"]):
            t = next((blocks[j]["id"] for j in (i + 1, i - 1) if 0 <= j < len(blocks) and blocks[j]["type"] == "table"), None)
        if t is not None:
            caps[t].append(b)
            c["T6 표 캡션 붙이기"] += 1
    for i, b in enumerate(blocks):                 # 표 바로 뒤(2블록 안, 같은 쪽)의 표 주석('Note:', '*** p < 0.01', 'Source:')을 표에 붙임(본문 사이에 끼던 실측 2411.10728v3)
        if b["type"] == "table":
            for j in range(i + 1, min(len(blocks), i + 3)):
                x = blocks[j]
                if x.get("page") != b.get("page") or x["type"] not in ("paragraph", "footnote", "list_item"):
                    break
                if TAB_NOTE.match(x["text"]) and len(x["text"].split()) <= 120 and x["id"] not in {y["id"] for ys in caps.values() for y in ys}:
                    caps.setdefault(b["id"], [])
                    x["_note"] = True
                    caps[b["id"]].append(x)
                    c["T8 표 주석을 표에 붙이기"] += 1
    attached = {b["id"] for bs in caps.values() for b in bs}
    orphan = []                                    # T9 짝 없는 표 캡션·표 번호 줄: 같은 쪽에 표가 있으면 붙이고, 없으면 지운다(본문 사이에 끼던 실측 86개)
    for i, b in enumerate(blocks):
        if b["id"] in attached or b["type"] == "table":
            continue
        t = b["text"].strip()
        if (TAB_CAP_SHAPE.match(t) and len(t.split()) <= 60) or (TAB_LABEL.match(t)):
            near = lambda j: 0 <= j < len(blocks) and blocks[j]["type"] == "table" and blocks[j].get("page") == b.get("page")
            ahead = lambda j: (0 <= j < len(blocks) and blocks[j]["type"] == "table"
                               and (b.get("page") is None or blocks[j].get("page") in (b.get("page"), b["page"] + 1)))
            # 캡션은 보통 표 앞: 뒤쪽 4블록 안의 첫 표(다음 쪽까지) → 없으면 같은 쪽 앞쪽 4블록 안의 표
            # (가장 가까운 표를 고르면 앞 표에 붙던 실측: 판결문 '<표 2>'의 표가 다음 쪽에 있어 '<표 1>' 표에)
            tj = next((j for j in range(i + 1, i + 5) if ahead(j)), None)
            if tj is None:
                tj = next((j for j in range(i - 1, i - 5, -1) if near(j)), None)
            if tj is not None:
                tgt = blocks[tj]
                grp = [b]
                if TAB_LABEL.match(t) and i + 1 < len(blocks) and i + 1 != tj and blocks[i + 1]["type"] in ("paragraph", "heading") \
                        and len(blocks[i + 1]["text"].split()) <= 20 and blocks[i + 1]["id"] not in attached:
                    grp.append(blocks[i + 1])            # 단독 번호 줄('Table 1') 다음의 짧은 제목 줄도 함께
                caps.setdefault(tgt["id"], []).extend(grp)
                attached.update(x["id"] for x in grp)
                c["T9 짝 없는 표 캡션을 같은 쪽 표에 붙이기"] += 1
            else:
                orphan.append(b["id"])
                if dropped is not None:
                    dropped.append({"rule": "T9 짝 없는 표 캡션 제거", "page": b.get("page"), "type": b["type"], "text": b["text"]})
                c["T9 짝 없는 표 캡션 제거"] += 1
    attached |= set(orphan)
    blocks = [({**b, "text": "\n".join(x["text"] for x in caps[b["id"]] if not x.get("_note")) + ("\n" if any(not x.get("_note") for x in caps[b["id"]]) else "")
                + b["text"] + "".join("\n" + x["text"] for x in caps[b["id"]] if x.get("_note"))} if b["id"] in caps else b)
              for b in blocks if b["id"] not in attached]

    paras = [b for b in blocks if b["type"] == "paragraph" and b.get("geom") and b.get("boxes")]
    ws = sorted(b["boxes"][-1][3] - b["boxes"][-1][1] for b in paras if b["geom"]["rows"] >= 3)
    col_w = ws[len(ws) // 2] if ws else 0
    for b in paras:                              # 끝 영역이 속한 단의 오른쪽 여백 = 같은 쪽, 왼쪽 끝이 같은 본문 영역들의 가장 오른쪽 끝
        pg, l = b["boxes"][-1][0], b["boxes"][-1][1]
        lh = b["geom"]["line_h"]
        b["_margin"] = max(bx[3] for x in paras for bx in x["boxes"] if bx[0] == pg and abs(bx[1] - l) < lh)

    units, last = [], None
    for bi, b in enumerate(blocks):
        k = _kind(b)
        if k in ("heading", "footnote"):
            units.append(_unit(b))
            last = None
            continue
        if k in ("table", "caption"):
            units.append(_unit(b))
            continue
        if k == "formula":
            f = formula_text(b, c)
            nb = next((x for x in blocks[bi + 1:] if x["type"] != "formula"), None)
            f = trim_neighbor_dup(f, last["text"] if last is not None else "", nb["text"] if nb else "", c)
            c["Q1 수식을 문장 안으로"] += 1
            if last is not None:
                last["text"] = last["text"].rstrip() + " " + f
                last["block_ids"].append(b["id"])
            else:
                last = {**_unit(b), "text": f, "kind": "para"}
                units.append(last)
            last["after_formula"] = True
            continue
        how = continues(last, b, c, col_w) if last is not None else None
        if how:
            last["after_formula"] = False
            last["text"] = (last["text"].rstrip()[:-1] + b["text"].lstrip()) if how == "hyphen" else (last["text"].rstrip() + " " + b["text"].lstrip())
            last["block_ids"].append(b["id"])
            last["_last"] = b
            for pg in (b.get("page_span") or sorted({bx[0] for bx in b.get("boxes", [])}) or [b.get("page")]):
                if pg is not None and pg not in last["pages"]:
                    last["pages"].append(pg)
            continue
        u = _unit(b)
        units.append(u)
        last = u if k == "para" else None

    glued, pending = [], []
    for u in units:
        if u["kind"] == "heading":
            pending.append(u)
            continue
        if pending and u["kind"] != "footnote":
            head = "\n".join(h["text"] for h in pending)
            u = {**u, "text": head + "\n" + u["text"], "heading": head, "block_ids": sum((h["block_ids"] for h in pending), []) + u["block_ids"]}
            pending = []
        elif pending:
            glued.extend(pending)
            pending = []
        glued.append(u)
    glued.extend(pending)

    toks = count_many([u["text"] for u in glued])
    out, carry = [], []
    for u, n in zip(glued, toks):
        if u["kind"] not in ("table", "footnote") and (not LETTER.search(u["text"]) or n <= 2):
            carry.append(u)
            c["S 조각 붙이기"] += 1
            continue
        if u["kind"] in ("para", "list") and n <= 5 and u["text"].rstrip().endswith(":"):
            carry.append(u)                       # S2 양식 항목('Postal code:')은 다음 단위 앞에
            c["S2 양식 항목 붙이기"] += 1
            continue
        if carry and u["kind"] != "footnote":
            u = {**u, "text": "\n".join(x["text"] for x in carry) + "\n" + u["text"], "block_ids": sum((x["block_ids"] for x in carry), []) + u["block_ids"]}
            carry = []
        out.extend(carry)
        carry = []
        out.append(u)
    out.extend(carry)
    final = []
    for u, n in zip(out, count_many([u["text"] for u in out])):
        final.append({"text": u["text"], "kind": u["kind"], "heading": u["heading"], "pages": u["pages"], "block_ids": u["block_ids"], "n_tokens": n})
    return final, c


def process(doc_id, ds, out, count_many):
    """문서 하나: meta_text_v2 → processed_para. 제외된 문서(G1)는 None."""
    doc = json.loads((out / "meta_text_v2" / ds / f"{doc_id}.json").read_text(encoding="utf-8"))
    dropped = []
    units, c = build_units(doc, count_many, dropped)
    if units is None:
        return None, c
    rec = {"doc_id": doc["doc_id"], "dataset": ds, "lang": doc.get("lang"), "title": doc.get("title"), "fmt": doc.get("fmt"),
           "source": f"meta_text_v2/{ds}", "n_pages": doc.get("n_pages"), "n_units": len(units),
           "n_tokens": sum(u["n_tokens"] for u in units),
           "log": {k: v for k, v in sorted(c.items()) if not k.startswith("words:")}, "units": units}
    out_dir = out / "processed_para" / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{doc_id}.dropped.json").write_text(json.dumps(dropped, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / f"{doc_id}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec, c


def main():
    from common.models import make_token_counter

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="all", help="all 또는 " + ", ".join(DATASETS))
    ap.add_argument("--doc", action="append", help="이 doc_id만(여러 번 줄 수 있음)")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=str(DATA_DIR), help="입출력 루트(기본 data/)")
    ns = ap.parse_args()
    out = Path(ns.out)
    count_many = make_token_counter()
    names = list(DATASETS) if ns.dataset == "all" else [ns.dataset]
    docs = [d for ds in names for d in documents(ds) if not ns.doc or d["doc_id"] in ns.doc]
    docs = [d for k, d in enumerate(docs) if k % ns.workers == ns.worker]
    for d in docs:
        rec, c = process(d["doc_id"], d["dataset"], out, count_many)
        if rec is None:
            print(f"{d['dataset']}:{d['doc_id']}: 제외됨 {dict(c)}", flush=True)
        else:
            print(f"{d['dataset']}:{d['doc_id'][:40]}: 단위 {rec['n_units']}, 토큰 {rec['n_tokens']:,}", flush=True)


if __name__ == "__main__":
    main()
