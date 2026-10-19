"""clean_text v5 문단 정리: clean_text.build가 만든 문단(v4)을 의미상 깔끔한 문단으로 다시 묶는다. stage2.process가 build 바로 뒤에 부른다.

2026-10-19 사용자와 정함(시험 구현·검사 tests/clean_text/v5.py, 결과 tests/clean_text/out/v5/). 순서대로 적용:
  toc      목차 줄: 쪽 번호로 끝나는 짧은 줄 3개 이상 연속, 끝 숫자가 줄어들지 않고, (같은 쪽 번호가 되풀이되거나 바로 앞이 'Contents' 제목)
           → '\\n'으로 이은 한 문단(kind toc). 앞의 'Contents' 제목 줄도 함께
  refs     참고문헌으로 표시되지 않은 참고문헌(stage2 R2가 놓친 것: 제목 없음·문서 앞 절반·이니셜 앞 저자 양식): 짧은 항목(120낱말 이하, 연도 +
           참고문헌 표지 2개 이상, 저자(성 앞·이니셜 앞)·[n]·번호로 시작)이 3개 이상 이어지면(사이 조각 2개까지, 문서 앞 20% 이후), 앞뒤의 references
           문단·참고문헌 비슷한 항목까지 넓혀 '\\n'으로 이은 한 문단(kind references). pdf에서 묶음 바로 뒤(캡션·각주·표 건너뜀)에 소문자로 시작하는
           본문 조각이 오거나 바로 앞 본문이 문장 끝 없이 끝나면(2단 배치 읽기 순서 문제) 그 묶음을 문서 끝(끝의 각주 묶음 앞)으로 옮긴다
  lists    목록: 항목(list 문단, 또는 '(a)' '(i)' '(2)' 'a)'로 시작하는 80낱말 이하 문단)이 2개 이상 이어지거나 도입 문장(':' '—' '–' 또는
           means/includes/following으로 끝나는 본문) 뒤에 오면 '\\n'으로 이은 한 문단(kind list). 윗단계 글머리(□ ■ Ⅰ. 1. 가. Part Division …)는
           새 묶음의 시작, 묶음이 200낱말에 이르면 다음 항목부터 새 묶음
  formula  수식·코드: 앞 본문·목록 문단 뒤에 공백으로 붙이고(문장 안에 들어간 것), 뒤 문단이 이어지는 모양(소문자, where/with …)이면 뒤 문단도 공백으로.
           앞이 캡션·표·각주·참고문헌·목차거나, 앞 문단이 400낱말을 넘고 문장이 끝났으면 수식이 새 문단의 시작(kind para)
  heading  단독 제목: heading 문단, 또는 절 번호(3.2 / IV. / A. / Ⅰ. / 가. / Part·Chapter·Division·Appendix / 붙임)로 시작하거나 모두 대문자인 12낱말
           이하 한 줄(문장 끝 기호 없음, 날짜·쪽 번호 끝·이메일·URL·=·Figure/Table·소속 기관·글자 60% 미만 제외)을 뒤 본문·목록 앞에 '\\n'으로(연달아 있으면 모두)
  broken   끊긴 문단: 앞 본문(3낱말 이상)이 문장 끝 없이 끝나고 다음 본문(사이 캡션·각주·표는 건너뜀)이 소문자나 '('로 시작하면 공백으로 잇고,
           건너뛴 것은 이은 문단 뒤에 둔다('(a)' 조항 번호, '(단위' '(Table' '(Fig' '(그림' '(표' 등은 제외)
글자는 지우지 않는다: 공백을 뺀 글자 구성이 그대로이고, 순서는 refs 이동만 바뀐다(stage2가 검사).
문단 = {"text", "kind", 그 밖의 목록 값(pages, block_ids …)}: 합칠 때 목록 값은 이어 붙인다(pages는 겹치지 않게).
"""
import re
from collections import Counter

from preprocess.stage2 import REF_START, REF_STRICT, YEAR     # stage2는 이 모듈을 process 안에서 불러 순환 import가 없다

RULES = ("toc", "refs", "lists", "formula", "heading", "broken")
SKIP = ("caption", "footnote", "table")
SENT_END = re.compile(r"[.!?:;)\]\"'”’…。]\s*$")


def join(grp, kind, sep):
    """문단 여러 개 → 한 문단(글은 sep로 잇고, 목록 값은 이어 붙임)."""
    out = {"text": sep.join(p["text"] for p in grp), "kind": kind}
    for k in grp[0]:
        if isinstance(grp[0][k], list):
            vals = []
            for p in grp:
                vals += [v for v in p.get(k, []) if not (k == "pages" and v in vals)]
            out[k] = vals
        elif k not in out:
            out[k] = grp[0][k]
    return out


# ---------- toc ----------
PAGE_END = re.compile(r"(\.{3,}|…+|\s)\s*((?:[A-Z]{0,3}-?\d{1,3}-)?(\d{1,4}))\s*$")
CONTENTS = re.compile(r"^\s*((table\s+of\s+)?contents|목\s*차|차\s*례)\s*:?\s*$", re.I)


def toc_num(p):
    if "\n" in p or len(p.split()) > 15 or re.search(r"doi|http|Eq\.|\(\d{3}\)|arxiv", p, re.I):
        return None
    if re.search(r"(19|20)\d\d\)?\.?\s*$", p):
        return None
    m = PAGE_END.search(p)
    return int(m.group(3)) if m else None


def rule_toc(paras, fmt):
    nums = [toc_num(p["text"]) if p["kind"] in ("para", "list", "heading") else None for p in paras]
    out, i, n = [], 0, len(paras)
    while i < n:
        j = i
        while j < n and nums[j] is not None:
            j += 1
        if j - i >= 3:
            v = nums[i:j]
            mono = all(b >= a for a, b in zip(v, v[1:]))
            head = bool(out) and CONTENTS.match(out[-1]["text"])
            if mono and (len(set(v)) < len(v) or head):
                grp = ([out.pop()] if head else []) + paras[i:j]
                out.append(join(grp, "toc", "\n"))
                i = j
                continue
        if j > i:
            out += paras[i:j]; i = j
        else:
            out.append(paras[i]); i += 1
    return out


# ---------- refs ----------
INITIALS = re.compile(r"^\s*(\[\d+\]\s*|\d+\.\s+)?(?:[A-Z][a-z]?\.\s?(?:-?[A-Z][a-z]?\.\s?)*[A-Z][A-Za-z'\-]+)(?:,|\s+and\s|\s+et al)")


JOURNAL = re.compile(r"\b(phys|rev|lett|astron|astrophys|nucl|chem|proc|jhep|jcap|mnras|prd|prl)\.?(?=[A-Z]|\s+[A-Z\d])", re.I)


def marks(t):                                    # 참고문헌 표지 수: stage2 REF_STRICT + 띄어쓰기 없는 학술지 약어('Phys.Rev.Lett.', 'JHEP')
    return len(REF_STRICT.findall(t)) + len(JOURNAL.findall(t))


def refy(t):
    if len(t.split()) > 120 or not (YEAR.search(t) and marks(t) >= 2):
        return False
    return bool(REF_START.match(t) or re.match(r"^\s*(\[\d+\]|\d+\.\s)", t) or INITIALS.match(t))


def refish(p):                                   # 묶음을 넓힐 때: references 문단, 또는 조건을 덜 채운 참고문헌 비슷한 항목
    t = p["text"]
    if p["kind"] == "references":
        return True
    if p["kind"] not in ("para", "list") or len(t.split()) > 120:
        return False
    return (bool(YEAR.search(t)) and marks(t) >= 1) or (p["kind"] == "list" and bool(INITIALS.match(t) or REF_START.match(t)))


def body(p):                                     # 끼어듦 판정에 쓰는 본문 문단
    t = p["text"]
    return p["kind"] == "para" and len(t.split()) >= 12 and not re.search(r"https?://|www\.|@", t) and not refish(p)


def piece(p):                                    # 참고문헌 묶음 사이·끝에 낄 수 있는 조각: 짧거나, 소문자 시작이면서 연도·표지가 있는 것(긴 본문 조각 제외)
    t = p["text"].strip()
    return p["kind"] in ("para", "list") and (len(t.split()) < 25 or bool(re.match(r"^[a-z(]", t) and (YEAR.search(t) or marks(t) >= 1)))


def rule_refs(paras, fmt):
    n = len(paras)
    flag = [p["kind"] in ("para", "list") and refy(p["text"]) for p in paras]
    blocks, i = [], int(n * 0.2)
    while i < n:
        if not flag[i]:
            i += 1; continue
        j, items, gap = i, [], []
        while j < n:
            if flag[j]:
                items += gap + [j]; gap = []
            elif piece(paras[j]) and len(gap) < 2:
                gap.append(j)
            else:
                break
            j += 1
        last_open = bool(items) and not SENT_END.search(paras[items[-1]]["text"])
        if gap and re.match(r"^[a-z]", paras[gap[0]]["text"]) and (last_open or YEAR.search(paras[gap[0]]["text"])):
            items.append(gap[0])                              # 끝 항목의 쪼개진 뒷부분(앞 항목이 덜 끝났거나 조각에 연도)
        if sum(flag[k] for k in items) >= 3:
            s, e = items[0], items[-1] + 1
            lo = blocks[-1][1] if blocks else 0
            while s > lo and (refish(paras[s - 1]) or (piece(paras[s - 1]) and s - 1 > lo and refish(paras[s - 2]))):
                s -= 1                                       # 참고문헌 영역 전체로 넓힘(앞)
            while e < n and (refish(paras[e]) or flag[e] or (re.match(r"^[a-z]", paras[e]["text"]) and piece(paras[e])
                                                             and ((e + 1 < n and refish(paras[e + 1])) or YEAR.search(paras[e]["text"])))):
                e += 1                                       # (뒤)
            blocks.append((s, e))
            i = e
        else:
            i = j + 1
    if not blocks:
        return paras
    moved, out, last = [], [], 0
    for s, e in blocks:
        out += paras[last:s]
        merged = join(paras[s:e], "references", "\n")
        k = e
        while k < n and paras[k]["kind"] in SKIP:
            k += 1
        after_frag = k < n and paras[k]["kind"] == "para" and bool(re.match(r"^[a-z]", paras[k]["text"])) and not refish(paras[k])
        b = len(out) - 1
        while b >= 0 and out[b]["kind"] in SKIP:
            b -= 1
        before_open = b >= 0 and body(out[b]) and not SENT_END.search(out[b]["text"])
        if fmt == "pdf" and (after_frag or before_open):      # 끼어듦은 pdf 단 배치의 읽기 순서 문제(docx에는 없음)
            moved.append(merged)
        else:
            out.append(merged)
        last = e
    out += paras[last:]
    if moved:
        t = len(out)
        while t > 0 and out[t - 1]["kind"] == "footnote":
            t -= 1
        out = out[:t] + moved + out[t:]
    return out


# ---------- lists ----------
ENUM = re.compile(r"^\s*(\((?:[a-z]{1,2}|[ivxlc]{1,6}|\d{1,3})\)|[a-z]\))\s")
LEAD = re.compile(r"([:—–]|\b(means|includes?|following))\s*$")
LIST_CAP = 200                                   # 합친 목록이 이 낱말 수에 이르면 다음 항목부터 새 묶음(θ 550토큰의 약 절반)
TOP = re.compile(r"^\s*([□■◆◇▣●◎]|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\s*\.|[IVX]+\.\s|\d{1,3}\.\s|\d{1,3}\s+[A-Z가-힣]|[가-하]\.\s|"
                 r"(Part|Division|Subdivision|Chapter|Section|Schedule|Appendix)\s)")     # 윗단계 글머리: 새 묶음의 시작


def is_item(p):
    return p["kind"] == "list" or (p["kind"] == "para" and bool(ENUM.match(p["text"])) and len(p["text"].split()) <= 80)


def rule_lists(paras, fmt):
    out, i, n = [], 0, len(paras)
    while i < n:
        if not is_item(paras[i]):
            out.append(paras[i]); i += 1
            continue
        lead = bool(out) and out[-1]["kind"] == "para" and bool(LEAD.search(out[-1]["text"])) and not TOP.match(paras[i]["text"])
        size = len(paras[i]["text"].split()) + (len(out[-1]["text"].split()) if lead else 0)
        j = i + 1
        while j < n and is_item(paras[j]) and not TOP.match(paras[j]["text"]) and size < LIST_CAP:
            size += len(paras[j]["text"].split())
            j += 1
        if j - i >= 2 or lead:
            grp = ([out.pop()] if lead else []) + paras[i:j]
            out.append(join(grp, "list", "\n"))
        else:
            out += paras[i:j]
        i = j
    return out


# ---------- formula ----------
CONT = re.compile(r"^\s*([a-z]|(where|with|and|or|in which|for which|which|then|so that|thus|hence|i\.e\.|e\.g\.)\b)")
FORMULA_CAP = 400       # 앞 문단이 이 낱말 수를 넘고 문장이 끝났으면 수식은 새 문단의 시작(증명·유도가 끝없이 한 문단이 되는 것 방지)


def rule_formula(paras, fmt):
    out, join_next = [], False
    for p in paras:
        if p["kind"] in ("formula", "code"):
            long_done = out and len(out[-1]["text"].split()) >= FORMULA_CAP and re.search(r"[.!?]\s*$", out[-1]["text"])
            if out and (join_next or out[-1]["kind"] in ("para", "list")) and not long_done:
                out[-1] = join([out[-1], p], out[-1]["kind"], " ")
            else:
                out.append({**p, "kind": "para"})
            join_next = True
            continue
        if join_next and p["kind"] in ("para", "list") and CONT.match(p["text"]):
            out[-1] = join([out[-1], p], out[-1]["kind"], " ")
        else:
            out.append(p)
        join_next = False
    return out


# ---------- heading ----------
SECNUM = re.compile(r"^\s*(\d{1,2}(\.\d{1,2}){0,3}\.?\s+[A-Z가-힣][a-z가-힣]|[IVXLC]{1,5}\.\s|[A-Z]\.\s|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\s*\.?\s*\S|[가-하]\.\s|"
                    r"(Part|Chapter|Section|Division|Subdivision|Appendix|Schedule|Article|Annex)\s|붙임|\[참고)")
MONTH = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2},?\s*)?(19|20)\d\d\b|^\s*\d{4}[.\-/년]", re.I)
AFFIL = re.compile(r"Department|University|Universit|Institut|School|Cent(er|re)|Laborator|College|Faculty|Hospital|Academy", re.I)


def is_heading(p):
    t = p["text"].strip()
    if p["kind"] not in ("para", "heading") or "\n" in t or not t or len(t.split()) > 12:
        return False
    if re.search(r"[.!?;,]\s*$", t) and not re.match(r"^\s*([IVXLC]{1,5}|[A-Z]|\d{1,2})\.?\s*$", t):
        return False
    if MONTH.search(t) or toc_num(t) is not None or re.search(r"https?://|www\.|@|=", t) or re.match(r"^(fig(ure)?|table|tab)\.?\s*\d", t, re.I):
        return False
    if AFFIL.search(t) or sum(ch.isalpha() for ch in t) < 0.6 * len(t.replace(" ", "")):
        return False
    caps = t.isupper() and sum(len(w) >= 3 and w.isalpha() for w in t.split()) >= 2 and len(t.split()) <= 8
    return p["kind"] == "heading" or bool(SECNUM.match(t)) or caps


def rule_heading(paras, fmt):
    out, i, n = [], 0, len(paras)
    while i < n:
        if is_heading(paras[i]):
            j = i
            while j < n and is_heading(paras[j]):
                j += 1
            if j < n and paras[j]["kind"] in ("para", "list") and toc_num(paras[j]["text"]) is None:
                out.append(join(paras[i:j + 1], paras[j]["kind"], "\n"))
                i = j + 1
                continue
        out.append(paras[i]); i += 1
    return out


# ---------- broken ----------
NOT_CONT = re.compile(r"^\s*\((단위|table|tab\.|fig|그림|표|source|출처|note)", re.I)


def continues(p):
    t = p["text"]
    return p["kind"] in ("para", "list") and bool(re.match(r"^\s*([a-z]|\()", t)) and not ENUM.match(t) and not NOT_CONT.match(t)


def rule_broken(paras, fmt):
    out, i, n = [], 0, len(paras)
    while i < n:
        p = paras[i]
        if p["kind"] == "para" and len(p["text"].split()) >= 3 and not SENT_END.search(p["text"]):
            j = i + 1
            while j < n and paras[j]["kind"] in SKIP and j - i <= 3:
                j += 1
            if j < n and continues(paras[j]):
                out.append(join([p, paras[j]], p["kind"], " "))
                out += paras[i + 1:j]                         # 건너뛴 캡션·각주·표는 이은 문단 뒤
                i = j + 1
                continue
        out.append(p); i += 1
    return out


FUNCS = {"toc": rule_toc, "refs": rule_refs, "lists": rule_lists, "formula": rule_formula, "heading": rule_heading, "broken": rule_broken}


def tidy(paras, fmt, upto="broken"):
    """문단 목록 → (정리된 문단 목록, 규칙별 문단 수 변화 기록)."""
    log = Counter()
    for name in RULES[:RULES.index(upto) + 1]:
        before = len(paras)
        paras = FUNCS[name](paras, fmt)
        log[f"P {name} 문단 줄어듦"] += before - len(paras)
    return paras, log
