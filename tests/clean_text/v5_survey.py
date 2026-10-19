"""clean_text v5 문단 정리 전 조사(tests 전용, 읽기만 함): 목차 줄 묶음 후보, 참고문헌으로 표시되지 않은 참고문헌 모양 문단 묶음.

  PYTHONPATH=src .venv311/bin/python tests/clean_text/v5_survey.py

목차 후보: 짧은 줄(15낱말 이하, 줄바꿈 없음)이 쪽 번호 꼴로 끝나고(' 12', ' 3-117', ' 1-6', '.... 12') 3개 이상 연속,
  끝 숫자가 줄어들지 않음(목차 쪽 번호), doi·http·참고문헌 모양·'Eq.' 줄 제외
참고문헌 모양: stage2 R2와 같은 판정(연도 + 참고문헌 표지 2개 이상, 120낱말 이하는 항목 번호·저자로 시작, 넘으면 50낱말당 연도 1개 이상)
  R2 저자 판정(REF_START)은 성이 앞('Lee, T. D.')인 양식만 봄 → 이니셜이 앞인 양식('T. D. Lee and …')을 따로 세어 R2가 놓친 몫을 본다
  끼어듦(순서 문제): 묶음 바로 뒤 1~2문단 안에 소문자로 시작하는 본문 조각이 있거나, 묶음 바로 앞 문단이 문장 끝 없이 끝남
출력: tests/clean_text/out/v5_survey.md
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chunking.sentences import paragraphs  # noqa: E402
from preprocess.stage2 import REF_START, REF_STRICT, YEAR  # noqa: E402

PAGE_END = re.compile(r"(\.{3,}|…+|\s)\s*((?:[A-Z]{0,3}-?\d{1,3}-)?(\d{1,4}))\s*$")
INITIALS = re.compile(r"^\s*(\[\d+\]\s*|\d+\.\s+)?(?:[A-Z]\.\s?(?:-?[A-Z]\.\s?)*[A-Z][A-Za-z'\-]+)(?:,|\s+and\s|\s+et al)")
SENT_END = re.compile(r"[.!?:;)\]\"'”’…]\s*$")


def toc_num(p):
    if "\n" in p or len(p.split()) > 15 or re.search(r"doi|http|Eq\.|\(\d{3}\)", p) or is_refy(p):
        return None
    m = PAGE_END.search(p)
    return int(m.group(3)) if m and not re.search(r"(19|20)\d\d\)?\.?\s*$", p) else None


def toc_flags(ps):
    nums = [toc_num(p) for p in ps]
    flags = [n is not None for n in nums]
    for s, e in runs(flags):                     # 끝 숫자가 줄어들면 목차가 아님
        v = nums[s:e]
        if any(b < a for a, b in zip(v, v[1:])):
            for i in range(s, e):
                flags[i] = False
    return flags


def is_refy(p, initials=False):
    if not (YEAR.search(p) and len(REF_STRICT.findall(p)) >= 2):
        return False
    n = len(p.split())
    if n > 120:
        return len(YEAR.findall(p)) >= n / 50
    start = REF_START.match(p) or re.match(r"^\s*(\[\d+\]|\d+\.\s)", p)
    return bool(start or (initials and INITIALS.match(p)))


def runs(flags, min_len=3):
    out, s = [], None
    for i, f in enumerate(flags + [False]):
        if f and s is None:
            s = i
        elif not f and s is not None:
            if i - s >= min_len:
                out.append((s, i))
            s = None
    return out


def main():
    rnd = random.Random(0)
    L = ["# clean_text v5 조사: 목차 줄, 참고문헌으로 표시되지 않은 참고문헌 모양 문단", ""]
    ex_toc, ex_ref = defaultdict(list), defaultdict(list)
    for ds in ("vectara", "allganize", "docx_corpus"):
        files = [p for p in sorted((ROOT / "data" / "clean_text" / ds).glob("*.json")) if not p.name.endswith(".dropped.json")]
        c = Counter()
        for f in files:
            r = json.loads(f.read_text(encoding="utf-8"))
            ps, kd = paragraphs(r["text"]), [x["kind"] for x in r["paras"]]
            c["문서"] += 1; c["문단"] += len(ps)
            tr = runs(toc_flags(ps))
            if tr:
                c["목차 묶음 있는 문서"] += 1
            for s, e in tr:
                c["목차 묶음"] += 1; c["목차 줄"] += e - s
                ex_toc[ds].append((r["doc_id"], s / len(ps), ps[max(0, s - 1)][:70], ps[s:min(e, s + 4)], e - s, ps[e][:70] if e < len(ps) else ""))
            for tag, ini in (("지금 R2 판정", False), ("+ 이니셜 앞 양식", True)):
                rr = runs([is_refy(p, ini) and k != "references" for p, k in zip(ps, kd)])
                if rr:
                    c[f"[{tag}] 묶음 있는 문서"] += 1
                for s, e in rr:
                    nxt = [i for i in range(e, min(e + 2, len(ps))) if kd[i] == "para" and re.match(r"^[a-z]", ps[i])]
                    inter = bool(nxt) or (s > 0 and kd[s - 1] == "para" and not SENT_END.search(ps[s - 1]))
                    c[f"[{tag}] 묶음"] += 1; c[f"[{tag}] 문단 수"] += e - s; c[f"[{tag}] 끼어듦"] += inter
                    if ini:
                        ex_ref[ds].append((r["doc_id"], s / len(ps), Counter(kd[s:e]).most_common(2), ps[s][:90], inter,
                                           ps[nxt[0]][:90] if nxt else (ps[s - 1][-90:] if inter else ""), e - s))
        L += [f"## {ds}", "", "| 항목 | 값 |", "|---|---|"] + [f"| {k} | {v:,} |" for k, v in c.items()] + [""]
        if ex_toc[ds]:
            L += ["### 목차 묶음 예시(무작위 4)", ""]
            for d, pos, before, items, n, after in rnd.sample(ex_toc[ds], min(4, len(ex_toc[ds]))):
                L.append(f"- [{d[:24]}] 문서 위치 {pos:.0%}, {n}줄. 앞: {before!r}")
                L += [f"    - {x[:80]!r}" for x in items] + [f"    - 뒤: {after!r}"]
            L.append("")
        if ex_ref[ds]:
            L += ["### 참고문헌 모양 묶음 예시(무작위 5, 끼어듦 우선)", ""]
            pool = sorted(ex_ref[ds], key=lambda x: -x[4])
            for d, pos, kinds, first, inter, nxt, nb in pool[:3] + rnd.sample(pool[3:], min(2, len(pool[3:]))):
                L.append(f"- [{d[:24]}] 문서 위치 {pos:.0%}, 종류 {kinds}, 첫 항목 {first!r}")
                L.append(f"    - {nb}문단, 끼어듦: {inter}" + (f" (조각: {nxt!r})" if inter else ""))
            L.append("")
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "v5_survey.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
