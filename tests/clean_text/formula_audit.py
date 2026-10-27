"""오류 수식 새 기준(formula_fix.py) 결과 전수 점검(tests 전용, 2026-10-21).

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/clean_text/formula_audit.py

바뀐 수식 전부와 바뀌지 않은 수식 전부에 자동 점검을 돌리고, 걸린 것은 표본을 남긴다.
  (바뀐 수식은 실제 전처리처럼 Q6 이웃 본문 반복 자르기까지 적용한 뒤 점검)
  T(글자 층으로): 글자 층이 LaTeX의 보이는 내용보다 너무 짧음(글자 수 비 < SHORT) / 깨진 글자 / 앞·뒤 블록 본문과 8낱말 묶음 겹침
  S(띄어 쓴 구간 지움): 지운 비율 > TOO_MUCH / 남은 한 글자 연속(6개 이상) / 빈 \\intertext
  그대로: 700토큰 초과(모양별: 괄호 짝, 반복, \\text 비율) / 한 글자 연속 6~11개 / 괄호 짝 안 맞음(300토큰 이하)
출력: tests/clean_text/out/formula_audit.md
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
import formula_fix as F  # noqa: E402
from common.models import make_token_counter  # noqa: E402
from preprocess import stage2  # noqa: E402

SHORT, TOO_MUCH = 0.25, 0.5
SP6 = re.compile(r"(?:(?<=\s)|^)(?:[A-Za-z]\s+){6,}")
TEXTCMD = re.compile(r"\\(?:text|intertext|mathrm|mbox)\s*\{([^{}]*)\}")


def visible(latex):
    """LaTeX에서 보이는 글자만(명령·괄호·공백 뺌) - 글자 층과 길이 비교용."""
    t = re.sub(r"\\[a-zA-Z]+", "", latex)
    return re.sub(r"[\s{}_^&\\$]", "", t)


def grams(t, n=8):
    w = re.sub(r"[^a-z0-9 ]+", " ", t.lower()).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    count = make_token_counter()
    recs = []
    for ds in ("vectara", "allganize"):
        for p in sorted((ROOT / "data" / "meta_text_v2" / ds).glob("*.json")):
            x = json.loads(p.read_text(encoding="utf-8"))
            bl = x["blocks"] if isinstance(x, dict) else x
            for k, b in enumerate(bl):
                if b.get("type") == "formula" and b.get("text"):
                    prv = next((y.get("text", "") for y in reversed(bl[:k]) if y.get("type") != "formula"), "")
                    nxt = next((y.get("text", "") for y in bl[k + 1:] if y.get("type") != "formula"), "")
                    recs.append((ds, p.stem, b, prv, nxt))
    olds = [stage2.formula_text(b, Counter()) for _, _, b, _, _ in recs]
    T0 = np.array(count(olds))
    out = [F.new_text(b, o, t, with_s=False) for (_, _, b, _, _), o, t in zip(recs, olds, T0)]   # S는 Q6 뒤에(아래)
    # 실제 전처리처럼 바뀐 수식 글 뒤에 Q6(이웃 본문 반복 자르기)를 적용해서 본다
    news = [stage2.trim_formula(n, prv, nxt, Counter()) if w else n for (_, _, _, prv, nxt), (n, w) in zip(recs, out)]
    whys = [w for _, w in out]
    reverted = 0
    for i, (o, n, w) in enumerate(zip(olds, news, whys)):           # T: Q6 뒤 너무 짧아지면 LaTeX 유지(formula_fix.py 설명)
        if w and w.startswith("T") and len(re.sub(r"\s", "", n)) < SHORT * len(visible(o[3:-3])):
            news[i], whys[i] = o, None
            reverted += 1
    for i, ((_, _, _, prv, nxt), o) in enumerate(zip(recs, olds)):     # S: Q6 뒤 남은 띄어 쓴 \intertext 지움
        base_txt = news[i] if whys[i] else stage2.trim_formula(o, prv, nxt, Counter())
        s = F.s_clean(base_txt)
        if s:
            olds_s = whys[i]
            news[i], whys[i] = s, (olds_s + " + S" if olds_s else "S 글자 띄어쓰기 → 띄어 쓴 \\intertext 지움(Q6 뒤)")
    T1 = np.array(count(news))
    # 최종 효과: 실제 전처리 순서(지금 = Q3 → Q6, 새 = Q3 → 새 검사 → Q6 → 되돌림)로 비교
    cur = [stage2.trim_formula(o, prv, nxt, Counter()) for (_, _, _, prv, nxt), o in zip(recs, olds)]
    fin = [n if w else c for n, w, c in zip(news, whys, cur)]
    TC, TF = np.array(count(cur)), np.array(count(fin))
    is_v = np.array([r[0] == "vectara" for r in recs])
    per_doc = Counter(recs[i][1] for i, w in enumerate(whys) if w)
    E = ["## 최종 효과(실제 전처리 순서: 지금 = Q3 → Q6, 새 = Q3 → 새 검사 → Q6 → 되돌림)", "",
         "| | 지금 전처리 | 새 기준 |", "|---|---|---|",
         f"| 수식 토큰 합 | {TC.sum():,} | {TF.sum():,} ({TF.sum() / TC.sum():.1%}) |",
         f"| 700토큰 넘는 수식 | {(TC > 700).sum():,} | {(TF > 700).sum():,} |",
         f"| 300토큰 넘는 수식 | {(TC > 300).sum():,} | {(TF > 300).sum():,} |",
         f"| 수식 토큰 중앙값 / 90% / 최대 | {int(np.median(TC))} / {int(np.percentile(TC, 90))} / {TC.max()} | {int(np.median(TF))} / {int(np.percentile(TF, 90))} / {TF.max()} |",
         f"| 바뀐 수식 / 문서 | – | {sum(1 for w in whys if w):,} / {len(per_doc)}편 (vectara 문서당 최대 {max(per_doc.values()) if per_doc else 0}개, 상위 {per_doc.most_common(3)}) |", ""]
    # 이상 검사(새 기준 결과 전부)
    odd = {}
    for i, (c, f, w) in enumerate(zip(cur, fin, whys)):
        if not w:
            continue
        core = f[3:-3] if f.startswith("$$ ") and f.endswith(" $$") else f
        if len(re.sub(r"\s", "", core)) < 3:
            odd.setdefault("빈 수식(보이는 글자 3개 미만)", []).append(i)
        if f.count("$$") % 2:
            odd.setdefault("'$$' 개수가 홀수", []).append(i)
        if f.startswith("$$ ") and core.count("$$"):
            odd.setdefault("'$$ … $$' 안에 또 '$$'", []).append(i)
        if w.startswith("S"):
            if not F.unbalanced(c[3:-3] if c.startswith("$$ ") else c) and F.unbalanced(core):
                odd.setdefault("S: 지운 뒤 새로 괄호 짝이 깨짐", []).append(i)
            if c.count("\\begin") == c.count("\\end") and core.count("\\begin") != core.count("\\end"):
                odd.setdefault("S: 지운 뒤 새로 \\begin/\\end 짝이 깨짐", []).append(i)
        if TF[i] > TC[i] * 1.1 + 5:
            odd.setdefault("바꾼 뒤 토큰이 10% 넘게 늘어남", []).append(i)
        if "\n\n" in f:
            odd.setdefault("빈 줄(문단 경계)이 들어감", []).append(i)
    E += ["## 이상 검사(바뀐 수식 전부)", "", "| 검사 | 개수 |", "|---|---|"] + \
         [f"| {k} | {len(v)} |" for k, v in odd.items()] + (["| (걸린 것 없음) | 0 |"] if not odd else []) + [""]
    rnd0 = random.Random(1)
    for k, v in odd.items():
        E += [f"### {k} (무작위 {min(5, len(v))}개)", ""]
        for i in rnd0.sample(v, min(5, len(v))):
            E += [f"- [{recs[i][1]} {recs[i][2].get('id')}] {whys[i]} {TC[i]}→{TF[i]}토큰",
                  f"  - 전: `{cur[i][:200]}` … `{cur[i][-90:]}`", f"  - 후: `{fin[i][:200]}` … `{fin[i][-90:]}`"]
        E.append("")
    flags = {}

    def flag(name, i, note=""):
        flags.setdefault(name, []).append((i, note))

    for i, ((ds, d, b, prv, nxt), o, n, w) in enumerate(zip(recs, olds, news, whys)):
        if w and w.startswith("T"):
            r = len(re.sub(r"\s", "", n)) / max(1, len(visible(o[3:-3])))
            if r < SHORT:
                flag("T: 글자 층이 너무 짧음(LaTeX 보이는 글자의 25% 미만)", i, f"비 {r:.2f}")
            bad = sum(ch in F.BAD for ch in n) / max(1, len(n))
            if bad > 0.01:
                flag("T: 글자 층에 깨진 글자 1% 넘음", i, f"{bad:.1%}")
            g = grams(n)
            if g and (len(g & grams(prv)) + len(g & grams(nxt))) >= 3:
                flag("T: 글자 층이 앞·뒤 본문과 겹침(8낱말 묶음 3개 이상)", i, f"{len(g & grams(prv))}+{len(g & grams(nxt))}")
        elif w and w.startswith("S"):
            cut = 1 - len(n.split()) / max(1, len(o.split()))
            if cut > TOO_MUCH:
                flag("S: 수식의 절반 넘게 지움", i, f"{cut:.0%}")
            if SP6.search(n[3:-3] + " "):
                flag("S: 지운 뒤에도 한 글자 연속 6개 이상 남음", i)
            if re.search(r"\\intertext\s*\{\s*\}", n):
                flag("S: 빈 \\intertext 남음", i)
        elif o.startswith("$$ "):
            inner = o[3:-3]
            if T0[i] > 700:
                txt = sum(len(m.group(1)) for m in TEXTCMD.finditer(inner)) / max(1, len(inner))
                shape = "괄호 짝 안 맞음" if F.unbalanced(inner) else ("반복 > 0.5" if F.rep6(inner) > 0.5 else ("\\text 글 30% 넘음" if txt > 0.3 else "정상 모양"))
                flag(f"그대로(700토큰 초과): {shape}", i)
            if SP6.search(inner + " "):
                flag("그대로: 한 글자 연속 6~11개", i)
            if F.unbalanced(inner) and T0[i] <= 300:
                flag("그대로: 괄호 짝 안 맞음(300토큰 이하)", i)
    ch = [i for i, w in enumerate(whys) if w]
    L = ["# 오류 수식 새 기준 전수 점검", "",
         f"수식 블록 {len(recs):,}개. 바뀜 {len(ch):,}개({Counter(w.split(' →')[0] for w in whys if w)}). T 중 Q6 뒤 너무 짧아 LaTeX 유지 {reverted}개. 자동 점검에 걸린 수:", "",
         "| 점검 | 개수 | 대상 중 비율 |", "|---|---|---|"]
    base = {"T": sum(1 for w in whys if w and w.startswith("T")), "S": sum(1 for w in whys if w and w.startswith("S")),
            "그": sum(1 for w, o in zip(whys, olds) if not w and o.startswith("$$ "))}
    for k, v in sorted(flags.items()):
        L.append(f"| {k} | {len(v):,} | {len(v) / base[k[0]]:.1%} |")
    L = L[:4] + E + L[4:]
    rnd = random.Random(0)
    for k, v in sorted(flags.items()):
        L += ["", f"## {k} (무작위 {min(8, len(v))}개)", ""]
        for i, note in rnd.sample(v, min(8, len(v))):
            ds, d, b, prv, nxt = recs[i]
            L.append(f"- [{d} {b.get('id')}] {note} {T0[i]}→{T1[i]}토큰")
            L.append(f"  - 전: `{olds[i][:200]}` … `{olds[i][-100:]}`")
            if news[i] != olds[i]:
                L.append(f"  - 후: `{news[i][:200]}` … `{news[i][-100:]}`")
            if k.startswith("T: 글자 층이 앞"):
                L.append(f"  - 앞 블록 끝: `{prv[-120:]}` / 뒤 블록 시작: `{nxt[:120]}`")
    (HERE / "out" / "formula_audit.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:4] + E))


if __name__ == "__main__":
    main()
