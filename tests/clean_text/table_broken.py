"""깨진 표 조사(tests 전용, 2026-10-23): clean_text의 표 문단(kind table)에 깨짐 신호를 재고 데이터셋별로 세고 예시를 남긴다.

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/clean_text/table_broken.py

신호(표 문단 하나 기준; 표 행 = '|'로 시작하는 줄, 구분 줄 '| - | - |' 제외)
  S1 흩어진 머리 줄: 표 행 앞의 표 아닌 줄 중 3낱말 이하 짧은 줄이 SCATTER(3)개 이상 (첫 줄 캡션은 셈에서 뺌)
  S2 빈 칸: 전체 칸 중 빈 칸 비율 > EMPTY(0.5)
  S3 열 수 불일치: 가장 흔한 칸 수와 다른 행의 비율 > COLS(0.3)
  S4 칸 하나에 값 여러 개: 숫자 4개 이상 또는 '±' 2개 이상이 든 칸이 숫자 칸의 MULTI(30%) 이상
  S5 너무 긴 칸: 칸 하나가 LONG(400)글자 초과
  S6 행 부족: 데이터 행(머리·구분 줄 뺀) 1개 이하
출력: tests/clean_text/out/table_broken.md (데이터셋별 표 수·신호별 수·신호 조합 + 신호별 무작위 예시)
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common.models import make_token_counter  # noqa: E402

SCATTER, EMPTY, COLS, MULTI, LONG = 3, 0.5, 0.3, 0.3, 400
SEP = re.compile(r"^\s*\|(\s*:?-+:?\s*\|)+\s*$|^\s*\|(\s*-\s*\|)+\s*$")
NUM = re.compile(r"[-+−]?\d+(?:[.,]\d+)?")


def cells(row):
    r = row.strip()
    if r.startswith("|"):
        r = r[1:]
    if r.endswith("|"):
        r = r[:-1]
    return [c.strip() for c in r.split("|")]


def signals(t):
    lines = t.split("\n")
    rows = [l for l in lines if l.lstrip().startswith("|")]
    first_row = next((i for i, l in enumerate(lines) if l.lstrip().startswith("|")), len(lines))
    pre = [l for l in lines[1:first_row] if l.strip()]              # 첫 줄(캡션)은 뺌
    data = [r for r in rows if not SEP.match(r)]
    body = data[1:] if data else []                                  # 첫 행은 머리 줄로 봄
    C = [cells(r) for r in data]
    allc = [c for row in C for c in row]
    ncol = Counter(len(row) for row in C)
    common = ncol.most_common(1)[0][0] if ncol else 0
    numc = [c for c in allc if NUM.search(c)]
    multi = [c for c in numc if len(NUM.findall(c)) >= 4 or c.count("±") >= 2]
    s = {}
    s["S1 흩어진 머리 줄"] = sum(len(l.split()) <= 3 for l in pre) >= SCATTER
    s["S2 빈 칸 많음"] = bool(allc) and sum(not c for c in allc) / len(allc) > EMPTY
    s["S3 열 수 불일치"] = bool(C) and sum(len(row) != common for row in C) / len(C) > COLS
    s["S4 칸에 값 여러 개"] = len(numc) >= 4 and len(multi) / len(numc) >= MULTI
    s["S5 너무 긴 칸"] = any(len(c) > LONG for c in allc)
    s["S6 행 부족"] = len(body) <= 1
    return s


def main():
    count = make_token_counter()
    rnd = random.Random(0)
    L = ["# 깨진 표 조사", "", f"신호 기준: 흩어진 머리 줄 ≥{SCATTER} / 빈 칸 >{EMPTY:.0%} / 열 수 불일치 행 >{COLS:.0%} / 값 여러 개 칸 ≥{MULTI:.0%} / 칸 >{LONG}글자 / 데이터 행 ≤1", ""]
    for ds in ("vectara", "docx_corpus", "allganize"):
        recs = []
        for f in sorted((ROOT / "data" / "clean_text" / ds).glob("*.json")):
            if f.name.endswith(".dropped.json"):
                continue
            x = json.loads(f.read_text(encoding="utf-8"))
            P = x["text"].split("\n\n")
            for i, m in enumerate(x["paras"]):
                if m["kind"] == "table":
                    recs.append((f.stem, i, P[i], signals(P[i])))
        T = count([r[2] for r in recs])
        n = len(recs)
        any_ = [k for k, r in enumerate(recs) if any(r[3].values())]
        L += [f"## {ds}: 표 {n}개, 신호 하나 이상 {len(any_)}개 ({len(any_) / max(1, n):.0%})", "",
              "| 신호 | 표 수 | 비율 | 그중 700토큰 초과 |", "|---|---|---|---|"]
        for key in recs[0][3] if recs else []:
            ks = [k for k, r in enumerate(recs) if r[3][key]]
            L.append(f"| {key} | {len(ks)} | {len(ks) / max(1, n):.1%} | {sum(T[k] > 700 for k in ks)} |")
        combo = Counter(" + ".join(k.split()[0] for k, v in r[3].items() if v) for r in recs if any(r[3].values()))
        L += ["", "신호 조합 상위: " + ", ".join(f"{k} {v}" for k, v in combo.most_common(8)), ""]
        for key in recs[0][3] if recs else []:
            ks = [k for k, r in enumerate(recs) if r[3][key]]
            if not ks:
                continue
            L += [f"### {ds} · {key} 예시(무작위 {min(4, len(ks))}개)", ""]
            for k in rnd.sample(ks, min(4, len(ks))):
                d, i, t, s = recs[k]
                on = ", ".join(x.split()[0] for x, v in s.items() if v)
                L.append(f"- [{d[:40]} #{i}] {T[k]}토큰, 신호 {on}")
                L.append("  ```")
                for line in t.split("\n")[:7]:
                    L.append("  " + line[:220])
                L += ["  ```", ""]
    (HERE / "out" / "table_broken.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(l for l in L if not l.startswith("  ") and not l.startswith("- [")))


if __name__ == "__main__":
    main()
