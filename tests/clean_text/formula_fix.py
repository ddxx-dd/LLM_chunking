"""오류 수식 판정 기준 시험(tests 전용, 2026-10-21): 지금 stage2.formula_text(Q3) vs 새 기준을 같은 수식 블록에 적용해 비교.

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/clean_text/formula_fix.py

지금 Q3: 원본 글자 층(text_layer)이 있고 (3~40낱말 단위가 4번 이상 반복 or LaTeX 길이 > 6 × 글자 층 + 80) → 글자 층.
놓치는 것(2026-10-21 거대 문단 조사): 짧은 반복('( - ( - ( -'), 출력 길이 한계로 잘린 수식(괄호 짝이 안 맞음).
새 기준(지금 Q3에 더해서, LaTeX로 남은 수식에만):
  R  고리: 1~40낱말 조각이 4번 이상 연달아 반복되는 구간이 LOOP_MIN(40)낱말 이상이고 수식의 LOOP_SHARE(40%) 이상
     (1차 시험의 '6낱말 묶음 반복 비율 > 0.5'는 띄어 쓴 LaTeX의 정상 반복 구조 - 합 기호 첨자, 행렬 - 를 잘못 잡아서 바꿈)
  S  글자 띄어쓰기: 한 글자 낱말(영문)이 SPACED(12)개 이상 연달아(본문을 수식으로 잘못 읽음: 'g u i n g u i n g e s a m a …')
  T  잘림: 이스케이프 안 된 '{' '}' 수 또는 '\\left' '\\right' 수가 다르고, bge-m3 토큰 > CUT_TOK(300)
  → R·T: 글자 층이 쓸 만하면 글자 층, 아니면 R은 반복 단위를 한 번만 남김(1~40낱말 단위가 4번 이상 이어지면 첫 번만), T는 그대로.
  → S: 수식은 그대로 두고 띄어 쓴 글자 구간만 지움(대부분 정상 LaTeX 뒤에 붙은 '\intertext { b h o m o … }' 꼬리; 2차 시험 표본에서 확인).
     지우는 범위는 \intertext{…} 하나 통째로(띄어 쓴 구간이 있거나 내용 낱말의 절반 이상이 한 글자일 때). 그 밖의 띄어 쓴 글자는 그대로
     (전수 점검 formula_audit.py: 구간만 지우면 '\,' '$ x $'로 끊긴 조각이 43% 남았고, \intertext 밖까지 지우면 첨자 속 변수 이름
      'v _ { o x y g e n U p t a k e }'이나 본문 \text를 지워 수식이 '$$ in $$'가 되는 경우가 있었다).
  → T에서 글자 층에 Q6(이웃 본문 반복 자르기)을 적용한 뒤 LaTeX 보이는 글자의 KEEP(25%)보다 짧아지면 LaTeX를 그대로 둠
     (글자 층은 첨자가 낱말처럼 펼쳐져 Q6이 수식까지 잘라 'f 2='만 남는 경우가 있었다 - stage2 반영 때 Q6 뒤에 같은 검사를 둔다).
글자 층이 쓸 만하지 않음 = 글꼴 인코딩이 깨진 글자(¨ ´ ă ÿ ˆ ˘ ˙ ‰ “ \` 등)가 글자 층의 BADCH(3%) 이상(2차 표본 '0 “ 1tΛi¨pt´qă8u').
출력: tests/clean_text/out/formula_fix.md (집계 + 바뀌는 수식 무작위 40개 + 경계 근처 정상 판정 수식 15개)
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common.models import make_token_counter  # noqa: E402
from preprocess import stage2  # noqa: E402

REP, CUT_TOK = 0.5, 300         # REP는 경계 근처 표본 출력에만 씀
LOOP_MIN, LOOP_SHARE, SPACED, BADCH = 40, 0.4, 12, 0.03
BAD = set("¨´ăÿˆ˘˙‰“`˝ˇ")
SPACED_RUN = re.compile(r"(?:(?<=\s)|^)(?:[A-Za-z]\s+){%d,}" % SPACED)
RUN = re.compile(r"((?:\S+\s+){1,40}?)\1{3,}")
INTERTEXT = re.compile(r"\\intertext\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}")


def spaced_box(t):
    """\\intertext{…}·\\text{…} 내용 낱말('\\,' '\\ ' '$'도 구분자)의 절반 이상이 한 글자 영문이면 띄어 쓴 쓰레기 글로 본다."""
    w = [x for x in re.split(r"\s+|\\[,;: ]|\$", t) if x]
    return len(w) >= 6 and sum(len(x) == 1 and x.isalpha() for x in w) >= 0.5 * len(w)


def rep6(t):
    w = t.split()
    g = [" ".join(w[i:i + 6]) for i in range(len(w) - 5)]
    return 1 - len(set(g)) / len(g) if g else 0.0


def unbalanced(t):
    o = len(re.findall(r"(?<!\\)\{", t)); cl = len(re.findall(r"(?<!\\)\}", t))
    return o != cl or t.count("\\left") != t.count("\\right")


def s_clean(f):
    """S를 Q6(이웃 본문 반복 자르기) 뒤에 적용: '$$ … $$' 안의 띄어 쓴 \\intertext{…}만 지운 글, 바뀐 것이 없으면 None.
    (Q6 앞에 지우면 Q6의 판단이 달라져 정상 수식 끝 '\\right \\} . \\\\'을 잘라 먹는 경우가 4건 있었고 - formula_audit 이상 검사 -
     지금 전처리에서도 Q6이 쓰레기 꼬리 상당수를 이미 자르고 있어서, Q6 뒤에 남은 것만 지운다)"""
    if not (f.startswith("$$ ") and f.endswith(" $$")):
        return None
    inner = f[3:-3]
    if not SPACED_RUN.search(inner + " "):
        return None
    t = INTERTEXT.sub(lambda m: " " if spaced_box(m.group(1)) or SPACED_RUN.search(m.group(1) + " ") else m.group(0), inner + " ")
    t = " ".join(t.split())
    return "$$ " + t + " $$" if t != " ".join(inner.split()) else None


def new_text(b, old, ntok, with_s=True):
    """old가 LaTeX('$$ … $$')로 남았을 때만 새 기준을 본다. 반환: (새 글, 이유 or None).
    with_s=False: S는 여기서 보지 않는다(실제 순서는 Q3 → R·T → Q6 → T 되돌림 → S, s_clean)."""
    if not (old.startswith("$$ ") and old.endswith(" $$")):
        return old, None
    inner = old[3:-3]
    n = len(inner.split())
    loop = max((len(m.group(0).split()) for m in RUN.finditer(inner + " ")), default=0)
    if loop >= LOOP_MIN and loop >= LOOP_SHARE * n:
        why = "R 고리"
    elif with_s and SPACED_RUN.search(inner + " "):
        why = "S 글자 띄어쓰기"
    elif unbalanced(inner) and ntok > CUT_TOK:
        why = "T 잘림"
    else:
        why = None
    if why is None:
        return old, None
    if why == "S 글자 띄어쓰기":
        t = INTERTEXT.sub(lambda m: " " if spaced_box(m.group(1)) or SPACED_RUN.search(m.group(1) + " ") else m.group(0), inner + " ")
        if " ".join(t.split()) == " ".join(inner.split()):
            return old, None                       # \intertext 밖의 띄어 쓴 글자(첨자 속 변수 이름 등)는 진짜 낱말일 수 있어 그대로
        return "$$ " + " ".join(t.split()) + " $$", why + " → 띄어 쓴 \\intertext 지움"
    layer = b.get("text_layer") or ""
    if layer and sum(ch in BAD for ch in layer) < BADCH * len(layer):
        return layer, why + " → 글자 층"
    if why == "R 고리":
        return "$$ " + RUN.sub(r"\1", inner + " ").strip() + " $$", why + " → 반복 줄임"
    return old, None


def main():
    count = make_token_counter()
    recs = []
    for ds in ("vectara", "allganize"):
        for p in sorted((ROOT / "data" / "meta_text_v2" / ds).glob("*.json")):
            x = json.loads(p.read_text(encoding="utf-8"))
            for b in (x["blocks"] if isinstance(x, dict) else x):
                if b.get("type") == "formula" and b.get("text"):
                    recs.append((ds, p.stem, b))
    olds = [stage2.formula_text(b, Counter()) for _, _, b in recs]
    T0 = np.array(count(olds))
    news, whys = zip(*[new_text(b, o, t) for (_, _, b), o, t in zip(recs, olds, T0)])
    T1 = np.array(count(list(news)))
    q3 = np.array([not o.startswith("$$ ") for o in olds])
    ch = np.array([w is not None for w in whys])
    L = ["# 오류 수식 판정 기준 시험", "", f"수식 블록 {len(recs):,}개(vectara·allganize meta_text_v2). 기준: 반복 > {REP}, 잘림 = 괄호 짝 안 맞음 & {CUT_TOK}토큰 초과.", "",
         "| 데이터셋 | 수식 | 지금 Q3가 글자 층으로 | 새 기준이 더 바꿈 | 이유 | 문서 |", "|---|---|---|---|---|---|"]
    for ds in ("vectara", "allganize"):
        m = np.array([r[0] == ds for r in recs])
        why = Counter(w for w, k in zip(whys, m) if k and w)
        L.append(f"| {ds} | {m.sum():,} | {q3[m].sum():,} ({q3[m].mean():.1%}) | {ch[m].sum():,} ({ch[m].mean():.1%}) | "
                 f"{', '.join(f'{k} {v}' for k, v in why.most_common())} | {len({r[1] for r, k in zip(recs, m & ch) if k})} |")
    L += ["", "| | 지금 | 새 기준 |", "|---|---|---|",
          f"| 수식 토큰 합 | {T0.sum():,} | {T1.sum():,} ({T1.sum() / T0.sum():.0%}) |",
          f"| 700토큰 넘는 수식 | {(T0 > 700).sum():,} | {(T1 > 700).sum():,} |",
          f"| 300토큰 넘는 수식 | {(T0 > 300).sum():,} | {(T1 > 300).sum():,} |",
          f"| 바뀐 수식 토큰(중앙값) | {int(np.median(T0[ch])) if ch.any() else 0} | {int(np.median(T1[ch])) if ch.any() else 0} |", ""]
    left = [(r, o, t) for r, o, t, k in zip(recs, olds, T0, ch) if not k and o.startswith("$$ ") and t > 700]
    L += [f"새 기준 뒤에도 LaTeX로 남은 700토큰 초과 수식: {len(left)}개(반복 비율 중앙값 {np.median([rep6(o) for _, o, _ in left]) if left else 0:.2f})", ""]
    rnd = random.Random(0)
    idx = [i for i in range(len(recs)) if ch[i]]
    L += ["## 바뀌는 수식 무작위 40개", ""]
    for i in rnd.sample(idx, min(40, len(idx))):
        ds, d, b = recs[i]
        L += [f"- [{ds} {d} {b.get('id')}] {whys[i]} | {T0[i]} → {T1[i]}토큰 | 반복 {rep6(olds[i]):.2f}",
              f"  - 전: `{olds[i][:220]}` … `{olds[i][-80:]}`", f"  - 후: `{news[i][:220]}`"]
    near = [i for i in range(len(recs)) if not ch[i] and olds[i].startswith("$$ ") and rep6(olds[i]) > REP]
    L += ["", f"## 1차 기준(반복 > 0.5)에는 걸리지만 새 기준은 그대로 두는 수식 {len(near)}개 중 무작위 15개", ""]
    for i in rnd.sample(near, min(15, len(near))):
        ds, d, b = recs[i]
        L.append(f"- [{ds} {d} {b.get('id')}] {T0[i]}토큰 반복 {rep6(olds[i]):.2f}: `{olds[i][:200]}`")
    (Path(__file__).parent / "out" / "formula_fix.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:22]))


if __name__ == "__main__":
    main()
