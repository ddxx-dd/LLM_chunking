"""LongBench 문장 나누기 검증(tests 전용, 구현 전): 공백 보정 정규식 + 현재 split(nltk + 학술 약어 + 영어 보정)을 세 데이터셋 표본에 적용한다.

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/sent_split/lb_check.py            # 표본·후보 만들기
  PYTHONPATH=src .venv311/bin/python tests/sent_split/lb_check.py --score                     # 판정(out/lb_labels.json)으로 채점

표본(시드 7): HotpotQA 줄 60(Passage 머리글 뺀 원문 줄, 25낱말 이상), NarrativeQA 책 문단 60(빈 줄 블록, 대본 제외),
  GovReport 150낱말 창 40. 모두 원문에 공백 보정만 한 글.
후보 경계: nltk 경계 ∪ 느슨한 정규식 경계('.!?' 뒤 공백 + 대문자·숫자·따옴표·괄호). 후보마다 사람이 판정(1 경계 / 0 아님 / null 애매).
출력: out/lb_samples.jsonl, out/lb_candidates.jsonl, out/lb_summary.md
"""
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from compare import boundaries  # noqa: E402
from preprocess.sentences import split  # noqa: E402

OUT = HERE / "out"
LB = ROOT / "data" / "raw" / "longbench"
# 공백 보정: [소문자·숫자·닫는 괄호·따옴표].[대문자][소문자] → 점 뒤에 공백. 짧은 대문자 조각을 점으로 이은 약어(D.Ed.Min, Prof.Dr, H.Con.Res)는 제외
MISSING = re.compile(r"(?<=[a-z0-9)\"'”’])\.(?=[A-Z][a-z])")
DOTTED = re.compile(r"(?:^|[\s(\[])(?:[A-Z][A-Za-z]{0,3}\.)+[A-Z][A-Za-z]{0,3}$")
LOOSE = re.compile(r"[.!?…][\"'”’)\]]*\s+(?=[A-Z0-9\"'“‘(\[])")


def fix_space(t):
    out, last, n = [], 0, 0
    for m in MISSING.finditer(t):
        if DOTTED.search(t[max(0, m.start() - 30):m.start()]):
            continue
        out.append(t[last:m.end()] + " ")
        last, n = m.end(), n + 1
    return "".join(out) + t[last:], n


def samples():
    rnd = random.Random(7)
    S = []
    hp = [json.loads(x) for x in (LB / "hotpotqa.jsonl").read_text(encoding="utf-8").splitlines()]
    lines = [(r["_id"][:10], ln.strip()) for r in hp for ln in r["context"].split("\n")
             if ln.strip() and not re.match(r"Passage \d+:$", ln.strip()) and len(ln.split()) >= 25]
    S += [("hotpotqa", d, t) for d, t in rnd.sample(lines, 60)]
    nq = [json.loads(x) for x in (LB / "narrativeqa.jsonl").read_text(encoding="utf-8").splitlines()]
    books = [c for c in dict.fromkeys(r["context"] for r in nq) if len(re.findall(r"</?[a-zA-Z]+[^>]*>", c)) < 5]
    paras = [(f"book{k:02d}", " ".join(b.split())) for k, c in enumerate(books) for b in re.split(r"\n\s*\n", c) if len(b.split()) >= 25]
    S += [("narrativeqa", d, t) for d, t in rnd.sample(paras, 60)]
    gov = [json.loads(x) for x in (LB / "gov_report.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in rnd.sample(gov, 40):
        w = r["context"].split()
        s = rnd.randrange(0, max(1, len(w) - 160))
        S.append(("gov_report", r["_id"][:10], " ".join(w[s:s + 150])))
    return S


def build():
    S, cands, nfix = samples(), [], 0
    rows = []
    for i, (src, d, raw) in enumerate(S):
        t, n = fix_space(raw)
        nfix += n
        sents = split(t, "en")
        B = boundaries(t, sents)
        L = {m.end() for m in LOOSE.finditer(t)}
        rows.append({"i": i, "src": src, "doc": d, "text": t, "n_fix": n, "sents": sents})
        for b in sorted(B | L):
            cands.append({"id": len(cands), "i": i, "src": src, "pos": b, "nltk": b in B, "loose": b in L,
                          "context": t[max(0, b - 70):b] + " ‖ " + t[b:b + 45]})
    with open(OUT / "lb_samples.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT / "lb_candidates.jsonl", "w", encoding="utf-8") as fh:
        for c in cands:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    for src in ("hotpotqa", "narrativeqa", "gov_report"):
        cs = [c for c in cands if c["src"] == src]
        print(src, "후보", len(cs), "둘 다", sum(c["nltk"] and c["loose"] for c in cs), "nltk만", sum(c["nltk"] and not c["loose"] for c in cs),
              "정규식만", sum(c["loose"] and not c["nltk"] for c in cs), "| 공백 보정", sum(r["n_fix"] for r in rows if r["src"] == src))


def score():
    cands = [json.loads(x) for x in (OUT / "lb_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    lab = {int(k): v for k, v in json.loads((OUT / "lb_labels.json").read_text(encoding="utf-8")).items()}
    L = ["# LongBench 문장 나누기 검증(공백 보정 + nltk)", "", "| 데이터셋 | 판정 후보 | 진짜 경계 | 애매 | 방법 | 자른 곳 | 정밀도 | 재현율 | F1 |", "|---|---|---|---|---|---|---|---|---|"]
    for src in ("hotpotqa", "narrativeqa", "gov_report"):
        cs = [c for c in cands if c["src"] == src and lab.get(c["id"]) is not None]
        true = [c for c in cs if lab[c["id"]] == 1]
        amb = sum(1 for c in cands if c["src"] == src and lab.get(c["id"], 0) is None)
        for m in ("nltk", "loose"):
            cut = [c for c in cs if c[m]]
            tp = sum(lab[c["id"]] == 1 for c in cut)
            p, r = tp / max(1, len(cut)), tp / max(1, len(true))
            L.append(f"| {src} | {len(cs)} | {len(true)} | {amb} | {'공백 보정+nltk' if m == 'nltk' else '(참고) 느슨한 정규식'} | {len(cut)} | {p:.1%} | {r:.1%} | {2 * p * r / max(1e-9, p + r):.1%} |")
    (OUT / "lb_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    score() if "--score" in sys.argv else build()
