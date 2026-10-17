"""문장 나누기 검증(tests 전용): 영어 SaT·spaCy, 한국어 SaT·Kiwi+구조 표시·spaCy를 새 표본으로 비교한다.

  source env.sh && .venv311/bin/python tests/sent_split/validate.py            # 후보 경계 만들기 → out/v2_candidates.jsonl
  .venv311/bin/python tests/sent_split/validate.py --score                     # 판정(out/v2_labels.json)으로 정밀도·재현율

표본(시드 7, compare.py 표본과 겹치지 않게): 영어 80문단(vectara 30, docx 20, GovReport 15(2026-10-17 원본 삭제로 이제 뽑지 않음), NarrativeQA 15), 한국어 allganize 60(개조식 30)
후보 경계: 비교 대상 + 보조 도구(영어 regex·pysbd·nltk, 한국어 regex·Kiwi 기본)가 자른 위치의 합집합. 사람이 후보마다 진짜 경계인지 판정한다.
지표: 정밀도 = 자른 곳 중 진짜 경계, 재현율 = 진짜 경계 중 자른 곳(진짜 경계는 후보 안에서 판정된 것).
"""
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare import LB, boundaries, load_units  # noqa: E402

OUT = HERE / "out"
MARK = re.compile(r"\s(?=(?:[□■▪◦○●ㅇ※▣◆◇]\s?|\*\s(?![의을를이가은는에와과로도만]\s)|[①-⑳]|[가-하]\.\s|\d{1,2}\)\s)\S)")
METHODS = {"en": ["sat", "spacy_en"], "ko": ["sat", "kiwi_plus", "spacy_ko"]}


def samples():
    old = {json.loads(x)["text"] for x in (OUT / "samples.jsonl").read_text(encoding="utf-8").splitlines()}
    rnd = random.Random(7)
    pick = lambda xs, n: rnd.sample([x for x in xs if x[1] not in old], n)
    S = []
    v = [x for x in load_units("vectara") if 30 <= len(x[1].split()) <= 200]
    S += [("en", "vectara", d, t) for d, t in pick(v, 30)]
    dx = [x for x in load_units("docx_corpus") if 20 <= len(x[1].split()) <= 200]
    S += [("en", "docx", d, t) for d, t in pick(dx, 20)]
    nq = [json.loads(x) for x in (LB / "narrativeqa.jsonl").read_text(encoding="utf-8").splitlines()]
    paras = [(r["_id"][:12], " ".join(p.split())) for r in nq for p in re.split(r"\n\s*\n", r["context"]) if 40 <= len(p.split()) <= 150]
    S += [("en", "narrativeqa", d, t) for d, t in pick(paras, 15)]
    ko = [x for x in load_units("allganize") if 15 <= len(x[1].split()) <= 120]
    lst = [x for x in ko if len(re.findall(r"(함|음|임|됨)\s*($|[,.\s])", x[1])) >= 2 or re.search(r"[○□■▪◦·ㅇ]\s", x[1])]
    rest = [x for x in ko if x not in lst]
    S += [("ko", "allganize", d, t) for d, t in pick(lst, 30) + pick(rest, 30)]
    return S


def splitters(lang):
    import spacy
    from wtpsplit import SaT
    sat = SaT("sat-3l-sm")
    sat.half().to("cuda")
    T = {"sat": lambda ts: [list(x) for x in sat.split(ts)], "regex": lambda ts: [re.split(r"(?<=[.?!])\s+", t) for t in ts]}
    if lang == "en":
        import nltk
        import pysbd
        nlp, seg = spacy.load("en_core_web_sm"), pysbd.Segmenter(language="en", clean=False)
        T.update(spacy_en=lambda ts: [[s.text for s in d.sents] for d in nlp.pipe(ts)],
                 pysbd=lambda ts: [seg.segment(t) for t in ts], nltk=lambda ts: [nltk.sent_tokenize(t) for t in ts])
    else:
        from kiwipiepy import Kiwi
        kiwi, nlp = Kiwi(), spacy.load("ko_core_news_sm")
        T.update(kiwi=lambda ts: [[s.text for s in kiwi.split_into_sents(t)] for t in ts],
                 kiwi_plus=lambda ts: [[p for s in kiwi.split_into_sents(t) for p in MARK.split(s.text) if p and p.strip()] for t in ts],
                 spacy_ko=lambda ts: [[s.text for s in d.sents] for d in nlp.pipe(ts)])
    return T


def build():
    S = samples()
    cands = []
    for lang in ("en", "ko"):
        idx = [i for i, x in enumerate(S) if x[0] == lang]
        texts = [S[i][3] for i in idx]
        outs = {name: f(texts) for name, f in splitters(lang).items()}
        for k, i in enumerate(idx):
            t = S[i][3]
            B = {name: boundaries(t, [s for s in o[k] if s.strip()]) for name, o in outs.items()}
            for b in sorted(set().union(*B.values())):
                cands.append({"id": len(cands), "i": i, "lang": lang, "src": S[i][1], "pos": b,
                              "cut_by": sorted(n for n, bs in B.items() if b in bs), "n_tools": len(B),
                              "context": t[max(0, b - 80):b] + " ‖ " + t[b:b + 60]})
    with open(OUT / "v2_samples.jsonl", "w", encoding="utf-8") as fh:
        for lang, src, d, t in S:
            fh.write(json.dumps({"lang": lang, "src": src, "doc": d, "text": t}, ensure_ascii=False) + "\n")
    with open(OUT / "v2_candidates.jsonl", "w", encoding="utf-8") as fh:
        for c in cands:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    for lang in ("en", "ko"):
        cs = [c for c in cands if c["lang"] == lang]
        print(lang, "문단", sum(x[0] == lang for x in S), "후보 경계", len(cs), "모든 도구가 자름", sum(len(c["cut_by"]) == c["n_tools"] for c in cs))


def score():
    cands = [json.loads(x) for x in (OUT / "v2_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    lab = {int(k): v for k, v in json.loads((OUT / "v2_labels.json").read_text(encoding="utf-8")).items()}   # id -> 1(경계) / 0(아님) / None(애매, 제외)
    lines = ["# 문장 나누기 검증(새 표본) — 정밀도·재현율", ""]
    for lang in ("en", "ko"):
        cs = [c for c in cands if c["lang"] == lang and lab.get(c["id"]) is not None]
        true = [c for c in cs if lab[c["id"]] == 1]
        lines += [f"## {'영어' if lang == 'en' else '한국어'}: 판정 후보 {len(cs)}곳(진짜 경계 {len(true)}곳), 애매해서 뺀 곳 {sum(1 for c in cands if c['lang'] == lang and lab.get(c['id'], 'x') is None)}곳", "",
                  "| 방법 | 자른 곳 | 정밀도(자른 곳 중 진짜) | 재현율(진짜 중 자른 곳) | F1 |", "|---|---|---|---|---|"]
        for m in METHODS[lang] + (["kiwi"] if lang == "ko" else ["pysbd", "nltk"]) + ["regex"]:
            cut = [c for c in cs if m in c["cut_by"]]
            tp = sum(lab[c["id"]] == 1 for c in cut)
            p, r = tp / max(1, len(cut)), tp / max(1, len(true))
            lines.append(f"| {m} | {len(cut)} | {p:.1%} | {r:.1%} | {2 * p * r / max(1e-9, p + r):.1%} |")
        lines.append("")
    (OUT / "v2_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    score() if "--score" in sys.argv else build()
