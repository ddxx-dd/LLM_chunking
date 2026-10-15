"""문장 나누기 도구 비교(tests 전용, A2-3).

  source env.sh && .venv311/bin/python tests/sent_split/compare.py

표본(고정 시드): 영어 = vectara 논문 60(약어·번호가 있는 문단 절반), docx 30, GovReport 30(줄바꿈 없는 원문에서 약 200낱말 창),
  NarrativeQA 30(빈 줄로 문단 복원), 한국어 = allganize 80(개조식 문단 절반). 우리 문단은 data/processed_para.
도구: 영어 regex(LangChain SemanticChunker 기본 '(?<=[.?!])\\s+'), pysbd, nltk punkt, spaCy(en_core_web_sm), SaT(sat-3l-sm, sat-3l)
      한국어 regex, Kiwi(kiwipiepy), kss(pecab), SaT(sat-3l-sm, sat-3l)
자동 지표: 조각(3낱말 이하) 비율, 긴 문장(영어 80낱말·한국어 50어절 초과) 비율, 약어·번호 뒤에서 자른 의심 수, 속도(문단/초)
불일치: 도구마다 문장 경계 위치(글자 위치)를 구해, 도구끼리 다른 경계를 앞뒤 글과 함께 out/disagree.jsonl로(사람 판정용)
출력: tests/sent_split/out/{samples.jsonl, splits.jsonl, summary.md, disagree.jsonl}
"""
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"
PARA = ROOT / "data" / "processed_para"
LB = ROOT / "data" / "raw" / "longbench"
ABBR_EN = re.compile(r"(\bet al\.|\b(Fig|Figs|Eq|Eqs|Ref|Refs|Sec|Tab|No|Vol|pp|vs|cf|e\.g|i\.e|etc|approx|resp|Dr|Mr|Mrs|Ms|Prof|St|Inc|Ltd|Jr|Sr|U\.S)\.|\b[A-Z]\.|\d\.)$")
ABBR_KO = re.compile(r"(\d\.|\b[A-Za-z]\.)$")
KO_LIST_END = re.compile(r"(함|음|임|됨|있음|없음|필요|예정|추진|강화|확대|마련|지원)\s*$")


def load_units(ds):
    out = []
    for f in sorted((PARA / ds).glob("*.json")):
        if f.name.endswith(".dropped.json"):
            continue
        for u in json.loads(f.read_text(encoding="utf-8"))["units"]:
            if u["kind"] in ("para", "list"):
                t = u["text"].split("\n", 1)[1] if u.get("heading") and "\n" in u["text"] else u["text"]   # 붙인 제목 줄은 뺀다
                out.append((f.stem, " ".join(t.split())))
    return out


def samples():
    rnd = random.Random(0)
    S = []
    v = [x for x in load_units("vectara") if 30 <= len(x[1].split()) <= 200]
    hard = [x for x in v if re.search(r"et al\.|Fig\.|Eq\.|e\.g\.|i\.e\.|\d\.\d|\b[A-Z]\. [A-Z]", x[1])]
    S += [("en", "vectara", d, t) for d, t in rnd.sample(hard, 30) + rnd.sample(v, 30)]
    dx = [x for x in load_units("docx_corpus") if 20 <= len(x[1].split()) <= 200]
    S += [("en", "docx", d, t) for d, t in rnd.sample(dx, 30)]
    gov = [json.loads(x) for x in (LB / "gov_report.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in rnd.sample(gov, 30):
        w = r["context"].split()
        s = rnd.randrange(0, max(1, len(w) - 220))
        S.append(("en", "gov_report", r["_id"][:12], " ".join(w[s:s + 200])))
    nq = [json.loads(x) for x in (LB / "narrativeqa.jsonl").read_text(encoding="utf-8").splitlines()]
    paras = []
    for r in nq:
        for p in re.split(r"\n\s*\n", r["context"]):
            t = " ".join(p.split())
            if 40 <= len(t.split()) <= 200:
                paras.append((r["_id"][:12], t))
    S += [("en", "narrativeqa", d, t) for d, t in rnd.sample(paras, 30)]
    ko = [x for x in load_units("allganize") if 15 <= len(x[1].split()) <= 150]
    lst = [x for x in ko if len(re.findall(r"(함|음|임|됨)\s*($|[,.\s])", x[1])) >= 2 or re.search(r"[○□■▪◦·-]\s", x[1])]
    rest = [x for x in ko if x not in lst]
    S += [("ko", "allganize", d, t) for d, t in rnd.sample(lst, 40) + rnd.sample(rest, 40)]
    return S


def tools(lang):
    if lang == "en":
        import nltk
        import pysbd
        import spacy
        seg = pysbd.Segmenter(language="en", clean=False)
        nlp = spacy.load("en_core_web_sm")
        T = {"regex": lambda ts: [re.split(r"(?<=[.?!])\s+", t) for t in ts],
             "pysbd": lambda ts: [seg.segment(t) for t in ts],
             "nltk_punkt": lambda ts: [nltk.sent_tokenize(t) for t in ts],
             "spacy": lambda ts: [[s.text for s in d.sents] for d in nlp.pipe(ts)]}
    else:
        import kss
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        T = {"regex": lambda ts: [re.split(r"(?<=[.?!])\s+", t) for t in ts],
             "kiwi": lambda ts: [[s.text for s in kiwi.split_into_sents(t)] for t in ts],
             "kss": lambda ts: [kss.split_sentences(t, backend="pecab") for t in ts]}
    from wtpsplit import SaT
    for name in ("sat-3l-sm", "sat-3l"):
        m = SaT(name)
        m.half().to("cuda")
        T[name.replace("-", "_")] = (lambda mm: (lambda ts: [list(x) for x in mm.split(ts)]))(m)
    return T


def boundaries(text, sents):
    """문장 끝 위치(다음 문장 첫 글자 위치) 집합. 공백 차이는 무시하고 순서대로 찾는다."""
    out, pos = set(), 0
    for s in sents[:-1]:
        s = s.strip()
        if not s:
            continue
        k = text.find(s, pos)
        if k < 0:
            k = text.find(s[:20], pos)
            if k < 0:
                continue
            end = k + len(s)
        else:
            end = k + len(s)
        while end < len(text) and text[end].isspace():
            end += 1
        out.add(end)
        pos = end
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = samples()
    with open(OUT / "samples.jsonl", "w", encoding="utf-8") as fh:
        for lang, src, d, t in S:
            fh.write(json.dumps({"lang": lang, "src": src, "doc": d, "text": t}, ensure_ascii=False) + "\n")
    res = defaultdict(dict)                         # (lang) -> tool -> [문장 목록]
    speed = {}
    for lang in ("en", "ko"):
        texts = [t for l, _, _, t in S if l == lang]
        for name, f in tools(lang).items():
            f(texts[:3])                             # 준비(모델 올리기·캐시)
            t0 = time.time()
            res[lang][name] = f(texts)
            speed[(lang, name)] = len(texts) / max(1e-6, time.time() - t0)
            print(f"{lang} {name}: {speed[(lang, name)]:.1f}문단/초", flush=True)
    rows = []
    for lang in ("en", "ko"):
        idx = [i for i, x in enumerate(S) if x[0] == lang]
        for name, outs in res[lang].items():
            for k, i in enumerate(idx):
                rows.append({"i": i, "lang": lang, "src": S[i][1], "tool": name, "sents": [s.strip() for s in outs[k] if s.strip()]})
    with open(OUT / "splits.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    lines = ["# 문장 나누기 도구 비교", "", f"표본: 영어 {sum(x[0] == 'en' for x in S)}문단, 한국어 {sum(x[0] == 'ko' for x in S)}문단", ""]
    for lang in ("en", "ko"):
        long_n, abbr = (80, ABBR_EN) if lang == "en" else (50, ABBR_KO)
        srcs = sorted({x[1] for x in S if x[0] == lang})
        lines += [f"## {'영어' if lang == 'en' else '한국어'}", "",
                  "| 도구 | 문장 수 | 조각(3낱말 이하) | 긴 문장 | 약어·번호 뒤 자름 의심 | " + ("한국어 개조식 끝(~함 등) 뒤 안 자름 | " if lang == "ko" else "") + "속도(문단/초) |",
                  "|---|---|---|---|---|" + ("---|" if lang == "ko" else "") + "---|"]
        for name in res[lang]:
            rs = [r for r in rows if r["lang"] == lang and r["tool"] == name]
            ss = [s for r in rs for s in r["sents"]]
            frag = sum(len(s.split()) <= 3 for s in ss)
            lng = sum(len(s.split()) > long_n for s in ss)
            sus = sum(bool(abbr.search(s)) for r in rs for s in r["sents"][:-1])
            extra = ""
            if lang == "ko":
                miss = sum(len(re.findall(r"(함|음|임|됨)\s+(?=\S)", s)) for s in ss)
                extra = f"{miss} | "
            lines.append(f"| {name} | {len(ss)} | {frag} ({frag / max(1, len(ss)):.1%}) | {lng} ({lng / max(1, len(ss)):.1%}) | {sus} | {extra}{speed[(lang, name)]:.1f} |")
        lines += ["", "출처별 문장 수: " + "; ".join(f"{src}: " + ", ".join(
            f"{name} {sum(len(r['sents']) for r in rows if r['lang'] == lang and r['tool'] == name and r['src'] == src)}" for name in res[lang]) for src in srcs), ""]
    # 불일치 경계
    dis = []
    for lang in ("en", "ko"):
        for i in [i for i, x in enumerate(S) if x[0] == lang]:
            text = S[i][3]
            B = {r["tool"]: boundaries(text, r["sents"]) for r in rows if r["i"] == i}
            allb = set().union(*B.values())
            for b in sorted(allb):
                who = sorted(t for t, bs in B.items() if b in bs)
                if len(who) != len(B):
                    dis.append({"i": i, "lang": lang, "src": S[i][1], "pos": b, "cut_by": who,
                                "not_cut_by": sorted(set(B) - set(who)), "context": text[max(0, b - 70):b] + " ‖ " + text[b:b + 50]})
    with open(OUT / "disagree.jsonl", "w", encoding="utf-8") as fh:
        for d in dis:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    lines += [f"불일치 경계: {len(dis)}곳 (영어 {sum(d['lang'] == 'en' for d in dis)}, 한국어 {sum(d['lang'] == 'ko' for d in dis)})"]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
