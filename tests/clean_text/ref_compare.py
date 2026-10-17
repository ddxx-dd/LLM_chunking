"""clean_text(vectara)를 Vectara 공식 코퍼스(Open RAG Benchmark, Mistral OCR)와 비교한다(tests 전용).

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/clean_text/ref_compare.py [--root data] [--low 10]

정답 = data/raw/vectara_pdf/_meta/corpus/<doc_id>.json 절 텍스트를 이은 것. clean_text = <root>/clean_text/vectara/<doc_id>.json의 text.
비교 전: 양쪽 모두 수식($…$, $$…$$, \\begin…\\end, 남은 \\명령어)·그림/표 자리표시·마크다운 제목 기호·각주 표시를 빼고, 우리 쪽은 표 문단을 뺀다.
낱말 = NFKC 소문자 3글자 이상 알파벳.
  본문 재현      정답 낱말 중 우리에게 있는 비율(우리 글 전체 기준)
  잡음           우리 낱말 중 정답에 없는 비율(정답 첫 8낱말이 나오는 문단 앞 = 제목·저자·소속은 뺌)
  순서 일치      우리 6낱말 묶음(정답에서 한 번만 나오는 것)의 정답 위치 최장 증가 부분열 비율
  문단 시작 재현·정밀  정답 본문 문단(8낱말 이상) 첫 6낱말과 우리 문단(제목 줄 다음 본문 줄) 첫 6낱말의 일치
  종류별 문단 시작    우리 문단 종류(paras.kind)마다 시작이 정답에서 문단 시작 / 정답 문단 안 줄 시작 / 그 밖인 비율
정답 낱말이 200개 미만인 문서(정답 불완전)는 뺀다. 2026-10-17 v4 형식 결정 때 수치: 재현 98.4%, 잡음 2.0%, 순서 98.2%, 문단 시작 74.9%/79.2%.
출력: tests/clean_text/out/ref_compare.md
"""
import argparse
import bisect
import glob
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORPUS = ROOT / "data" / "raw" / "vectara_pdf" / "_meta" / "corpus"
STRIP = [re.compile(p, re.S) for p in (r"\$\$.+?\$\$", r"\$[^$\n]+?\$", r"!\[[^\]]*\]\([^)]*\)", r"\[\^\d+\]:?", r"(?m)^#+\s*",
                                       r"\\begin\{[^}]*\}.*?\\end\{[^}]*\}", r"\\[A-Za-z]+")]
W = re.compile(r"[a-z]{3,}")
ROOTDIR = None


def words(t):
    for p in STRIP:
        t = p.sub(" ", t)
    return W.findall(unicodedata.normalize("NFKC", t).lower())


def lis_ratio(seq):
    tails = []
    for x in seq:
        k = bisect.bisect_left(tails, x)
        tails[k:k + 1] = [x]
    return len(tails) / max(1, len(seq))


def key(t):
    w = words(t)
    return " ".join(w[:6]) if len(w) >= 6 else None


def score(doc_id):
    rec = json.loads((ROOTDIR / "clean_text" / "vectara" / f"{doc_id}.json").read_text(encoding="utf-8"))
    r = json.loads((CORPUS / f"{doc_id}.json").read_text(encoding="utf-8"))
    rtext = "\n\n".join(s["text"] for s in r["sections"])
    rw = words(rtext)
    if len(rw) < 200:
        return None
    rparas = [p.strip() for p in re.split(r"\n\s*\n", rtext) if p.strip()]
    body, starts, kinds = [], [], []
    for p, m in zip(rec["text"].split("\n\n"), rec["paras"]):
        if m["kind"] == "table":
            continue
        body.append(p)
        lines = p.split("\n")
        starts.append(p if m["kind"] == "references" else (lines[-1] if len(lines) > 1 else p))
        kinds.append(m["kind"])
    ow_all = [words(p) for p in body]
    k8 = " ".join(rw[:8])
    k0 = next((i for i, w in enumerate(ow_all) if k8 in " ".join(sum(ow_all[i:i + 3], []))), 0)
    ow = sum(ow_all[k0:], [])
    rc, oc, oca = Counter(rw), Counter(ow), Counter(sum(ow_all, []))
    grams = Counter(tuple(rw[i:i + 6]) for i in range(len(rw) - 5))
    pos = {g: i for i, g in enumerate(tuple(rw[i:i + 6]) for i in range(len(rw) - 5)) if grams[g] == 1}
    seq = [pos[g] for g in (tuple(ow[i:i + 6]) for i in range(len(ow) - 5)) if g in pos]
    rkeys = {key(p) for p in rparas if len(words(p)) >= 8} - {None}
    lkeys = {key(ln) for p in rparas for ln in p.split("\n")[1:]} - {None}
    rtxt = " ".join(rw)
    okeys = [(key(s), kd) for s, kd in zip(starts[k0:], kinds[k0:]) if len(words(s)) >= 8]
    okeys_in = {k for k, _ in okeys if k and k in rtxt}
    by = defaultdict(Counter)
    for k, kd in okeys:
        if k and k in rtxt:
            by[kd]["정답 문단 시작" if k in rkeys else "정답 줄 시작" if k in lkeys else "그 밖"] += 1
    return {"doc": doc_id, "recall": sum((rc & oca).values()) / len(rw), "noise": 1 - sum((rc & oc).values()) / max(1, len(ow)),
            "order": lis_ratio(seq), "pstart_r": len(rkeys & {k for k, _ in okeys}) / max(1, len(rkeys)),
            "pstart_p": len(rkeys & okeys_in) / max(1, len(okeys_in)), "by": {k: dict(v) for k, v in by.items()}}


def _init(root):
    global ROOTDIR
    ROOTDIR = root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data"), help="clean_text가 있는 루트(기본 data/)")
    ap.add_argument("--low", type=int, default=10, help="본문 재현 하위 몇 편을 적을지")
    ns = ap.parse_args()
    root = Path(ns.root).resolve()
    ids = sorted(Path(f).stem for f in glob.glob(str(root / "clean_text" / "vectara" / "*.json")))
    ids = [d for d in ids if (CORPUS / f"{d}.json").exists()]
    with Pool(16, initializer=_init, initargs=(root,)) as pool:
        S = [s for s in pool.map(score, ids, chunksize=8) if s]
    m = lambda k: sum(s[k] for s in S) / len(S)
    by = defaultdict(Counter)
    for s in S:
        for kd, c in s["by"].items():
            by[kd].update(c)
    L = ["# clean_text vs Vectara 공식 코퍼스", "", f"문서 {len(S)}편(정답 낱말 200개 미만 {len(ids) - len(S)}편 제외)", "",
         "| 본문 재현 | 잡음 | 순서 일치 | 문단 시작 재현 | 문단 시작 정밀 |", "|---|---|---|---|---|",
         f"| {m('recall'):.1%} | {m('noise'):.1%} | {m('order'):.1%} | {m('pstart_r'):.1%} | {m('pstart_p'):.1%} |", "",
         "| 종류 | 비교 문단 | 정답 문단 시작 | 정답 문단 안 줄 시작 | 그 밖 |", "|---|---|---|---|---|"]
    for kd, c in sorted(by.items(), key=lambda x: -sum(x[1].values())):
        n = sum(c.values())
        L.append(f"| {kd} | {n} | {c['정답 문단 시작'] / n:.1%} | {c['정답 줄 시작'] / n:.1%} | {c['그 밖'] / n:.1%} |")
    L += ["", "본문 재현 하위: " + ", ".join(f"{s['doc']}({s['recall']:.0%})" for s in sorted(S, key=lambda s: s["recall"])[:ns.low])]
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "ref_compare.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
