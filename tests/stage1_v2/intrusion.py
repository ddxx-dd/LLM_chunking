"""읽기 순서 대조(tests 전용): Vectara 공식 텍스트를 이어 붙인 글과 우리 결과를 이어 붙인 글을 낱말 단위로 정렬(diff)해
순서가 어긋난 곳(같은 글이 다른 자리에 있음), 본문 사이에 끼어든 잡음, 빠진 글을 찾는다.

  .venv311/bin/python tests/stage1_v2/intrusion.py [--list 목록] [--workers 16]

양쪽 글: ref_eval과 같은 정리(기준: 참고문헌·수식·표·캡션 제외 / 우리: 표·제목 앞부분 제외), 3글자 이상 영어 낱말(소문자).
정렬: difflib.SequenceMatcher(낱말 목록). 차이 구간 중 8낱말 이상만 본다.
  순서 어긋남  우리 쪽에만 있는 구간인데, 그 글(6낱말 묶음 절반 이상)이 기준의 다른 자리에 있음 = 다른 자리 글이 끼어듦
  잡음 끼어듦  우리 쪽에만 있고 기준 어디에도 없음(그림 글자·각주·머리말·수식 조각 등). 영어 낱말 비율이 낮으면 '수식 조각'
  빠짐         기준 쪽에만 있고 우리 어디에도 없음
출력: out/intrusion/docs.jsonl(문서마다 구간 목록: 쪽·단위·앞뒤 글), out/intrusion/summary.md
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ref_eval import OUT, load_units, our_text, reference_text  # noqa: E402

PARA = ["processed_para"]
RES = OUT / "intrusion"
TOK = re.compile(r"[A-Za-z]{3,}")
N, MIN_SEG = 6, 8


def toks(t):
    return [w.lower() for w in TOK.findall(unicodedata.normalize("NFKC", t))]


def grams(ws):
    return {tuple(ws[i:i + N]) for i in range(len(ws) - N + 1)}


def compare(doc_id):
    rw = toks(reference_text(doc_id))
    units = load_units(doc_id, PARA[0])
    body = our_text(units, rw[:40])
    ow, owner = [], []                                 # 우리 낱말과 그 낱말의 단위 번호
    for ui, u in enumerate(units):
        if u["kind"] == "table" or u["text"] not in body:
            continue
        t = toks(u["text"])
        ow += t
        owner += [ui] * len(t)
    g_ref, g_our = grams(rw), grams(ow)
    sm = SequenceMatcher(a=rw, b=ow, autojunk=False)
    found = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if j2 - j1 >= MIN_SEG:                         # 우리 쪽에만 있는 구간
            seg = ow[j1:j2]
            gs = [tuple(seg[k:k + N]) for k in range(len(seg) - N + 1)]
            elsewhere = sum(g in g_ref for g in gs) / max(1, len(gs))
            raw = " ".join(units[ui]["text"] for ui in sorted(set(owner[j1:j2])))
            latin = len(TOK.findall(raw)) / max(1, len(raw.split()))
            kind = "순서 어긋남" if elsewhere >= 0.5 else ("수식 조각" if latin < 0.5 else "잡음 끼어듦")
            us = sorted(set(owner[j1:j2]))
            near = set(ow[max(0, j1 - 80):j1] + ow[j2:j2 + 80])
            near_g = grams(ow[max(0, j1 - 80):j1]) | grams(ow[j2:j2 + 80])
            if kind == "순서 어긋남" and sum(g in near_g for g in gs) / max(1, len(gs)) >= 0.5:
                kind = "중복(바로 옆 글 반복)"     # 끼어든 글이 앞뒤 80낱말 안에 이미 있음 = 같은 글이 두 번
            if all(units[ui]["kind"] == "footnote" for ui in us):
                kind = "각주(문서 끝으로 옮김: 방침)"
            elif j1 == 0 or all(ui <= owner[0] for ui in us) and kind == "잡음 끼어듦":
                kind = "앞부분(제목·저자)"
            found.append({"kind": kind, "n": j2 - j1, "units": us, "op": op, "pages": sorted({p for ui in us for p in units[ui]["pages"]}),
                          "before": " ".join(ow[max(0, j1 - 10):j1]), "text": " ".join(seg)[:300], "after": " ".join(ow[j2:j2 + 10]),
                          "at": round(j1 / max(1, len(ow)), 3)})
        if i2 - i1 >= MIN_SEG:                         # 기준 쪽에만 있는 구간
            seg = rw[i1:i2]
            gs = [tuple(seg[k:k + N]) for k in range(len(seg) - N + 1)]
            if sum(g in g_our for g in gs) / max(1, len(gs)) < 0.5:   # 우리 다른 자리에 있으면 '순서 어긋남'의 짝이라 따로 안 셈
                found.append({"kind": "빠짐", "n": i2 - i1, "text": " ".join(seg)[:300], "at": round(i1 / max(1, len(rw)), 3),
                              "before": " ".join(rw[max(0, i1 - 10):i1]), "after": " ".join(rw[i2:i2 + 10])})
    match = sum(m.size for m in sm.get_matching_blocks())
    return {"doc_id": doc_id, "ref_words": len(rw), "our_words": len(ow), "aligned": round(match / max(1, len(rw)), 4),
            "counts": dict(Counter(f["kind"] for f in found)), "found": found}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--para", default="processed_para", help="out/vectara/ 아래 폴더, 또는 old(옛 data/meta_text)")
    ap.add_argument("--tag", default="")
    ns = ap.parse_args()
    global RES
    PARA[0] = ns.para
    if ns.tag:
        RES = RES.parent / (RES.name + ns.tag)
    ev = {json.loads(x)["doc_id"]: json.loads(x) for x in (OUT / "ref_eval" / "docs.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()}
    if ns.list:
        ids = [x.split(":", 1)[1] for x in Path(ns.list).read_text(encoding="utf-8").split("\n") if x.startswith("vectara:")]
    else:
        ids = sorted(d for d, r in ev.items() if not r["ref_incomplete"])
    RES.mkdir(parents=True, exist_ok=True)
    with Pool(ns.workers) as pool:
        rows = pool.map(compare, ids, chunksize=4)
    with open(RES / "docs.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds = ["순서 어긋남", "중복(바로 옆 글 반복)", "잡음 끼어듦", "수식 조각", "빠짐", "각주(문서 끝으로 옮김: 방침)", "앞부분(제목·저자)"]
    lines = [f"# 읽기 순서 대조 (vectara {len(rows)}편: 공식 텍스트 이어 붙임 ↔ 우리 결과 이어 붙임, 8낱말 이상 차이 구간)", "",
             "| 종류 | 구간 | 낱말 | 문서 |", "|---|---|---|---|"]
    for k in kinds:
        fs = [f for r in rows for f in r["found"] if f["kind"] == k]
        lines.append(f"| {k} | {len(fs)} | {sum(f['n'] for f in fs):,} | {sum(1 for r in rows if r['counts'].get(k))} |")
    body = lambda f: f.get("op") == "insert" and f["kind"] in ("순서 어긋남", "잡음 끼어듦", "수식 조각", "중복(바로 옆 글 반복)")
    fs = [f for r in rows for f in r["found"] if body(f)]
    lines += ["", "## 이어진 글 사이 끼어듦(기준 쪽은 끊김 없이 이어지는데 우리 쪽에만 8낱말 이상이 들어간 곳)", "",
              "| 끼어든 글 | 곳 | 낱말 | 문서 |", "|---|---|---|---|"]
    for k in ("순서 어긋남", "잡음 끼어듦", "수식 조각", "중복(바로 옆 글 반복)"):
        ks = [f for f in fs if f["kind"] == k]
        lines.append(f"| {k} | {len(ks)} | {sum(f['n'] for f in ks):,} | {sum(1 for r in rows if any(body(f) and f['kind'] == k for f in r['found']))} |")
    lines.append(f"| 합계 | {len(fs)} | {sum(f['n'] for f in fs):,} | {sum(1 for r in rows if any(body(f) for f in r['found']))} |")
    lines += ["", "## 순서 어긋남이 많은 문서"]
    for r in sorted(rows, key=lambda r: -r["counts"].get("순서 어긋남", 0))[:25]:
        if r["counts"].get("순서 어긋남"):
            lines.append(f"- {r['doc_id']}: {r['counts']}")
    (RES / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
