"""vectara 표본 진단(tests 전용): 공식 텍스트와 우리 결과를 6낱말 묶음으로 맞춰 '빠진 구간'과 '남은 구간'을 찾고 원인을 붙인다.

  .venv311/bin/python tests/stage1_v2/diagnose.py --pick        # 표본 고르기(하위 100편 + 나머지에서 무작위 50편) → out/diagnose/sample150.txt
  .venv311/bin/python tests/stage1_v2/diagnose.py               # 표본 진단 → out/diagnose/docs.jsonl, summary.md

빠진 구간(기준에는 있는데 우리에 없음, 연속 12낱말 이상)의 원인
  dropped:<규칙>  2차 규칙이 지움(.dropped.json에 있음)
  meta_only      1차 결과에는 있는데 2차 단위에 없음(지운 기록도 없음 = 단위 만들기에서 사라짐)
  docling_only   Docling 결과에는 있는데 1차 결과에 없음(1차가 버림: 구석 제거 등)
  not_found      Docling 결과에도 없음(Docling이 못 읽음, 또는 공식 텍스트 쪽 차이)
남은 구간(우리에는 있는데 기준에 없음, 연속 12낱말 이상)은 단위 종류·쪽·제목과 함께 기록한다(유형은 사람이 본다).
하위 순위: 문서마다 세 지표(재현, 잡음, 순서)의 하위 백분위 중 가장 나쁜 것으로 정렬.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ref_eval import OUT, our_text, reference_text, words  # noqa: E402

DIAG = OUT / "diagnose"
N = 6
MIN_RUN = 12


def grams(ws):
    return {tuple(ws[i:i + N]) for i in range(len(ws) - N + 1)}


def runs(ws, other):
    """ws 중 other에 없는 6낱말 묶음이 덮는 낱말 구간들(연속 MIN_RUN 이상)."""
    miss = [False] * len(ws)
    for i in range(len(ws) - N + 1):
        if tuple(ws[i:i + N]) not in other:
            for j in range(i, i + N):
                miss[j] = True
    out, s = [], None
    for i, m in enumerate(miss + [False]):
        if m and s is None:
            s = i
        elif not m and s is not None:
            if i - s >= MIN_RUN:
                out.append((s, i))
            s = None
    return out


def pick():
    rows = [json.loads(x) for x in (OUT / "ref_eval" / "docs.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()]
    rows = [r for r in rows if not r["ref_incomplete"]]
    pct = {}
    for i, worse_high in ((0, False), (1, True), (2, False)):
        vals = sorted(rows, key=lambda r: (-(r["new"][i] or 0) if worse_high else (r["new"][i] if r["new"][i] is not None else 1)))
        for k, r in enumerate(vals):
            pct.setdefault(r["doc_id"], []).append(k / len(vals))
    order = sorted(rows, key=lambda r: min(pct[r["doc_id"]]))
    bottom = [r["doc_id"] for r in order[:100]]
    rest = [r["doc_id"] for r in order[100:]]
    rnd = random.Random(0).sample(rest, 50)
    DIAG.mkdir(parents=True, exist_ok=True)
    (DIAG / "bottom100.txt").write_text("\n".join(f"vectara:{d}" for d in bottom) + "\n", encoding="utf-8")
    (DIAG / "random50.txt").write_text("\n".join(f"vectara:{d}" for d in rnd) + "\n", encoding="utf-8")
    (DIAG / "sample150.txt").write_text("\n".join(f"vectara:{d}" for d in bottom + rnd) + "\n", encoding="utf-8")
    print("하위 100 + 무작위 50 →", DIAG / "sample150.txt")


def diagnose(doc_id):
    ref = words(reference_text(doc_id))
    pp = OUT / "vectara" / "processed_para"
    rec = json.loads((pp / f"{doc_id}.json").read_text(encoding="utf-8"))
    units = rec["units"]
    ours = words(our_text(units, ref[:40]))
    dropped = json.loads((pp / f"{doc_id}.dropped.json").read_text(encoding="utf-8"))
    meta = json.loads((OUT / "vectara" / "meta_text_v2" / f"{doc_id}.json").read_text(encoding="utf-8"))
    dl = json.loads((OUT / "vectara" / "docling" / f"{doc_id}.docling.json").read_text(encoding="utf-8"))
    g_ours, g_ref = grams(ours), grams(ref)
    g_drop = defaultdict(set)
    for d in dropped:
        g_drop[d["rule"]] |= grams(words(d["text"]))
    g_all_units = grams(words("\n".join(u["text"] for u in units)))      # 표·앞부분 포함 전체 단위
    g_meta = grams(words("\n".join(b["text"] for b in meta["blocks"])))
    g_dl = grams(words("\n".join((t.get("text") or "") + "\n" + (t.get("orig") or "") for t in dl["texts"])))
    missing = []
    for s, e in runs(ref, g_ours):
        seg = ref[s:e]
        gs = grams(seg)
        cause = Counter()
        for g in gs:
            if g in g_all_units:
                cause["in_units(front/table)"] += 1
                continue
            hit = next((r for r, st in g_drop.items() if g in st), None)
            if hit:
                cause[f"dropped:{hit}"] += 1
            elif g in g_meta:
                cause["meta_only"] += 1
            elif g in g_dl:
                cause["docling_only"] += 1
            else:
                cause["not_found"] += 1
        missing.append({"pos": round(s / max(1, len(ref)), 3), "n": e - s, "cause": cause.most_common(1)[0][0], "text": " ".join(seg)[:300]})
    # 남은 구간: 단위별로 찾는다
    extra = []
    for i, u in enumerate(units):
        if u["kind"] == "table":
            continue
        ws = words(u["text"])
        for s, e in runs(ws, g_ref):
            extra.append({"unit": i, "kind": u["kind"], "pages": u["pages"], "heading": (u.get("heading") or "")[:60], "n": e - s,
                          "text": " ".join(ws[s:e])[:300]})
    return {"doc_id": doc_id, "ref_words": len(ref), "our_words": len(ours),
            "missing_words": sum(m["n"] for m in missing), "extra_words": sum(x["n"] for x in extra), "missing": missing, "extra": extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", action="store_true")
    ap.add_argument("--list", default=str(DIAG / "sample150.txt"))
    ns = ap.parse_args()
    if ns.pick:
        pick()
        return
    ids = [x.split(":", 1)[1] for x in Path(ns.list).read_text(encoding="utf-8").split("\n") if x.strip()]
    bottom = {x.split(":", 1)[1] for x in (DIAG / "bottom100.txt").read_text(encoding="utf-8").split("\n") if x.strip()}
    rows = []
    for d in ids:
        r = diagnose(d)
        r["group"] = "bottom" if d in bottom else "random"
        rows.append(r)
        print(d, r["missing_words"], r["extra_words"], flush=True)
    with open(DIAG / "docs.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    lines = [f"# vectara 표본 진단 ({len(rows)}편)", "", "## 빠진 구간 원인(낱말 수 / 문서 수)", "", "| 원인 | 하위 100 | 무작위 50 |", "|---|---|---|"]
    for cause in sorted({m["cause"] for r in rows for m in r["missing"]}):
        cells = []
        for g in ("bottom", "random"):
            rs = [r for r in rows if r["group"] == g]
            cells.append(f"{sum(m['n'] for r in rs for m in r['missing'] if m['cause'] == cause):,} / {sum(any(m['cause'] == cause for m in r['missing']) for r in rs)}")
        lines.append(f"| {cause} | " + " | ".join(cells) + " |")
    lines += ["", "## 남은 구간(단위 종류별 낱말 수 / 문서 수)", "", "| 종류 | 하위 100 | 무작위 50 |", "|---|---|---|"]
    for kind in sorted({x["kind"] for r in rows for x in r["extra"]}):
        cells = []
        for g in ("bottom", "random"):
            rs = [r for r in rows if r["group"] == g]
            cells.append(f"{sum(x['n'] for r in rs for x in r['extra'] if x['kind'] == kind):,} / {sum(any(x['kind'] == kind for x in r['extra']) for r in rs)}")
        lines.append(f"| {kind} | " + " | ".join(cells) + " |")
    (DIAG / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
