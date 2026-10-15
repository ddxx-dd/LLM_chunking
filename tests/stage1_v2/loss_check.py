"""본문 삭제 검사(tests 전용): 전처리 규칙이 본문을 지우지 않았는지 본다. 규칙을 바꿀 때마다 돌려 기준값과 비교한다.

  .venv311/bin/python tests/stage1_v2/loss_check.py            # intrusion.py를 먼저 돌려 둘 것(vectara 빠짐 구간을 쓴다)

A. vectara: 공식 텍스트에는 있는데 우리 결과에 없는 구간(8낱말 이상, intrusion.py의 '빠짐')마다 사라진 단계·규칙을 붙인다.
   2차 <규칙>       .dropped.json에 있음(그 규칙이 지움)
   2차 단위에서 사라짐  1차 결과에는 있는데 단위에 없고 지운 기록도 없음(문단 잇기·수식 처리 등에서 사라짐)
   1차가 버림       Docling 결과에는 있는데 1차 결과에 없음
   Docling에 없음    Docling 결과에도 없음(Docling이 못 읽음 또는 공식 텍스트 쪽 차이)
   방침: F1 그림 캡션, R1·R2 참고문헌, N3 첫 쪽 정보는 일부러 지우는 것이라 따로 묶는다.
B. 모든 데이터셋: 2차 규칙이 지운 블록 중 본문처럼 보이는 것(30낱말 이상, 문장 끝맺음, 참고문헌·캡션 모양 아님) 수를 규칙별로 센다.
출력: out/loss_check/summary.md, out/loss_check/vectara.jsonl, out/loss_check/prose_dropped.jsonl
"""
import json
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from intrusion import toks  # noqa: E402
from ref_eval import OUT  # noqa: E402
sys.path.insert(0, str(HERE.parents[1] / "src"))
from preprocess import stage2  # noqa: E402

RES = OUT / "loss_check"
POLICY = ("F1", "R1", "R2", "N3")
END = re.compile(r"[.?!다요]\s*[\"'”’)]?$")


def g6(t):
    w = toks(t)
    return {tuple(w[i:i + 6]) for i in range(len(w) - 5)}


def vectara_doc(line):
    r = json.loads(line)
    d = r["doc_id"]
    miss = [f for f in r["found"] if f["kind"] == "빠짐"]
    if not miss:
        return []
    base = OUT / "vectara"
    drop = json.loads((base / "processed_para" / f"{d}.dropped.json").read_text(encoding="utf-8"))
    units = json.loads((base / "processed_para" / f"{d}.json").read_text(encoding="utf-8"))["units"]
    meta = json.loads((base / "meta_text_v2" / f"{d}.json").read_text(encoding="utf-8"))
    dl = json.loads((base / "docling" / f"{d}.docling.json").read_text(encoding="utf-8"))
    gd = [(x["rule"], g6(x["text"])) for x in drop]
    gu = set().union(*[g6(u["text"]) for u in units]) if units else set()
    gm = g6("\n".join(b["text"] for b in meta["blocks"]))
    gl = g6("\n".join((t.get("text") or "") + " " + (t.get("orig") or "") for t in dl["texts"]))
    out = []
    for f in miss:
        seg = f["text"].split()
        c = Counter()
        for g in {tuple(seg[i:i + 6]) for i in range(len(seg) - 5)}:
            if g in gu:
                c["단위에 있음(표·앞부분, 채점 범위 밖)"] += 1
                continue
            hit = next((rl for rl, st in gd if g in st), None)
            c[f"2차 {hit}" if hit else ("2차 단위에서 사라짐" if g in gm else ("1차가 버림" if g in gl else "Docling에 없음"))] += 1
        cause = c.most_common(1)[0][0] if c else "?"
        out.append({"doc_id": d, "cause": cause, "policy": cause.split(" ")[1][:2] in POLICY if cause.startswith("2차 ") else False,
                    "n": f["n"], "at": f["at"], "text": f["text"][:300]})
    return out


def prose_dropped(f):
    ds = f.parent.parent.name
    out = []
    for x in json.loads(f.read_text(encoding="utf-8")):
        t = x["text"].strip()
        n = len(t.split()) if ds == "vectara" else max(len(t.split()), len(t) // 4)
        if n < 30 or not END.search(t):
            continue
        head = " ".join(t.split()[:30])
        if stage2.YEAR.search(t) and (len(stage2.REF_STRICT.findall(t)) >= 1 or stage2.REF_START.match(t) or head.count(",") >= 4 or "———" in t):
            continue                              # 참고문헌 항목(연도 + 표지·저자 시작·이름 나열 쉼표)
        if stage2.CAP_SHAPE.match(t) or stage2.TAB_CAP_SHAPE.match(t):
            continue
        out.append({"dataset": ds, "doc_id": f.name[: -len(".dropped.json")], "rule": x["rule"], "page": x.get("page"), "text": t[:300]})
    return out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    lines_in = (OUT / "intrusion" / "docs.jsonl").read_text(encoding="utf-8").splitlines()
    files = sorted(OUT.glob("*/processed_para/*.dropped.json"))
    with Pool(16) as pool:
        va = [x for rs in pool.map(vectara_doc, lines_in, chunksize=8) for x in rs]
        pd = [x for rs in pool.map(prose_dropped, files, chunksize=16) for x in rs]
    with open(RES / "vectara.jsonl", "w", encoding="utf-8") as fh:
        for x in va:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(RES / "prose_dropped.jsonl", "w", encoding="utf-8") as fh:
        for x in pd:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    agg = defaultdict(lambda: [0, 0, set()])
    for x in va:
        a = agg[x["cause"]]
        a[0] += 1
        a[1] += x["n"]
        a[2].add(x["doc_id"])
    lines = ["# 본문 삭제 검사", "", "## A. vectara: 공식 텍스트에 있는데 우리 결과에 없는 구간(8낱말 이상)의 원인", "",
             "| 원인 | 방침 | 구간 | 낱말 | 문서 |", "|---|---|---|---|---|"]
    for c, (k, n, ds) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        pol = "방침" if c.startswith("2차 ") and c.split(" ")[1][:2] in POLICY else ""
        lines.append(f"| {c} | {pol} | {k} | {n:,} | {len(ds)} |")
    nonpol = [x for x in va if not x["policy"] and not x["cause"].startswith("단위에 있음")]
    lines += ["", f"방침 아닌 원인 합계: {len(nonpol)}구간 {sum(x['n'] for x in nonpol):,}낱말 {len({x['doc_id'] for x in nonpol})}편", "",
              "## B. 모든 데이터셋: 2차 규칙이 지운 블록 중 본문처럼 보이는 것(30낱말 이상·문장 끝맺음·참고문헌·캡션 모양 아님)", "",
              "| 규칙 | vectara | allganize | docx_corpus |", "|---|---|---|---|"]
    by = defaultdict(Counter)
    for x in pd:
        by[x["rule"]][x["dataset"]] += 1
    for rl, c in sorted(by.items(), key=lambda kv: -sum(kv[1].values())):
        lines.append(f"| {rl} | {c['vectara']} | {c['allganize']} | {c['docx_corpus']} |")
    (RES / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
