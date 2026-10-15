"""문단 순서 검사(tests 전용): 우리 단위(문단)마다 Vectara 공식 텍스트 안의 위치를 구해, 원문 순서에서 벗어난 단위와
한 단위 안에서 위치가 뒤섞인 곳을 찾는다. 낱말 diff보다 수식·표기 차이에 덜 흔들린다.

  .venv311/bin/python tests/stage1_v2/order_check.py [--list 목록]

위치: 단위 글의 6낱말 묶음 중 기준 텍스트에 딱 한 번 나오는 것의 기준 위치(낱말 번호). 묶음 3개 이상인 단위만 본다.
  문단 자리 이동 : 단위 위치(중앙값)들의 최장 증가 부분열에서 벗어난 단위(본문 단위만: 표·각주 제외)
  문단 안 뒤섞임 : 한 단위 안에서 이어지는 묶음의 기준 위치가 뒤로 50낱말 넘게 돌아가거나 앞으로 300낱말 넘게 건너뛴 곳
                   (단 넘김을 잘못 읽어 다른 단의 글이 문단 가운데 끼거나, 문단 사이 글이 섞임)
출력: out/order_check/docs.jsonl, out/order_check/summary.md
"""
import argparse
import bisect
import json
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from intrusion import toks  # noqa: E402
from ref_eval import OUT, load_units, our_text, reference_text  # noqa: E402

PARA = ["processed_para"]
RES = OUT / "order_check"
N = 6
BACK, FWD = 50, 300


def lis_keep(seq):
    tails, tails_i, prev = [], [], [-1] * len(seq)
    for k, x in enumerate(seq):
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
            tails_i.append(k)
        else:
            tails[i], tails_i[i] = x, k
        prev[k] = tails_i[i - 1] if i else -1
    keep = [False] * len(seq)
    k = tails_i[-1] if tails_i else -1
    while k >= 0:
        keep[k] = True
        k = prev[k]
    return keep


def check(doc_id):
    rw = toks(reference_text(doc_id))
    pos = {}
    for i in range(len(rw) - N + 1):
        pos.setdefault(tuple(rw[i:i + N]), []).append(i)
    units = load_units(doc_id, PARA[0])
    body = our_text(units, rw[:40])
    placed, jumps = [], []
    for ui, u in enumerate(units):
        if u["kind"] in ("table", "footnote") or u["text"] not in body:
            continue
        w = toks(u["text"])
        anc = [(i, pos[g][0]) for i in range(len(w) - N + 1) for g in [tuple(w[i:i + N])] if len(pos.get(g, [])) == 1]
        if len(anc) < 3:
            continue
        placed.append((ui, statistics.median(p for _, p in anc), len(w)))
        for (i1, p1), (i2, p2) in zip(anc, anc[1:]):
            d = p2 - p1 - (i2 - i1)
            if d < -BACK or d > FWD:
                jumps.append({"unit": ui, "pages": u["pages"], "jump": int(d), "at_word": i2,
                              "before": " ".join(w[max(0, i2 - 12):i2]), "after": " ".join(w[i2:i2 + 12])})
                break                                  # 단위마다 첫 뒤섞임만(같은 원인이 이어서 여러 번 잡힘)
    keep = lis_keep([p for _, p, _ in placed])
    moved = [{"unit": ui, "pages": units[ui]["pages"], "words": n, "ref_pos": round(p / max(1, len(rw)), 3),
              "prev": units[placed[k - 1][0]]["text"][-80:] if k else "", "text": units[ui]["text"][:160]}
             for k, ((ui, p, n), ok) in enumerate(zip(placed, keep)) if not ok]
    return {"doc_id": doc_id, "units_placed": len(placed), "moved": moved, "jumps": jumps}


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
        rows = pool.map(check, ids, chunksize=4)
    with open(RES / "docs.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    nm = sum(len(r["moved"]) for r in rows)
    nj = sum(len(r["jumps"]) for r in rows)
    lines = [f"# 문단 순서 검사 (vectara {len(rows)}편, 위치를 정한 단위 {sum(r['units_placed'] for r in rows):,}개)", "",
             f"- 문단 자리 이동: {nm}개 단위({sum(m['words'] for r in rows for m in r['moved']):,}낱말), {sum(1 for r in rows if r['moved'])}편",
             f"- 문단 안 뒤섞임: {nj}곳, {sum(1 for r in rows if r['jumps'])}편", "", "## 많은 문서"]
    for r in sorted(rows, key=lambda r: -(len(r["moved"]) + len(r["jumps"])))[:25]:
        lines.append(f"- {r['doc_id']}: 이동 {len(r['moved'])}, 뒤섞임 {len(r['jumps'])}")
    (RES / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
