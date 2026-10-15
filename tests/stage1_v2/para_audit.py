"""문단 나누기 점검(tests 전용): 원본 PDF의 줄 위치로 '눈에 보이는 문단 경계 신호'를 뽑아 1차 v2 블록과 비교한다.

  .venv311/bin/python tests/stage1_v2/para_audit.py [--list out/para_sample.txt] [--cases 60]

대상: 표본의 PDF 문서, 본문 블록(paragraph, list_item). 줄 = PyMuPDF line을 같은 높이끼리 묶은 '행'(블록 영역 안, 영역마다 따로).
문서 기준값: 행 간격(위 끝 차) 중앙값 = pitch, 행 높이 중앙값 = lh.

A. 블록 안에 숨은 경계(덜 나뉨) - 같은 영역 안 행 i(첫 행 제외)에서
   gap     줄 사이 빈 공간 > 보통 빈 공간 + 0.6 lh (빈 줄. 위 끝 차가 아니라 빈 공간으로 재서 인라인 수식으로 높아진 줄은 제외)
   indent  행 왼쪽이 블록의 왼쪽 끝보다 0.5~4 lh 들어가고, 앞 행이 오른쪽 끝보다 1.5 lh 넘게 짧게 끝남
   short   앞 행이 블록 폭의 75% 미만에서 끝나고 마침표류로 끝남, 이 행은 들여쓰기 없음(약한 신호)
B. 블록 사이(Docling이 나눈 곳) - 전체 블록 순서에서 바로 이웃한 두 본문 블록(참고문헌 영역 제외)
   cont    잘못 끊김 후보: 앞 블록 마지막 행이 단 오른쪽 여백까지 차 있고, 마침표류로 안 끝나고, 뒤 블록이 들여쓰기 아님,
           (같은 쪽·같은 단이면) 보통보다 더 빈 공간이 0.6 lh 미만
   break   나머지(진짜 경계로 보임). 이때 gap(더 빈 공간 ≥ 0.6 lh)·indent(뒤 첫 행 들여씀)·none 으로 '이 문서가 문단을 나누는 방식'을 센다
출력: out/para_audit/docs.jsonl(문서별), out/para_audit/summary.md, out/para_audit/cases/NN.png + NN.md(무작위 사례: 쪽 이미지에 블록 영역(파랑)·신호 행(빨강))
"""
import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"
AUD = OUT / "para_audit"
END = re.compile(r"[.?!:;。다요함음임됨]\s*[\"'”’)\]]?$")
BODY = ("paragraph", "list_item")


def page_lines(doc, cache, pg):
    if pg not in cache:
        lines = []
        for blk in doc[pg - 1].get_text("dict")["blocks"]:
            for ln in blk.get("lines", []):
                t = "".join(s["text"] for s in ln["spans"])
                if t.strip():
                    lines.append((ln["bbox"], t))
        cache[pg] = lines
    return cache[pg]


def rows_in(lines, l, t, r, b):
    """영역 안(가운데 점 기준) 줄을 같은 높이끼리 묶은 행 [x0, y0, x1, y1, 글] (위→아래)."""
    inside = [(bb, tx) for bb, tx in lines if l - 2 <= (bb[0] + bb[2]) / 2 <= r + 2 and t - 2 <= (bb[1] + bb[3]) / 2 <= b + 2]
    rows = []
    for bb, tx in sorted(inside, key=lambda x: (x[0][1], x[0][0])):
        if rows and min(rows[-1][3], bb[3]) - max(rows[-1][1], bb[1]) > 0.5 * min(rows[-1][3] - rows[-1][1], bb[3] - bb[1]):
            rw = rows[-1]
            rows[-1] = [min(rw[0], bb[0]), min(rw[1], bb[1]), max(rw[2], bb[2]), max(rw[3], bb[3]), (rw[4] + " " + tx) if bb[0] > rw[0] else (tx + " " + rw[4])]
        else:
            rows.append([bb[0], bb[1], bb[2], bb[3], tx])
    return rows


def audit_doc(spec):
    import pymupdf

    ds, doc_id = spec
    v = json.loads((OUT / ds / "meta_text_v2" / f"{doc_id}.json").read_text(encoding="utf-8"))
    REFS = re.compile(r"^\s*(\d+\.?\s*)?(references|bibliography|참고\s*문헌|참고자료)\s*$", re.I)
    in_refs, pos, blocks = False, {}, []
    for k, b in enumerate(v["blocks"]):
        if b["type"] == "heading":
            in_refs = bool(REFS.match(b["text"]))
        if b["type"] in BODY and b.get("boxes") and not in_refs:
            pos[len(blocks)] = k
            blocks.append(b)
    cache, inner, between = {}, [], []
    with pymupdf.open(str(ROOT / v["source_file"])) as d:
        rows_of = []
        for b in blocks:
            rows_of.append([(bx[0], bx, rows_in(page_lines(d, cache, bx[0]), *bx[1:])) for bx in b["boxes"]])
        pitches = [rs[i + 1][1] - rs[i][1] for parts in rows_of for _, _, rs in parts for i in range(len(rs) - 1)]
        heights = [rw[3] - rw[1] for parts in rows_of for _, _, rs in parts for rw in rs]
        if len(pitches) < 5:
            return None
        pitch = statistics.median(pitches)
        lh = statistics.median(heights)
        lead = statistics.median([rs[i + 1][1] - rs[i][3] for parts in rows_of for _, _, rs in parts for i in range(len(rs) - 1)])
        # 쪽마다 단의 오른쪽 여백: 왼쪽 끝이 같은 본문 영역들의 가장 오른쪽 끝
        boxes_all = [bx for b in blocks for bx in b["boxes"]]

        def margin(bx):
            return max(o[3] for o in boxes_all if o[0] == bx[0] and abs(o[1] - bx[1]) < lh)

        # A. 블록 안
        for bi, (b, parts) in enumerate(zip(blocks, rows_of)):
            for pg, bx, rs in parts:
                if len(rs) < 2:
                    continue
                left = min(rw[0] for rw in rs)
                right = max(rw[2] for rw in rs)
                width = right - left
                for i in range(1, len(rs)):
                    p, c = rs[i - 1], rs[i]
                    kind = None
                    if c[1] - p[3] > lead + 0.6 * lh:      # 줄 사이 빈 공간이 보통보다 글자 높이의 60% 넘게 큼 = 빈 줄(인라인 수식으로 줄이 높아진 것은 제외)
                        kind = "gap"
                    elif 0.5 * lh < c[0] - left < 4 * lh and p[2] < right - 1.5 * lh:
                        kind = "indent"
                    elif p[2] < left + 0.75 * width and END.search(p[4].strip()) and c[0] - left < 0.5 * lh:
                        kind = "short"
                    if kind:
                        inner.append({"block": bi, "page": pg, "kind": kind, "prev": p[4].strip()[-70:], "row": c[4].strip()[:70],
                                      "row_box": [round(x, 1) for x in c[:4]], "block_box": bx, "ratio": round((c[1] - p[3] - lead) / lh, 2)})
        # B. 블록 사이
        for bi in range(len(blocks) - 1):
            if pos[bi + 1] != pos[bi] + 1:              # 사이에 수식·표·제목·그림 캡션 등이 끼어 있으면 비교하지 않는다
                continue
            a, b = blocks[bi], blocks[bi + 1]
            pa, pb = rows_of[bi], rows_of[bi + 1]
            la = next((x for x in reversed(pa) if x[2]), None)
            fb = next((x for x in pb if x[2]), None)
            if not la or not fb:
                continue
            (pga, bxa, rsa), (pgb, bxb, rsb) = la, fb
            last, first = rsa[-1], rsb[0]
            full = last[2] >= margin(bxa) - lh
            ends = bool(END.search(a["text"].strip()))
            leftb = min(rw[0] for rw in rsb)
            indent = len(rsb) > 1 and first[0] - leftb > 0.5 * lh
            same_col = pga == pgb and abs(bxa[1] - bxb[1]) < lh
            gap = (first[1] - last[3] - lead) / lh if same_col and first[1] > last[1] else None   # 보통보다 더 빈 공간(글자 높이 배수)
            if full and not ends and not indent and (gap is None or gap < 0.6):
                kind = "cont"
            else:
                kind = "break:" + ("gap" if gap is not None and gap >= 0.6 else "indent" if indent else "none")
            between.append({"a": bi, "b": bi + 1, "page": pgb, "kind": kind, "prev": a["text"].strip()[-70:], "next": b["text"].strip()[:70],
                            "gap": None if gap is None else round(gap, 2), "box_a": bxa, "box_b": bxb})
    style = Counter(x["kind"].split(":")[1] for x in between if x["kind"].startswith("break") and x["gap"] is not None)
    return {"dataset": ds, "doc_id": doc_id, "pages": v["n_pages"], "blocks": len(blocks), "pitch": round(pitch, 1), "lh": round(lh, 1),
            "inner": inner, "between": between, "style": dict(style)}


def render_case(n, ds, doc_id, page, boxes_blue, box_red, md):
    import pymupdf

    v = json.loads((OUT / ds / "meta_text_v2" / f"{doc_id}.json").read_text(encoding="utf-8"))
    with pymupdf.open(str(ROOT / v["source_file"])) as d:
        p = d[page - 1]
        for bx in boxes_blue:
            if bx and bx[0] == page:
                p.draw_rect(pymupdf.Rect(*bx[1:]), color=(0, 0, 1), width=1)
        if box_red:
            p.draw_rect(pymupdf.Rect(*box_red), color=(1, 0, 0), width=1.5)
        p.get_pixmap(dpi=70).save(str(AUD / "cases" / f"{n:02d}.png"))
    (AUD / "cases" / f"{n:02d}.md").write_text(md, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default=str(OUT / "para_sample.txt"))
    ap.add_argument("--cases", type=int, default=60)
    ns = ap.parse_args()
    specs = [tuple(x.split(":", 1)) for x in Path(ns.list).read_text(encoding="utf-8").split("\n") if x.strip()]
    (AUD / "cases").mkdir(parents=True, exist_ok=True)
    docs = []
    with open(AUD / "docs.jsonl", "w", encoding="utf-8") as fh:
        for s in specs:
            if not (OUT / s[0] / "meta_text_v2" / f"{s[1]}.json").exists():
                continue
            r = audit_doc(s)
            if r:
                docs.append(r)
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    lines = ["# 문단 나누기 점검", "", f"문서 {len(docs)}편 (vectara {sum(d['dataset'] == 'vectara' for d in docs)}, allganize {sum(d['dataset'] == 'allganize' for d in docs)})", ""]
    for ds in ("vectara", "allganize"):
        dd = [d for d in docs if d["dataset"] == ds]
        if not dd:
            continue
        inner = Counter(x["kind"] for d in dd for x in d["inner"])
        betw = Counter(x["kind"] for d in dd for x in d["between"])
        styles = Counter(max(d["style"], key=d["style"].get) if d["style"] else "판단 불가" for d in dd)
        lines += [f"## {ds}", f"- 본문 블록 {sum(d['blocks'] for d in dd):,}",
                  f"- A 블록 안 숨은 경계 신호: {dict(inner)} (신호가 있는 문서 {sum(bool(d['inner']) for d in dd)}편)",
                  f"- B 블록 사이: {dict(betw)}",
                  f"- 문서가 문단을 나누는 방식(블록 사이 진짜 경계 중 다수): {dict(styles)}", ""]
    rnd = random.Random(0)
    pool = [("A", d, x) for d in docs for x in d["inner"]] + [("B", d, x) for d in docs for x in d["between"] if x["kind"] == "cont"]
    by = {}
    for p in pool:
        by.setdefault((p[0], p[2]["kind"]), []).append(p)
    picks = []
    for k, v in sorted(by.items()):
        rnd.shuffle(v)
        picks += v[: max(5, ns.cases * len(v) // max(1, len(pool)))]
    rnd.shuffle(picks)
    picks = picks[: ns.cases]
    lines += ["## 사례 (이미지: cases/NN.png, 파랑 = 블록 영역, 빨강 = 신호 행/뒤 블록)", ""]
    for n, (part, d, x) in enumerate(picks):
        if part == "A":
            md = f"# {n:02d} A-{x['kind']} {d['dataset']}/{d['doc_id']} p{x['page']} (간격 {x['ratio']}배)\n\n앞 행: …{x['prev']}\n\n신호 행: {x['row']}…\n"
            render_case(n, d["dataset"], d["doc_id"], x["page"], [x["block_box"]], x["row_box"], md)
            lines.append(f"- {n:02d} A-{x['kind']} {d['doc_id'][:24]} p{x['page']}: …{x['prev'][-40:]!r} ▶ {x['row'][:40]!r}")
        else:
            md = f"# {n:02d} B-cont {d['dataset']}/{d['doc_id']} p{x['page']}\n\n앞 블록 끝: …{x['prev']}\n\n뒤 블록 시작: {x['next']}…\n"
            render_case(n, d["dataset"], d["doc_id"], x["page"], [x["box_a"]], x["box_b"][1:] if x["box_b"][0] == x["page"] else None, md)
            lines.append(f"- {n:02d} B-cont {d['doc_id'][:24]} p{x['page']}: …{x['prev'][-40:]!r} ▶ {x['next'][:40]!r}")
    (AUD / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
