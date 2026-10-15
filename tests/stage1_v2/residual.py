"""processed_para(현재 1·2차 시험 결과)에 남은 문제를 유형별로 센다(tests 전용). 설계의 근거 자료.

  .venv311/bin/python tests/stage1_v2/residual.py

유형
  P 문단      끊긴 채 남음(앞 단위가 끝맺음 없이 끝나고 다음 단위가 소문자/한글로 시작), 아주 긴 단위(1,500토큰 초과)
  F 수식      변환 글에 남은 LaTeX 명령(\\frac 등), 같은 묶음 반복
  G 그림      Docling이 찾은 그림 영역과 절반 넘게 겹치는 블록이 단위에 남음, 한 글자 라틴 낱말 3개 이상 연속(그림 낱글자)
  S 조각      5토큰 이하 단위(표·각주 제외)
  H 반복      같은 줄(숫자 무시, 4~30낱말)이 서로 다른 3쪽 이상의 단위에 나옴(머리말·바닥글 후보)
  N 잡음 문구 arXiv 세로 도장, 저작권·라이선스, 이메일·URL만 있는 줄
  R 참고문헌  참고문헌처럼 보이는 단위(연도 + 저자/학술지 표지 2개 이상)
출력: out/residual/summary.md, out/residual/docs.jsonl
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RES = OUT / "residual"
END = re.compile(r"[.?!:;)\]\"'”’…。다요까함음임됨□■∎]$")
LATEX = re.compile(r"\\[a-zA-Z]{2,}")
LETTERS = re.compile(r"(?:\b[A-Za-z]\b\s+){3,}")
ARXIV = re.compile(r"arXiv:\d{4}\.\d{4,5}(v\d+)?\s*\[[^\]]+\]")
NOISE = re.compile(r"Permission to make digital|©\s*\d{4}|Copyright\s+(held|\d{4}|©)|All rights reserved|Licensed under|Creative Commons|"
                   r"^\s*\S+@\S+\.\S+\s*$|^\s*https?://\S+\s*$", re.I | re.M)
REFY = re.compile(r"\b(19|20)\d{2}[a-z]?\b")
REFM = re.compile(r"et al\.|vol\.|pp\.|doi|arxiv|journal|proc\.|conference|press|\[\d+\]", re.I)


def inside(a, b):
    """영역 a(l,t,r,b)가 영역 b와 겹치는 넓이 / a의 넓이."""
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return w * h / area


def pictures(ds, doc_id):
    f = OUT / ds / "docling" / f"{doc_id}.docling.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text(encoding="utf-8"))
    pages = {int(k): v["size"]["height"] for k, v in d["pages"].items()}
    out = defaultdict(list)
    for p in d["pictures"]:
        for pr in p.get("prov", []):
            b, h = pr["bbox"], pages[pr["page_no"]]
            top_left = b.get("coord_origin") == "TOPLEFT"
            t, bt = (b["t"], b["b"]) if top_left else (h - b["t"], h - b["b"])
            out[pr["page_no"]].append((b["l"], t, b["r"], bt))
    return out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    rows, ex = [], defaultdict(list)
    for f in sorted(OUT.glob("*/processed_para/*.json")):
        if f.name.endswith(".dropped.json"):
            continue
        ds, doc_id = f.parent.parent.name, f.stem
        r = json.loads(f.read_text(encoding="utf-8"))
        meta = json.loads((OUT / ds / "meta_text_v2" / f"{doc_id}.json").read_text(encoding="utf-8"))
        units = r["units"]
        c = Counter()
        paras = [u for u in units if u["kind"] == "para"]
        for a, b in zip(paras, paras[1:]):
            p, n = a["text"].rstrip(), b["text"].lstrip()
            if p and n and not END.search(p) and not b["heading"] and (re.match(r"[a-z]", n) or (re.search(r"[가-힣]$", p) and re.match(r"[가-힣]", n))):
                c["P 끊긴 채 남음"] += 1
                ex["P"].append((doc_id, p[-50:], n[:50]))
        for u in units:
            t = u["text"]
            if u["n_tokens"] > 1500:
                c["P 1500토큰 넘는 단위"] += 1
                ex["P긴"].append((doc_id, u["n_tokens"], t[:80]))
            if u["kind"] not in ("table", "footnote") and u["n_tokens"] <= 5:
                c["S 5토큰 이하 단위"] += 1
                ex["S"].append((doc_id, t[:60]))
            if LATEX.search(t) and u["kind"] != "table":
                c["F LaTeX 명령 남음"] += 1
                ex["F"].append((doc_id, LATEX.findall(t)[:5]))
            m = LETTERS.search(t)
            if m and u["kind"] != "table":
                c["G 그림 낱글자 연속"] += 1
                ex["G낱글자"].append((doc_id, t[max(0, m.start() - 30):m.end() + 30]))
            for k, pat in (("N arXiv 도장", ARXIV), ("N 저작권·이메일·URL 줄", NOISE)):
                if pat.search(t):
                    c[k] += 1
                    ex[k].append((doc_id, pat.search(t).group(0)[:70]))
            if u["kind"] in ("para", "list") and len(REFY.findall(t)) >= 1 and len(REFM.findall(t)) >= 2 and u["n_tokens"] < 200:
                c["R 참고문헌 같은 단위"] += 1
                ex["R"].append((doc_id, t[:90]))
        # H 반복 줄
        seen = defaultdict(set)
        for u in units:
            for line in u["text"].split("\n"):
                s = re.sub(r"\d+", "#", " ".join(line.split()))
                if 4 <= len(s.split()) <= 30:
                    for pg in u["pages"]:
                        seen[s].add(pg)
        for s, pgs in seen.items():
            if len(pgs) >= 3:
                c["H 3쪽 이상 반복 줄"] += 1
                ex["H"].append((doc_id, s[:80]))
        # G 그림 영역 안 블록이 단위에 남았나
        pics = pictures(ds, doc_id) if meta["fmt"] == "pdf" else {}
        used = {bid.split(".")[0] for u in units for bid in u["block_ids"]}
        for b in meta["blocks"]:
            if b["type"] in ("paragraph", "list_item") and b["id"] in used and b.get("boxes"):
                pg, *box = b["boxes"][0]
                if any(inside(box, pb) > 0.5 for pb in pics.get(pg, [])):
                    c["G 그림 영역 안 블록 남음"] += 1
                    ex["G영역"].append((doc_id, b["text"][:60]))
        rows.append({"dataset": ds, "doc_id": doc_id, "units": len(units), "tokens": r["n_tokens"], **c})
    with open(RES / "docs.jsonl", "w", encoding="utf-8") as fh:
        for x in rows:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    keys = sorted({k for x in rows for k in x if k[:2] in ("P ", "F ", "G ", "S ", "H ", "N ", "R ")})
    lines = ["# processed_para 남은 문제 (현재 1·2차 시험 결과)", "",
             "| 유형 | " + " | ".join(f"{ds} 합계(문서)" for ds in ("vectara", "allganize", "docx_corpus")) + " |", "|---|---|---|---|"]
    for k in keys:
        cells = []
        for ds in ("vectara", "allganize", "docx_corpus"):
            rs = [x for x in rows if x["dataset"] == ds]
            cells.append(f"{sum(x.get(k, 0) for x in rs)} ({sum(bool(x.get(k)) for x in rs)}/{len(rs)})")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += ["", f"단위 합계: " + ", ".join(f"{ds} {sum(x['units'] for x in rows if x['dataset'] == ds):,}" for ds in ("vectara", "allganize", "docx_corpus")), ""]
    import random
    rnd = random.Random(0)
    for k, v in ex.items():
        rnd.shuffle(v)
        lines += [f"## 예: {k} ({len(v)})"] + [f"- {x}" for x in v[:8]] + [""]
    (RES / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
