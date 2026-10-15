"""사람 검토용 페이지 만들기(tests 전용): 원본과 processed_para를 나란히 보는 HTML.

  .venv311/bin/python tests/stage1_v2/make_review.py --list tests/stage1_v2/out/review40.txt [--out human_review]

출력: tests/stage1_v2/out/human_review/index.html (문서 목록) + 문서마다 <번호>.html
  PDF : 쪽마다 [원본 쪽 이미지 | 그 쪽에서 시작하는 결과 단위], 아래에 그 쪽에서 지운 블록(규칙 이름과 함께)
  docx: [원본 글(python-docx로 본문 순서대로, 문단 스타일 표시) | 결과 단위], 맨 아래 지운 블록
이미지는 HTML 안에 넣는다(파일 하나로 열림). JupyterLab에서 열면 'Trust HTML'이 필요할 수 있다.
"""
import argparse
import base64
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"
REV = OUT / "human_review"

CSS = """
:root { --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --line:#d2d2d7; --unit:#f5f7fb; --drop:#fff1f0; --dropfg:#a8071a; --head:#e8f0fe; }
@media (prefers-color-scheme: dark) { :root { --bg:#161618; --fg:#ececec; --muted:#9a9aa0; --line:#3a3a3e; --unit:#1f2228; --drop:#2a1618; --dropfg:#ff8a80; --head:#1c2a40; } }
body { background:var(--bg); color:var(--fg); font:14px/1.55 -apple-system, "Noto Sans KR", "Malgun Gothic", sans-serif; margin:0 16px 40px; }
h1 { font-size:20px; margin:18px 0 6px; } h2 { font-size:16px; margin:22px 0 8px; border-bottom:1px solid var(--line); padding-bottom:4px; }
.meta { color:var(--muted); font-size:13px; } a { color:#2f6fde; }
.page { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:14px; border-top:1px solid var(--line); padding:12px 0; }
@media (max-width: 900px) { .page { grid-template-columns:1fr; } }
.page img { width:100%; border:1px solid var(--line); }
.unit { background:var(--unit); border-left:3px solid #2f6fde; padding:6px 10px; margin:0 0 8px; white-space:pre-wrap; word-break:break-word; }
.unit .tag { color:var(--muted); font-size:12px; display:block; margin-bottom:2px; }
.unit.table { border-left-color:#13a10e; font-family: ui-monospace, monospace; font-size:12px; }
.unit.footnote { border-left-color:#999; }
.drop { background:var(--drop); color:var(--dropfg); font-size:12px; padding:4px 8px; margin:0 0 4px; white-space:pre-wrap; word-break:break-word; }
.orig { white-space:pre-wrap; word-break:break-word; font-size:13px; padding:2px 0; border-bottom:1px dashed var(--line); }
.orig .st { color:var(--muted); font-size:11px; margin-right:6px; }
table.idx { border-collapse:collapse; width:100%; font-size:13px; } table.idx td, table.idx th { border:1px solid var(--line); padding:4px 6px; text-align:left; }
table.idx th { background:var(--head); }
.check { margin:10px 0; padding:8px 10px; background:var(--head); font-size:13px; }
"""

CHECK = ("<div class='check'><b>볼 것</b>: ① 본문이 빠졌나(원본 쪽에 있는데 오른쪽에 없음) ② 쓰레기 글이 남았나(그림 글자, 머리말, 쪽 번호, 참고문헌) "
         "③ 순서가 원문과 같은가 ④ 다른 문단을 잘못 이었나 ⑤ 빨간 '지운 블록'에 본문이 섞였나</div>")


def unit_html(i, u):
    t = html.escape(u["text"])
    return f"<div class='unit {u['kind']}'><span class='tag'>[{i}] {u['kind']} · 쪽 {u['pages']} · {u['n_tokens']}토큰</span>{t}</div>"


def drop_html(d):
    return f"<div class='drop'>✕ {html.escape(d['rule'])} · {d['type']} · {html.escape(d['text'])}</div>"


def page_images(pdf, n):
    import pymupdf

    out = []
    with pymupdf.open(str(pdf)) as d:
        for i in range(min(n, len(d))):
            pix = d[i].get_pixmap(dpi=72)
            out.append(base64.b64encode(pix.tobytes("jpeg", jpg_quality=60)).decode())
    return out


def docx_original(path):
    """본문 순서대로 (스타일, 글). 표는 행마다 칸을 ' | '로."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(path))
    out = []
    for el in d.element.body.iterchildren():
        tag = el.tag.split("}")[-1]
        if tag == "p":
            p = Paragraph(el, d)
            if p.text.strip():
                out.append((p.style.name if p.style is not None else "", p.text))
        elif tag == "tbl":
            for row in Table(el, d).rows:
                cells = []
                for c in row.cells:
                    t = " ".join(c.text.split())
                    if not cells or cells[-1] != t:
                        cells.append(t)
                out.append(("표", " | ".join(cells)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", default="human_review", help="out/ 아래 폴더 이름")
    ns = ap.parse_args()
    global REV
    REV = OUT / ns.out
    REV.mkdir(parents=True, exist_ok=True)
    specs = [x.split(":", 1) for x in Path(ns.list).read_text(encoding="utf-8").split("\n") if x.strip()]
    rows = []
    for n, (ds, doc_id) in enumerate(specs, 1):
        r = json.loads((OUT / ds / "processed_para" / f"{doc_id}.json").read_text(encoding="utf-8"))
        meta = json.loads((OUT / ds / "meta_text_v2" / f"{doc_id}.json").read_text(encoding="utf-8"))
        dropped = json.loads((OUT / ds / "processed_para" / f"{doc_id}.dropped.json").read_text(encoding="utf-8"))
        units = r["units"]
        logs = ", ".join(f"{k} {v}" for k, v in r["log"].items() if not k.startswith(("W", "C1", "C2", "빈")))
        head = (f"<h1>{n:02d}. {html.escape(ds)} / {html.escape(doc_id)}</h1><div class='meta'>{r['fmt']} · 쪽 {r.get('n_pages')} · 단위 {len(units)} · "
                f"토큰 {r['n_tokens']:,} · <a href='index.html'>목록</a></div><div class='meta'>규칙: {html.escape(logs)}</div>{CHECK}")
        body = []
        if r["fmt"] == "pdf":
            by_page = defaultdict(list)
            for i, u in enumerate(units):
                by_page[(u["pages"] or [0])[0]].append(unit_html(i, u))
            drops = defaultdict(list)
            for dd in dropped:
                drops[dd.get("page") or 0].append(drop_html(dd))
            imgs = page_images(ROOT / meta["source_file"], meta["n_pages"])
            for pg, img in enumerate(imgs, 1):
                right = "".join(by_page.get(pg, [])) or "<div class='meta'>(이 쪽에서 시작하는 단위 없음)</div>"
                if drops.get(pg):
                    right += f"<div class='meta'>지운 블록 {len(drops[pg])}개</div>" + "".join(drops[pg])
                body.append(f"<div class='page'><div><div class='meta'>원본 {pg}쪽</div><img loading='lazy' src='data:image/jpeg;base64,{img}'></div><div>{right}</div></div>")
            if by_page.get(0) or drops.get(0):
                body.append("<h2>쪽 정보 없는 단위·지운 블록(각주 등)</h2>" + "".join(by_page.get(0, [])) + "".join(drops.get(0, [])))
        else:
            orig = docx_original(ROOT / meta["source_file"])
            left = "".join(f"<div class='orig'><span class='st'>{html.escape(st)}</span>{html.escape(t)}</div>" for st, t in orig)
            right = "".join(unit_html(i, u) for i, u in enumerate(units))
            body.append(f"<div class='page'><div><div class='meta'>원본(Word 문단 순서, 회색 = 스타일 이름)</div>{left}</div><div><div class='meta'>결과 단위</div>{right}</div></div>")
            body.append(f"<h2>지운 블록 {len(dropped)}개</h2>" + "".join(drop_html(dd) for dd in dropped))
        page = (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
                f"<title>검토 {n:02d}</title><style>{CSS}</style></head><body>{head}{''.join(body)}</body></html>")
        (REV / f"{n:02d}.html").write_text(page, encoding="utf-8")
        rows.append((n, ds, doc_id, r["fmt"], r.get("n_pages"), len(units), r["n_tokens"], len(dropped)))
        print(f"{n:02d} {ds}:{doc_id[:40]}", flush=True)
    idx = ["<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
           f"<title>전처리 검토</title><style>{CSS}</style></head><body><h1>processed_para 사람 검토: {html.escape(ns.out)} ({len(rows)}편)</h1>", CHECK,
           "<table class='idx'><tr><th>#</th><th>데이터셋</th><th>문서</th><th>형식</th><th>쪽</th><th>단위</th><th>토큰</th><th>지운 블록</th></tr>"]
    for n, ds, doc_id, fmt, pg, nu, nt, nd in rows:
        idx.append(f"<tr><td>{n:02d}</td><td>{ds}</td><td><a href='{n:02d}.html'>{html.escape(doc_id)}</a></td><td>{fmt}</td><td>{pg or ''}</td>"
                   f"<td>{nu}</td><td>{nt:,}</td><td>{nd}</td></tr>")
    idx.append("</table></body></html>")
    (REV / "index.html").write_text("\n".join(idx), encoding="utf-8")


if __name__ == "__main__":
    main()
