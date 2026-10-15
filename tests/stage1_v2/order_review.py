"""순서 오류 후보 검토 페이지(tests 전용): order_check.py가 찾은 '자리 이동' 단위 중 쪽 배치로 맞다고 판정되지 않은 후보를
원본 쪽 이미지 · 우리 순서(앞 단위 → 이동 단위 → 다음 단위) · Vectara 원문에서 그 글이 있는 자리의 앞뒤와 나란히 보여 준다.

  .venv311/bin/python tests/stage1_v2/order_review.py        # order_check.py와 배치 판정(out/order_check/geo_verdict.json)을 먼저 만들 것

출력: out/order_review/index.html(후보 목록) + 후보마다 <번호>.html, out/order_review/candidates.json(판정 기록용)
"""
import base64
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from make_review import CSS  # noqa: E402
from ref_eval import CORPUS, OUT  # noqa: E402

RES = OUT / "order_review"
GROUPS = ["같은 단·위(우리 오류)", "오른→왼 단(우리 오류?)", "앞 쪽으로 되돌아감", "다른 쪽(쪽 넘김)", "앞 단위 없음"]
WORD = re.compile(r"[A-Za-z]{3,}")


def vectara_raw(doc_id):
    d = json.loads((CORPUS / f"{doc_id}.json").read_text(encoding="utf-8"))
    return "\n\n".join(s["text"] for s in d["sections"])


def locate(raw, text):
    """우리 단위 글의 낱말 5개 연속을 원문에서 찾는다(사이에 낱말 아닌 글자 허용). 앞쪽부터 여러 곳을 시도."""
    ws = WORD.findall(text)
    for k in range(0, max(1, len(ws) - 4), 3):
        pat = r"\W+(?:\S*\W+){0,2}?".join(map(re.escape, ws[k:k + 5]))
        m = re.search(pat, raw, re.I)
        if m:
            return m.start()
    return None


def page_png(pdf, pg):
    import pymupdf
    with pymupdf.open(str(pdf)) as d:
        if 1 <= pg <= len(d):
            return base64.b64encode(d[pg - 1].get_pixmap(dpi=80).tobytes("png")).decode()
    return None


def main():
    RES.mkdir(parents=True, exist_ok=True)
    geo = json.loads((OUT / "order_check" / "geo_verdict.json").read_text(encoding="utf-8"))
    cands = [(g, *x) for g in GROUPS for x in geo.get(g, [])]
    rows = []
    for n, (group, doc_id, pg, ui, _) in enumerate(cands, 1):
        units = json.loads((OUT / "vectara" / "processed_para" / f"{doc_id}.json").read_text(encoding="utf-8"))["units"]
        meta = json.loads((OUT / "vectara" / "meta_text_v2" / f"{doc_id}.json").read_text(encoding="utf-8"))
        raw = vectara_raw(doc_id)
        u = units[ui]
        prev = next((units[j] for j in range(ui - 1, -1, -1) if units[j]["kind"] not in ("table", "footnote")), None)
        nxt = next((units[j] for j in range(ui + 1, len(units)) if units[j]["kind"] not in ("table", "footnote")), None)
        pos = locate(raw, u["text"])
        ppos = locate(raw, prev["text"][-400:]) if prev else None
        vctx = (html.escape(raw[max(0, pos - 700):pos]) + "<mark>" + html.escape(raw[pos:pos + 500]) + "</mark>" + html.escape(raw[pos + 500:pos + 800])) if pos is not None else "(원문에서 못 찾음)"
        pages = sorted({p for x in (prev, u) if x for p in x["pages"][:2]})
        imgs = "".join(f"<div><div class='meta'>원본 {p}쪽</div><img src='data:image/png;base64,{b}'></div>"
                       for p in pages[:3] for b in [page_png(ROOT / meta["source_file"], p)] if b)
        unit = lambda x, tag: (f"<div class='unit'><span class='tag'>{tag} · {x['kind']} · 쪽 {x['pages']}</span>{html.escape(x['text'][:900])}</div>" if x else "")
        body = (f"<h1>{n:02d}. {doc_id} — {html.escape(group)}</h1><div class='meta'><a href='index.html'>목록</a> · 이동 단위 [{ui}] · "
                f"원문 위치: 이동 단위 {pos}, 앞 단위 끝 {ppos} {'(앞 단위가 원문에서 더 뒤)' if pos is not None and ppos is not None and ppos > pos else ''}</div>"
                f"<div class='check'><b>볼 것</b>: 원본 쪽에서 읽는 순서대로 하면 '앞 단위 → 이동 단위'가 맞나? 맞으면 Vectara 쪽 차이, 아니면 우리 순서 오류</div>"
                f"<div class='page'><div>{imgs}</div><div><h2>우리 순서</h2>{unit(prev, '앞 단위')}{unit(u, '이동 단위')}{unit(nxt, '다음 단위')}"
                f"<h2>Vectara 원문(노란색 = 이동 단위 글이 있는 자리)</h2><div class='orig'>{vctx}</div></div></div>")
        (RES / f"{n:02d}.html").write_text(f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
                                           f"<title>순서 검토 {n:02d}</title><style>{CSS} mark{{background:#ffe58f}}</style></head><body>{body}</body></html>", encoding="utf-8")
        rows.append({"n": n, "group": group, "doc_id": doc_id, "page": pg, "unit": ui, "vectara_pos": pos, "prev_pos": ppos,
                     "prev": prev["text"][-300:] if prev else "", "text": u["text"][:600], "next": nxt["text"][:300] if nxt else "",
                     "vectara_ctx": raw[max(0, pos - 500):pos + 400] if pos is not None else ""})
    (RES / "candidates.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    idx = [f"<tr><td><a href='{r['n']:02d}.html'>{r['n']:02d}</a></td><td>{html.escape(r['group'])}</td><td>{r['doc_id']}</td><td>{r['page']}</td><td>{html.escape(r['text'][:80])}</td></tr>" for r in rows]
    (RES / "index.html").write_text(f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>순서 검토</title>"
                                    f"<style>{CSS}</style></head><body><h1>순서 오류 후보 {len(rows)}개</h1><table class='idx'><tr><th>#</th><th>판정 그룹</th><th>문서</th><th>쪽</th><th>이동 단위</th></tr>"
                                    + "".join(idx) + "</table></body></html>", encoding="utf-8")
    print(len(rows), "후보 →", RES / "index.html")


if __name__ == "__main__":
    main()
