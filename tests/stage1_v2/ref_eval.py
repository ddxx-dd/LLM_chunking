"""vectara 전처리 자동 채점(tests 전용): Vectara 공식 절 텍스트(Mistral OCR, data/raw/vectara_pdf/_meta/corpus)를 기준 답안으로
우리 결과(processed_para)와 옛 결과(data/meta_text 블록 이음)를 비교한다.

  .venv311/bin/python tests/stage1_v2/ref_eval.py [--list 목록]

기준 답안 만들기: 공식 절 중 참고문헌 절(제목이 References/Bibliography) 제외, 수식($…$, $$…$$, 남은 LaTeX 명령)·그림 표시·표 자리 표시·표 줄·그림/표 캡션 줄('Fig. 3:'처럼 번호 뒤 구두점) 제거,
절 안의 참고문헌 제목부터 다음 제목 전까지 제외.
우리 쪽 범위: 표 단위 제외, 기준 답안 첫 40낱말(또는 Abstract·Introduction)이 나오는 단위 앞(제목·저자·소속) 제외. 공식 텍스트가 우리보다 많이 짧으면(60% 미만) '기준 불완전'으로 따로 빼고 통계에서 제외
(Vectara는 '#' 제목에서만 절을 나눠, 제목 없는 본문을 버린 문서가 있다: 실측 2411.13556v1은 참고문헌만 남음).
비교 낱말: NFKC 소문자, 3글자 이상 알파벳 낱말(수식 기호·숫자 표기 차이에 흔들리지 않게).
  본문 재현  = 기준 낱말 중 우리 결과에 있는 비율(빠뜨림이 적을수록 높음)
  잡음 비율  = 우리 결과 낱말 중 기준에 없는 비율(그림 글자·머리말·참고문헌 등이 남을수록 높음)
  순서 일치  = 우리 결과의 연속 6낱말 묶음 중 기준에서 딱 한 번 나오는 것들의 기준 위치가 증가하는 비율(최장 증가 부분열 / 전체)
  순서 끊김  = 그 부분열에서 벗어난 덩어리(묶음 3개 이상) 수와 위치. 긴 문서에 한 문단이 끼어도 '1곳'으로 드러난다
출력: out/ref_eval/docs.jsonl, out/ref_eval/summary.md
"""
import argparse
import bisect
import json
import re
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
ROOT = HERE.parents[1]
OUT = HERE / "out"
CORPUS = ROOT / "data" / "raw" / "vectara_pdf" / "_meta" / "corpus"
W = re.compile(r"[a-z]{3,}")
REFS = re.compile(r"^#+\s*\**\s*(\d+\.?\s*)?((bibliographical\s+)?references?(\s+list)?|bibliography|literature\s+cited)\s*\**\s*:?\s*$", re.I | re.M)
CAPTION = re.compile(r"^\s*\**(sup(p|plementary)?\.?\s*)?(extended data\s*)?(fig(ure)?s?\.?|table)\s*([A-Z]\.?)?\d+(\.\d+)*[a-z]?\**\s*[:.|](?!\d)", re.I)  # 'Fig. 5 shows'·'Figure 4.5 illustrates'(본문)는 남김
HEAD = re.compile(r"^\s*#+\s")
REFS_LINE = re.compile(r"^\s*(#+\s*)?\**\s*(\d+\.?\s*)?((bibliographical\s+)?references?(\s+list)?|bibliography|literature\s+cited)\s*\**\s*:?\s*$", re.I)  # 2차 R1과 같은 제목 범위('Reference:' 포함)
FRONT_END = re.compile(r"^\s*(#+\s*)?((\d+\.?|[IVX]+\.)\s*)?(abstract|introduction)\b", re.I)


REF_ITEM_START = re.compile(r"^\s*(-\s*)?(\[\d+\]\s*|\d+\.\s+)?([A-Z][A-Za-z'´`\-]+,\s*([A-Z]\.|[A-Z][a-z])|[A-Z][A-Za-z'\-]+\s[A-Z]{1,2}[,.]|[A-Z]\.\s*([A-Z]\.\s*)?[A-Z][a-z]+,)")
REF_MARK = re.compile(r"et al\.|\bvol\.|\bpp\.|\bdoi\b|arxiv|journal|proc\.|conference|press|phys\.|rev\.|lett\.|\[\d+\]|[A-Z][a-z]+,\s*[A-Z]\.", re.I)
YEAR_RE = re.compile(r"\b(19|20)\d{2}[a-z]?\b")


def ref_item_line(l):
    """제목 없이 절 안에 들어 있는 참고문헌 항목 줄('- [3] Smith, J., … 2020.', 'M. Campisi, R. …, Phys. Rev. Lett. … (2020)'): 우리 2차 R2와 같은 판별식."""
    from preprocess.stage2 import REF_START, REF_STRICT, YEAR
    l = re.sub(r"^\s*-\s+", "", l)
    n = len(l.split())
    return bool(n <= 120 and (REF_START.match(l) or re.match(r"^\s*(\[\d+\]|\d+\.\s)", l)) and YEAR.search(l) and len(REF_STRICT.findall(l)) >= 2)


def our_text(units, ref_head=None):
    """우리 결과 중 기준 답안과 같은 범위: 표 단위 제외, 초록·서론 앞의 제목·저자·소속 단위 제외(기준 답안은 Abstract부터)."""
    start = next((i for i, u in enumerate(units) if FRONT_END.match(u["text"]) or FRONT_END.match(u.get("heading") or "")), 0)
    if ref_head:                                   # 기준 답안 첫 부분(초록 앞 40낱말)이 처음 나오는 단위부터(초록에 'Abstract' 표시가 없는 문서)
        g = {tuple(ref_head[i:i + 6]) for i in range(len(ref_head) - 5)}
        hit = next((i for i, u in enumerate(units) if u["kind"] != "table" and g & {tuple(w[j:j + 6]) for w in [words(u["text"])] for j in range(len(w) - 5)}), None)
        if hit is not None:
            start = min(start, hit)
    return "\n".join(u["text"] for u in units[start:] if u["kind"] != "table")


def load_units(doc_id, para="processed_para"):
    """채점할 단위 목록. para='old'면 옛 전처리(data/meta_text 블록)를 단위처럼(표·각주 종류 유지) 읽는다."""
    if para == "old":
        blocks = json.loads((ROOT / "data" / "meta_text" / "vectara" / f"{doc_id}.json").read_text(encoding="utf-8"))["blocks"]
        kind = {"table": "table", "footnote": "footnote"}
        return [{"text": b["text"], "kind": kind.get(b["type"], "para"), "pages": [b["page"]] if b.get("page") else [], "block_ids": [b["id"]],
                 "heading": ""} for b in blocks]
    return json.loads((OUT / "vectara" / para / f"{doc_id}.json").read_text(encoding="utf-8"))["units"]


def words(t):
    return W.findall(unicodedata.normalize("NFKC", t).lower())


def reference_text(doc_id):
    f = CORPUS / f"{doc_id}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    parts = []
    for s in d["sections"]:
        t = s["text"]
        if REFS.match(t.lstrip().split("\n")[0]):
            continue
        keep, in_refs = [], False                  # 절 안의 참고문헌 제목(예: '# ACKNOWLEDGMENTS' 절 안 '## REFERENCES')부터 다음 제목 전까지 제외
        for l in t.split("\n"):
            if HEAD.match(l) or REFS_LINE.match(l):  # 제목 표시(#) 없이 'Reference:' 한 줄로 시작하는 참고문헌도
                in_refs = bool(REFS_LINE.match(l))
            if not in_refs and not l.lstrip().startswith("|") and not ref_item_line(l):   # 표 줄(| … |)·참고문헌 항목 줄도 제외: 우리 쪽도 뺀다
                keep.append(l)
        t = "\n".join(keep)
        t = re.sub(r"\$\$.*?\$\$|\$[^$]*\$", " ", t, flags=re.S)          # 수식
        t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)                        # 그림·표 자리 표시
        t = re.sub(r"\\[A-Za-z]+", " ", t)                                   # 짝이 안 맞아 남은 LaTeX 명령(\mathrm 등)
        t = "\n".join(l for l in t.split("\n") if not CAPTION.match(l))  # 그림·표 캡션 줄(우리는 그림 캡션을 지우고 표 캡션은 표에 붙임)
        parts.append(t)
    return "\n".join(parts)


def lis_ratio(seq):
    tails = []
    for x in seq:
        i = bisect.bisect_left(tails, x)
        tails[i:i + 1] = [x]
    return len(tails) / max(1, len(seq))


def lis_keep(seq):
    """최장 증가 부분열에 드는 원소 표시(True/False 목록)."""
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


def order_breaks(ref, ours, min_run=3):
    """순서 끊김: 우리 글의 6낱말 묶음(기준에 한 번만 나오는 것) 중 최장 증가 부분열에서 벗어난 연속 덩어리(min_run 묶음 이상) 수와 위치.
    문서 전체 비율(순서 일치)은 긴 문서에서 한 문단이 끼어도 거의 안 내려가므로, 끊긴 곳을 '몇 곳'으로 센다."""
    rw, ow = words(ref), words(ours)
    pos = {}
    for i in range(len(rw) - 5):
        pos.setdefault(tuple(rw[i:i + 6]), []).append(i)
    idx = [(i, pos[k][0]) for i in range(len(ow) - 5) for k in [tuple(ow[i:i + 6])] if len(pos.get(k, [])) == 1]
    keep = lis_keep([p for _, p in idx])
    out, run = [], []
    for (i, _), k in zip(idx + [(None, None)], keep + [True]):
        if not k:
            run.append(i)
            continue
        if len(run) >= min_run:
            out.append({"at": round(run[0] / max(1, len(ow)), 3), "n": len(run), "text": " ".join(ow[run[0]:run[-1] + 6])[:160]})
        run = []
    return out


def score(ref, ours):
    rw, ow = words(ref), words(ours)
    rc, oc = Counter(rw), Counter(ow)
    recall = sum(min(v, oc[k]) for k, v in rc.items()) / max(1, len(rw))
    noise = 1 - sum(min(v, rc[k]) for k, v in oc.items()) / max(1, len(ow))
    pos = {}
    for i in range(len(rw) - 5):
        pos.setdefault(tuple(rw[i:i + 6]), []).append(i)
    seq = [pos[k][0] for k in (tuple(ow[i:i + 6]) for i in range(len(ow) - 5)) if len(pos.get(k, [])) == 1]
    return round(recall, 4), round(noise, 4), round(lis_ratio(seq), 4) if len(seq) >= 20 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list")
    ap.add_argument("--para", default="processed_para", help="out/vectara/ 아래 채점할 폴더(회귀 비교용 기준 사본 등)")
    ap.add_argument("--tag", default="", help="결과 파일 이름 뒤에 붙일 표시")
    ns = ap.parse_args()
    if ns.list:
        ids = [x.split(":", 1)[1] for x in Path(ns.list).read_text(encoding="utf-8").split("\n") if x.startswith("vectara:")]
    else:
        ids = sorted(p.stem for p in (OUT / "vectara" / ns.para).glob("*.json") if not p.name.endswith(".dropped.json"))
    (OUT / "ref_eval").mkdir(parents=True, exist_ok=True)
    rows = []
    for d in ids:
        ref = reference_text(d)
        pp = OUT / "vectara" / ns.para / f"{d}.json"
        old = ROOT / "data" / "meta_text" / "vectara" / f"{d}.json"
        if ref is None or not pp.exists():
            continue
        new_text = our_text(json.loads(pp.read_text(encoding="utf-8"))["units"], words(ref)[:40])
        old_text = "\n".join(b["text"] for b in json.loads(old.read_text(encoding="utf-8"))["blocks"])
        ratio = len(words(ref)) / max(1, len(words(new_text)))
        brk = order_breaks(ref, new_text)
        rows.append({"doc_id": d, "new": score(ref, new_text), "old": score(ref, old_text), "ref_words": len(words(ref)),
                     "ref_ratio": round(ratio, 2), "ref_incomplete": ratio < 0.6, "order_breaks": len(brk), "breaks": brk})
    with open(OUT / "ref_eval" / f"docs{ns.tag}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    bad_ref = [r for r in rows if r["ref_incomplete"]]
    rows_all, rows = rows, [r for r in rows if not r["ref_incomplete"]]
    med = lambda k, i: statistics.median(r[k][i] for r in rows if r[k][i] is not None)
    lines = [f"# vectara 자동 채점 (기준: Vectara 공식 절 텍스트, {len(rows)}편; 기준 불완전으로 제외 {len(bad_ref)}편)", "",
             "기준 불완전(공식 텍스트 낱말 수가 우리 결과의 60% 미만): " + ", ".join(f"{r['doc_id']}({r['ref_ratio']})" for r in bad_ref), "",
             "| | 본문 재현(높을수록 좋음) | 잡음 비율(낮을수록 좋음) | 순서 일치(높을수록 좋음) |", "|---|---|---|---|",
             f"| 옛 전처리(meta_text 블록 그대로) | {med('old', 0):.1%} | {med('old', 1):.1%} | {med('old', 2):.1%} |",
             f"| 새 전처리(processed_para) | {med('new', 0):.1%} | {med('new', 1):.1%} | {med('new', 2):.1%} |", "",
             f"순서 끊김(최장 증가 부분열에서 벗어난 6낱말 묶음 3개 이상 덩어리): 끊김 있는 문서 {sum(r['order_breaks'] > 0 for r in rows)}/{len(rows)}, "
             f"합계 {sum(r['order_breaks'] for r in rows)}곳, 3곳 이상 문서 {sum(r['order_breaks'] >= 3 for r in rows)}", "",
             "## 새 전처리 하위 문서(본문 재현 낮은 순 / 잡음 높은 순 / 순서 낮은 순)"]
    for i, name in ((0, "재현"), (1, "잡음"), (2, "순서")):
        key = (lambda r: r["new"][i] if r["new"][i] is not None else 1) if i != 1 else (lambda r: -r["new"][1])
        lines.append(f"- {name}: " + ", ".join(f"{r['doc_id']}({r['new'][i]})" for r in sorted(rows, key=key)[:8]))
    (OUT / "ref_eval" / f"summary{ns.tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
