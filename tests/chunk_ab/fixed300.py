"""청커 A/B(tests 전용): 같은 고정 글자 수 분할로 Vectara 원문 JSON 텍스트와 우리 전처리(processed_para)를 각각 청킹해
같은 질문·같은 임베딩으로 검색 지표를 비교한다. 전처리가 검색에 쓸 만한지(원문과 비슷한지) 보는 확인용.

  source env.sh && .venv311/bin/python tests/chunk_ab/fixed300.py [--size 300] [--overlap 0]

입력
  vectara : data/raw/vectara_pdf/_meta/corpus/<id>.json의 sections[].text를 순서대로 이어 붙임(그림 자리 표시 '![..](..)'만 뺌)
  ours    : tests/stage1_v2/out/vectara/processed_para/<id>.json의 units[].text를 이어 붙임
  문서: ref_eval에서 기준 불완전이 아닌 984편, 질문: 그 문서를 정답으로 하는 질문(queries.json + qrels.json)
청커: LangChain CharacterTextSplitter(separator="", chunk_size, chunk_overlap) = 순수 글자 수 분할
검색: bge-m3 fp16 임베딩, FAISS IndexFlatIP(정확 내적 = 코사인), 전체 청크 한 인덱스, 상위 10
지표
  문서 적중률@k : 상위 k 청크 중 정답 문서 청크가 있음
  절 적중률@k   : 상위 k 청크 중 정답 절 글을 담은 청크가 있음(청크의 6낱말 묶음 절반 이상이 정답 절 텍스트에 있음; 양쪽 같은 판정)
  MRR@10        : 정답 문서 청크의 첫 순위 역수(문서 기준)
출력: tests/chunk_ab/out/fixed<size>/{chunks_*.jsonl, results.jsonl, summary.md}
"""
import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
META = ROOT / "data" / "raw" / "vectara_pdf" / "_meta"
PARA = ROOT / "tests" / "stage1_v2" / "out" / "vectara" / "processed_para"
EVAL = ROOT / "tests" / "stage1_v2" / "out" / "ref_eval" / "docs.jsonl"
TOK = re.compile(r"[A-Za-z]{3,}")


def toks(t):
    return [w.lower() for w in TOK.findall(unicodedata.normalize("NFKC", t))]


def no_math(t):
    """절 적중 판정용: LaTeX 수식($…$, $$…$$, \\[…\\], \\begin…\\end, 남은 \\명령)을 지운 글. Vectara 원문은 수식이 LaTeX라 명령어(mathrm, left, frac …)가
    낱말처럼 세어져 원문 청크만 정답 절과 쉽게 겹친다(실측: 우리만 놓친 질문의 '없는 부분' 68%가 수식 줄). 양쪽에 같은 처리를 한다."""
    t = re.sub(r"\$\$.*?\$\$|\$[^$]*\$|\\\[.*?\\\]|\\begin\{.*?\\end\{[^}]*\}", " ", t, flags=re.S)
    return re.sub(r"\\[A-Za-z]+", " ", t)


def grams(ws):
    return {tuple(ws[i:i + 6]) for i in range(len(ws) - 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=0)
    ns = ap.parse_args()
    out = HERE / "out" / (f"fixed{ns.size}" + (f"_ov{ns.overlap}" if ns.overlap else ""))
    out.mkdir(parents=True, exist_ok=True)

    from langchain_text_splitters import CharacterTextSplitter
    splitter = CharacterTextSplitter(separator="", chunk_size=ns.size, chunk_overlap=ns.overlap)

    ev = [json.loads(x) for x in EVAL.read_text(encoding="utf-8").splitlines() if x.strip()]
    docs = sorted(r["doc_id"] for r in ev if not r["ref_incomplete"])
    qs = json.loads((META / "queries.json").read_text(encoding="utf-8"))
    qrels = json.loads((META / "qrels.json").read_text(encoding="utf-8"))
    dset = set(docs)
    qids = [q for q in qs if qrels[q]["doc_id"] in dset]
    sections = {}
    corpora = {"vectara": [], "ours": []}           # (doc_id, 청크 글)
    for d in docs:
        c = json.loads((META / "corpus" / f"{d}.json").read_text(encoding="utf-8"))
        sections[d] = [s["text"] for s in c["sections"]]
        vt = "\n\n".join(re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s["text"]) for s in c["sections"])
        ot = "\n\n".join(u["text"] for u in json.loads((PARA / f"{d}.json").read_text(encoding="utf-8"))["units"])
        corpora["vectara"] += [(d, t) for t in splitter.split_text(vt)]
        corpora["ours"] += [(d, t) for t in splitter.split_text(ot)]
    for name, ch in corpora.items():
        with open(out / f"chunks_{name}.jsonl", "w", encoding="utf-8") as fh:
            for d, t in ch:
                fh.write(json.dumps({"doc_id": d, "text": t}, ensure_ascii=False) + "\n")
    print(f"문서 {len(docs)}, 질문 {len(qids)}, 청크 vectara {len(corpora['vectara']):,} / ours {len(corpora['ours']):,}", flush=True)

    import faiss
    from common.models import load_embedder, make_encode
    model = load_embedder()
    encode = make_encode(model, batch_size=128)
    qv = encode([qs[q]["query"] for q in qids])     # float32, 길이 1
    gold_grams = {}
    rows = {q: {"qid": q, "doc_id": qrels[q]["doc_id"], "section_id": qrels[q]["section_id"], "type": qs[q]["type"], "source": qs[q]["source"]} for q in qids}
    for name, ch in corpora.items():
        t0 = time.time()
        order = sorted(range(len(ch)), key=lambda i: len(ch[i][1]))   # 길이순으로 묶어 임베딩(빠르게), 결과는 원래 순서로
        vec = np.zeros((len(ch), 1024), np.float32)
        step = 20000
        for s in range(0, len(order), step):
            idx = order[s:s + step]
            vec[idx] = encode([ch[i][1] for i in idx])
        index = faiss.IndexFlatIP(vec.shape[1])     # FAISS 정확 내적 검색(정규화된 벡터 → 코사인)
        index.add(vec)
        _, top = index.search(qv, 10)
        del index, vec
        for qi, q in enumerate(qids):
            d, sid = qrels[q]["doc_id"], qrels[q]["section_id"]
            key = (d, sid)
            if key not in gold_grams:
                gold_grams[key] = (grams(toks(sections[d][sid])) if sid < len(sections[d]) else set(),
                                   grams(toks(no_math(sections[d][sid]))) if sid < len(sections[d]) else set())
            gg, gm = gold_grams[key]
            docs_hit, sec_hit, sec_nm = [], [], []
            for i in top[qi]:
                cd, ct = ch[i]
                docs_hit.append(cd == d)
                g = grams(toks(ct))
                sec_hit.append(cd == d and bool(g) and len(g & gg) / len(g) >= 0.5)
                g2 = grams(toks(no_math(ct)))
                sec_nm.append(cd == d and len(g2) >= 3 and len(g2 & gm) / len(g2) >= 0.5)   # 수식 뺀 판정
            rows[q][name] = {"doc": docs_hit, "sec": sec_hit, "sec_nm": sec_nm, "top": [int(i) for i in top[qi]]}
        print(f"{name}: 임베딩·검색 {time.time() - t0:.0f}초", flush=True)
    with open(out / "results.jsonl", "w", encoding="utf-8") as fh:
        for r in rows.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def metric(name, kind, k):
        return np.mean([any(r[name][kind][:k]) for r in rows.values()])

    def mrr(name):
        return np.mean([next((1 / (i + 1) for i, h in enumerate(r[name]["doc"]) if h), 0) for r in rows.values()])

    def boot(name_a, name_b, f, n=2000):          # 질문 단위 짝 부트스트랩(차이의 95% 구간)
        rng = np.random.default_rng(0)
        a = np.array([f(r[name_a]) for r in rows.values()], float)
        b = np.array([f(r[name_b]) for r in rows.values()], float)
        diffs = [np.mean(b[ix] - a[ix]) for ix in (rng.integers(0, len(a), len(a)) for _ in range(n))]
        return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
    fns = {"문서 적중률@5": lambda x: any(x["doc"][:5]), "절 적중률@5(수식 뺀 판정)": lambda x: any(x["sec_nm"][:5]),
           "절 적중률@1(수식 뺀 판정)": lambda x: x["sec_nm"][0], "절 적중률@5(옛 판정: 수식 포함)": lambda x: any(x["sec"][:5]),
           "MRR@10(문서)": lambda x: next((1 / (i + 1) for i, h in enumerate(x["doc"]) if h), 0),
           "문서 적중률@1": lambda x: x["doc"][0], "절 적중률@1": lambda x: x["sec"][0]}
    lines = [f"# 고정 {ns.size}자 분할(겹침 {ns.overlap}) — Vectara 원문 vs 우리 전처리", "",
             f"문서 {len(docs)}편, 질문 {len(qids)}개, 청크 수: Vectara 원문 {len(corpora['vectara']):,} / 우리 {len(corpora['ours']):,}", "",
             "| 지표 | Vectara 원문 | 우리 processed_para | 차이(우리-원문) | 차이 95% 구간 |", "|---|---|---|---|---|"]
    for lab, f in fns.items():
        va = np.mean([f(r["vectara"]) for r in rows.values()])
        ou = np.mean([f(r["ours"]) for r in rows.values()])
        lo, hi = boot("vectara", "ours", f)
        lines.append(f"| {lab} | {va:.3f} | {ou:.3f} | {ou - va:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    for src in ("text", "text-table", "text-image", "text-table-image"):
        rs = [r for r in rows.values() if r["source"] == src]
        if rs:
            lines.append(f"| 절 적중률@5(수식 뺀 판정) · 질문 출처 {src} ({len(rs)}) | {np.mean([any(r['vectara']['sec_nm'][:5]) for r in rs]):.3f} | "
                         f"{np.mean([any(r['ours']['sec_nm'][:5]) for r in rs]):.3f} | | |")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
