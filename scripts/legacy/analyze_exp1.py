"""실험 1 결과 정리·분석 (scripts/legacy/exp1_main.sh 의 결과를 읽는다). 새 청킹은 하지 않고, 검색은 크기 편향 보정용 k 변경에서만 다시 한다.

  source env.sh && .venv311/bin/python scripts/legacy/analyze_exp1.py [--no-budget]

출력 results/exp1/analysis.json 과 콘솔 표:
  1 요약 표(청크 수·평균 토큰·대표 지표·MRR·F1·top-3 토큰)      2 질문 유형별
  3 문서 단위 묶음 paired bootstrap 95% 구간(청커 쌍 차이)        4 청크 구조 지표(chunking/stats) + smart·semantic 청킹 통계
  5 크기 편향 보정: (a) docx 근거 정밀도·IoU(글자 단위)  (b) 같은 토큰 예산(fixed의 k를 키워 top-k 토큰 합을 semantic 수준으로)
  6 smart vs semantic 질문별 승패와 top-3 청크 일치율
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chunking.v1.stats import compute_stats  # noqa: E402
from common.config import CHUNKS_DIR, PROCESSED_DIR, RESULTS_DIR  # noqa: E402
from common.docjson import load_doc  # noqa: E402
from indexing.index import load_chunks  # noqa: E402
from qa_build.build_qa import load_qa  # noqa: E402
from tasks.retrieval import PRIMARY, span_coverage  # noqa: E402

DATASETS = ["vectara", "allganize", "docx_corpus"]
TAGS = {"fixed": "fixed__c500", "semantic": "semantic__p85-min128-max512", "smart": "smart__p85-min128-max512"}
OUT = RESULTS_DIR / "exp1"
MIN, MAX = 128, 512


def load_results():
    return {ds: {n: json.loads((RESULTS_DIR / ds / "retrieval" / f"{t}.json").read_text(encoding="utf-8")) for n, t in TAGS.items()} for ds in DATASETS}


def metric_of(row, prim):
    return {"hit": float(row["hit"]), "mrr": row["mrr"], "f1": row["f1"], **({prim: float(row[prim])} if prim in row else {})}


def bootstrap(rows_a, rows_b, key, B=2000, seed=0):
    """질문 쌍(qid 기준)의 평균 차이(a-b)의 문서 단위 묶음 부트스트랩 95% 구간."""
    ra, rb = {r["qid"]: r for r in rows_a}, {r["qid"]: r for r in rows_b}
    qids = [q for q in ra if q in rb and key in ra[q] and key in rb[q]]
    by_doc = {}
    for q in qids:
        by_doc.setdefault(ra[q]["doc_id"], []).append(float(ra[q][key]) - float(rb[q][key]))
    docs = list(by_doc.values())
    flat = np.array([x for d in docs for x in d])
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(B):
        pick = rng.integers(0, len(docs), len(docs))
        v = np.concatenate([docs[i] for i in pick])
        means.append(v.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff": float(flat.mean()), "ci": [float(lo), float(hi)], "n": len(flat), "docs": len(docs)}


def fmt(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def chunk_spans_by_id(ds, tag):
    return {c["chunk_id"]: c for c in load_chunks(ds, tag)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-budget", action="store_true", help="같은 토큰 예산 재검색(GPU)을 건너뜀")
    args = ap.parse_args()
    res = load_results()
    out = {"summary": {}, "by_qtype": {}, "bootstrap": {}, "structure": {}, "precision": {}, "budget": {}, "smart_vs_semantic": {}}

    # 1 요약 -----------------------------------------------------------------
    print("## 1. 요약 (top-3, 질문 전체)")
    for ds in DATASETS:
        prim = PRIMARY.get(ds)
        print(f"\n[{ds}] 질문 {res[ds]['smart']['n_questions']}개, 문서 {res[ds]['smart']['n_docs']}개")
        print("| 청커 | 청크 | 평균토큰 | 최대 | hit@3 | MRR | F1 | " + (f"{prim}@3 | " if prim else "") + "top-3 토큰 |")
        print("|---|---|---|---|---|---|---|" + ("---|" if prim else "") + "---|")
        for n in TAGS:
            r = res[ds][n]; o = r["overall"]
            out["summary"].setdefault(ds, {})[n] = {"n_chunks": r["index"]["n_chunks"], "mean_tokens": r["index"]["chunk_tokens_mean"], "max_tokens": r["index"]["chunk_tokens_max"], **{k: v for k, v in o.items()}}
            print(f"| {n} | {r['index']['n_chunks']} | {r['index']['chunk_tokens_mean']:.0f} | {r['index']['chunk_tokens_max']} | {fmt(o['hit@k'])} | {fmt(o['mrr'])} | {fmt(o['f1'])} | "
                  + (f"{fmt(o.get(prim + '@k'))} | " if prim else "") + f"{o['top_k_tokens_mean']:.0f} |")

    # 2 질문 유형별 -----------------------------------------------------------
    print("\n## 2. 질문 유형별 (대표 지표 또는 MRR)")
    for ds in DATASETS:
        prim = PRIMARY.get(ds)
        key = (prim + "@k") if prim else "mrr"
        types = sorted(res[ds]["smart"]["by_qtype"])
        print(f"\n[{ds}] 지표 {key}")
        print("| 유형 | n | " + " | ".join(TAGS) + " |\n|---|---|" + "---|" * len(TAGS))
        for t in types:
            vals = [res[ds][n]["by_qtype"][t].get(key) for n in TAGS]
            out["by_qtype"].setdefault(ds, {}).setdefault(t, {n: v for n, v in zip(TAGS, vals)})
            print(f"| {t} | {res[ds]['smart']['by_qtype'][t]['n']} | " + " | ".join(fmt(v) for v in vals) + " |")

    # 3 부트스트랩 -------------------------------------------------------------
    print("\n## 3. 청커 쌍 차이의 95% 구간 (문서 단위 묶음 부트스트랩, 2000회)")
    pairs = [("smart", "semantic"), ("smart", "fixed"), ("semantic", "fixed")]
    for ds in DATASETS:
        prim = PRIMARY.get(ds)
        keys = ["hit", "mrr", "f1"] + ([{"section_hit": "section_hit", "evidence_recall": "evidence_recall"}[prim]] if prim else [])
        print(f"\n[{ds}]")
        print("| 비교 | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys))
        for a, b in pairs:
            cells = []
            for k in keys:
                bs = bootstrap(res[ds][a]["rows"], res[ds][b]["rows"], k)
                out["bootstrap"].setdefault(ds, {}).setdefault(f"{a}-{b}", {})[k] = bs
                sig = "*" if (bs["ci"][0] > 0 or bs["ci"][1] < 0) else ""
                cells.append(f"{bs['diff']:+.3f} [{bs['ci'][0]:+.3f},{bs['ci'][1]:+.3f}]{sig}")
            print(f"| {a}-{b} | " + " | ".join(cells) + " |")
    print("(* = 95% 구간이 0을 포함하지 않음)")

    # 4 구조 지표 ---------------------------------------------------------------
    print("\n## 4. 청크 구조 지표 (실제 토큰 기준, min 128 / max 512)")
    for ds in DATASETS:
        docs = {}
        for n, t in TAGS.items():
            ch = load_chunks(ds, t)
            for d in {c["doc_id"] for c in ch} - set(docs):
                docs[d] = load_doc(PROCESSED_DIR / ds / f"{d}.json")
        print(f"\n[{ds}]")
        print("| 청커 | under_min | over_max | 문장중간끝 | 제목만남음 | 표 갈림 | 표 수 |\n|---|---|---|---|---|---|---|")
        for n, t in TAGS.items():
            st = compute_stats(load_chunks(ds, t), docs, MIN, MAX)
            out["structure"].setdefault(ds, {})[n] = st
            print(f"| {n} | {fmt(st.get('under_min_rate'), 4)} | {fmt(st.get('over_max_rate'), 4)} | {fmt(st.get('sentence_cut_rate'), 4)} | {fmt(st.get('orphan_heading_rate'), 4)} | {fmt(st.get('table_split_rate'), 4)} | {st.get('n_tables')} |")
        for n in ("semantic", "smart"):
            m = json.loads((CHUNKS_DIR / ds / f"{TAGS[n]}.meta.json").read_text(encoding="utf-8"))["info_totals"]
            out["structure"][ds][n + "_engine"] = m
            keep = {k.split("_", 1)[1]: v for k, v in m.items() if k.split("_", 1)[1] in ("units", "chunks", "threshold_cuts", "size_cuts", "under_min", "over_max_single_unit", "min_shortfall", "orig_sections", "section_merges", "cross_section_chunks")}
            print(f"  {n} 엔진 통계: {keep}")

    # 5 크기 편향 보정 ----------------------------------------------------------
    print("\n## 5. 크기 편향 보정")
    qa_docx = {q["qid"]: q for q in load_qa("docx_corpus")}
    print("\n(a) docx 근거 정밀도·IoU (top-3 청크의 글자 범위 대비 근거 구간, 정답 문서 청크만)")
    print("| 청커 | evidence_recall | 정밀도(근거/가져온 글자) | IoU |\n|---|---|---|---|")
    for n, t in TAGS.items():
        ch = chunk_spans_by_id("docx_corpus", t)
        rec, pre, iou = [], [], []
        for r in res["docx_corpus"][n]["rows"]:
            ev = qa_docx[r["qid"]]["evidence"]
            got = [ch[c] for c in r["top_k"] if ch[c]["doc_id"] == r["doc_id"]]
            if not got:
                rec.append(0.0); pre.append(0.0); iou.append(0.0); continue
            ivs = sorted((c["start"], c["end"]) for c in got)
            merged = []
            for s, e in ivs:
                if merged and s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            total = sum(e - s for s, e in merged)
            inter = sum(max(0, min(e, ev["end"]) - max(s, ev["start"])) for s, e in merged)
            rec.append(inter / (ev["end"] - ev["start"])); pre.append(inter / total if total else 0.0)
            iou.append(inter / (total + (ev["end"] - ev["start"]) - inter))
        out["precision"][n] = {"recall": float(np.mean(rec)), "precision": float(np.mean(pre)), "iou": float(np.mean(iou))}
        print(f"| {n} | {np.mean(rec):.3f} | {np.mean(pre):.3f} | {np.mean(iou):.3f} |")

    if not args.no_budget:
        from common.models import get_embeddings
        from indexing.index import get_index, make_encoder
        from tasks.retrieval import evaluate
        encode = make_encoder(get_embeddings()._client)
        print("\n(b) 같은 토큰 예산: fixed의 k를 키워 top-k 토큰 합을 semantic top-3 수준으로 맞춤")
        print("| 데이터셋 | 설정 | k | top-k 토큰 | hit | MRR | 대표 지표 |\n|---|---|---|---|---|---|---|")
        for ds in DATASETS:
            prim = PRIMARY.get(ds)
            target = res[ds]["semantic"]["overall"]["top_k_tokens_mean"]
            fixed_mean = res[ds]["fixed"]["index"]["chunk_tokens_mean"]
            kk = max(3, int(round(target / fixed_mean)))
            tag = TAGS["fixed"]
            docs = sorted({c["doc_id"] for c in load_chunks(ds, tag)})
            idx = get_index(ds, tag, encode, doc_ids=docs, max_seq_length=8192)
            r = evaluate(idx, load_qa(ds, doc_ids=docs), k=kk)
            o = r["overall"]; s = res[ds]["semantic"]["overall"]; m = res[ds]["smart"]["overall"]
            out["budget"][ds] = {"k_fixed": kk, "fixed": o, "semantic": s, "smart": m}
            pk = (prim + "@k") if prim else None
            for name, oo, k in (("fixed(예산 맞춤)", o, kk), ("semantic", s, 3), ("smart", m, 3)):
                print(f"| {ds} | {name} | {k} | {oo['top_k_tokens_mean']:.0f} | {oo['hit@k']:.3f} | {oo['mrr']:.3f} | " + (f"{oo.get(pk):.3f}" if pk and pk in oo else "-") + " |")

    # 6 smart vs semantic -------------------------------------------------------
    print("\n## 6. smart vs semantic 질문별")
    for ds in DATASETS:
        prim = PRIMARY.get(ds)
        key = prim if prim else "mrr"
        sm, se = {r["qid"]: r for r in res[ds]["smart"]["rows"]}, {r["qid"]: r for r in res[ds]["semantic"]["rows"]}
        win = lose = tie = 0
        for q in sm:
            a, b = float(sm[q].get(key, 0)), float(se[q].get(key, 0))
            win += a > b + 1e-9; lose += a < b - 1e-9; tie += abs(a - b) <= 1e-9
        # top-3 청크가 같은 글자 범위인가 (청크 id가 달라도 [start,end)가 같으면 같은 청크로 본다)
        cs, ce = chunk_spans_by_id(ds, TAGS["smart"]), chunk_spans_by_id(ds, TAGS["semantic"])
        same = []
        for q in sm:
            A = {(cs[c]["start"], cs[c]["end"]) for c in sm[q]["top_k"]}
            B = {(ce[c]["start"], ce[c]["end"]) for c in se[q]["top_k"]}
            same.append(len(A & B) / 3)
        # 청크 자체가 같은 비율
        sa = {(c["doc_id"], c["start"], c["end"]) for c in cs.values()}
        sb = {(c["doc_id"], c["start"], c["end"]) for c in ce.values()}
        ident = len(sa & sb) / len(sa)
        big = [c for c in cs.values() if c["n_tokens"] > MAX]
        out["smart_vs_semantic"][ds] = {"metric": key, "smart_better": win, "semantic_better": lose, "tie": tie, "top3_same_chunk_share": float(np.mean(same)),
                                        "identical_chunk_share": ident, "smart_chunks_over_max": len(big)}
        print(f"[{ds}] {key}: smart 우세 {win} / semantic 우세 {lose} / 동점 {tie} | top-3 청크 일치율 {np.mean(same):.2f} | 두 청커에서 똑같은 청크 비율(smart 기준) {ident:.2f} | smart max 초과 청크 {len(big)}개")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n[저장] {OUT / 'analysis.json'}")


if __name__ == "__main__":
    main()
