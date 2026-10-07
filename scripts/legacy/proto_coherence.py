"""실험용 프로토타입(파이프라인에 넣지 않음): 청크 응집도 + 구조 비용 연속 군집화 DP vs 현재 semantic·smart.

  source env.sh
  .venv311/bin/python scripts/legacy/proto_coherence.py selftest     # 작은 입력에서 DP == 전수 탐색
  .venv311/bin/python scripts/legacy/proto_coherence.py run [--limit N]

방법 (연속 군집화 = 청크는 항상 연속한 문장 구간):
  노드 = 문장(smart와 같은 단위 규칙: 제목 붙임, 표 한 덩어리, 캡션 붙임; 섹션 분할·문단 단위 없음).
  구간 [a, b)의 응집도 비용 = 구간 안 임베딩이 평균에서 퍼진 정도 = m - ||S||^2 / m   (m = 문장 수, S = 임베딩 합; 누적합으로 O(1))
  총 비용 = 청크들의 응집도 비용 합 + lam * (청크 수) - 구조 보너스(제목 앞 경계를 자르면 beta * 2^-(수준-1))
  제약은 현재와 같은 사전식: 청크는 max 이하(단일 단위 예외) > min 부족분 합 최소 > 총 비용 최소.
  lam은 "청크 하나를 더 만들 만큼 응집도가 좋아져야 하는 양"이다. 청크 크기가 결과를 좌우하므로 lam은 직접 고르지 않고
  **청크 수가 smart와 같아지도록 이분 탐색**한다(같은 크기에서 비교). 튜닝하는 것은 beta(구조 보너스 세기) 배수 4개뿐이다.
평가: docx 94편(질문 422개), vectara 앞서 실험 1에 쓴 100편(질문은 실험 1 결과의 질문). 검색 지표는 실험 1과 같은 코드(tasks.retrieval.evaluate, k=3).
선택 편향 방지: beta는 tune 분할 대표 지표로 고르고 eval 분할로 보고하며, 전체 결과도 함께 기록한다.
"""
import argparse
import multiprocessing as mp
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "legacy"))

MIN, MAX = 128, 512
W = 5                                   # Delta 계산용 좌우 창(문장 수)
BETA_MULTS = [0.0, 0.5, 1.0, 2.0]       # beta = mult * lam (lam은 청크 수를 smart와 맞추도록 이분 탐색)
OUT = ROOT / "results" / "proto_coherence"


# ---------------------------------------------------------------------------
# DP
# ---------------------------------------------------------------------------
def segment(E, own, gap, bonus, lam, min_t, max_t):
    """E: (n, d) 단위 임베딩(정규화), own[i]: 단위 토큰 수, gap[i]: 단위 i 시작~i+1 시작 사이 토큰 수(마지막은 0),
    bonus[a]: 단위 a 앞 경계를 자를 때의 구조 보너스(양수). -> 그룹 [(a, b)] (사전식 최적)."""
    n = len(E)
    P = np.vstack([np.zeros((1, E.shape[1])), np.cumsum(E.astype(np.float64), 0)])
    pg = np.concatenate([[0], np.cumsum(gap)])        # gap 누적: sum(gap[a:b-1]) = pg[b-1] - pg[a]
    f_sf = np.full(n + 1, np.inf); f_cost = np.full(n + 1, np.inf)
    f_sf[0] = 0.0; f_cost[0] = 0.0
    prev = np.zeros(n + 1, dtype=int)
    lo = 0
    for j in range(1, n + 1):
        while lo < j - 1 and (pg[j - 1] - pg[lo]) + own[j - 1] > max_t:
            lo += 1
        a = np.arange(lo, j)
        size = (pg[j - 1] - pg[a]) + own[j - 1]
        m = (j - a).astype(np.float64)
        S = P[j] - P[a]
        scatter = m - (S * S).sum(1) / m
        short = np.maximum(0, min_t - size)
        cost = scatter + lam - np.where(a > 0, bonus[a], 0.0)
        tot_sf = f_sf[a] + short
        tot_cost = f_cost[a] + cost
        best_sf = tot_sf.min()
        ok = np.where(tot_sf <= best_sf + 1e-9)[0]
        k = ok[np.argmin(tot_cost[ok])]
        f_sf[j], f_cost[j], prev[j] = tot_sf[k], tot_cost[k], a[k]
    groups, j = [], n
    while j > 0:
        groups.append((int(prev[j]), j)); j = int(prev[j])
    return groups[::-1], (float(f_sf[n]), float(f_cost[n]))


def local_deltas(E, w=W):
    """경계마다 좌우 w문장 창의 응집도 개선량 Delta."""
    n = len(E); out = []
    for i in range(n - 1):
        L, R = E[max(0, i - w + 1):i + 1], E[i + 1:i + 1 + w]
        mL, mR = len(L), len(R)
        out.append(mL * mR / (mL + mR) * float(np.sum((L.mean(0) - R.mean(0)) ** 2)))
    return out


def selftest():
    rng = np.random.default_rng(1)
    for trial in range(40):
        n = int(rng.integers(2, 9)); d = 6
        E = rng.normal(size=(n, d)); E /= np.linalg.norm(E, axis=1, keepdims=True)
        own = rng.integers(5, 70, n).astype(float); gap = np.concatenate([own[:-1] * 0 + 1, [0]])
        bonus = rng.random(n) * 0.5; bonus[0] = 0
        lam = float(rng.random() * 2); mn, mx = int(rng.integers(20, 50)), int(rng.integers(80, 160))
        groups, (sf, cost) = segment(E, own, gap, bonus, lam, mn, mx)
        pg = np.concatenate([[0], np.cumsum(gap)])
        size = lambda a, b: (pg[b - 1] - pg[a]) + own[b - 1]
        best = None
        for mask in range(2 ** (n - 1)):
            cuts = [i for i in range(n - 1) if mask >> i & 1]
            b = [0] + [c + 1 for c in cuts] + [n]; segs = list(zip(b, b[1:]))
            if any(size(a, e) > mx and e - a > 1 for a, e in segs):
                continue
            s = sum(max(0, mn - size(a, e)) for a, e in segs)
            c = sum((e - a) - np.sum(E[a:e].sum(0) ** 2) / (e - a) + lam - (bonus[a] if a > 0 else 0) for a, e in segs)
            if best is None or s < best[0] - 1e-9 or (abs(s - best[0]) <= 1e-9 and c < best[1] - 1e-9):
                best = (s, c)
        assert abs(sf - best[0]) < 1e-6 and abs(cost - best[1]) < 1e-6, (trial, sf, cost, best)
    print("[selftest] 40개 무작위 입력에서 DP == 전수 탐색: 통과")


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def build_doc(doc, models, embed):
    """문서 -> (text, pieces, units, E, own, gap, level_at) 단위·임베딩·크기·제목 수준(단위가 제목으로 시작하면 수준, 아니면 0)."""
    from chunking.v1.units import Rules, build_units, caption_glue
    from common.docjson import render
    text, pieces = render(doc)
    rules = Rules(heading_glue=True, table_atomic=True, caption_glue=True)
    units, _ = build_units(text, pieces, caption_glue(doc, pieces), rules, count=models["count"], max_tokens=MAX)
    blocks = {b["id"]: b for b in doc["blocks"]}
    hs = {p.start: blocks[p.block_ids[0]].get("level", 1) for p in pieces if p.kind == "heading" and p.block_ids}
    level_at = np.array([hs.get(s, 0) for s, _ in units])
    E = np.asarray(embed.encode([text[s:e] for s, e in units], normalize_embeddings=True), dtype=np.float64)
    own = np.array([models["count"](text[s:e]) for s, e in units], dtype=float)
    gap = np.array([models["count"](text[units[i][0]:units[i + 1][0]]) for i in range(len(units) - 1)] + [0], dtype=float)
    return text, pieces, units, E, own, gap, level_at


_BUILT = None


def _dp_one(args):
    i, lam, beta = args
    text, pieces, units, E, own, gap, level_at = _BUILT[i]
    bonus = np.where(level_at > 0, beta * 0.5 ** np.maximum(level_at - 1, 0), 0.0)
    return segment(E, own, gap, bonus, lam, MIN, MAX)[0]


def main_run(limit):
    import datetime
    from compare_size_control import CachedEmbed
    from chunking.v1.meta import attach_meta
    from chunking.v1.stats import compute_stats
    from common.config import PROCESSED_DIR, RESULTS_DIR
    from common.docjson import load_doc
    from common.models import get_embeddings, get_token_counter
    from indexing.index import build_index, make_encoder
    from qa_build.build_qa import load_qa
    from tasks.retrieval import evaluate
    emb = get_embeddings(); embed = CachedEmbed(emb._client); encode = make_encoder(emb._client)
    count = get_token_counter("bge-m3"); models = {"count": count}
    print(f"[장치] 임베딩 모델: {emb._client.device}", flush=True)
    exp1 = {"docx_corpus": {n: json.loads((RESULTS_DIR / "docx_corpus" / "retrieval" / f"{t}.json").read_text(encoding="utf-8"))
                            for n, t in (("semantic", "semantic__p85-min128-max512"), ("smart", "smart__p85-min128-max512"))},
            "vectara": {n: json.loads((RESULTS_DIR / "vectara" / "retrieval" / f"{t}.json").read_text(encoding="utf-8"))
                        for n, t in (("semantic", "semantic__p85-min128-max512"), ("smart", "smart__p85-min128-max512"))}}
    results = {}
    t0 = time.time()
    for ds in ("docx_corpus", "vectara"):
        res1 = exp1[ds]["smart"]
        doc_ids = sorted({r["doc_id"] for r in res1["rows"]})
        if limit:
            doc_ids = doc_ids[:limit]
        docs = [load_doc(PROCESSED_DIR / ds / f"{d}.json") for d in doc_ids]
        qas_all = {q["qid"]: q for q in load_qa(ds)}
        qids = {r["qid"] for r in res1["rows"] if r["doc_id"] in set(doc_ids)}
        qas = [qas_all[q] for q in qids if q in qas_all]
        prim = {"docx_corpus": "evidence_recall", "vectara": "section_hit"}[ds]
        doc_ids_set = set(doc_ids)
        print(f"[{ds}] 문서 {len(docs)}개, 질문 {len(qas)}개 ({time.time() - t0:.0f}s)", flush=True)
        built = [build_doc(d, models, embed) for d in docs]
        print(f"  단위·임베딩 준비 끝: 문장 {sum(len(b[2]) for b in built)}개 ({time.time() - t0:.0f}s)", flush=True)
        from indexing.index import load_chunks
        target_n = len(load_chunks(ds, "smart__p85-min128-max512", doc_ids))       # 청크 수를 smart와 맞춘다(크기 효과 통제)
        deltas = np.concatenate([local_deltas(b[3]) for b in built if len(b[3]) > 1])
        print(f"  목표 청크 수(smart와 같게) {target_n} | Delta 분위수 { {q: round(float(np.percentile(deltas, q)), 3) for q in (60, 75, 90)} }", flush=True)
        results[ds] = {"target_n": target_n, "grid": {}}

        global _BUILT
        _BUILT = built                                                              # fork된 워커가 상속한다(복사 없음)
        pool = mp.get_context("fork").Pool(min(24, len(built)))

        def run_dp(lam, beta):
            return pool.map(_dp_one, [(i, lam, beta) for i in range(len(built))], chunksize=1)

        for bm in BETA_MULTS:
            lo, hi = 0.05, 40.0                                                      # lam 이분 탐색(로그 척도): lam이 클수록 청크가 적다
            for it in range(14):
                lam = (lo * hi) ** 0.5
                n = sum(len(g) for g in run_dp(lam, bm * lam))
                lo, hi = (lam, hi) if n > target_n else (lo, lam)
                print(f"    [{ds}] b{bm:g} 이분 탐색 {it + 1}/14 lam {lam:.3f} 청크 {n} (목표 {target_n}) ({time.time() - t0:.0f}s)", flush=True)
            lam = (lo * hi) ** 0.5; beta = bm * lam
            groups_all = run_dp(lam, beta)
            chunks = []
            for doc, (text, pieces, units, E, own, gap, level_at), groups in zip(docs, built, groups_all):
                spans = [(units[a][0], units[b - 1][1]) for a, b in groups]
                chunks += attach_meta(doc, text, pieces, spans, count, "proto", {"lam": lam, "beta": beta})
            idx = build_index(chunks, encode, 8192)
            r = evaluate(idx, qas, k=3)
            st = compute_stats(chunks, {d["doc_id"]: d for d in docs}, MIN, MAX)
            key = f"b{bm:g}"
            results[ds]["grid"][key] = {"lam": lam, "beta": beta, "overall": r["overall"], "stats": st, "n_chunks": len(chunks),
                                         "rows": [{k: v for k, v in x.items() if k in ("qid", "doc_id", "hit", "mrr", "f1", "section_hit", "evidence_recall")} for x in r["rows"]]}
            o = r["overall"]
            print(f"  [{ds}] {key:6} lam {lam:.3f} 청크 {len(chunks):5} 평균 {st['tokens_mean']:.0f}토큰 under_min {st.get('under_min_rate')} | hit {o['hit@k']:.3f} MRR {o['mrr']:.3f} {prim}@3 {o.get(prim + '@k', float('nan')):.3f} | top3토큰 {o['top_k_tokens_mean']:.0f} ({time.time() - t0:.0f}s)", flush=True)
        pool.close()
        results[ds]["qids"] = sorted(q["qid"] for q in qas)
        results[ds]["splits"] = {q["qid"]: q["split"] for q in qas}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"grid_{ds}.json").write_text(json.dumps(results[ds], ensure_ascii=False, default=float), encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "grid.json").write_text(json.dumps(results, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"[저장] {OUT / 'grid.json'} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "run"])
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    selftest() if a.cmd == "selftest" else main_run(a.limit)
