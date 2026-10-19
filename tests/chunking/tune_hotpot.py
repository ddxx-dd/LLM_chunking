"""HotpotQA 튜닝 → 평가(tests 전용): 문서 200편을 섞어 20%(40편)로 청커마다 평균 청크 크기를 고르고, 나머지 80%(160편)로 잰다.

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/chunking/tune_hotpot.py

청커마다 같은 손잡이 하나: 목표 평균 청크 토큰 GRID. θ 상한 = max(550, 2·목표), 최소 = min(100, 목표/2)(fixed·recursive·semantic 제외)
  fixed·recursive: chunk_size(글자) 이분 탐색 · semantic: LangChain number_of_chunks = 문서 토큰 / 목표
  asc_src: src advanced_semantic(흩어짐 n·(1−‖m‖²) + 문단 보너스 0.5β + 제목 규칙) · asc_src_win: 같은 것, 판단용 임베딩 = 앞·뒤 문장 포함
  v2_sum_win: 구조 규칙 없음 + 이웃 문맥 · lumber_win: θ까지 모아 그룹 안 최적 두 덩어리 자리에서 자름(θ 탐색) + 이웃 문맥
  lumber_v2_win: 2θ까지 모아 그룹 안을 asc_src 방식으로 최적 분할해 첫 청크만 확정, 반복(variants.lumber_dp) + 이웃 문맥
튜닝 점수: 튜닝 문서 전체 FAISS 색인에서 정답 문자열 포함률(예산 512·1024 평균). 같으면 작은 목표.
평가: 평가 문서 전체 FAISS 색인, 같은 예산(넘치는 청크는 잘라 정확히 같은 양), 95% 구간과 recursive 대비 짝지은 차이(부트스트랩 1000회)
출력: tests/chunking/out/tune_hotpot.md
"""
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
import compare as C  # noqa: E402
import variants as Vr  # noqa: E402
from chunking import advanced_semantic as A, fixed, recursive, semantic  # noqa: E402
from chunking.embeddings import default_encode  # noqa: E402
from chunking.sentences import split_structured  # noqa: E402
from common.models import make_token_counter  # noqa: E402
from preprocess import longbench  # noqa: E402

GRID = [100, 150, 200, 300, 400, 550]
NAMES = ["fixed", "recursive", "semantic", "asc_src", "asc_src_win", "v2_sum_win", "lumber_win", "lumber_v2_win"]
BUDGETS = (512, 1024)


def prep(text, lang, encode, count):
    items = split_structured(text, lang)
    units = [u for u, _, _ in items]
    seps = [s for _, s, _ in items]
    return dict(text=text, lang=lang, units=units, seps=seps, ps=np.array([s == "\n\n" for s in seps]),
                hd=np.array([h for _, _, h in items]), toks=count(units),
                emb=np.asarray(encode(units), dtype=np.float32), wemb=Vr.window_emb(units, encode))


def chunk_with(name, D, target, encode, count):
    theta, mn = max(550, 2 * target), min(100, target // 2)
    toks, units, seps = D["toks"], D["units"], D["seps"]
    join = lambda r: [units[i] + "".join(seps[k] + units[k] for k in range(i + 1, j)) for i, j in r]
    srch = lambda run, lo, hi, **kw: Vr.search(run, toks, target, lo, hi, **kw)[0]
    C.TARGET = target
    if name == "fixed":
        c = C.size_search(lambda t, cs: fixed.chunk(t, cs), D["text"], count)
    elif name == "recursive":
        c = C.size_search(lambda t, cs: recursive.chunk(t, cs), D["text"], count)
    elif name == "semantic":
        c = semantic.chunk(D["text"], encode, lang=D["lang"], number_of_chunks=max(1, round(sum(toks) / target)))
    elif name in ("asc_src", "asc_src_win"):
        e = D["emb"] if name == "asc_src" else D["wemb"]
        c = join(A.segment_target(toks, e, theta, target, mn, para_start=D["ps"], bonus_frac=0.5, heading=D["hd"])[0])
    elif name == "v2_sum_win":
        c = join(srch(lambda b: Vr.dp(toks, D["wemb"], b, theta, mn), 1e-6, 50, iters=20))
    elif name == "lumber_win":
        c = join(srch(lambda th: Vr.lumber(toks, D["wemb"], th, mn), target * 0.5, target * 8))
    else:
        c = join(srch(lambda b: Vr.lumber_dp(toks, D["wemb"], b, theta, mn, para_start=D["ps"], bonus=0.5 * b, heading=D["hd"]),
                      1e-4, 50, iters=14))
    return [x for x in c if x.strip()]


def evaluate(docs, chunks_by_doc, encode, count, faiss):
    """docs: [(doc_id, D, [(q, answers)])], chunks_by_doc[doc_id] → 청크 목록 → {B: [점수…]}, 크기, 경계."""
    texts, toks, owner, embs = [], [], [], []
    for d, D, _ in docs:
        cs = chunks_by_doc[d]
        texts += cs; t = count(cs); toks += t; owner += [d] * len(cs)
        embs.append(np.asarray(encode(cs), dtype=np.float32))
    E = np.concatenate(embs); owner = np.array(owner)
    index = faiss.IndexFlatIP(E.shape[1]); index.add(E)
    qs = [(d, D, q, a) for d, D, ql in docs for q, a in ql]
    Q = np.asarray(encode([q for _, _, q, _ in qs]), dtype=np.float32)
    _, top = index.search(Q, 128)
    out = {B: [] for B in BUDGETS}
    for (d, D, q, a), row in zip(qs, top):
        ws = C.words(D["text"])
        dn = " " + " ".join(ws) + " "
        for B in BUDGETS:
            got = C.fill(row, texts, toks, B)
            out[B].append(C.score("hotpotqa", a, " ".join(t for k, t in got if owner[k] == d), None, dn))
    bnd = []
    for d, D, _ in docs:
        cs, nst = chunks_by_doc[d], "".join(D["text"].split())
        if len("".join("".join(cs).split())) != len(nst):
            continue
        ref = np.array([m.start() for m in re.finditer(r"Passage\d+:", nst) if m.start() > 0])
        hyp, pos = [], 0
        for c in cs[:-1]:
            pos += len("".join(c.split())); hyp.append(pos)
        if len(ref):
            bnd.append(C.boundary_scores(ref, np.array(hyp)))
    return out, toks, bnd


def main():
    import argparse
    import faiss
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=None, help="시험용으로 앞 n편만(섞은 뒤)")
    ap.add_argument("--out", default=str(HERE / "out" / "tune_hotpot.md"))
    ns = ap.parse_args()
    encode, count = default_encode(), make_token_counter()
    docs = longbench.documents("hotpotqa")
    qmap = defaultdict(list)
    for q in longbench.questions("hotpotqa"):
        qmap[q["doc_id"]].append((q["question"], q["answers"]))
    random.Random(0).shuffle(docs)
    docs = docs[:ns.n_docs] if ns.n_docs else docs
    n_tune = len(docs) // 5
    t0 = time.time()
    prepped = [(x["doc_id"], prep(x["text"], x["lang"], encode, count), qmap[x["doc_id"]]) for x in docs]
    print(f"준비 {len(prepped)}편 {time.time() - t0:.0f}초", flush=True)
    tune, test = prepped[:n_tune], prepped[n_tune:]

    L = ["# HotpotQA 튜닝(20%) → 평가(80%)", "",
         f"문서 {len(docs)}편을 seed 0으로 섞어 튜닝 {len(tune)}편 / 평가 {len(test)}편. 손잡이: 목표 평균 청크 토큰 {GRID}. "
         "θ = max(550, 2·목표), 최소 = min(100, 목표/2). 점수 = FAISS 전체 색인 정답 문자열 포함률(예산 512·1024 평균).", "",
         "## 1. 튜닝 (튜닝 문서, 512·1024 평균)", "", "| 청커 | " + " | ".join(map(str, GRID)) + " | 고른 값 |", "|---" * (len(GRID) + 2) + "|"]
    chosen, secs = {}, defaultdict(float)
    for name in NAMES:
        row = []
        for g in GRID:
            t0 = time.time()
            cb = {d: chunk_with(name, D, g, encode, count) for d, D, _ in tune}
            secs[name] += time.time() - t0
            sc, _, _ = evaluate(tune, cb, encode, count, faiss)
            v = np.mean([np.mean([x for x in sc[B] if x is not None]) for B in BUDGETS])
            row.append(v)
        chosen[name] = GRID[int(np.argmax(row))]
        L.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in row) + f" | **{chosen[name]}** |")
        print(name, [round(v, 3) for v in row], "→", chosen[name], flush=True)

    rng = np.random.default_rng(0)
    res, sizes, bnds = {}, {}, {}
    for name in NAMES:
        t0 = time.time()
        cb = {d: chunk_with(name, D, chosen[name], encode, count) for d, D, _ in test}
        secs[name] += time.time() - t0
        res[name], sizes[name], bnds[name] = evaluate(test, cb, encode, count, faiss)
        print(name, "평가 완료", flush=True)
    keep = [i for i, v in enumerate(res["recursive"][BUDGETS[0]]) if v is not None]
    L += ["", f"## 2. 평가 (평가 문서 {len(test)}편, 질문 {len(keep)}개, FAISS 전체 색인, 평균 ± 95% 구간)", "",
          "| 청커 | 고른 목표 | 실제 평균 토큰 | 50토큰 미만 | 512 | 1024 | recursive 대비 512 | recursive 대비 1024 |", "|---|---|---|---|---|---|---|---|"]
    for name in NAMES:
        cells = []
        for B in BUDGETS:
            v = np.array([res[name][B][i] for i in keep], float)
            m, h = C.ci(v, rng)
            cells.append(f"{m:.3f} ±{h:.3f}")
        for B in BUDGETS:
            dlt = np.array([res[name][B][i] for i in keep], float) - np.array([res["recursive"][B][i] for i in keep], float)
            m, h = C.ci(dlt, rng)
            cells.append("–" if name == "recursive" else f"{m:+.3f}" + ("*" if abs(m) > h else ""))
        s = np.array(sizes[name])
        L.append(f"| {name} | {chosen[name]} | {s.mean():.0f} | {np.mean(s < 50):.0%} | " + " | ".join(cells) + " |")
    L += ["", "\\* = 짝지은 부트스트랩 95% 구간이 0을 포함하지 않음.", "",
          "## 3. 패시지 경계 (평가 문서, 고른 크기에서)", "", "| 청커 | 경계 재현율(±10자) | 순도 |", "|---|---|---|"]
    for name in NAMES:
        b = bnds[name]
        L.append(f"| {name} | {np.mean([x[0] for x in b]):.3f} | {np.mean([x[1] for x in b]):.3f} |")
    L += ["", "## 4. 청킹 시간 (초, 튜닝 격자 + 평가, 단위 임베딩 제외)", "", "| " + " | ".join(NAMES) + " |", "|---" * len(NAMES) + "|",
          "| " + " | ".join(f"{secs[n]:.0f}" for n in NAMES) + " |"]
    Path(ns.out).parent.mkdir(exist_ok=True)
    Path(ns.out).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
