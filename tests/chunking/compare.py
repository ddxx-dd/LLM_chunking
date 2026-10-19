"""청커 비교(tests 전용, 파일럿 표본·임시 크기 값): 크기를 맞춘 fixed·recursive·semantic vs advanced_semantic 개선 후보.

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/chunking/compare.py [--n-vectara 50 --n-docx 40 --n-hotpot 120]

모든 청커의 평균 청크 크기를 문서마다 TARGET(bge-m3 토큰)에 맞춘다(같은 조건 비교)
  fixed·recursive: chunk_size(글자)를 이분 탐색 · semantic: LangChain number_of_chunks = 문서 토큰 / TARGET
  asc_src: src advanced_semantic 기본(흩어짐 n·(1−‖m‖²) + 문단 보너스 0.5β + 제목 규칙), β 탐색
  v2_sum: 같은 동적 계획법, 구조 규칙 없음 · v2_mean: 흩어짐 1−‖m‖²(문장 수 안 곱함, 나눌수록 합이 커져 β를 음수까지 탐색)
  v2_sum_win: 판단용 임베딩 = 앞·뒤 문장 포함 · v2_sum_win_size: + 크기 고르기 벌점 β·((크기−TARGET)/TARGET)²
  lumber / lumber_win: θ까지 모아 그룹 안 최적 두 덩어리 자리에서 자르고 남은 뒤를 다시 채움(tests/chunking/variants.py), θ 탐색
  공통: θ 상한 THETA(lumber는 θ 자체를 탐색), 하한 MIN_TOK
검색(같은 토큰 예산 512 / 1024, 예산을 넘는 마지막 청크는 낱말 비율로 잘라 모두 정확히 같은 양)
  문서 안: 질문과 정답 문서의 청크만 비교 · FAISS: 청커마다 표본 문서 전체 청크를 IndexFlatIP 하나에 넣고 검색(다른 문서와 경쟁, 정답 문서 청크만 점수)
  vectara·docx = 정답 근거 낱말 5-gram 재현율(우리 문서에 있는 5-gram만 분모), hotpotqa = 정답 문자열 포함
  95% 구간: 질문 단위 부트스트랩 1000회. recursive와의 차이는 같은 질문끼리 짝지은 부트스트랩
경계(HotpotQA 'Passage N:'): 경계 재현율(±10자 안에 청크 경계) · 순도(두 패시지에 걸치지 않는 청크 비율)
출력: tests/chunking/out/compare.md
"""
import argparse
import json
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
import variants as Vr  # noqa: E402
from chunking import advanced_semantic as A, fixed, recursive, semantic  # noqa: E402
from chunking.embeddings import default_encode  # noqa: E402
from chunking.sentences import split_structured  # noqa: E402
from common.models import make_token_counter  # noqa: E402
from preprocess import longbench  # noqa: E402

THETA, MIN_TOK, TARGET = 550, 100, 200
NAMES = ["fixed", "recursive", "semantic", "asc_src", "v2_sum", "v2_mean", "v2_sum_win", "v2_sum_win_size", "lumber", "lumber_win"]
BUDGETS = (512, 1024)
W = re.compile(r"\w+")


def words(t):
    return W.findall(t.lower())


def grams(ws, n=5):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def size_search(split, text, count, iters=12):
    lo, hi, best = 50.0, 20000.0, None
    for _ in range(iters):
        cs = int((lo * hi) ** 0.5)
        c = [x for x in split(text, cs) if x.strip()]
        avg = sum(count(c)) / len(c)
        if best is None or abs(avg - TARGET) < abs(best[1] - TARGET):
            best = (c, avg)
        lo, hi = (cs, hi) if avg < TARGET else (lo, cs)
    return best[0]


def run_chunkers(text, lang, encode, count):
    items = split_structured(text, lang)
    units = [u for u, _, _ in items]
    seps = [s for _, s, _ in items]
    ps = np.array([s == "\n\n" for s in seps])
    hd = np.array([h for _, _, h in items])
    toks = count(units)
    t0 = time.time()
    emb = np.asarray(encode(units), dtype=np.float32)
    t_emb = time.time() - t0
    t0 = time.time()
    wemb = Vr.window_emb(units, encode)
    t_win = time.time() - t0
    join = lambda r: [units[i] + "".join(seps[k] + units[k] for k in range(i + 1, j)) for i, j in r]
    dp = lambda e, **kw: Vr.search(lambda b: Vr.dp(toks, e, b, THETA, MIN_TOK, target=TARGET, **kw), toks, TARGET, 1e-6, 50, iters=20)[0]
    # 문장 수를 곱하지 않으면 나눌수록 흩어짐 합이 커져 β ≥ 0으로는 작은 청크가 안 나온다 → β를 음수(청크마다 보상)까지 탐색
    dp_mean = lambda e: Vr.search(lambda b: Vr.dp(toks, e, b, THETA, MIN_TOK, cost="mean"), toks, TARGET, -2.0, 2.0, iters=24, geometric=False)[0]
    lb = lambda e: Vr.search(lambda th: Vr.lumber(toks, e, th, MIN_TOK), toks, TARGET, TARGET * 0.5, TARGET * 8)[0]
    out, secs = {}, {}
    for name in NAMES:
        t0 = time.time()
        if name == "fixed":
            c = size_search(lambda t, cs: fixed.chunk(t, cs), text, count)
        elif name == "recursive":
            c = size_search(lambda t, cs: recursive.chunk(t, cs), text, count)
        elif name == "semantic":
            c = semantic.chunk(text, encode, lang=lang, number_of_chunks=max(1, round(sum(toks) / TARGET)))
        elif name == "asc_src":
            c = join(A.segment_target(toks, emb, THETA, TARGET, MIN_TOK, para_start=ps, bonus_frac=0.5, heading=hd)[0])
        elif name == "v2_sum":
            c = join(dp(emb))
        elif name == "v2_mean":
            c = join(dp_mean(emb))
        elif name == "v2_sum_win":
            c = join(dp(wemb))
        elif name == "v2_sum_win_size":
            c = join(dp(wemb, size_lambda=1.0))
        elif name == "lumber":
            c = join(lb(emb))
        else:
            c = join(lb(wemb))
        secs[name] = time.time() - t0 + (t_emb if name not in ("fixed", "recursive", "semantic") else 0) + (t_win if "win" in name else 0)
        out[name] = [x for x in c if x.strip()]
    return out, secs


def fill(order, texts, toks, budget):
    """순위대로 예산을 정확히 채운다(넘치는 청크는 낱말 비율로 자름) → [(위치, 글)]."""
    got, used = [], 0
    for k in order:
        rem = budget - used
        if rem <= 0:
            break
        if toks[k] <= rem:
            got.append((k, texts[k])); used += toks[k]
        else:
            ws = texts[k].split()
            got.append((k, " ".join(ws[:max(1, int(len(ws) * rem / toks[k]))]))); used = budget
    return got


def score(ds, gold, got_text, doc_g, doc_norm):
    if ds == "hotpotqa":
        ans = [" " + " ".join(words(a)) + " " for a in gold]
        ans = [a for a in ans if a.strip() and a in doc_norm]
        if not ans:
            return None
        g = " " + " ".join(words(got_text)) + " "
        return float(any(a in g for a in ans))
    eg = grams(words(gold)) & doc_g
    return len(eg & grams(words(got_text))) / len(eg) if eg else None


def boundary_scores(ref_b, hyp_b, tol=10):
    hyp_b = np.sort(hyp_b)
    hit = [len(hyp_b) and np.min(np.abs(hyp_b - r)) <= tol for r in ref_b]
    starts = np.concatenate([[0], hyp_b]); ends = np.concatenate([hyp_b, [np.inf]])
    pure = [not np.any((ref_b > a + tol) & (ref_b < b - tol)) for a, b in zip(starts, ends)]
    return float(np.mean(hit)), float(np.mean(pure))


def ci(x, rng, n=1000):
    x = np.asarray(x, float)
    m = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), (np.percentile(m, 97.5) - np.percentile(m, 2.5)) / 2


def main():
    import faiss
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-vectara", type=int, default=50)
    ap.add_argument("--n-docx", type=int, default=40)
    ap.add_argument("--n-hotpot", type=int, default=120)
    ap.add_argument("--out", default=str(HERE / "out" / "compare.md"))
    ns = ap.parse_args()
    encode, count = default_encode(), make_token_counter()
    rnd = random.Random(0)
    jobs = []
    for ds, n in (("vectara", ns.n_vectara), ("docx_corpus", ns.n_docx)):
        qs = defaultdict(list)
        for ln in open(ROOT / "data" / "qa" / f"{ds}.jsonl", encoding="utf-8"):
            q = json.loads(ln)
            ev = q.get("evidence") or {}
            if not q.get("excluded") and ev.get("text") and ev.get("kind") in ("section_text", "span"):
                qs[q["doc_id"]].append((q["question"], ev["text"]))
        ids = sorted(d for d in qs if (ROOT / "data" / "clean_text" / ds / f"{d}.json").exists())
        for d in rnd.sample(ids, min(n, len(ids))):
            r = json.loads((ROOT / "data" / "clean_text" / ds / f"{d}.json").read_text(encoding="utf-8"))
            jobs.append((ds, d, r["text"], r["lang"], qs[d]))
    hq = defaultdict(list)
    for q in longbench.questions("hotpotqa"):
        hq[q["doc_id"]].append((q["question"], q["answers"]))
    for x in rnd.sample(longbench.documents("hotpotqa"), ns.n_hotpot):
        jobs.append(("hotpotqa", x["doc_id"], x["text"], x["lang"], hq[x["doc_id"]]))

    sizes = defaultdict(lambda: defaultdict(list))
    secs = defaultdict(float)
    bnd = defaultdict(list)
    pool = defaultdict(lambda: {"texts": [], "toks": [], "emb": [], "doc": []})   # (ds, name)
    qrecs = defaultdict(list)                                                    # ds → [(doc, 질문, 정답, 문서 5-gram, 문서 정규화, q벡터)]
    for n_job, (ds, d, text, lang, qs) in enumerate(jobs):
        chunks, sec = run_chunkers(text, lang, encode, count)
        for k, v in sec.items():
            secs[k] += v
        doc_ws = words(text)
        dg, dn = grams(doc_ws), " " + " ".join(doc_ws) + " "
        qemb = np.asarray(encode([q for q, _ in qs]), dtype=np.float32)
        for (q, gold), qe in zip(qs, qemb):
            qrecs[ds].append((d, gold, dg, dn, qe))
        nst = "".join(text.split())
        for name, cs in chunks.items():
            ct = count(cs)
            sizes[ds][name] += ct
            P = pool[(ds, name)]
            P["texts"] += cs; P["toks"] += ct; P["doc"] += [d] * len(cs)
            P["emb"].append(np.asarray(encode(cs), dtype=np.float32))
            if ds == "hotpotqa":
                if len("".join("".join(cs).split())) == len(nst):
                    ref = np.array([m.start() for m in re.finditer(r"Passage\d+:", nst) if m.start() > 0])
                    hyp, pos = [], 0
                    for c in cs[:-1]:
                        pos += len("".join(c.split())); hyp.append(pos)
                    if len(ref):
                        bnd[name].append(boundary_scores(ref, np.array(hyp)))
                else:
                    print("경계 계산 제외(글자 수 불일치):", d, name, flush=True)
        print(f"[{n_job + 1}/{len(jobs)}] {ds} {d[:20]} 완료", flush=True)

    res = defaultdict(lambda: defaultdict(list))    # (mode, ds, B) → name → [점수(질문 순서 고정, None은 제외 표시)]
    for (ds, name), P in pool.items():
        E = np.concatenate(P["emb"])
        docs = np.array(P["doc"])
        index = faiss.IndexFlatIP(E.shape[1]); index.add(E)
        Q = np.stack([r[4] for r in qrecs[ds]])
        _, top = index.search(Q, 128)
        for qi, (d, gold, dg, dn, qe) in enumerate(qrecs[ds]):
            mine = np.where(docs == d)[0]
            in_order = mine[np.argsort(-(E[mine] @ qe))]
            for B in BUDGETS:
                for mode, order in (("문서 안", in_order), ("FAISS 전체", top[qi])):
                    got = fill(order, P["texts"], P["toks"], B)
                    txt = " ".join(t for k, t in got if docs[k] == d)
                    res[(mode, ds, B)][name].append(score(ds, gold, txt, dg, dn))

    rng = np.random.default_rng(0)
    L = ["# 청커 비교 (파일럿 표본, 임시 크기 값, 평균 청크 크기 맞춤)", "",
         f"표본: vectara {ns.n_vectara}편, docx {ns.n_docx}편, HotpotQA {ns.n_hotpot}편(seed 0). 목표 평균 {TARGET}토큰, 상한 θ {THETA}, 하한 {MIN_TOK}(fixed·recursive·semantic 제외).", "",
         "## 1. 청크 크기 (bge-m3 토큰)", "", "| 데이터셋 | " + " | ".join(NAMES) + " |", "|---" * (len(NAMES) + 1) + "|"]
    for ds, s in sizes.items():
        L.append(f"| {ds} 평균 | " + " | ".join(f"{np.mean(s[n]):.0f}" for n in NAMES) + " |")
        L.append(f"| {ds} 50토큰 미만 | " + " | ".join(f"{np.mean(np.array(s[n]) < 50):.0%}" for n in NAMES) + " |")
        L.append(f"| {ds} 최대 | " + " | ".join(f"{max(s[n])}" for n in NAMES) + " |")
    for mode in ("문서 안", "FAISS 전체"):
        L += ["", f"## 2. 검색: {mode} (평균 ± 95% 구간, 높을수록 좋음, 굵게 = 1위)", "",
              "| 데이터셋 · 예산 | 질문 | " + " | ".join(NAMES) + " |", "|---" * (len(NAMES) + 2) + "|"]
        D = ["", f"### recursive 대비 차이 ({mode}, 짝지은 부트스트랩, * = 95% 구간이 0을 포함하지 않음)", "",
             "| 데이터셋 · 예산 | " + " | ".join(n for n in NAMES if n != "recursive") + " |", "|---" * len(NAMES) + "|"]
        allrow = defaultdict(list)
        for ds in ("vectara", "docx_corpus", "hotpotqa"):
            for B in BUDGETS:
                r = res[(mode, ds, B)]
                keep = [i for i, v in enumerate(r["recursive"]) if v is not None]
                vals = {n: np.array([r[n][i] for i in keep], float) for n in NAMES}
                stats = {n: ci(vals[n], rng) for n in NAMES}
                top = max(m for m, _ in stats.values())
                L.append(f"| {ds} · {B} | {len(keep)} | " + " | ".join(
                    (f"**{m:.3f}**" if m == top else f"{m:.3f}") + f" ±{h:.3f}" for m, h in stats.values()) + " |")
                cells = []
                for n in NAMES:
                    if n == "recursive":
                        continue
                    dlt = vals[n] - vals["recursive"]
                    m, h = ci(dlt, rng)
                    cells.append(f"{m:+.3f}" + ("*" if abs(m) > h else ""))
                    allrow[n].append(m)
                D.append(f"| {ds} · {B} | " + " | ".join(cells) + " |")
        D.append("| 6칸 평균 | " + " | ".join(f"{np.mean(allrow[n]):+.3f}" for n in NAMES if n != "recursive") + " |")
        L += D
    L += ["", "vectara·docx = 정답 근거 5-gram 재현율, hotpotqa = 정답 문자열 포함 비율.", "",
          "## 3. HotpotQA 패시지 경계 (높을수록 좋음)", "", "| 지표 | " + " | ".join(NAMES) + " |", "|---" * (len(NAMES) + 1) + "|",
          "| 경계 재현율(±10자) | " + " | ".join(f"{np.mean([x[0] for x in bnd[n]]):.3f}" for n in NAMES) + " |",
          "| 순도(한 패시지 안 청크) | " + " | ".join(f"{np.mean([x[1] for x in bnd[n]]):.3f}" for n in NAMES) + " |",
          "", "## 4. 청킹 시간 (초, 표본 전체, 크기 탐색 포함, 임베딩 계열은 단위 임베딩 포함)", "",
          "| " + " | ".join(NAMES) + " |", "|---" * len(NAMES) + "|", "| " + " | ".join(f"{secs[n]:.0f}" for n in NAMES) + " |"]
    Path(ns.out).parent.mkdir(exist_ok=True)
    Path(ns.out).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
