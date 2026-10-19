"""같은 LumberChunker 알고리즘을 clean_text v4 문단과 v5 문단(tests/clean_text/v5.py)에 돌려 문단 정리의 효과를 본다(tests 전용).

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/chunking/v4_v5_lumber.py [--skip-gemma]

알고리즘(θ = 550 bge-m3 토큰, 단위 = 빈 줄 문단)
  gemma   원본 LumberChunker(tests/chunking/lumber_gemma.py, 논문 부록 B 프롬프트). 개발 문서 vectara 10 + docx 10.
          v4 결과는 out/lumber_gemma/를 쓰고, v5는 새로 돌려 out/lumber_gemma_v5/에 둔다(있으면 건너뜀)
  cosine  문단 단위 코사인 lumber(tests/chunking/variants.lumber_min_sim): θ까지 모아 이웃 문단 코사인 최저 자리에서 자름.
          표본 vectara 50 + docx 40(compare.py와 같은 seed 0 표본)
재는 것
  청크 크기(bge-m3 토큰), 잘못된 자리 절단 비율(앞 청크가 문장 끝 없이 끝나거나 다음 청크가 소문자·where·'$$'로 시작하는 경계의 비율)
  검색: 정답 근거 낱말 5-gram 재현율(같은 토큰 예산 512·1024, 문서 안 + FAISS 전체 색인), v5 − v4 짝지은 부트스트랩
출력: tests/chunking/out/v4_v5_lumber.md
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
sys.path.insert(0, str(ROOT / "tests" / "clean_text"))
import compare as C  # noqa: E402
import lumber_gemma as G  # noqa: E402
import v5 as V5  # noqa: E402
import variants as Vr  # noqa: E402
from chunking.embeddings import default_encode  # noqa: E402
from chunking.sentences import paragraphs  # noqa: E402
from common.models import make_token_counter  # noqa: E402

THETA = 550
BUDGETS = (512, 1024)
END = re.compile(r"[.!?:;)\]\"'”’…。]\s*$")
BAD_START = re.compile(r"^\s*([a-z]|\$\$|(where|with|which|and|or)\b)")
OUT5 = HERE / "out" / "lumber_gemma_v5"


def versions(ds, d):
    r = json.loads((ROOT / "data" / "clean_text" / ds / f"{d}.json").read_text(encoding="utf-8"))
    p4 = paragraphs(r["text"])
    p5 = [p["text"] for p in V5.apply([{"text": t, "kind": x["kind"], "src": [k], "fmt": r["fmt"]}
                                         for k, (t, x) in enumerate(zip(p4, r["paras"]))], "broken")]
    return r, {"v4": p4, "v5": p5}


def safe_encoder():
    """GPU를 다른 작업과 함께 쓰므로 긴 글(4,000자 초과)은 2개씩, 나머지는 32개씩 인코딩(벡터는 같고 메모리만 덜 씀)."""
    from common.models import load_embedder, make_encode
    model = load_embedder()
    small, big = make_encode(model, batch_size=32), make_encode(model, batch_size=2)

    def encode(texts):
        texts = list(texts)
        out = [None] * len(texts)
        for f, idx in ((small, [i for i, t in enumerate(texts) if len(t) <= 4000]), (big, [i for i, t in enumerate(texts) if len(t) > 4000])):
            if idx:
                for i, v in zip(idx, f([texts[i] for i in idx])):
                    out[i] = v
        return np.asarray(out, dtype=np.float32)
    return encode


def bad_rate(chunks):
    b = [(not END.search(a.rstrip())) or bool(BAD_START.match(c)) for a, c in zip(chunks, chunks[1:])]
    return float(np.mean(b)) if b else 0.0


def qa_sets():
    qs = {}
    for ds in ("vectara", "docx_corpus"):
        m = defaultdict(list)
        for ln in open(ROOT / "data" / "qa" / f"{ds}.jsonl", encoding="utf-8"):
            q = json.loads(ln)
            ev = q.get("evidence") or {}
            if not q.get("excluded") and ev.get("text") and ev.get("kind") in ("section_text", "span"):
                m[q["doc_id"]].append((q["question"], ev["text"]))
        qs[ds] = m
    return qs


def evaluate(docs_chunks, qmap, encode, count, faiss, rng):
    """docs_chunks: {ver: {(ds, d): [청크]}} → 검색 점수 {(mode, ds, B): {ver: [점수]}} (질문 순서 같음)."""
    res = defaultdict(lambda: defaultdict(list))
    for ver, dc in docs_chunks.items():
        for ds in ("vectara", "docx_corpus"):
            items = [(d, cs) for (x, d), cs in dc.items() if x == ds]
            texts, toks, owner, embs = [], [], [], []
            for d, cs in items:
                texts += cs; toks += count(cs); owner += [d] * len(cs)
                embs.append(np.asarray(encode(cs), dtype=np.float32))
            E = np.concatenate(embs); owner = np.array(owner)
            index = faiss.IndexFlatIP(E.shape[1]); index.add(E)
            qs = [(d, q, ev) for d, _ in items for q, ev in qmap[ds][d]]
            Q = np.asarray(encode([q for _, q, _ in qs]), dtype=np.float32)
            _, top = index.search(Q, 128)
            for (d, q, ev), qe, row in zip(qs, Q, top):
                dtext = " ".join(cs for dd, cs2 in items if dd == d for cs in cs2)
                dg = C.grams(C.words(dtext))
                mine = np.where(owner == d)[0]
                for B in BUDGETS:
                    for mode, order in (("문서 안", mine[np.argsort(-(E[mine] @ qe))]), ("FAISS 전체", row)):
                        got = " ".join(t for k, t in C.fill(order, texts, toks, B) if owner[k] == d)
                        res[(mode, ds, B)][ver].append(C.score(ds, ev, got, dg, None))
    return res


def report(title, chunks, res, count, rng):
    L = [f"## {title}", "", "| 판 | 데이터셋 | 문서 | 문단(단위) | 청크 | 평균 토큰 | 50토큰 미만 | 잘못된 자리 절단 |", "|---|---|---|---|---|---|---|---|"]
    for ver, dc in chunks.items():
        for ds in ("vectara", "docx_corpus"):
            cs = [c for (x, _), v in dc.items() if x == ds for c in v]
            if not cs:
                continue
            t = np.array(count(cs))
            units = sum(1 for (x, _), v in UNITS[ver].items() if x == ds for _ in v)
            br = np.mean([bad_rate(v) for (x, _), v in dc.items() if x == ds])
            L.append(f"| {ver} | {ds} | {sum(1 for (x, _) in dc if x == ds)} | {units:,} | {len(t):,} | {t.mean():.0f} | {np.mean(t < 50):.1%} | {br:.1%} |")
    L += ["", "| 검색 · 데이터셋 · 예산 | 질문 | v4 | v5 | v5 − v4 (95% 구간) |", "|---|---|---|---|---|"]
    for (mode, ds, B), r in sorted(res.items()):
        keep = [i for i in range(len(r["v4"])) if r["v4"][i] is not None and r["v5"][i] is not None]
        a, b = (np.array([r[v][i] for i in keep], float) for v in ("v4", "v5"))
        m, h = C.ci(b - a, rng)
        L.append(f"| {mode} · {ds} · {B} | {len(keep)} | {a.mean():.3f} | {b.mean():.3f} | {m:+.3f} (±{h:.3f}){' *' if abs(m) > h else ''} |")
    return L + [""]


UNITS = {}


def main():
    import faiss
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gemma", action="store_true")
    ap.add_argument("--skip-cosine", action="store_true")
    ap.add_argument("--out", default=str(HERE / "out" / "v4_v5_lumber.md"))
    ns = ap.parse_args()
    count = make_token_counter()
    dev = [(ds, d) for ds, d, _ in G.dev_docs() if ds in ("vectara", "docx_corpus")]
    if not ns.skip_gemma and any(not (OUT5 / ds / f"{d}.json").exists() for ds, d in dev):
        # GPU를 함께 쓰는 다른 작업이 있어 모델을 하나씩 올린다: Gemma로 v5 기준을 먼저 만들고 내린 뒤 bge-m3를 올린다
        import torch
        from common.models import load_llm
        llm = load_llm()
        for ds, d in dev:
            p = OUT5 / ds / f"{d}.json"
            if p.exists():
                continue
            _, vs = versions(ds, d)
            t0 = time.time()
            rg, log = G.lumber(llm, vs["v5"], count(vs["v5"]))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"doc_id": d, "dataset": ds, "theta": THETA, "n_paras": len(vs["v5"]), "ranges": rg, "calls": log,
                                     "seconds": round(time.time() - t0, 1)}, ensure_ascii=False), encoding="utf-8")
            print(f"gemma v5 {ds} {d[:24]} 청크 {len(rg)} {time.time() - t0:.0f}초", flush=True)
        del llm
        torch.cuda.empty_cache()
    encode = safe_encoder()
    qmap = qa_sets()
    rng = np.random.default_rng(0)
    L = ["# 같은 LumberChunker를 clean_text v4 vs v5 문단에 (θ = 550)", "",
         "잘못된 자리 절단 = 앞 청크가 문장 끝 없이 끝나거나 다음 청크가 소문자·where·'$$'로 시작하는 경계 비율(문서 평균). * = 95% 구간이 0을 포함하지 않음.", ""]

    # 1) 코사인 lumber: compare.py와 같은 표본
    rnd = random.Random(0)
    if ns.skip_cosine:
        qmap_ids = {}
    sample = []
    for ds, n in ((("vectara", 50), ("docx_corpus", 40)) if not ns.skip_cosine else ()):
        ids = sorted(d for d in qmap[ds] if (ROOT / "data" / "clean_text" / ds / f"{d}.json").exists())
        sample += [(ds, d) for d in rnd.sample(ids, min(n, len(ids)))]
    chunks = {"v4": {}, "v5": {}}
    UNITS.update({"v4": {}, "v5": {}})
    for ds, d in sample:
        r, vs = versions(ds, d)
        for ver, ps in vs.items():
            pe = np.asarray(encode(ps), dtype=np.float32)
            rg = Vr.lumber_min_sim(count(ps), pe, THETA)
            chunks[ver][(ds, d)] = ["\n\n".join(ps[i:j]) for i, j in rg]
            UNITS[ver][(ds, d)] = ps
    print("코사인 lumber 청킹 완료", flush=True)
    if not ns.skip_cosine:
        L += report("문단 단위 코사인 lumber (표본 vectara 50 + docx 40)", chunks, evaluate(chunks, qmap, encode, count, faiss, rng), count, rng)

    # 2) Gemma 원본 LumberChunker: 개발 문서 20편
    if not ns.skip_gemma:
        gch = {"v4": {}, "v5": {}}
        for ds, d in dev:
            r, vs = versions(ds, d)
            g4 = json.loads((HERE / "out" / "lumber_gemma" / ds / f"{d}.json").read_text(encoding="utf-8"))
            assert g4["n_paras"] == len(vs["v4"])
            gch["v4"][(ds, d)] = ["\n\n".join(vs["v4"][i:j]) for i, j in g4["ranges"]]
            g5 = json.loads((OUT5 / ds / f"{d}.json").read_text(encoding="utf-8"))
            gch["v5"][(ds, d)] = ["\n\n".join(vs["v5"][i:j]) for i, j in g5["ranges"]]
            UNITS["v4"][(ds, d)] = vs["v4"]; UNITS["v5"][(ds, d)] = vs["v5"]
        for v in ("v4", "v5"):                      # 보고서의 단위 수는 이 문서들만으로
            UNITS[v] = {k: UNITS[v][k] for k in gch[v]}
        L += report("원본 LumberChunker(Gemma) (개발 문서 vectara 10 + docx 10)", gch, evaluate(gch, qmap, encode, count, faiss, rng), count, rng)
    Path(ns.out).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
