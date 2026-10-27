"""LumberChunker A형(Chonkie식 '앞 문단 글 vs 다음 문단') vs fixed·recursive·semantic, top-3 검색 품질(tests 전용).

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/chunking/lumber_a.py

A형(2026-10-20 사용자 확정 조건)
  단위 = 문단(docx_corpus: data/clean_text v5의 text를 '\\n\\n'으로 나눔, NarrativeQA: 원문 빈 줄 문단), 문단마다 bge-m3 따로 인코딩
  sim(a, j) = cos( bge-m3(문단 a … j−1을 '\\n\\n'으로 이은 글), bge-m3(문단 j) )
  thr(문서마다) = 문서 곡선 sim(max(0, j−K), j)의 골짜기 값 하위 Q 백분위 (K = 2, Q = 0.15; 골짜기 = 양옆보다 낮음, 끝 자리는 한쪽만 비교)
  그룹 = s부터 토큰 합이 θ(550)를 넘게 만드는 문단 m까지 [s, m] (원본 LumberChunker)
  그룹 곡선 sim_g(j) = sim(max(s, j−K), j), j = s+1 … m  (앞 글은 그룹 안으로만)
  자르기 = 그룹 곡선의 골짜기 중 sim_g ≤ thr 인 자리들 가운데 sim_g가 가장 낮은 j → 청크 [s, j), s = j
           없으면 자르지 않음: θ 안 문단 [s, m)을 한 청크로 두고 s = m (550토큰 상한 유지)
  θ보다 큰 문단은 혼자 청크. min_tokens 없음
비교: fixed(CharacterTextSplitter 글자 수) · recursive(RecursiveCharacterTextSplitter 글자 수)는 A형 평균 토큰에 맞도록 데이터셋마다 글자 수 하나를
      이분 탐색 · semantic(LangChain 기본 + 공용 문장 분리, 거리 백분위 85 = 유사도 하위 15%, 크기 조건 없음)
검색: 청커마다 데이터셋 전체 청크를 FAISS(IndexFlatIP) 하나에, 질문마다 top-3(정답 문서의 청크만 점수)
  docx_corpus: 정답 근거 낱말 5-gram 재현율, 근거 적중(재현율 ≥ 0.5) · NarrativeQA: 정답 낱말 재현율, 정답 문자열 포함
  평균 ± 95% 구간과 A형 대비 짝지은 차이(부트스트랩 1000회), top-3 평균 토큰
출력: tests/chunking/out/lumber_a.md
"""
import json
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
from narrative_lumber import answer_scores  # noqa: E402
from v4_v5_lumber import qa_sets, safe_encoder  # noqa: E402
from chunking import fixed, recursive, semantic  # noqa: E402
from chunking.sentences import paragraphs  # noqa: E402
from common.models import make_token_counter  # noqa: E402
from preprocess import longbench  # noqa: E402

THETA, K, Q, TOPK = 550, 2, 0.15, 3
NAMES = ["lumber_A", "fixed", "recursive", "semantic"]


def valleys(v):
    """양옆보다 낮은 자리(끝 자리는 한쪽만 비교)."""
    n = len(v)
    return [i for i in range(n) if (i == 0 or v[i] < v[i - 1]) and (i == n - 1 or v[i] <= v[i + 1]) and n > 1]


def lumber_a(units, toks, encode):
    n = len(units)
    e = encode(units)
    cache = {}

    def sim(a, j):
        if (a, j) not in cache:
            cache[(a, j)] = encode(["\n\n".join(units[a:j])])[0]
        return float(cache[(a, j)] @ e[j])

    # 문서 곡선은 한 번에 인코딩(창 글 묶음)
    pairs = [(max(0, j - K), j) for j in range(1, n)]
    if pairs:
        vec = encode(["\n\n".join(units[a:j]) for a, j in pairs])
        for (a, j), v in zip(pairs, vec):
            cache[(a, j)] = v
    doc = np.array([sim(a, j) for a, j in pairs])
    vi = valleys(list(doc))
    thr = float(np.quantile(doc[vi], Q)) if vi else (float(np.quantile(doc, Q)) if len(doc) else 0.0)
    tok = np.concatenate([[0], np.cumsum(toks)])
    out, s, stats = [], 0, defaultdict(int)
    while s < n:
        if toks[s] > THETA:
            out.append((s, s + 1)); s += 1; stats["θ 초과 문단"] += 1
            continue
        m = int(np.searchsorted(tok, tok[s] + THETA, side="right")) - 1
        if m >= n:
            out.append((s, n)); break
        if toks[m] > THETA:
            out.append((s, m)); s = m; stats["큰 문단 앞에서 끊음"] += 1
            continue
        js = list(range(s + 1, m + 1))
        g = [sim(max(s, j - K), j) for j in js]
        ok = [i for i in valleys(g) if g[i] <= thr]
        if ok:
            cut = js[min(ok, key=lambda i: g[i])]; stats["의미 경계에서 자름"] += 1
        else:
            cut = m; stats["경계 없음 → θ 안 전체"] += 1
        out.append((s, cut)); s = cut
    return ["\n\n".join(units[i:j]) for i, j in out], thr, stats


def size_search(split, texts, count, target, iters=14):
    lo, hi, best = 50.0, 20000.0, None
    for _ in range(iters):
        cs = int((lo * hi) ** 0.5)
        ch = [[c for c in split(t, cs) if c.strip()] for t in texts]
        avg = np.mean([x for c in ch for x in count(c)])
        if best is None or abs(avg - target) < abs(best[1] - target):
            best = (ch, avg, cs)
        lo, hi = (cs, hi) if avg < target else (lo, cs)
    return best


def main():
    import faiss
    count = make_token_counter()
    encode = safe_encoder()
    rng = np.random.default_rng(0)
    qa = qa_sets()
    data = {}
    docx = []
    for f in sorted((ROOT / "data" / "clean_text" / "docx_corpus").glob("*.json")):
        if f.name.endswith(".dropped.json"):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        docx.append((r["doc_id"], r["text"], r["text"].split("\n\n"), r["lang"]))
    data["docx_corpus"] = (docx, {d: [(q, ev) for q, ev in qa["docx_corpus"].get(d, [])] for d, *_ in docx})
    nqa = longbench.documents("narrativeqa")
    qn = defaultdict(list)
    for q in longbench.questions("narrativeqa"):
        qn[q["doc_id"]].append((q["question"], q["answers"]))
    data["narrativeqa"] = ([(b["doc_id"], b["text"], paragraphs(b["text"]), b["lang"]) for b in nqa], qn)

    L = ["# LumberChunker A형 vs fixed·recursive·semantic (top-3 검색)", "",
         f"θ = {THETA}, K = {K}, Q = {Q}(골짜기 하위 15%), min_tokens 없음, 경계 없으면 자르지 않음(θ 안 전체). semantic 거리 백분위 85. top-{TOPK}.", ""]
    for ds, (docs, qmap) in data.items():
        t0 = time.time()
        chunks, stats, thrs = {n: {} for n in NAMES}, defaultdict(int), []
        for d, text, units, lang in docs:
            cs, thr, st = lumber_a(units, count(units), encode)
            chunks["lumber_A"][d] = cs; thrs.append(thr)
            for k, v in st.items():
                stats[k] += v
        target = np.mean([x for cs in chunks["lumber_A"].values() for x in count(cs)])
        print(ds, "lumber_A 완료", f"평균 {target:.0f}토큰", f"{time.time() - t0:.0f}초", flush=True)
        texts = [t for _, t, _, _ in docs]
        for name, split in (("fixed", lambda t, cs: fixed.chunk(t, cs)), ("recursive", lambda t, cs: recursive.chunk(t, cs))):
            ch, avg, cs = size_search(split, texts, count, target)
            for (d, *_), c in zip(docs, ch):
                chunks[name][d] = c
            stats[f"{name} 글자 수"] = cs
        for d, text, units, lang in docs:
            chunks["semantic"][d] = [c for c in semantic.chunk(text, encode, lang=lang, breakpoint_threshold_amount=85) if c.strip()]
        print(ds, "기준 청커 완료", flush=True)

        L += [f"## {ds} (문서 {len(docs)}편)", "",
              "A형 판단: " + ", ".join(f"{k} {v:,}" for k, v in stats.items() if "글자" not in k) +
              f" | 문서 기준 thr 평균 {np.mean(thrs):.3f} | fixed {stats['fixed 글자 수']}자, recursive {stats['recursive 글자 수']}자", "",
              "| 청커 | 청크 | 평균 토큰 | 중앙값 | 최대 | 50토큰 미만 |", "|---|---|---|---|---|---|"]
        for nme in NAMES:
            t = np.array([x for cs in chunks[nme].values() for x in count(cs)])
            L.append(f"| {nme} | {len(t):,} | {t.mean():.0f} | {np.median(t):.0f} | {t.max()} | {np.mean(t < 50):.1%} |")

        res = defaultdict(lambda: defaultdict(list))
        for nme in NAMES:
            texts_, toks_, owner, embs = [], [], [], []
            for d, cs in chunks[nme].items():
                texts_ += cs; toks_ += count(cs); owner += [d] * len(cs); embs.append(encode(cs))
            E = np.concatenate(embs); owner = np.array(owner)
            index = faiss.IndexFlatIP(E.shape[1]); index.add(E)
            qs = [(d, q, g) for d in chunks[nme] for q, g in qmap.get(d, [])]
            Qv = encode([q for _, q, _ in qs])
            _, top = index.search(Qv, TOPK)
            for (d, q, g), row in zip(qs, top):
                got = " ".join(texts_[i] for i in row if owner[i] == d)
                res["top-3 토큰"][nme].append(sum(toks_[i] for i in row))
                if ds == "narrativeqa":
                    rec, hit = answer_scores(g, got)
                    res["정답 낱말 재현율"][nme].append(rec); res["정답 문자열 포함"][nme].append(hit)
                else:
                    dg = C.grams(C.words(" ".join(chunks[nme][d])))
                    sc = C.score(ds, g, got, dg, None)
                    res["근거 5-gram 재현율"][nme].append(sc)
                    res["근거 적중(재현율 ≥ 0.5)"][nme].append(None if sc is None else float(sc >= 0.5))
            print(ds, nme, "검색 완료", flush=True)
        L += ["", f"### 검색 (top-{TOPK}, FAISS 전체 색인, 질문 {len(res['top-3 토큰']['lumber_A'])}개)", "",
              "| 지표 | " + " | ".join(NAMES) + " | " + " | ".join(f"{n} − A형" for n in NAMES[1:]) + " |", "|---" * (2 * len(NAMES)) + "|"]
        for metric, r in res.items():
            keep = [i for i in range(len(r["lumber_A"])) if all(r[n][i] is not None for n in NAMES)]
            v = {n: np.array([r[n][i] for i in keep], float) for n in NAMES}
            if metric == "top-3 토큰":
                L.append(f"| {metric}(평균) | " + " | ".join(f"{v[n].mean():.0f}" for n in NAMES) + " |" + " – |" * (len(NAMES) - 1))
                continue
            s = {n: C.ci(v[n], rng) for n in NAMES}
            top = max(m for m, _ in s.values())
            cells = [(f"**{m:.3f}**" if m == top else f"{m:.3f}") + f" ±{h:.3f}" for m, h in s.values()]
            for n in NAMES[1:]:
                m, h = C.ci(v[n] - v["lumber_A"], rng)
                cells.append(f"{m:+.3f}" + ("*" if abs(m) > h else ""))
            L.append(f"| {metric} | " + " | ".join(cells) + " |")
        L += ["", "\\* = 짝지은 부트스트랩 95% 구간이 0을 포함하지 않음.", ""]
    (HERE / "out" / "lumber_a.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
