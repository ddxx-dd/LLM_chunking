"""LumberChunker식(앞쪽 최근 수준 대비 하락) 확정 실험: vs 지난 A형·채우기만·fixed·recursive·semantic, RAG 검색 지표(tests 전용).

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/chunking/lumber_d.py

설정(2026-10-21 사용자 결정, 나머지는 추천값)
  데이터  NarrativeQA 14권(원문 빈 줄 문단) + vectara(QA 있는 문서 100편 + 방해 문서 100편, clean_text v5 문단). seed 0
          개발/평가: vectara QA 100편 중 20편, NarrativeQA 3권을 개발용(λ 고르기). 색인에는 항상 데이터셋 전체(방해 문서 포함)
  그룹    s부터 토큰 합이 θ(700)를 넘게 만드는 문단 m까지 [s, m] (m 포함). 혼자 θ를 넘는 문단은 혼자 청크, 남은 글이 θ 안이면 마지막 청크
  유사도  sim(j) = cos( bge-m3(그룹 안 앞 2문단을 공백으로 이은 글), bge-m3(문단 j) )
  판단    최근 수준(j) = 그룹 안 바로 앞 3자리 sim 평균(앞 자리가 2개 이상일 때만 판단), 하락(j) = 최근 수준 − sim(j)
          후보 = 하락 ≥ λ·σ_doc (σ_doc = 문서 전체 곡선 sim(max(0, j−2), j)의 표준편차) 이고 다시 올라옴(sim(j) < sim(j+1))
                 이고 앞 청크 [s, j)가 min_tokens(100) 이상
          후보 중 하락이 가장 큰 j에서 자름. 없으면 자르지 않음(θ 안 [s, m)이 한 청크, s = m). λ ∈ {0.5, 1, 1.5}
비교    A형(θ 700: sim 값이 문서 골짜기 하위 15% 이하인 골짜기 중 가장 낮은 곳, 없으면 안 자름) · 채우기만(항상 [s, m)) ·
        fixed·recursive(글자 수를 고른 λ 판 평균 토큰에 맞춤) · semantic(LangChain, number_of_chunks로 같은 평균 크기)
지표    FAISS(IndexFlatIP) 데이터셋 전체 색인, 질문마다 top-10
        문서 Hit@3, 문서 MRR@10 · 근거 Hit@3, 근거 Precision@3, 근거 MRR@10 · 근거 재현율(top-3, 토큰 예산 1,500)
        근거 청크: vectara = 청크와 정답 근거의 5-gram 겹침이 작은 쪽의 50% 이상 / NarrativeQA = 정답 낱말 50% 이상이 든 청크(대리 기준)
        평균 ± 95% 구간, recursive 대비 짝지은 차이(부트스트랩 1000회)
출력: tests/chunking/out/lumber_d.md
"""
import json
import random
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
from lumber_a import size_search, valleys  # noqa: E402
from narrative_lumber import answer_scores, ans_words  # noqa: E402
from v4_v5_lumber import qa_sets, safe_encoder  # noqa: E402
from chunking import fixed, recursive, semantic  # noqa: E402
from chunking.sentences import paragraphs  # noqa: E402
from common.models import make_token_counter  # noqa: E402
from preprocess import longbench  # noqa: E402

THETA, K, RECENT, MIN_PREV, MIN_TOK, BUDGET = 700, 2, 3, 2, 100, 1500
LAMBDAS = (0.5, 1.0, 1.5)
TOPK, DEPTH = 3, 10


class Doc:
    """문단·토큰·벡터와 '앞 글 vs 문단' 유사도(캐시)."""

    def __init__(self, units, count, encode):
        self.u, self.t, self.enc = units, np.array(count(units)), encode
        self.e = encode(units)
        self.cache = {}
        pairs = [(max(0, j - K), j) for j in range(1, len(units))]
        self._fill(pairs)
        self.doc_curve = np.array([self.sim(a, j) for a, j in pairs])
        self.sigma = float(self.doc_curve.std()) if len(self.doc_curve) else 0.0

    def _fill(self, pairs):
        need = [p for p in pairs if p not in self.cache]
        if need:
            for p, v in zip(need, self.enc([" ".join(self.u[a:j]) for a, j in need])):
                self.cache[p] = v

    def sim(self, a, j):
        if (a, j) not in self.cache:
            self._fill([(a, j)])
        return float(self.cache[(a, j)] @ self.e[j])


def run(D, rule, lam=None, thr_a=None):
    """rule: 'D'(LumberChunker식) / 'A'(지난 A형) / 'fill'(채우기만) → (청크 글 목록, 통계)."""
    u, t = D.u, D.t
    n = len(u)
    tok = np.concatenate([[0], np.cumsum(t)])
    out, s, st = [], 0, defaultdict(int)
    while s < n:
        if t[s] > THETA:
            out.append((s, s + 1)); s += 1; st["혼자 θ 초과"] += 1
            continue
        m = int(np.searchsorted(tok, tok[s] + THETA, side="right")) - 1
        if m >= n:
            out.append((s, n)); break
        if t[m] > THETA:
            out.append((s, m)); s = m; st["큰 문단 앞에서 끊음"] += 1
            continue
        js = list(range(s + 1, m + 1))
        g = [D.sim(max(s, j - K), j) for j in js]
        cut = None
        if rule == "D":
            best = None
            for i, j in enumerate(js):
                if i < MIN_PREV or i + 1 >= len(js):           # 앞 자리 2개 이상 + 다음 자리가 있어야(다시 올라옴 확인)
                    continue
                drop = np.mean(g[max(0, i - RECENT):i]) - g[i]
                if drop >= lam * D.sigma and g[i] < g[i + 1] and tok[j] - tok[s] >= MIN_TOK:
                    if best is None or drop > best[0]:
                        best = (drop, j)
            cut = best[1] if best else None
        elif rule == "A":
            ok = [i for i in valleys(g) if g[i] <= thr_a]
            cut = js[min(ok, key=lambda i: g[i])] if ok else None
        if cut is None:
            cut = m; st["자르지 않음(θ 안 전체)"] += 1
        else:
            st["의미 경계에서 자름"] += 1
        out.append((s, cut)); s = cut
    return ["\n\n".join(u[i:j]) for i, j in out], st


def a_threshold(D):
    v = valleys(list(D.doc_curve))
    return float(np.quantile(D.doc_curve[v], 0.15)) if v else float(np.quantile(D.doc_curve, 0.15))


def overlap_rel(chunk_g, ev_g):
    if not chunk_g or not ev_g:
        return False
    return len(chunk_g & ev_g) / min(len(chunk_g), len(ev_g)) >= 0.5


def evaluate(ds, chunks, qrecs, count, encode, faiss, doc_text):
    """chunks: {doc_id: [청크]} (데이터셋 전체), qrecs: [(doc_id, 질문, 정답)] → 지표별 질문 점수 목록."""
    texts, toks, owner, embs = [], [], [], []
    for d, cs in chunks.items():
        texts += cs; toks += count(cs); owner += [d] * len(cs); embs.append(encode(cs))
    E = np.concatenate(embs); owner = np.array(owner)
    index = faiss.IndexFlatIP(E.shape[1]); index.add(E)
    Qv = encode([q for _, q, _ in qrecs])
    _, top = index.search(Qv, max(DEPTH, 64))
    grams = [C.grams(C.words(x)) for x in texts]
    doc_g = {}
    res = defaultdict(list)
    for (d, q, gold), row in zip(qrecs, top):
        if ds == "narrativeqa":
            rel = lambda i: owner[i] == d and (answer_scores(gold, texts[i])[0] or 0) >= 0.5
            if not any(ans_words(a) for a in gold):
                continue
        else:
            if d not in doc_g:
                doc_g[d] = C.grams(C.words(doc_text[d]))          # 원문으로(청커마다 질문이 같게 걸러지도록)
            ev = C.grams(C.words(gold)) & doc_g[d]
            if not ev:
                continue
            rel = lambda i, ev=ev: owner[i] == d and overlap_rel(grams[i], ev)
        r10 = row[:DEPTH]
        doc_rank = next((k for k, i in enumerate(r10) if owner[i] == d), None)
        rel_rank = next((k for k, i in enumerate(r10) if rel(i)), None)
        res["문서 Hit@3"].append(float(doc_rank is not None and doc_rank < TOPK))
        res["문서 MRR@10"].append(0.0 if doc_rank is None else 1 / (doc_rank + 1))
        res["근거 Hit@3"].append(float(rel_rank is not None and rel_rank < TOPK))
        res["근거 Precision@3"].append(sum(rel(i) for i in r10[:TOPK]) / TOPK)
        res["근거 MRR@10"].append(0.0 if rel_rank is None else 1 / (rel_rank + 1))
        got3 = " ".join(texts[i] for i in r10[:TOPK] if owner[i] == d)
        gotB = " ".join(x for i, x in C.fill(row, texts, toks, BUDGET) if owner[i] == d)
        if ds == "narrativeqa":
            res["정답 낱말 재현율(top-3)"].append(answer_scores(gold, got3)[0])
            res[f"정답 낱말 재현율(예산 {BUDGET})"].append(answer_scores(gold, gotB)[0])
        else:
            res["근거 재현율(top-3)"].append(len(ev & C.grams(C.words(got3))) / len(ev))
            res[f"근거 재현율(예산 {BUDGET})"].append(len(ev & C.grams(C.words(gotB))) / len(ev))
        res["top-3 토큰"].append(sum(toks[i] for i in r10[:TOPK]))
        res["_doc"].append(d)
    return res


def main():
    import faiss
    count = make_token_counter()
    encode = safe_encoder()
    rng = np.random.default_rng(0)
    rnd = random.Random(0)
    qa = qa_sets()

    # ---- 데이터
    data = {}
    vq = sorted(d for d in qa["vectara"] if (ROOT / "data" / "clean_text" / "vectara" / f"{d}.json").exists())
    allv = sorted(f.stem for f in (ROOT / "data" / "clean_text" / "vectara").glob("*.json") if not f.name.endswith(".dropped.json"))
    qdocs = rnd.sample(vq, 100)
    distract = rnd.sample([d for d in allv if d not in qa["vectara"]], 100)
    dev_v = set(qdocs[:20])
    vdocs = {d: json.loads((ROOT / "data" / "clean_text" / "vectara" / f"{d}.json").read_text(encoding="utf-8"))["text"] for d in qdocs + distract}
    data["vectara"] = ({d: (t, t.split("\n\n"), "en") for d, t in vdocs.items()},
                       [(d, q, ev) for d in qdocs for q, ev in qa["vectara"][d]], dev_v)
    books = longbench.documents("narrativeqa")
    order = sorted(b["doc_id"] for b in books)
    dev_n = set(rnd.sample(order, 3))
    qn = [(q["doc_id"], q["question"], q["answers"]) for q in longbench.questions("narrativeqa")]
    data["narrativeqa"] = ({b["doc_id"]: (b["text"], paragraphs(b["text"]), b["lang"]) for b in books}, qn, dev_n)

    L = ["# LumberChunker식(최근 수준 대비 하락) vs 비교 청커: RAG 검색 지표", "",
         f"θ = {THETA}, 앞 글 = 그룹 안 앞 {K}문단(공백으로 이음) vs 문단, 최근 수준 = 앞 {RECENT}자리 평균(앞 자리 {MIN_PREV}개 이상), "
         f"기준 = λ·σ_doc, 다시 올라옴 확인, min_tokens {MIN_TOK}, 가장 큰 하락에서 자름, 없으면 안 자름. top-{TOPK}, MRR@{DEPTH}, 예산 {BUDGET}.", ""]
    for ds, (docs, qrecs, dev) in data.items():
        t0 = time.time()
        DS = {d: Doc(units, count, encode) for d, (text, units, lang) in docs.items()}
        dtext = {d: text for d, (text, units, lang) in docs.items()}
        print(ds, "문서 벡터 완료", f"{time.time() - t0:.0f}초", flush=True)
        variants, stats = {}, {}
        for lam in LAMBDAS:
            name = f"lumber λ={lam}"
            variants[name], stats[name] = {}, defaultdict(int)
            for d, D in DS.items():
                cs, st = run(D, "D", lam=lam); variants[name][d] = cs
                for k, v in st.items():
                    stats[name][k] += v
        for name, rule in (("A형(θ 700)", "A"), ("채우기만", "fill")):
            variants[name], stats[name] = {}, defaultdict(int)
            for d, D in DS.items():
                cs, st = run(D, rule, thr_a=a_threshold(D) if rule == "A" else None); variants[name][d] = cs
                for k, v in st.items():
                    stats[name][k] += v
        print(ds, "lumber·A·채우기 완료", flush=True)

        # 개발 질문으로 λ 고르기(근거 Hit@3)
        dev_q = [r for r in qrecs if r[0] in dev]
        test_q = [r for r in qrecs if r[0] not in dev]
        dev_score = {}
        for lam in LAMBDAS:
            r = evaluate(ds, variants[f"lumber λ={lam}"], dev_q, count, encode, faiss, dtext)
            dev_score[lam] = float(np.mean(r["근거 Hit@3"]))
        best = max(LAMBDAS, key=lambda l: (dev_score[l], -l))
        target = np.mean([x for cs in variants[f"lumber λ={best}"].values() for x in count(cs)])
        texts = [docs[d][0] for d in docs]
        for name, split in (("fixed", lambda t, cs: fixed.chunk(t, cs)), ("recursive", lambda t, cs: recursive.chunk(t, cs))):
            ch, avg, cs = size_search(split, texts, count, target)
            variants[name] = dict(zip(docs, ch)); stats[name] = {"글자 수": cs}
        variants["semantic"] = {}
        for d, (text, units, lang) in docs.items():
            nk = max(1, round(sum(count([text])) / target))
            variants["semantic"][d] = [c for c in semantic.chunk(text, encode, lang=lang, number_of_chunks=nk) if c.strip()]
        stats["semantic"] = {}
        print(ds, "기준 청커 완료, 고른 λ", best, flush=True)

        names = [f"lumber λ={best}", "A형(θ 700)", "채우기만", "fixed", "recursive", "semantic"] + [f"lumber λ={l}" for l in LAMBDAS if l != best]
        res = {n: evaluate(ds, variants[n], test_q, count, encode, faiss, dtext) for n in names}
        print(ds, "평가 완료", flush=True)

        L += [f"## {ds} (색인 문서 {len(docs)}편, 개발 문서 {len(dev)}편 / 평가 질문 {len(res[names[0]]['문서 Hit@3'])}개)", "",
              "개발 질문 근거 Hit@3: " + ", ".join(f"λ={l} {v:.3f}" for l, v in dev_score.items()) + f" → **λ = {best}**", "",
              "### 청크", "", "| 청커 | 청크 | 평균 토큰 | 중앙값 | 50토큰 미만 | 의미 경계에서 자름 / 안 자름 | 비고 |", "|---|---|---|---|---|---|---|"]
        for n in names:
            tt = np.array([x for cs in variants[n].values() for x in count(cs)])
            st = stats[n]
            cut = f"{st.get('의미 경계에서 자름', 0):,} / {st.get('자르지 않음(θ 안 전체)', 0):,}" if "의미 경계에서 자름" in st or "자르지 않음(θ 안 전체)" in st else "–"
            note = f"{st['글자 수']}자" if "글자 수" in st else ""
            L.append(f"| {n} | {len(tt):,} | {tt.mean():.0f} | {np.median(tt):.0f} | {np.mean(tt < 50):.1%} | {cut} | {note} |")
        metrics = [k for k in res[names[0]] if not k.startswith("_")]
        L += ["", "### 검색 (평가 질문, 평균 ± 95% 구간, 괄호 = recursive 대비 짝지은 차이, * = 유의)", "",
              "| 지표 | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for mt in metrics:
            vals = {n: np.array(res[n][mt], float) for n in names}
            if mt == "top-3 토큰":
                L.append(f"| {mt}(평균) | " + " | ".join(f"{vals[n].mean():.0f}" for n in names) + " |")
                continue
            top = max(v.mean() for v in vals.values())
            cells = []
            for n in names:
                m, h = C.ci(vals[n], rng)
                cell = (f"**{m:.3f}**" if np.isclose(m, top) else f"{m:.3f}") + f" ±{h:.3f}"
                if n != "recursive":
                    dm, dh = C.ci(vals[n] - vals["recursive"], rng)
                    cell += f" ({dm:+.3f}{'*' if abs(dm) > dh else ''})"
                cells.append(cell)
            L.append(f"| {mt} | " + " | ".join(cells) + " |")
        L.append("")
    (HERE / "out" / "lumber_d.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
