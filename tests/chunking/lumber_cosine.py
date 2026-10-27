"""LumberChunker 코사인 판: 문단 벡터 만드는 법 비교(tests 전용) — 따로 / 1 그룹 late / 3 이웃 이어 인코딩.

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/chunking/lumber_cosine.py [--check-only]

틀(원본 LumberChunker와 같음): 단위 = v5 문단(data/clean_text의 text를 빈 줄로 나눔, NarrativeQA는 원문 빈 줄 문단), θ = 550 bge-m3 토큰.
  s부터 문단을 모아 토큰 합이 θ를 넘을 때까지 그룹 [s, m](m = 넘게 만든 문단) → 그룹 안 자리 j(s+1 ≤ j ≤ m) 중 하나에서 자르고 j부터 반복.
  θ보다 큰 문단은 혼자 청크(앞에서 끊음). 남은 글이 θ 안이면 마지막 청크.
판단(세 판 공통, 그룹 안만 봄 — LLM도 그룹만 본다)
  sim(j) = cos(그룹 안 앞 W문단 [j−W, j)의 토큰 가중 평균 벡터, 뒤 W문단 [j, j+W)의 평균 벡터)
  depth(j) = (왼쪽 봉우리 − sim(j)) + (오른쪽 봉우리 − sim(j))       (TextTiling, 그룹 안 곡선)
  후보 = 앞 청크 [s, j)가 min_tokens 이상인 자리(없으면 모든 자리) → depth ≥ ρ·(후보 중 최대 depth)인 첫 자리(최대 depth가 0이면 m)
벡터
  sep    문단마다 따로 bge-m3 [CLS](기준선)
  late   그룹 글('\\n\\n'로 이음)을 한 번에 넣고 문단 범위 토큰 출력의 평균(정규화). [CLS]는 첫 문단, [SEP]는 마지막 문단 평균에 넣음
         (Late Chunking 원문 4.1절). 그룹이 바뀔 때마다 다시 인코딩
  neigh  '앞 문단 + 문단 + 뒤 문단'을 이어 인코딩한 [CLS]
참고: minsim = 이웃 문단 코사인 최저 자리(variants.lumber_min_sim과 같은 판단), gemma = 원본(out/lumber_gemma_v5, NarrativeQA는 out/lumber_gemma)
재는 것
  같은 그룹 일치율: Gemma가 판단한 그룹을 그대로 주고 같은 자리를 고르는 비율(정확히 / ±1문단)
  자유 실행: Gemma 경계 재현·정밀·F1(문단 경계 번호, ±1 허용 포함), 청크 크기, 잘못된 자리 절단
  검색: 정답 근거 5-gram 재현율(같은 토큰 예산 512·1024, 문서 안 + FAISS 전체). 개발 문서(Gemma 포함) + 큰 표본(vectara 50 + docx 40)
값 고르기: ρ ∈ {0.5, 0.7, 0.85, 1.0}, min_tokens ∈ {0, 100, 200}을 개발 문서 절반(튜닝)의 같은 그룹 일치율로 판마다 고르고 나머지 절반(평가)에서 잰다.
출력: tests/chunking/out/lumber_cosine.md
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
import compare as C  # noqa: E402
import lumber_gemma as G  # noqa: E402
from v4_v5_lumber import bad_rate, qa_sets, safe_encoder  # noqa: E402
from chunking.sentences import paragraphs  # noqa: E402
from common.models import load_embedder, make_token_counter  # noqa: E402
from preprocess import longbench  # noqa: E402

THETA, W = 550, 2
RHOS, MINS = (0.5, 0.7, 0.85, 1.0), (0, 100, 200)
VARS = ("sep", "late", "neigh")
BUDGETS = (512, 1024)


# ---------- 벡터 ----------
class Late:
    """그룹 글을 한 번에 넣어 문단별 토큰 평균 벡터를 만든다(bge-m3 transformer 출력)."""

    def __init__(self, st):
        import torch
        self.torch, self.model, self.tok = torch, st[0].auto_model, st.tokenizer

    def __call__(self, units):
        text, spans, pos = "", [], 0
        for k, u in enumerate(units):
            if k:
                text += "\n\n"; pos += 2
            spans.append((pos, pos + len(u))); text += u; pos += len(u)
        enc = self.tok(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=8192)
        offs = enc.pop("offset_mapping")[0].tolist()
        with self.torch.no_grad():
            h = self.model(**{k: v.to(self.model.device) for k, v in enc.items()}).last_hidden_state[0].float().cpu().numpy()
        n = len(offs)
        owner = np.full(n, -1)
        for t, (a, b) in enumerate(offs):
            if b > a:
                for k, (x, y) in enumerate(spans):
                    if a < y and b > x:
                        owner[t] = k
                        break
        owner[0] = 0                                  # [CLS] → 첫 문단
        owner[n - 1] = len(units) - 1                 # [SEP] → 마지막 문단
        out = np.zeros((len(units), h.shape[1]), dtype=np.float32)
        for k in range(len(units)):
            sel = owner == k
            v = h[sel].mean(0) if sel.any() else h[0]
            out[k] = v / (np.linalg.norm(v) + 1e-9)
        return out


# ---------- 판단 ----------
def depth_curve(vecs, toks, s, m):
    """그룹 [s, m] 안 자리 j = s+1..m 의 depth(배열 길이 m−s)."""
    J = range(s + 1, m + 1)
    sims = []
    for j in J:
        a, b = max(s, j - W), min(m + 1, j + W)
        L = np.average(vecs[a:j], axis=0, weights=toks[a:j]); R = np.average(vecs[j:b], axis=0, weights=toks[j:b])
        sims.append(float(L @ R / (np.linalg.norm(L) * np.linalg.norm(R) + 1e-9)))
    sims = np.array(sims)
    dep = np.zeros(len(sims))
    for i in range(len(sims)):
        lp = sims[i]; k = i - 1
        while k >= 0 and sims[k] >= lp:
            lp = sims[k]; k -= 1
        rp = sims[i]; k = i + 1
        while k < len(sims) and sims[k] >= rp:
            rp = sims[k]; k += 1
        dep[i] = (lp - sims[i]) + (rp - sims[i])
    return dep


def decide(dep, toks, s, m, rho, min_tok):
    js = np.arange(s + 1, m + 1)
    pre = np.array([sum(toks[s:j]) for j in js])
    ok = pre >= min_tok
    if not ok.any():
        ok[:] = True
    d = np.where(ok, dep, -1)
    if d.max() <= 0:
        return m
    return int(js[np.argmax(d >= rho * d.max())])


def minsim_decide(vecs, s, m):
    js = np.arange(s + 1, m + 1)
    return int(js[np.argmin(np.einsum("ij,ij->i", vecs[js - 1], vecs[js]))])


def group_vecs(var, units, s, m, base):
    """그룹 [s, m]의 문단 벡터(late는 그룹 글로 새로 인코딩, 나머지는 문서 단위로 미리 만든 벡터의 조각)."""
    return LATE(units[s:m + 1]) if var == "late" else base[var][s:m + 1]


def run_doc(var, units, toks, base, rho, min_tok):
    """자유 실행: 청크 범위."""
    n = len(units)
    tok = np.concatenate([[0], np.cumsum(toks)])
    out, s = [], 0
    while s < n:
        if toks[s] > THETA:
            out.append((s, s + 1)); s += 1
            continue
        m = int(np.searchsorted(tok, tok[s] + THETA, side="right")) - 1
        if m >= n:
            out.append((s, n)); break
        if toks[m] > THETA:
            out.append((s, m)); s = m
            continue
        if var == "minsim":
            cut = minsim_decide(base["sep"], s, m)
        else:
            gv = group_vecs(var, units, s, m, base)
            dep = depth_curve(gv, toks[s:m + 1], 0, m - s)
            cut = s + decide(dep, toks[s:m + 1], 0, m - s, rho, min_tok)
        out.append((s, cut)); s = cut
    return out


# ---------- 데이터 ----------
def dev_set():
    out = []
    for ds, d, _ in G.dev_docs():
        if ds == "narrativeqa":
            b = next(x for x in longbench.documents("narrativeqa") if x["doc_id"] == d)
            units, g = paragraphs(b["text"]), json.loads((HERE / "out" / "lumber_gemma" / ds / f"{d}.json").read_text(encoding="utf-8"))
        else:
            r = json.loads((ROOT / "data" / "clean_text" / ds / f"{d}.json").read_text(encoding="utf-8"))
            units, g = r["text"].split("\n\n"), json.loads((HERE / "out" / "lumber_gemma_v5" / ds / f"{d}.json").read_text(encoding="utf-8"))
        assert g["n_paras"] == len(units), (ds, d)
        out.append((ds, d, units, g))
    return out


def boundaries(ranges):
    return {j for _, j in ranges[:-1]}


def prf(pred, gold, tol=0):
    hit_p = sum(any(abs(p - g) <= tol for g in gold) for p in pred)
    hit_g = sum(any(abs(p - g) <= tol for p in pred) for g in gold)
    P = hit_p / len(pred) if pred else 0.0
    R = hit_g / len(gold) if gold else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


LATE = None


def main():
    global LATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="bge-m3 late 벡터 점검만")
    ns = ap.parse_args()
    import faiss
    count = make_token_counter()
    st = load_embedder()
    LATE = Late(st)
    from common.models import make_encode
    enc = make_encode(st, batch_size=8)
    L = ["# LumberChunker 코사인 판: 문단 벡터 비교 (따로 / 1 그룹 late / 3 이웃 이어 인코딩)", "",
         f"θ = {THETA}, 덩어리 W = {W}, 단위 = v5 문단(NarrativeQA는 원문 빈 줄 문단). 판단 = 그룹 안 깊이(TextTiling), 깊이 ≥ ρ·최대인 첫 자리.", ""]

    # 0) 점검: 원문 표 1(Berlin) 재현 + 짧은 문단 이웃 유사도
    berlin = ["Berlin is the capital and largest city of Germany, both by area and by population.",
              "Its more than 3.85 million inhabitants make it the European Union's most populous city, as measured by population within city limits.",
              "The city is also one of the states of Germany, and is the third smallest state in the country in terms of area."]
    q = enc(["Berlin"])[0]
    sep_v, late_v = enc(berlin), LATE(berlin)
    L += ["## 0. 점검: bge-m3 late 벡터", "", "원문(Late Chunking) 표 1 재현: 'Berlin'과 각 문장의 코사인", "", "| 문장 | 따로 | late |", "|---|---|---|"]
    L += [f"| {t[:60]}… | {float(sep_v[k] @ q):.3f} | {float(late_v[k] @ q):.3f} |" for k, t in enumerate(berlin)]
    L.append("")
    print("\n".join(L[-7:]), flush=True)
    if ns.check_only:
        print("\n".join(L)); return

    dev = dev_set()
    base = {}
    for ds, d, units, g in dev:
        toks = np.array(count(units))
        base[(ds, d)] = {"toks": toks, "sep": enc(units),
                         "neigh": np.asarray(enc(["\n\n".join(units[max(0, i - 1):i + 2]) for i in range(len(units))]), dtype=np.float32)}
    print("개발 문서 벡터 완료", flush=True)

    # 짧은 문단(20토큰 미만)의 이웃 유사도: 따로 vs late(그 문단이 든 Gemma 그룹으로)
    sims = defaultdict(list)
    for ds, d, units, g in dev:
        B = base[(ds, d)]
        for c in g["calls"][:30]:
            s, m = c["group"]
            lv = LATE(units[s:m + 1])
            for i in range(s + 1, m + 1):
                if B["toks"][i] < 20:
                    sims["따로"].append(float(B["sep"][i - 1] @ B["sep"][i]))
                    sims["late"].append(float(lv[i - s - 1] @ lv[i - s]))
                    sims["이웃"].append(float(B["neigh"][i - 1] @ B["neigh"][i]))
    L += [f"짧은 문단(20토큰 미만, {len(sims['따로'])}개)과 앞 문단의 코사인 평균: 따로 {np.mean(sims['따로']):.3f} / late {np.mean(sims['late']):.3f} / 이웃 이어 {np.mean(sims['이웃']):.3f}", ""]

    # 1) 같은 그룹 일치율 (튜닝 절반 → 값 고르기, 평가 절반)
    order = sorted(range(len(dev)), key=lambda k: (dev[k][0], dev[k][1]))
    tune = set(order[0::2]); evalset = set(order[1::2])
    cache = {}

    def group_depth(var, k, s, m):
        key = (var, k, s, m)
        if key not in cache:
            ds, d, units, g = dev[k]
            B = base[(ds, d)]
            gv = group_vecs(var, units, s, m, B)
            cache[key] = depth_curve(gv, B["toks"][s:m + 1], 0, m - s)
        return cache[key]

    def tf_acc(var, ks, rho, mt):
        ex = near = n = 0
        for k in ks:
            ds, d, units, g = dev[k]
            B = base[(ds, d)]
            for c in g["calls"]:
                s, m = c["group"]
                if var == "minsim":
                    cut = minsim_decide(B["sep"], s, m)
                else:
                    cut = s + decide(group_depth(var, k, s, m), B["toks"][s:m + 1], 0, m - s, rho, mt)
                ex += cut == c["cut"]; near += abs(cut - c["cut"]) <= 1; n += 1
        return ex / n, near / n, n

    best = {}
    L += ["## 1. 같은 그룹 일치율 (Gemma 그룹을 그대로 주고 같은 자리를 고르는 비율)", "",
          "| 판 | 고른 ρ | 고른 min_tokens | 튜닝 정확 | 평가 정확 | 평가 ±1문단 | 평가 그룹 수 |", "|---|---|---|---|---|---|---|"]
    for var in VARS + ("minsim",):
        if var == "minsim":
            best[var] = (None, None)
        else:
            grid = [(tf_acc(var, tune, r, mt)[0], r, mt) for r in RHOS for mt in MINS]
            _, r, mt = max(grid)
            best[var] = (r, mt)
        a_t = tf_acc(var, tune, *best[var])[0] if var != "minsim" else tf_acc(var, tune, 0, 0)[0]
        a_e, n_e, cnt_e = tf_acc(var, evalset, *(best[var] if var != "minsim" else (0, 0)))
        L.append(f"| {var} | {best[var][0]} | {best[var][1]} | {a_t:.1%} | **{a_e:.1%}** | {n_e:.1%} | {cnt_e} |")
        print(var, best[var], f"{a_e:.3f}", flush=True)
    # 우연 기준: 그룹 안 자리 중 무작위 / 항상 θ 경계(m)
    rnd = random.Random(0)
    ex_r = ex_m = n = 0
    for k in evalset:
        for c in dev[k][3]["calls"]:
            s, m = c["group"]
            ex_r += rnd.randint(s + 1, m) == c["cut"]; ex_m += m == c["cut"]; n += 1
    L += [f"| (무작위 자리) | – | – | – | {ex_r / n:.1%} | – | {n} |", f"| (항상 θ 경계) | – | – | – | {ex_m / n:.1%} | – | {n} |", ""]

    # 2) 자유 실행: Gemma 경계와 비교 (평가 절반)
    L += ["## 2. 자유 실행 (평가 문서): Gemma 경계와 비교, 청크 크기", "",
          "| 판 | 청크 | 평균 토큰 | 50토큰 미만 | 잘못된 자리 절단 | 경계 P/R/F1 (정확) | 경계 F1 (±1문단) |", "|---|---|---|---|---|---|---|"]
    free = defaultdict(dict)
    for var in VARS + ("minsim", "gemma"):
        P = []; F1n = []; sizes = []; bad = []
        for k in sorted(evalset | tune):
            ds, d, units, g = dev[k]
            B = base[(ds, d)]
            rg = g["ranges"] if var == "gemma" else run_doc(var, units, B["toks"], B, *(best.get(var) or (0, 0)))
            free[var][(ds, d)] = ["\n\n".join(units[i:j]) for i, j in rg]
            if k in evalset:
                gb = boundaries(g["ranges"]); pb = boundaries(rg)
                P.append(prf(pb, gb)); F1n.append(prf(pb, gb, 1)[2])
                sizes += [int(B["toks"][i:j].sum()) for i, j in rg]; bad.append(bad_rate(free[var][(ds, d)]))
        p, r, f = np.mean(P, axis=0)
        sz = np.array(sizes)
        L.append(f"| {var} | {len(sz):,} | {sz.mean():.0f} | {np.mean(sz < 50):.1%} | {np.mean(bad):.1%} | {p:.2f}/{r:.2f}/{f:.2f} | {np.mean(F1n):.2f} |")
        print(var, "자유 실행 완료", flush=True)
    L.append("")

    # 3) 검색: 개발 문서(Gemma 포함)
    qmap = qa_sets()
    rng = np.random.default_rng(0)

    def retrieval(chunks_by, title):
        res = defaultdict(lambda: defaultdict(list))
        names = list(chunks_by)
        for ds in ("vectara", "docx_corpus"):
            for name in names:
                items = [(d, cs) for (x, d), cs in chunks_by[name].items() if x == ds and d in qmap[ds]]
                if not items:
                    continue
                texts, toks, owner, embs = [], [], [], []
                for d, cs in items:
                    texts += cs; toks += count(cs); owner += [d] * len(cs); embs.append(SAFE(cs))
                E = np.concatenate(embs); owner = np.array(owner)
                index = faiss.IndexFlatIP(E.shape[1]); index.add(E)
                qs = [(d, qq, ev) for d, _ in items for qq, ev in qmap[ds][d]]
                Q = SAFE([qq for _, qq, _ in qs])
                _, top = index.search(Q, 128)
                for (d, qq, ev), qe, row in zip(qs, Q, top):
                    dg = C.grams(C.words(" ".join(texts[i] for i in np.where(owner == d)[0])))
                    mine = np.where(owner == d)[0]
                    for B_ in BUDGETS:
                        for mode, od in (("문서 안", mine[np.argsort(-(E[mine] @ qe))]), ("FAISS", row)):
                            got = " ".join(t for i, t in C.fill(od, texts, toks, B_) if owner[i] == d)
                            res[(mode, ds, B_)][name].append(C.score(ds, ev, got, dg, None))
        out = [f"## {title}", "", "| 검색 · 데이터셋 · 예산 | 질문 | " + " | ".join(names) + " |", "|---" * (len(names) + 2) + "|"]
        for key, r in sorted(res.items()):
            keep = [i for i in range(len(r[names[0]])) if all(r[n][i] is not None for n in names if n in r)]
            vals = {n: np.mean([r[n][i] for i in keep]) for n in names if n in r}
            top = max(vals.values())
            out.append(f"| {' · '.join(map(str, key))} | {len(keep)} | " + " | ".join(
                (f"**{vals[n]:.3f}**" if vals[n] == top else f"{vals[n]:.3f}") if n in vals else "–" for n in names) + " |")
        return out + [""]

    global SAFE
    SAFE = safe_encoder()
    L += retrieval({k: free[k] for k in ("gemma",) + VARS + ("minsim",)}, "3. 검색: 개발 문서 (vectara 10 + docx 10, Gemma 포함, 질문 수 적음)")
    print("개발 문서 검색 완료", flush=True)

    # 4) 검색: 큰 표본(vectara 50 + docx 40), 코사인 판끼리
    rnd = random.Random(0)
    big = defaultdict(dict)
    for ds, n in (("vectara", 50), ("docx_corpus", 40)):
        ids = sorted(d for d in qmap[ds] if (ROOT / "data" / "clean_text" / ds / f"{d}.json").exists())
        for d in rnd.sample(ids, min(n, len(ids))):
            units = json.loads((ROOT / "data" / "clean_text" / ds / f"{d}.json").read_text(encoding="utf-8"))["text"].split("\n\n")
            B = {"toks": np.array(count(units)), "sep": SAFE(units),
                 "neigh": SAFE(["\n\n".join(units[max(0, i - 1):i + 2]) for i in range(len(units))])}
            for var in VARS + ("minsim",):
                rg = run_doc(var, units, B["toks"], B, *(best.get(var) or (0, 0)))
                big[var][(ds, d)] = ["\n\n".join(units[i:j]) for i, j in rg]
    print("큰 표본 청킹 완료", flush=True)
    sz = {v: np.mean([len(c.split()) for cs in big[v].values() for c in cs]) for v in big}
    L += [f"큰 표본 청크 평균 낱말 수: " + ", ".join(f"{v} {s:.0f}" for v, s in sz.items()), ""]
    L += retrieval(dict(big), "4. 검색: 큰 표본 (vectara 50 + docx 40, 코사인 판끼리)")
    (HERE / "out" / "lumber_cosine.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


SAFE = None

if __name__ == "__main__":
    main()
