"""NarrativeQA(책 14편): 문단 단위 LumberChunker 코사인 판 vs fixed vs LangChain semantic (tests 전용).

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/chunking/narrative_lumber.py

lumber_para: 빈 줄 문단 단위, θ = 550 bge-m3 토큰까지 모아 그룹 안 이웃 문단 코사인 최소 자리에서 자르고 반복(variants.lumber_min_sim)
fixed: CharacterTextSplitter 500자(src chunking.fixed)
semantic: LangChain SemanticChunker 기본 + 공용 문장 분리(src chunking.semantic). 백분위 하나를 14편 공통으로,
  평균 청크 토큰이 (fixed 평균 + lumber_para 평균) / 2가 되도록 이분 탐색(LangChain과 같은 계산: 문장 사이 거리 > 백분위 값에서 자름)
검색: 같은 토큰 예산(512 / 1024 / 2048, 넘치는 청크는 낱말 비율로 자름). 책 안 검색 + 14편 전체 FAISS 색인(정답 책 청크만 점수)
지표: 정답 낱말 재현율(정답의 불용어 뺀 낱말 중 검색 글에 있는 비율, 정답 여러 개 중 최대), 정답 문자열 포함률. 95% 구간·짝지은 차이는 부트스트랩 1000회
출력: tests/chunking/out/narrative_lumber.md
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
import compare as C  # noqa: E402
import variants as Vr  # noqa: E402
from chunking import fixed, semantic  # noqa: E402
from chunking.embeddings import as_langchain, default_encode  # noqa: E402
from chunking.sentences import paragraphs, split_text  # noqa: E402
from common.models import make_token_counter  # noqa: E402
from preprocess import longbench  # noqa: E402

THETA, FIXED_CHARS = 550, 500
NAMES = ["fixed", "semantic", "lumber_para"]
BUDGETS = (512, 1024, 2048)
STOP = set("a an the of to in on at for and or but is are was were be been being he she it they them his her its their this that these "
           "those with as by from into about than then so not no do does did has have had i you we me my your our who whom which what "
           "when where why how there here up out over after before because while if".split())


def ans_words(a):
    return {w for w in C.words(a) if w not in STOP}


def answer_scores(answers, got):
    gw = set(C.words(got))
    g = " " + " ".join(C.words(got)) + " "
    rec = max((len(aw & gw) / len(aw) for aw in map(ans_words, answers) if aw), default=None)
    hit = float(any((" " + " ".join(C.words(a)) + " ") in g for a in answers if C.words(a)))
    return rec, hit


def semantic_distances(text, encode):
    """LangChain SemanticChunker와 같은 문장·거리(공용 문장 분리 + buffer 1)."""
    from langchain_experimental.text_splitter import SemanticChunker

    class _C(SemanticChunker):
        def _get_single_sentences_list(self, t):
            return split_text(t) or [t]

    sp = _C(as_langchain(encode))
    sents = sp._get_single_sentences_list(text)
    dist, _ = sp._calculate_sentence_distances(sents)
    return sents, np.asarray(dist)


def main():
    import faiss
    encode, count = default_encode(), make_token_counter()
    books = longbench.documents("narrativeqa")
    qmap = defaultdict(list)
    for q in longbench.questions("narrativeqa"):
        qmap[q["doc_id"]].append((q["question"], q["answers"]))
    chunks = {n: {} for n in NAMES}
    over = []
    for b in books:
        paras = paragraphs(b["text"])
        pt = count(paras)
        pe = np.asarray(encode(paras), dtype=np.float32)
        r = Vr.lumber_min_sim(pt, pe, THETA)
        chunks["lumber_para"][b["doc_id"]] = ["\n\n".join(paras[i:j]) for i, j in r]
        over += [t for t in pt if t > THETA]
        chunks["fixed"][b["doc_id"]] = [c for c in fixed.chunk(b["text"], FIXED_CHARS) if c.strip()]
        print(b["doc_id"], "lumber·fixed 완료", flush=True)
    avg = {n: np.mean([t for cs in chunks[n].values() for t in count(cs)]) for n in ("fixed", "lumber_para")}
    target = (avg["fixed"] + avg["lumber_para"]) / 2

    # 백분위 탐색: 청크 = 거리 > 백분위 값인 자리에서 자른 문장 묶음(LangChain과 같은 계산), 크기 = 문장 토큰 합
    cache = {}
    for b in books:
        sents, dist = semantic_distances(b["text"], encode)
        cache[b["doc_id"]] = (np.array(count(sents)), dist)
    def sem_avg(p):
        tot, nch = 0, 0
        for st, dist in cache.values():
            nch += 1 + (int(np.sum(dist > np.percentile(dist, p))) if len(dist) else 0)
            tot += st.sum()
        return tot / nch
    lo, hi = 0.0, 100.0
    for _ in range(30):
        p = (lo + hi) / 2
        lo, hi = (p, hi) if sem_avg(p) < target else (lo, p)
    pct = (lo + hi) / 2
    for b in books:
        chunks["semantic"][b["doc_id"]] = [c for c in semantic.chunk(b["text"], encode, breakpoint_threshold_amount=pct) if c.strip()]
    print("semantic 백분위", round(pct, 2), flush=True)

    # 검색
    rng = np.random.default_rng(0)
    res = defaultdict(lambda: defaultdict(list))    # (mode, B, 지표) → name → 점수
    stats = {}
    for name in NAMES:
        texts, toks, owner, embs = [], [], [], []
        for b in books:
            cs = chunks[name][b["doc_id"]]
            texts += cs; toks += count(cs); owner += [b["doc_id"]] * len(cs)
            embs.append(np.asarray(encode(cs), dtype=np.float32))
        E = np.concatenate(embs); owner = np.array(owner)
        st = np.array(toks)
        stats[name] = (len(st), st.mean(), np.median(st), st.max(), np.mean(st < 50))
        index = faiss.IndexFlatIP(E.shape[1]); index.add(E)
        qs = [(b["doc_id"], q, a) for b in books for q, a in qmap[b["doc_id"]]]
        Q = np.asarray(encode([q for _, q, _ in qs]), dtype=np.float32)
        _, top = index.search(Q, 256)
        for (d, q, a), qe, row in zip(qs, Q, top):
            mine = np.where(owner == d)[0]
            in_order = mine[np.argsort(-(E[mine] @ qe))]
            for B in BUDGETS:
                for mode, order in (("책 안", in_order), ("FAISS 전체", row)):
                    got = " ".join(t for k, t in C.fill(order, texts, toks, B) if owner[k] == d)
                    rec, hit = answer_scores(a, got)
                    res[(mode, B, "낱말 재현율")][name].append(rec)
                    res[(mode, B, "문자열 포함")][name].append(hit)
        print(name, "검색 완료", flush=True)

    nq = len(res[("책 안", BUDGETS[0], "낱말 재현율")]["fixed"])
    L = ["# NarrativeQA: 문단 단위 LumberChunker 코사인 판 vs fixed vs semantic", "",
         f"책 {len(books)}편(대본 제외), 질문 {nq}개. lumber_para θ = {THETA} bge-m3 토큰(문단 단위), fixed {FIXED_CHARS}자, "
         f"semantic 백분위 {pct:.2f}(목표 평균 {target:.0f}토큰 = fixed {avg['fixed']:.0f}와 lumber_para {avg['lumber_para']:.0f}의 평균).", "",
         f"θ({THETA})보다 큰 문단: {len(over)}개" + (f"(최대 {max(over)}토큰, 혼자 청크)" if over else "") + ".", "",
         "## 1. 청크 (bge-m3 토큰)", "", "| 청커 | 청크 수 | 평균 | 중앙값 | 최대 | 50토큰 미만 |", "|---|---|---|---|---|---|"]
    for n in NAMES:
        c, m, md, mx, sm = stats[n]
        L.append(f"| {n} | {c:,} | {m:.0f} | {md:.0f} | {mx} | {sm:.0%} |")
    for mode in ("책 안", "FAISS 전체"):
        for metric in ("낱말 재현율", "문자열 포함"):
            L += ["", f"## 검색: {mode} · 정답 {metric} (같은 토큰 예산, 평균 ± 95% 구간)", "",
                  "| 예산 | " + " | ".join(NAMES) + " | lumber−fixed | lumber−semantic |", "|---" * (len(NAMES) + 3) + "|"]
            for B in BUDGETS:
                r = res[(mode, B, metric)]
                keep = [i for i in range(nq) if all(r[n][i] is not None for n in NAMES)]
                v = {n: np.array([r[n][i] for i in keep], float) for n in NAMES}
                s = {n: C.ci(v[n], rng) for n in NAMES}
                top = max(m for m, _ in s.values())
                cells = [(f"**{m:.3f}**" if m == top else f"{m:.3f}") + f" ±{h:.3f}" for m, h in s.values()]
                for other in ("fixed", "semantic"):
                    m, h = C.ci(v["lumber_para"] - v[other], rng)
                    cells.append(f"{m:+.3f}" + ("*" if abs(m) > h else ""))
                L.append(f"| {B} | " + " | ".join(cells) + " |")
    L += ["", "\\* = 짝지은 부트스트랩 95% 구간이 0을 포함하지 않음."]
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "narrative_lumber.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
