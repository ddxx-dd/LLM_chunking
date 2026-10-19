"""청커 가벼운 비교(tests 전용, 파일럿 표본·임시 크기 값): 크기 분포, HotpotQA 패시지 경계(Pk·WindowDiff), 문서 안 검색.

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/chunking/compare_v2.py [--n-vectara 50 --n-docx 40 --n-hotpot 60]

청커(임시 값, 평균 약 200 bge-m3 토큰 근처로 맞춤)
  fixed 850자 · recursive 850자 · semantic(LangChain 기본, 백분위 95, 크기 통제 없음)
  advanced_semantic: src/chunking/advanced_semantic(θ 550, min 100, 평균 200토큰이 되도록 β 이분 탐색, 문단 보너스 0.5β, 제목 뒤 절단 금지)
  advanced_semantic(b=0): 같은 조건에서 문단 보너스·제목 규칙 없이(순수 의미 분할, 절제 실험)
  (2026-10-17 첫 실행 결과 out/compare_v2_first_run.md는 옛 판 asc_maxmin·asc_centroid와 시험 구현 v2·v2+구조 비교이고, 그 결과로 v2+구조가 advanced_semantic이 됐다)
검색(문서 안): 질문과 청크를 bge-m3로 인코딩해 코사인 순으로 고르고, 토큰 예산(512 / 1024) 안에 드는 청크만 넣는다(청크 크기 차이를 줄이려고).
  vectara·docx: 정답 근거 글의 낱말 5-gram 중 검색된 글에 있는 비율(우리 문서에 있는 5-gram만 분모)
  hotpotqa: 정답 문자열(정규화 낱말열)이 검색된 글에 있으면 1
경계: HotpotQA 원문의 'Passage N:' 자리를 정답 경계로 본다(공백 제외 글자 위치).
  경계 재현율 = 정답 경계 중 ±10자 안에 청크 경계가 있는 비율, 순도 = 두 패시지에 걸치지 않는 청크 비율(높을수록 좋음).
  Pk·WindowDiff는 정답 구간(패시지 수천 토큰)이 청크(200토큰)보다 훨씬 커서 잘게 자른 것만으로 벌점을 받아 뺐다(시험 실행에서 확인).
출력: tests/chunking/out/compare_v2.md
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
from chunking import advanced_semantic as A, fixed, recursive, semantic  # noqa: E402
from chunking.embeddings import default_encode  # noqa: E402
from common.models import make_token_counter  # noqa: E402
from preprocess import longbench  # noqa: E402

THETA, MIN_TOK, TARGET, CHARS = 550, 100, 200, 850
NAMES = ["fixed", "recursive", "semantic", "advanced_semantic", "advanced_semantic(b=0)"]
BUDGETS = (512, 1024)
W = re.compile(r"\w+")


def words(t):
    return W.findall(t.lower())


def grams(ws, n=5):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def run_chunkers(text, lang, encode, count):
    out, secs = {}, {}
    for name in NAMES:
        t0 = time.time()
        if name == "fixed":
            c = fixed.chunk(text, CHARS)
        elif name == "recursive":
            c = recursive.chunk(text, CHARS)
        elif name == "semantic":
            c = semantic.chunk(text, encode, lang=lang)
        elif name == "advanced_semantic":
            c = A.chunk(text, encode, count, THETA, target=TARGET, lang=lang, min_tokens=MIN_TOK)
        else:
            c = A.chunk(text, encode, count, THETA, target=TARGET, lang=lang, min_tokens=MIN_TOK, bonus_frac=0.0, heading_rule=False)
        secs[name] = time.time() - t0
        out[name] = [x for x in c if x.strip()]
    return out, secs


def retrieve(chunks, ctoks, cemb, qemb, budget):
    order = np.argsort(-(cemb @ qemb))
    got, used = [], 0
    for k in order:
        if got and used + ctoks[k] > budget:
            break
        got.append(chunks[k]); used += ctoks[k]
    return " ".join(got)


def boundary_scores(ref_b, hyp_b, tol=10):
    """→ (경계 재현율, 순도). ref_b·hyp_b: 공백 제외 글자 위치."""
    hyp_b = np.sort(hyp_b)
    hit = [len(hyp_b) and np.min(np.abs(hyp_b - r)) <= tol for r in ref_b]
    starts = np.concatenate([[0], hyp_b]); ends = np.concatenate([hyp_b, [np.inf]])
    pure = [not np.any((ref_b > a + tol) & (ref_b < b - tol)) for a, b in zip(starts, ends)]
    return float(np.mean(hit)), float(np.mean(pure))


def boundaries_ns(chunks):
    pos, b = 0, []
    for c in chunks[:-1]:
        pos += len("".join(c.split())); b.append(pos)
    return np.array(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-vectara", type=int, default=50)
    ap.add_argument("--n-docx", type=int, default=40)
    ap.add_argument("--n-hotpot", type=int, default=60)
    ns = ap.parse_args()
    encode, count = default_encode(), make_token_counter()
    rnd = random.Random(0)
    stats = defaultdict(lambda: defaultdict(list))      # stats[ds][name] → 청크 토큰 목록
    ret = defaultdict(lambda: defaultdict(list))        # ret[(ds, budget)][name] → 점수 목록
    bnd = defaultdict(list)                             # bnd[name] → (pk, wd)
    secs = defaultdict(float)

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

    for n_job, (ds, d, text, lang, qs) in enumerate(jobs):
        chunks, sec = run_chunkers(text, lang, encode, count)
        for k, v in sec.items():
            secs[k] += v
        qemb = np.asarray(encode([q for q, _ in qs]), dtype=np.float32)
        doc_ws = words(text)
        doc_g = grams(doc_ws)
        doc_norm = " " + " ".join(doc_ws) + " "
        for name, cs in chunks.items():
            ct = count(cs)
            stats[ds][name] += ct
            cemb = np.asarray(encode(cs), dtype=np.float32)
            for (q, gold), qe in zip(qs, qemb):
                for B in BUDGETS:
                    got = retrieve(cs, ct, cemb, qe, B)
                    if ds == "hotpotqa":
                        ans = [" " + " ".join(words(a)) + " " for a in gold]
                        ans = [a for a in ans if a.strip() and a in doc_norm]
                        if ans:
                            g = " " + " ".join(words(got)) + " "
                            ret[(ds, B)][name].append(float(any(a in g for a in ans)))
                    else:
                        eg = grams(words(gold)) & doc_g
                        if eg:
                            ret[(ds, B)][name].append(len(eg & grams(words(got))) / len(eg))
            if ds == "hotpotqa":
                nst = "".join(text.split())
                if len("".join("".join(cs).split())) == len(nst):
                    ref = np.array([m.start() for m in re.finditer(r"Passage\d+:", nst) if m.start() > 0])
                    if len(ref):
                        bnd[name].append(boundary_scores(ref, boundaries_ns(cs)))
                else:
                    print("경계 계산 제외(글자 수 불일치):", ds, d, name, flush=True)
        print(f"[{n_job + 1}/{len(jobs)}] {ds} {d[:20]} 완료", flush=True)

    L = ["# 청커 가벼운 비교 (파일럿 표본, 임시 크기 값)", "",
         f"표본: vectara {ns.n_vectara}편, docx {ns.n_docx}편, HotpotQA {ns.n_hotpot}편(무작위, seed 0). "
         f"fixed·recursive {CHARS}자, advanced_semantic θ {THETA}·min {MIN_TOK}·평균 {TARGET}토큰 목표.", "",
         "## 1. 청크 크기 (bge-m3 토큰)", "", "| 데이터셋 | " + " | ".join(NAMES) + " |", "|---" * (len(NAMES) + 1) + "|"]
    for ds, s in stats.items():
        L.append(f"| {ds} 평균 | " + " | ".join(f"{np.mean(s[n]):.0f}" for n in NAMES) + " |")
        L.append(f"| {ds} 50토큰 미만 | " + " | ".join(f"{np.mean(np.array(s[n]) < 50):.0%}" for n in NAMES) + " |")
        L.append(f"| {ds} 최대 | " + " | ".join(f"{max(s[n])}" for n in NAMES) + " |")
    L += ["", "## 2. 문서 안 검색 (토큰 예산 안 정답 포함 정도, 높을수록 좋음)", "", "| 데이터셋 · 예산 | 질문 수 | " + " | ".join(NAMES) + " |", "|---" * (len(NAMES) + 2) + "|"]
    for (ds, B), r in sorted(ret.items()):
        vals = [np.mean(r[n]) for n in NAMES]
        top = max(vals)
        L.append(f"| {ds} · {B} | {len(r[NAMES[0]])} | " + " | ".join(f"**{v:.3f}**" if v == top else f"{v:.3f}" for v in vals) + " |")
    L += ["", "vectara·docx = 정답 근거 5-gram 재현율, hotpotqa = 정답 문자열 포함 비율.", "",
          "## 3. HotpotQA 패시지 경계 (높을수록 좋음, 크기가 비슷한 청커끼리 비교)", "", "| 지표 | " + " | ".join(NAMES) + " |", "|---" * (len(NAMES) + 1) + "|"]
    L.append("| 경계 재현율(±10자) | " + " | ".join(f"{np.mean([x[0] for x in bnd[n]]):.3f}" for n in NAMES) + " |")
    L.append("| 순도(한 패시지 안 청크) | " + " | ".join(f"{np.mean([x[1] for x in bnd[n]]):.3f}" for n in NAMES) + " |")
    L += ["", "## 4. 청킹 시간 (초, 표본 전체, advanced_semantic은 단위 임베딩 포함)", "", "| " + " | ".join(NAMES) + " |", "|---" * len(NAMES) + "|",
          "| " + " | ".join(f"{secs[n]:.1f}" for n in NAMES) + " |"]
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "compare_v2.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
