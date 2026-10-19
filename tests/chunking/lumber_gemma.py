"""원본 LumberChunker(Duarte et al. 2024)를 Gemma로 재현해 '흉내 낼 기준' 절단 위치를 만든다(tests 전용).

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/chunking/lumber_gemma.py [--limit-docs N]

논문 3.1절·그림 1·부록 B 그대로
  단위 = 빈 줄 문단(chunking.sentences.paragraphs, 정리 없음), 문서 전체에 'ID 0001:' 번호
  s부터 문단을 더해 토큰 합(bge-m3)이 θ = 550을 넘을 때까지 그룹(넘게 만든 문단 포함) → 부록 B 프롬프트로 Gemma에 묻는다
  답 'Answer: ID XXXX' 문단 앞에서 자르고 그 문단부터 다시 그룹을 만든다. 남은 글이 θ 안이면 마지막 청크
추가 규칙(앞서 사용자와 정한 공통 규칙): θ보다 큰 문단(표 등)은 그 앞에서 끊고 혼자 청크(LLM 호출 없음)
잘못된 답(형식 오류, 그룹 밖, 첫 문단): 허용 범위를 덧붙여 한 번 다시 묻고, 그래도 안 되면 θ 경계(넘게 만든 문단 앞)에서 자른다(기록)
개발 문서: vectara 10 + docx 10 + NarrativeQA 3 (영어, seed 1). Gemma는 common.models.generate(탐욕 디코딩, 같은 입력 → 같은 출력)
출력: tests/chunking/out/lumber_gemma/<데이터셋>/<doc_id>.json (있으면 건너뜀), 요약 tests/chunking/out/lumber_gemma.md
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chunking.sentences import paragraphs  # noqa: E402
from common.models import generate, load_llm, make_token_counter  # noqa: E402
from preprocess import longbench  # noqa: E402

THETA = 550
OUT = HERE / "out" / "lumber_gemma"
PROMPT = ("You will receive as input an English document with paragraphs identified by 'ID XXXX: <text>'.\n\n"
          "Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.\n\n"
          "Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX'.\n\n"
          "Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts "
          "and keeping groups manageable.\n\nDocument:\n")
ANS = re.compile(r"ID\s*(\d{1,5})")


def dev_docs():
    rnd = random.Random(1)
    out = []
    for ds, n in (("vectara", 10), ("docx_corpus", 10)):
        files = sorted(p for p in (ROOT / "data" / "clean_text" / ds).glob("*.json") if not p.name.endswith(".dropped.json"))
        rows = [json.loads(p.read_text(encoding="utf-8")) for p in files]
        rows = [r for r in rows if r["lang"] == "en"]
        out += [(ds, r["doc_id"], r["text"]) for r in rnd.sample(rows, n)]
    books = longbench.documents("narrativeqa")
    out += [("narrativeqa", b["doc_id"], b["text"]) for b in rnd.sample(books, 3)]
    return out


def ask(llm, paras, s, m, extra=""):
    doc = "\n".join(f"ID {i + 1:04d}: {paras[i]}" for i in range(s, m + 1))
    raw = generate(llm, PROMPT + doc + extra, max_new_tokens=24)
    got = [int(x) - 1 for x in ANS.findall(raw)]
    idx = got[-1] if got else None
    return raw, (idx if idx is not None and s < idx <= m else None)


def lumber(llm, paras, toks):
    n = len(paras)
    tok = np.concatenate([[0], np.cumsum(toks)])
    ranges, log, s = [], [], 0
    while s < n:
        if toks[s] > THETA:
            ranges.append((s, s + 1)); s += 1
            continue
        m = int(np.searchsorted(tok, tok[s] + THETA, side="right")) - 1     # [s, m)이 θ 안, m이 넘게 만든 문단
        if m >= n:
            ranges.append((s, n)); break
        if toks[m] > THETA:
            ranges.append((s, m)); s = m
            continue
        raw, idx = ask(llm, paras, s, m)
        rec = {"group": [s, m], "raw": raw}
        if idx is None:
            raw2, idx = ask(llm, paras, s, m, f"\n\nThe answer must be one ID from ID {s + 2:04d} to ID {m + 1:04d}.")
            rec["retry"] = raw2
        if idx is None:
            idx = m
            rec["fallback"] = True
        rec["cut"] = idx
        log.append(rec)
        ranges.append((s, idx)); s = idx
    return ranges, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-docs", type=int, default=None)
    ns = ap.parse_args()
    count = make_token_counter()
    llm = load_llm()
    docs = dev_docs()[:ns.limit_docs] if ns.limit_docs else dev_docs()
    for k, (ds, d, text) in enumerate(docs):
        p = OUT / ds / f"{d}.json"
        if p.exists():
            continue
        t0 = time.time()
        paras = paragraphs(text)
        toks = count(paras)
        ranges, log = lumber(llm, paras, toks)
        assert ranges[0][0] == 0 and ranges[-1][1] == len(paras) and all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"doc_id": d, "dataset": ds, "theta": THETA, "n_paras": len(paras), "para_tokens": toks,
                                 "ranges": ranges, "calls": log, "seconds": round(time.time() - t0, 1)}, ensure_ascii=False), encoding="utf-8")
        print(f"[{k + 1}/{len(docs)}] {ds} {d[:24]} 문단 {len(paras)} 청크 {len(ranges)} 호출 {len(log)} "
              f"잘못된 답 {sum('retry' in r for r in log)} 대체 {sum(r.get('fallback', False) for r in log)} {time.time() - t0:.0f}초", flush=True)
    summarize(docs)


def summarize(docs):
    L = ["# 원본 LumberChunker(Gemma) 기준 절단", "", f"θ = {THETA} bge-m3 토큰, 단위 = 빈 줄 문단, 프롬프트 = 논문 부록 B. 문서별 결과는 lumber_gemma/<데이터셋>/<doc_id>.json.", "",
         "| 데이터셋 | 문서 | 문단 | 청크 | 평균 청크 토큰 | 50토큰 미만 | LLM 호출 | 다시 물음 | θ 경계 대체 | 초 |", "|---|---|---|---|---|---|---|---|---|---|"]
    tot = {}
    for ds, d, _ in docs:
        p = OUT / ds / f"{d}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        ct = [sum(r["para_tokens"][i:j]) for i, j in r["ranges"]]
        retry = sum("retry" in c for c in r["calls"]); fb = sum(c.get("fallback", False) for c in r["calls"])
        L.append(f"| {ds} | {d[:20]} | {r['n_paras']} | {len(ct)} | {np.mean(ct):.0f} | {np.mean(np.array(ct) < 50):.0%} | {len(r['calls'])} | {retry} | {fb} | {r['seconds']:.0f} |")
        t = tot.setdefault(ds, [[], 0, 0, 0])
        t[0] += ct; t[1] += len(r["calls"]); t[2] += retry; t[3] += fb
    L += ["", "## 데이터셋별", "", "| 데이터셋 | 청크 | 평균 토큰 | 중앙값 | 50토큰 미만 | LLM 호출 | 다시 물음 | θ 경계 대체 |", "|---|---|---|---|---|---|---|---|"]
    for ds, (ct, c, rt, fb) in tot.items():
        a = np.array(ct)
        L.append(f"| {ds} | {len(a)} | {a.mean():.0f} | {np.median(a):.0f} | {np.mean(a < 50):.0%} | {c} | {rt} | {fb} |")
    (HERE / "out" / "lumber_gemma.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
