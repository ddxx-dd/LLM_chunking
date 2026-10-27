"""기준값 크기 감각 확인(tests 전용, 짧은 진단): 문단 하나끼리 / 앞 3 vs 뒤 3문단 / 그룹을 둘로 나눈 덩어리끼리 유사도 분포 비교.

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/chunking/split_scale.py

표본: vectara 15편(clean_text v5 문단), NarrativeQA 4권(원문 빈 줄 문단). seed 0. 문단 벡터는 bge-m3 따로 인코딩, 덩어리 = 토큰 가중 평균 벡터.
  ① 이웃 문단:   cos(문단 j−1, 문단 j)                         (문서의 모든 자리)
  ② 앞3 vs 뒤3: cos(앞 3문단 평균, 뒤 3문단 평균)               (문서의 모든 자리, 끝은 있는 만큼)
  ③ 그룹 최솟값: 문단을 θ = 700까지 차례로 모은 그룹마다(채우기만 기준), 자리마다 [s, j) vs [j, m] 평균의 cos 중 가장 낮은 값
  '자를 비율' = ③이 각 문서의 ①·② 하위 20% 기준 이하인 그룹 비율(그 기준으로 실제 몇 그룹을 자르게 될지)
출력: tests/chunking/out/split_scale.md
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from v4_v5_lumber import safe_encoder  # noqa: E402
from chunking.sentences import paragraphs  # noqa: E402
from common.models import make_token_counter  # noqa: E402
from preprocess import longbench  # noqa: E402

THETA, Q = 700, 0.2


def mean_vec(E, T, a, b):
    v = (E[a:b] * T[a:b, None]).sum(0) / T[a:b].sum()
    return v / (np.linalg.norm(v) + 1e-9)


def main():
    count, encode = make_token_counter(), safe_encoder()
    rnd = random.Random(0)
    vfiles = [f for f in sorted((ROOT / "data" / "clean_text" / "vectara").glob("*.json")) if not f.name.endswith(".dropped.json")]
    sets = {"vectara": [json.loads(f.read_text(encoding="utf-8"))["text"].split("\n\n") for f in rnd.sample(vfiles, 15)],
            "narrativeqa": [paragraphs(b["text"]) for b in rnd.sample(longbench.documents("narrativeqa"), 4)]}
    L = ["# 기준값 크기 감각 확인", "", f"θ = {THETA}, 기준 = 각 문서의 하위 {Q:.0%}. 분포는 모든 문서 값을 합쳐서.", ""]
    for ds, docs in sets.items():
        a1, a2, a3, cut1, cut2, ng = [], [], [], 0, 0, 0
        for units in docs:
            T = np.array(count(units), float)
            E = np.asarray(encode(units), dtype=np.float32)
            n = len(units)
            adj = [float(E[j - 1] @ E[j]) for j in range(1, n)]
            blk = [float(mean_vec(E, T, max(0, j - 3), j) @ mean_vec(E, T, j, min(n, j + 3))) for j in range(1, n)]
            thr1, thr2 = np.quantile(adj, Q), np.quantile(blk, Q)
            a1 += adj; a2 += blk
            tok = np.concatenate([[0], np.cumsum(T)])
            s = 0
            while s < n:
                if T[s] > THETA:
                    s += 1; continue
                m = int(np.searchsorted(tok, tok[s] + THETA, side="right")) - 1
                if m >= n:
                    break
                if T[m] > THETA:
                    s = m; continue
                mn = min(float(mean_vec(E, T, s, j) @ mean_vec(E, T, j, m + 1)) for j in range(s + 1, m + 1))
                a3.append(mn); ng += 1; cut1 += mn <= thr1; cut2 += mn <= thr2
                s = m
        q = lambda a: " / ".join(f"{np.quantile(a, p):.2f}" for p in (0.1, 0.2, 0.5, 0.8))
        L += [f"## {ds} (문서 {len(docs)}편, 그룹 {ng}개)", "", "| 분포 | 개수 | 10% / 20% / 50% / 80% 지점 | 평균 |", "|---|---|---|---|",
              f"| ① 이웃 문단끼리 | {len(a1):,} | {q(a1)} | {np.mean(a1):.2f} |",
              f"| ② 앞 3문단 vs 뒤 3문단 | {len(a2):,} | {q(a2)} | {np.mean(a2):.2f} |",
              f"| ③ 그룹을 둘로 나눈 값 중 최솟값 | {len(a3):,} | {q(a3)} | {np.mean(a3):.2f} |", "",
              f"③이 기준 이하라서 '자르게 될' 그룹 비율: ① 기준이면 **{cut1 / ng:.0%}**, ② 기준이면 **{cut2 / ng:.0%}**", ""]
    (HERE / "out" / "split_scale.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
