"""LumberChunker 코사인 판 '둘로 나눠 항상 자르기'(tests 전용, 2026-10-21 사용자 확정: 작은 청크 제한 없음).

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/chunking/lumber_split.py      # 예시(문서 몇 편) 보기

  1. 문단을 θ(700) 토큰까지 모은다: s부터 토큰 합이 θ를 넘게 만드는 문단 m까지 그룹 [s, m]
     (문단 s 혼자 θ 초과 → 혼자 청크 / 남은 글이 θ 안 → 마지막 청크 / 문단 m 혼자 θ 초과 → [s, m) 청크, s = m)
  2. 그룹을 자리 j = s+1 … m 마다 둘로 나눠 본다: 앞 덩어리 [s, j) vs 뒤 덩어리 [j, m]
     덩어리 벡터 = 문단 bge-m3 벡터(문단마다 따로 인코딩)의 토큰 가중 평균, 앞뒤 유사도 = 코사인
  3. 점수가 가장 큰 자리에서 항상 자른다: 청크 [s, j), s = j (j = m이면 'θ 안 전체'와 같다)
     점수(j) = (1 − 앞뒤 코사인) × [(앞 토큰 × 뒤 토큰) / (앞 토큰 + 뒤 토큰)]^α,  α = 0.5
     = '앞뒤가 다른 정도 × 양쪽이 고르게 나뉜 정도'(Ward 군집 기준과 같은 꼴: 단위 벡터 평균 차의 제곱 = 2(1 − cos)).
     앞뒤 유사도만 쓰면 한쪽이 문단 1개인 나누기(양 끝)가 거의 늘 뽑혀서(2026-10-21 예시) 크기 균형을 곱했다
출력(예시): tests/chunking/out/lumber_split_demo.md
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

THETA = 700
ALPHA = 0.5     # 크기 균형 세기(2026-10-21 tests/chunking/split_variants.py 비교로 고정: 0은 양 끝 쏠림, 1은 앞쪽 경계를 놓침)


def lumber_split(toks, E, theta=THETA, alpha=ALPHA):
    """문단 토큰 수·벡터 → 청크 범위 [(i, j)]와 그룹마다 (그룹 크기, 자른 위치 순번)."""
    n = len(toks)
    T = np.asarray(toks, float)
    tok = np.concatenate([[0], np.cumsum(T)])
    S = np.concatenate([np.zeros((1, E.shape[1])), np.cumsum(E.astype(np.float64) * T[:, None], axis=0)])

    def mean(a, b):
        v = (S[b] - S[a]) / (tok[b] - tok[a])
        return v / (np.linalg.norm(v) + 1e-12)

    out, log, s = [], [], 0
    while s < n:
        if T[s] > theta:
            out.append((s, s + 1)); s += 1
            continue
        m = int(np.searchsorted(tok, tok[s] + theta, side="right")) - 1
        if m >= n:
            out.append((s, n)); break
        if T[m] > theta:
            out.append((s, m)); s = m
            continue
        js = range(s + 1, m + 1)
        sims = [float(mean(s, j) @ mean(j, m + 1)) for j in js]
        bal = [(tok[j] - tok[s]) * (tok[m + 1] - tok[j]) / (tok[m + 1] - tok[s]) for j in js]
        score = [(1 - c) * b ** alpha for c, b in zip(sims, bal)]
        k = int(np.argmax(score))
        cut = s + 1 + k
        log.append((m - s, k, sims[k]))                   # 그룹의 자리 수, 고른 자리 순번(0 = 첫 자리), 그 자리 앞뒤 유사도
        out.append((s, cut)); s = cut
    return out, log


def main():
    from v4_v5_lumber import safe_encoder
    from chunking.sentences import paragraphs
    from common.models import make_token_counter
    from preprocess import longbench
    count, encode = make_token_counter(), safe_encoder()
    rnd = random.Random(1)
    vfiles = [f for f in sorted((ROOT / "data" / "clean_text" / "vectara").glob("*.json")) if not f.name.endswith(".dropped.json")]
    docs = [("vectara", f.stem, json.loads(f.read_text(encoding="utf-8"))["text"].split("\n\n")) for f in rnd.sample(vfiles, 3)]
    docs += [("narrativeqa", b["doc_id"], paragraphs(b["text"])) for b in rnd.sample(longbench.documents("narrativeqa"), 2)]
    L = ["# '둘로 나눠 항상 자르기' 예시 (① 크기 균형 반영)", "", f"θ = {THETA}, 작은 청크 제한 없음. 점수 = (1 − 앞뒤 코사인) × 크기 균형. 자른 위치: 그룹 안 자리 순번(첫 자리 = 앞 덩어리가 문단 1개, 마지막 자리 = θ 안 전체와 같음).", ""]
    for ds, d, units in docs:
        T = count(units)
        E = np.asarray(encode(units), dtype=np.float32)
        rg, log = lumber_split(T, E)
        assert rg[0][0] == 0 and rg[-1][1] == len(units) and all(a[1] == b[0] for a, b in zip(rg, rg[1:]))
        ct = np.array([sum(T[i:j]) for i, j in rg])
        one = sum(g == 1 for g, _, _ in log)
        first = sum(k == 0 and g > 1 for g, k, _ in log); last = sum(k == g - 1 and g > 1 for g, k, _ in log)
        L += [f"## {ds} {d} (문단 {len(units)})", "",
              f"- 청크 {len(rg)}개, 평균 {ct.mean():.0f}토큰, 중앙값 {np.median(ct):.0f}, 최대 {ct.max()}, 50토큰 미만 {np.mean(ct < 50):.0%}",
              f"- 그룹 {len(log)}개(그룹당 자리 평균 {np.mean([g for g, _, _ in log]):.1f}개, 자리가 1개뿐인 그룹 {one}개): "
              f"첫 자리에서 자름 {first}개({first / max(1, len(log)):.0%}), 마지막 자리(= θ 안 전체) {last}개({last / max(1, len(log)):.0%}), "
              f"그 사이 {len(log) - first - last - one}개({(len(log) - first - last - one) / max(1, len(log)):.0%})",
              f"- 고른 자리의 앞뒤 유사도: 중앙값 {np.median([v for *_, v in log]):.2f}", "", "잘린 자리 예시(앞 청크 끝 ‖ 다음 청크 시작, 무작위 6곳):", ""]
        cuts = [(i, j) for (i, j) in rg[:-1]]
        for i, j in rnd.sample(cuts, min(6, len(cuts))):
            a = units[j - 1][-110:].replace("\n", "⏎"); b = units[j][:110].replace("\n", "⏎")
            L.append(f"- [{sum(T[i:j])}토큰, 문단 {j - i}개] …{a!r}\n    ‖ {b!r}")
        L.append("")
    (HERE / "out" / "lumber_split_demo.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
