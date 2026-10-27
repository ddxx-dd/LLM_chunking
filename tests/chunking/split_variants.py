"""'둘로 나눠 항상 자르기'의 크기 균형 세기 비교(tests 전용, 예시 문서 몇 편으로 보는 확인).

  source env.sh && PYTHONPATH=src .venv311/bin/python tests/chunking/split_variants.py

판(그룹 = 문단을 θ 700까지, 자리마다 앞 [s, j) vs 뒤 [j, m] 토큰 가중 평균 벡터)
  α=0     점수 = 다른 정도(1 − 앞뒤 코사인)
  α=0.5   점수 = 다른 정도 × 고른 정도^0.5     (고른 정도 = 앞 토큰 × 뒤 토큰 / 그룹 토큰)
  α=1     점수 = 다른 정도 × 고른 정도          (지금 판)
  2문단↑  앞·뒤 덩어리가 각각 문단 2개 이상인 자리 중 다른 정도가 가장 큰 곳(그런 자리가 없으면 모든 자리)
보는 것: ① 판마다 실제로 청킹한 결과(자른 위치: 첫 자리 / 가운데 / 마지막, 50토큰 미만, 고른 자리 코사인)
         ② 같은 그룹(문단을 700토큰씩 차례로 채운 그룹)에 네 판을 적용해 고른 자리가 다른 그룹의 문장
표본: tests/chunking/lumber_split.py 예시와 같음(seed 1: vectara 3편, NarrativeQA 2권)
출력: tests/chunking/out/split_variants.md
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
VARS = ["α=0", "α=0.5", "α=1", "2문단↑"]


class Group:
    def __init__(self, T, E):
        self.T = np.asarray(T, float)
        self.tok = np.concatenate([[0], np.cumsum(self.T)])
        self.S = np.concatenate([np.zeros((1, E.shape[1])), np.cumsum(E.astype(np.float64) * self.T[:, None], axis=0)])

    def mean(self, a, b):
        v = (self.S[b] - self.S[a]) / (self.tok[b] - self.tok[a])
        return v / (np.linalg.norm(v) + 1e-12)

    def choose(self, var, s, m):
        """그룹 [s, m]에서 자를 자리 j(s+1 … m)와 그 자리 코사인."""
        js = list(range(s + 1, m + 1))
        cos = [float(self.mean(s, j) @ self.mean(j, m + 1)) for j in js]
        diff = [1 - c for c in cos]
        tot = self.tok[m + 1] - self.tok[s]
        bal = [(self.tok[j] - self.tok[s]) * (self.tok[m + 1] - self.tok[j]) / tot for j in js]
        if var == "2문단↑":
            ok = [i for i, j in enumerate(js) if j - s >= 2 and m + 1 - j >= 2] or list(range(len(js)))
            k = max(ok, key=lambda i: diff[i])
        else:
            a = float(var.split("=")[1])
            k = int(np.argmax([d * b ** a for d, b in zip(diff, bal)]))
        return js[k], cos[k], k, len(js)


def groups_of(T, theta=THETA, start=0, stop=None):
    """θ 그룹 경계 찾기: s에서 시작하는 그룹의 (종류, m)."""
    n = len(T)
    tok = np.concatenate([[0], np.cumsum(T)])
    s = start
    if T[s] > theta:
        return "big", s
    m = int(np.searchsorted(tok, tok[s] + theta, side="right")) - 1
    if m >= n:
        return "end", n
    if T[m] > theta:
        return "bigm", m
    return "group", m


def run(var, G, T):
    out, log, s, n = [], [], 0, len(T)
    while s < n:
        kind, m = groups_of(T, start=s)
        if kind == "big":
            out.append((s, s + 1)); s += 1; continue
        if kind == "end":
            out.append((s, n)); break
        if kind == "bigm":
            out.append((s, m)); s = m; continue
        j, c, k, g = G.choose(var, s, m)
        out.append((s, j)); log.append((k, g, c)); s = j
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
    L = ["# 크기 균형 세기 비교 (예시 문서)", "", "점수 = 다른 정도 × 고른 정도^α. '2문단↑' = 앞·뒤 각 2문단 이상 자리 중 다른 정도 최대. θ = 700.", "",
         "## ① 판마다 실제로 청킹한 결과", "", "| 문서 | 판 | 청크 | 평균 토큰 | 50토큰 미만 | 첫 자리 / 가운데 / 마지막 | 고른 자리 코사인(중앙값) |", "|---|---|---|---|---|---|---|"]
    ex = []
    rnd2 = random.Random(5)
    for ds, d, units in docs:
        T = count(units)
        G = Group(T, np.asarray(encode(units), dtype=np.float32))
        for var in VARS:
            rg, log = run(var, G, T)
            ct = np.array([sum(T[i:j]) for i, j in rg])
            multi = [(k, g, c) for k, g, c in log if g > 1]
            f = sum(k == 0 for k, g, c in multi); l = sum(k == g - 1 for k, g, c in multi); mid = len(multi) - f - l
            L.append(f"| {ds} {d[:14]} | {var} | {len(rg)} | {ct.mean():.0f} | {np.mean(ct < 50):.0%} | "
                     f"{f / len(multi):.0%} / {mid / len(multi):.0%} / {l / len(multi):.0%} | {np.median([c for *_, c in log]):.2f} |")
        # ② 같은 그룹(채우기만 그룹)에서 고른 자리 비교
        s = 0
        while s < len(units):
            kind, m = groups_of(T, start=s)
            if kind != "group":
                s = s + 1 if kind == "big" else (m if kind == "bigm" else len(units)); continue
            picks = {v: G.choose(v, s, m) for v in VARS}
            if len({p[0] for p in picks.values()}) > 1:
                ex.append((ds, d, s, m, picks, units))
            s = m
    L += ["", "## ② 같은 그룹에서 판마다 고른 자리가 다른 경우(무작위 10개)", "",
          "자리 표시: '앞 문단 끝 ‖ 뒤 문단 시작'(앞 덩어리 문단 수 · 그 자리 앞뒤 코사인).", ""]
    for ds, d, s, m, picks, units in rnd2.sample(ex, min(10, len(ex))):
        L.append(f"### [{ds} {d[:16]}] 그룹 문단 {s}~{m} ({m - s + 1}개)")
        by = {}
        for v, (j, c, k, g) in picks.items():
            by.setdefault(j, []).append((v, c))
        for j, vs in sorted(by.items()):
            a = units[j - 1][-80:].replace("\n", "⏎"); b = units[j][:80].replace("\n", "⏎")
            L.append(f"- {', '.join(v for v, _ in vs)} → 앞 {j - s}문단 · 코사인 {vs[0][1]:.2f}: …{a!r} ‖ {b!r}")
        L.append("")
    (HERE / "out" / "split_variants.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
