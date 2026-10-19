"""advanced_semantic 개선 후보(tests 전용 시험 구현). 단위는 chunking.sentences.split_structured와 같다.

v2 계열(문서 전체 동적 계획법, src/chunking/advanced_semantic.segment의 일반화)
  점수 = Σ_청크 흩어짐 + β·(청크 수) [+ β·size_lambda·((청크 토큰 − target)/target)²]
  흩어짐: cost='sum'  → n·(1 − ‖m‖²)   (문장 수를 곱함, 지금 src)
          cost='mean' → 1 − ‖m‖²       (곱하지 않음)
  제약: min_tokens ≤ 청크 토큰 ≤ θ(θ보다 큰 단위 하나는 혼자 허용)
lumber 계열(사용자 제안, LumberChunker 틀)
  s부터 문장을 더해 토큰 합이 θ를 넘을 때까지 그룹 G = [s, e)를 만든다(넘게 만든 단위 포함)
  첫 단위가 θ보다 크면(큰 표 등) 그 단위 혼자 청크, 다음으로 건너뜀. 남은 글이 θ 안이면 그대로 마지막 청크.
  그 밖에는 G를 두 덩어리 [s, k) + [k, e)로 나눴을 때 흩어짐 합(n·(1 − ‖m‖²))이 가장 작은 k에서 자르고
  ([s, k)는 min_tokens 이상), 청크 = [s, k), s = k로 다시 θ까지 채워 반복한다.
이웃 문맥 임베딩(window_emb): 판단용 벡터를 '앞 문장 + 문장 + 뒤 문장'으로 만든다(LangChain SemanticChunker buffer_size=1과 같은 발상).
크기 맞추기: v2는 β, lumber는 θ를 평균 청크 토큰이 target에 가장 가깝도록 문서마다 이분 탐색한다.
"""
import numpy as np

BIG = 1e6


def window_emb(units, encode, w=1):
    texts = [" ".join(units[max(0, i - w):i + w + 1]) for i in range(len(units))]
    return np.asarray(encode(texts), dtype=np.float32)


def _prefix(tokens, emb):
    tok = np.concatenate([[0], np.cumsum(tokens)]).astype(np.float64)
    S = np.concatenate([np.zeros((1, emb.shape[1])), np.cumsum(emb.astype(np.float64), axis=0)])
    return tok, S


def _cost(S, i, j, mode="sum"):
    cnt = j - i
    v = S[j] - S[i]
    sq = np.einsum("...d,...d->...", v, v)
    return cnt - sq / cnt if mode == "sum" else 1 - sq / cnt ** 2


def dp(tokens, emb, beta, theta, min_tokens=0, cost="sum", size_lambda=0.0, target=200):
    n = len(tokens)
    tok, S = _prefix(tokens, emb)
    best = np.full(n + 1, np.inf)
    best[0] = 0.0
    frm = np.zeros(n + 1, dtype=np.int64)
    lo = 0
    for j in range(1, n + 1):
        while tok[j] - tok[lo] > theta and lo < j - 1:
            lo += 1
        i = np.arange(lo, j)
        size = tok[j] - tok[i]
        c = _cost(S, i, j, cost)
        pen = np.where(size < min_tokens, BIG, 0.0) + beta * size_lambda * ((size - target) / target) ** 2
        tot = best[i] + c + beta + pen
        k = int(np.argmin(tot))
        best[j], frm[j] = tot[k], i[k]
    out, j = [], n
    while j > 0:
        out.append((int(frm[j]), j))
        j = int(frm[j])
    return out[::-1]


def lumber(tokens, emb, theta, min_tokens=0):
    n = len(tokens)
    tok, S = _prefix(tokens, emb)
    out, s = [], 0
    while s < n:
        if tokens[s] > theta:                                   # 큰 표 등: 혼자 청크, 건너뜀
            out.append((s, s + 1)); s += 1
            continue
        m = int(np.searchsorted(tok, tok[s] + theta, side="right")) - 1   # [s, m)이 θ 안에 드는 가장 큰 m
        if m >= n:                                              # 남은 글이 θ 안
            out.append((s, n)); break
        if tokens[m] > theta:                                   # 그룹을 채우다 θ보다 큰 단위(표)를 만남: 앞까지 한 청크, 표는 다음 차례에 혼자
            out.append((s, m)); s = m
            continue
        g = m + 1                                               # 그룹 G = [s, g): θ를 넘게 만든 단위 m 포함
        k = np.arange(s + 1, g)                                 # 자를 자리(k ≤ m이라 [s, k)는 θ 안)
        ok = tok[k] - tok[s] >= min_tokens
        if not ok.any():
            cut = m
        else:
            tot = _cost(S, s, k) + _cost(S, k, g)
            tot = np.where(ok, tot, np.inf)
            cut = int(k[int(np.argmin(tot))])
        out.append((s, cut)); s = cut
    return out


def search(run, tokens, target, lo, hi, iters=16, geometric=True):
    """run(p) → 범위. 평균 청크 토큰이 target에 가장 가까운 p(p가 크면 청크가 커진다)."""
    total, best = sum(tokens), None
    for _ in range(iters):
        p = (lo * hi) ** 0.5 if geometric else (lo + hi) / 2
        r = run(p)
        avg = total / len(r)
        if best is None or abs(avg - target) < abs(best[1] - target):
            best = (r, avg, p)
        lo, hi = (p, hi) if avg < target else (lo, p)
    return best


def lumber_dp(tokens, emb, beta, theta, min_tokens=0, window=None, para_start=None, bonus=0.0, heading=None):
    """lumber + v2: s부터 window 토큰(기본 2θ)까지 그룹을 모으고, 그룹 안을 v2 동적 계획법(src advanced_semantic.segment,
    문단 보너스·제목 규칙 포함)으로 최적 분할해 첫 청크만 확정한 뒤, 남은 뒤부터 다시 그룹을 채워 반복한다."""
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parents[2] / "src"))
    from chunking.advanced_semantic import segment
    n = len(tokens)
    window = window or 2 * theta
    tokens = np.asarray(tokens)
    tok = np.concatenate([[0], np.cumsum(tokens)])
    ps = None if para_start is None else np.asarray(para_start)
    hd = None if heading is None else np.asarray(heading)
    out, s = [], 0
    while s < n:
        e = max(s + 1, min(n, int(np.searchsorted(tok, tok[s] + window, side="right")) - 1))
        r = segment(tokens[s:e], emb[s:e], beta, theta, min_tokens,
                    None if ps is None else ps[s:e], bonus, None if hd is None else hd[s:e])
        if e >= n:
            out += [(s + i, s + j) for i, j in r]
            break
        out.append((s, s + r[0][1]))
        s += r[0][1]
    return out


def lumber_min_sim(tokens, emb, theta):
    """문단 단위 LumberChunker 코사인 판(사용자 제안, 2026-10-19).
    s부터 단위를 모아 토큰 합이 θ를 넘을 때까지 그룹(넘게 만든 단위 m 포함) → 그룹 안에서 이웃한 두 단위의 코사인이 가장 낮은 자리 i
    (s+1 ≤ i ≤ m, 앞 청크 [s, i)는 θ 안)에서 자르고, i부터 다시 채워 반복한다. 남은 글이 θ 안이면 마지막 청크.
    θ보다 큰 단위(표 등): 그룹 시작이면 혼자 청크, 채우다 만나면 그 앞까지 청크로 끝내고 다음 차례에 혼자 청크."""
    n = len(tokens)
    tok = np.concatenate([[0], np.cumsum(tokens)])
    out, s = [], 0
    while s < n:
        if tokens[s] > theta:
            out.append((s, s + 1)); s += 1
            continue
        m = int(np.searchsorted(tok, tok[s] + theta, side="right")) - 1     # [s, m)이 θ 안
        if m >= n:
            out.append((s, n)); break
        if tokens[m] > theta:
            out.append((s, m)); s = m
            continue
        i = np.arange(s + 1, m + 1)
        sims = np.einsum("ij,ij->i", emb[i - 1], emb[i])
        cut = int(i[int(np.argmin(sims))])
        out.append((s, cut)); s = cut
    return out
