"""advanced_semantic_chunker: 문서 전체 최적 분할(Embed-KCPD 코사인 비용) + 문단 경계 보너스 + 크기 제약 + 제목 규칙.

2026-10-17 파일럿(tests/chunking/out/compare_v2.md)에서 앞에서부터 판단하던 옛 판(LumberChunker 틀 + Max–Min / 중심 벡터)보다
HotpotQA 패시지 경계·문서 안 검색이 나아서 이것으로 바꿨다(옛 판은 git 기록에 있다).

단위: chunking.sentences.split_structured(문장, 표·수식은 한 단위) + 단위마다 앞 구분('\\n\\n' 문단 시작 / '\\n' 새 줄 / ' ')과 제목 여부
점수(작을수록 좋음, 동적 계획법으로 문서 전체에서 최소인 자르기를 찾는다)
  Σ_청크 흩어짐(청크) + β·(청크 수) − b·(문단 시작 자리에서 자른 수)
  흩어짐([i, j)) = n·(1 − ‖평균 벡터‖²) = n − ‖Σe‖²/n   (n = 단위 수, e = 길이 1 임베딩, = 청크 안 Σ‖e − m‖²)
      Embed-KCPD(arXiv 2601.18788)의 구간 비용(코사인 커널). 청크 안 모든 단위 쌍의 코사인이 높을수록 작다.
  β: 자르기 한 번의 가격. 잘라서 줄어드는 흩어짐이 β보다 클 때만 자른다.
  b = bonus_frac·β: 빈 줄 문단 경계에서 자르면 깎아 주는 값(0이면 순수 의미 분할, 절제 실험용)
제약(어기면 큰 벌점 BIG): 청크 토큰 ≤ θ(θ보다 큰 단위 하나는 혼자 허용), 청크 토큰 ≥ min_tokens(가능한 한), heading_rule이면 제목 줄 바로 뒤에서 자르지 않음
β 정하기: beta를 주면 그 값, 주지 않으면 평균 청크 토큰이 target에 가장 가까운 β를 문서마다 이분 탐색(라벨 없이 크기만 맞춘다)
청크 글: 단위를 원래 앞 구분으로 다시 잇는다(문단·제목 줄 형식 유지). 크기 θ·target·min_tokens는 bge-m3 토큰(count_tokens).
"""
import numpy as np

from chunking.sentences import split_structured

BIG = 1e6


def segment(tokens, emb, beta, theta, min_tokens=0, para_start=None, bonus=0.0, heading=None):
    """단위별 토큰 수·임베딩 → 청크 범위 [(i, j)] (j 미포함)."""
    n = len(tokens)
    tok = np.concatenate([[0], np.cumsum(tokens)]).astype(np.float64)
    S = np.concatenate([np.zeros((1, emb.shape[1])), np.cumsum(emb.astype(np.float64), axis=0)])
    best = np.full(n + 1, np.inf)
    best[0] = 0.0
    frm = np.zeros(n + 1, dtype=np.int64)
    ps = np.zeros(n + 1) if para_start is None else np.concatenate([np.asarray(para_start, float), [1.0]])
    hd = np.zeros(n + 1, bool) if heading is None else np.concatenate([[False], np.asarray(heading, bool)])  # hd[i]: 단위 i-1이 제목
    lo = 0
    for j in range(1, n + 1):
        while tok[j] - tok[lo] > theta and lo < j - 1:   # θ 안에 드는 시작점만(단위 하나는 θ보다 커도 허용)
            lo += 1
        i = np.arange(lo, j)
        cnt = j - i
        v = S[j] - S[i]
        cost = cnt - np.einsum("ij,ij->i", v, v) / cnt
        pen = np.where(tok[j] - tok[i] < min_tokens, BIG, 0.0) + np.where(hd[i] & (i > 0), BIG, 0.0)
        tot = best[i] + cost + beta - bonus * ps[i] * (i > 0) + pen
        k = int(np.argmin(tot))
        best[j], frm[j] = tot[k], i[k]
    out, j = [], n
    while j > 0:
        out.append((int(frm[j]), j))
        j = int(frm[j])
    return out[::-1]


def segment_target(tokens, emb, theta, target, min_tokens=0, para_start=None, bonus_frac=0.0, heading=None, iters=18):
    """평균 청크 토큰이 target에 가장 가까운 β를 이분 탐색(β가 크면 청크가 커진다). → (범위, 평균 토큰, β)."""
    lo, hi, best = 1e-4, 50.0, None
    for _ in range(iters):
        beta = (lo * hi) ** 0.5
        r = segment(tokens, emb, beta, theta, min_tokens, para_start, bonus_frac * beta, heading)
        avg = sum(tokens) / len(r)
        if best is None or abs(avg - target) < abs(best[1] - target):
            best = (r, avg, beta)
        if avg < target:
            lo = beta
        else:
            hi = beta
    return best


def chunk(text, encode, count_tokens, theta, target=None, beta=None, lang="en", min_tokens=0, bonus_frac=0.5,
          heading_rule=True, return_ranges=False):
    """텍스트 → 청크 문자열 목록(return_ranges면 (청크 목록, 단위 목록, 범위 목록)).
    encode: 글 목록 → 길이 1 벡터, count_tokens: 글 목록 → 토큰 수 목록, theta: 청크 토큰 상한, target 또는 beta 중 하나."""
    if (target is None) == (beta is None):
        raise ValueError("target(평균 청크 토큰)과 beta 중 하나만 준다")
    items = split_structured(text, lang)
    if not items:
        return ([], [], []) if return_ranges else []
    units = [u for u, _, _ in items]
    seps = [s for _, s, _ in items]
    para_start = [s == "\n\n" for s in seps]
    heading = [h for _, _, h in items] if heading_rule else None
    tokens = count_tokens(units)
    emb = np.asarray(encode(units), dtype=np.float32)
    if beta is None:
        ranges = segment_target(tokens, emb, theta, target, min_tokens, para_start, bonus_frac, heading)[0]
    else:
        ranges = segment(tokens, emb, beta, theta, min_tokens, para_start, bonus_frac * beta, heading)
    chunks = [units[i] + "".join(seps[k] + units[k] for k in range(i + 1, j)) for i, j in ranges]
    return (chunks, units, ranges) if return_ranges else chunks
