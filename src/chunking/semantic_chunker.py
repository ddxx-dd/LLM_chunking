# 임베딩 유사도 기반 의미 분할 베이스라인

import re
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core import Chunk


SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# ── 1. 문장 분리 = 경계 후보 생성 ─────────────────────────────
# (문장, 시작, 끝) 목록을 반환한다.
def split_sentences(text, max_length=200):
    
    pieces, pos = [], 0
    for m in SPLIT_RE.finditer(text):
        pieces.append((text[pos:m.start()], pos))    # (조각, 원문에서의 시작위치)
        pos = m.end()
    if pos < len(text):
        pieces.append((text[pos:], pos))

    out = []
    for raw, off in pieces:
        stripped = raw.strip()
        if not stripped:
            continue
        # strip() 으로 앞쪽이 잘린 만큼 시작 위치를 보정해야 오프셋이 맞는다
        start = off + (len(raw) - len(raw.lstrip()))

        if len(stripped) <= max_length:
            out.append((stripped, start, start + len(stripped)))
            continue

        # 너무 긴 조각 -> 단어 단위로 쪼개되 위치를 계속 추적
        cur, cur_start = "", start
        for word in stripped.split(" "):
            if len(cur) + len(word) + 1 > max_length and cur:
                out.append((cur, cur_start, cur_start + len(cur)))
                cur_start = cur_start + len(cur) + 1      # +1 = 공백 한 칸
                cur = word
            else:
                cur = cur + " " + word if cur else word
        if cur:
            out.append((cur, cur_start, cur_start + len(cur)))
    return out

# ── 2. 인접 유사도  ────────────────────────────────
def calculate_similarities(vectors):
#similarities[i] = 문장 i 와 문장 i+1 사이의 코사인 유사도.


    V = np.asarray(vectors, dtype=float)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return [float(x) for x in (V[:-1] * V[1:]).sum(axis=1)]


# ── 3. 임계값  ────────────────────────────────────
def calculate_threshold(similarities, method="percentile", amount=5):
    if not similarities:
        return 0.0
    if method == "fixed":
        return float(amount)
    if method == "percentile":
        return float(np.percentile(similarities, amount))
    if method == "std":
        return float(np.mean(similarities)) - amount * float(np.std(similarities))
    raise ValueError("알 수 없는 method: " + method)
    

# ── 4. 경계에서 분할 ──────────────────────────
def split_at_boundaries(sents, similarities, threshold, text):
    chunks = []
    start = 0                                  
    last_cut = 0

    for i, sim in enumerate(similarities):
        if sim < threshold:
            end = sents[i + 1][1]
            chunks.append(Chunk(text[start:end], start, end))
            start = end
            last_cut = end

    chunks.append(Chunk(text[start:], start, len(text)))   
    return chunks

def semantic_chunking(text, model, method="percentile", amount=5,
                      max_sentence_length=200, return_debug=False):
#문장 분리 -> 임베딩 -> 유사도 -> 임계값 -> 경계에서 분할.

    sents = split_sentences(text, max_sentence_length)
    if len(sents) < 2:
        chunks = [Chunk(text, 0, len(text))] if text else []
        return (chunks, [], 0.0, sents) if return_debug else chunks

    vectors = model.encode([s for s, _, _ in sents])
    similarities = calculate_similarities(vectors)
    threshold = calculate_threshold(similarities, method, amount)
    chunks = split_at_boundaries(sents, similarities, threshold, text, min_gap)


    return (chunks, similarities, threshold, sents) if return_debug else chunks


def semantic_chunking_k(text, model, k, max_sentence_length=200, min_gap=0):
#유사도가 가장 낮은 k-1개 지점에서 자른다.

    sents = split_sentences(text, max_sentence_length)
    if len(sents) < 2:
        return [Chunk(text, 0, len(text))] if text else []

    sims = calculate_similarities(model.encode([s for s, _, _ in sents]))
    order = np.argsort(sims)                  # 유사도 오름차순 = 자를 후보 순
    cut, last = set(), 0
    for i in order:
        if len(cut) >= k - 1:
            break
        cut.add(int(i))
        last = sents[i][2]

    chunks, start = [], 0
    for i in sorted(cut):                     # 원문 순서대로 정렬해서 자른다
        end = sents[i + 1][1]
        chunks.append(Chunk(text[start:end], start, end))
        start = end
    chunks.append(Chunk(text[start:], start, len(text)))
    return chunks

