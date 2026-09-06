import re
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core import Chunk

SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

def split_sentences(text, max_length=200):
    pieces, pos = [], 0
    for m in SPLIT_RE.finditer(text):
        pieces.append((text[pos:m.start()], pos))
        pos = m.end()
    if pos < len(text):
        pieces.append((text[pos:], pos))

    out = []
    for raw, off in pieces:
        stripped = raw.strip()
        if not stripped:
            continue
        start = off + (len(raw) - len(raw.lstrip()))
        end = start + len(stripped)
        if len(stripped) <= max_length:
            out.append((stripped, start, end))
        else:
            cur, cur_start = "", start
            for word in stripped.split(" "):
                if len(cur) + len(word) + 1 > max_length and cur:
                    out.append((cur, cur_start, cur_start + len(cur)))
                    cur_start = cur_start + len(cur) + 1
                    cur = word
                else:
                    cur = cur + " " + word if cur else word
            if cur:
                out.append((cur, cur_start, cur_start + len(cur)))
    return out

def calculate_similarities(vectors):
    V = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    V_norm = V / norms
    return [float(x) for x in (V_norm[:-1] * V_norm[1:]).sum(axis=1)]

def calculate_threshold(similarities, method="percentile", amount=10):
    if not similarities:
        return 0.0
    if method == "fixed":
        return float(amount)
    if method == "percentile":
        return float(np.percentile(similarities, amount))
    if method == "std":
        return float(np.mean(similarities)) - amount * float(np.std(similarities))
    raise ValueError("알 수 없는 method: " + method)

def split_at_boundaries(sents, similarities, threshold, text, max_chunk_chars=None):
    """문장 사이 유사도가 임계값 아래로 떨어지는 지점에서 자른다. 크기 상한이
    주어지면, 다음 문장을 지금 청크에 포함시켰을 때 상한을 넘게 될 경우 그
    문장을 넣기 "전에" 끊는다 - 그래서 어떤 청크도 max_chunk_chars를 절대
    넘지 않는다(문장 하나 자체가 상한보다 긴 극단적인 경우만 예외 - 그 문장을
    또 쪼개면 단어 중간을 자르게 되므로 어쩔 수 없이 그대로 둔다).

    크기 상한도 이 함수 안에서 같은 한 번의 순회로 처리한다 - 예전에는 의미 기반
    경계로 다 자른 뒤 상한을 넘는 청크만 골라 고정 글자수 분할로 "다시" 쪼갰는데,
    그러면 상한 때문에 잘리는 자리는 결국 단어 중간을 잘라버리는 Fixed 분할의
    한계를 다시 끌어들이는 셈이었다. 지금은 문장 단위로만 누적하다가 자르므로,
    상한 때문에 끊기더라도 항상 문장 경계에서 끊긴다."""
    chunks = []
    start = 0
    seg_start_idx = 0  # 지금 열려 있는 청크에 포함된 첫 문장의 인덱스
    n = len(sents)

    for i in range(n):
        # 문장 i를 지금 청크에 포함시키면 상한을 넘는가? seg_end는 실제로
        # 자를 때 쓰는 경계(다음 문장 시작 지점, 또는 마지막 문장이면 텍스트 끝)와
        # 정확히 같은 기준이어야 한다 - 문장 자체의 끝(sents[i][2])으로 검사하면
        # 문장 사이 구분자(공백/줄바꿈) 한두 글자만큼 실제 청크 길이와 어긋난다.
        seg_end = sents[i + 1][1] if i + 1 < n else len(text)
        # 문장 i를 지금 청크에 포함시키면 상한을 넘는가?
        # (i > seg_start_idx: 이 청크의 "첫" 문장은 그 자체가 상한보다 길어도
        #  일단 받아들인다 - 안 그러면 청크를 하나도 못 만드는 경우가 생긴다)
        if max_chunk_chars and i > seg_start_idx and (seg_end - start) > max_chunk_chars:
            cut_at = sents[i][1]  # 문장 i가 시작하는 자리 = 앞 청크의 끝
            chunk_text = text[start:cut_at]
            if chunk_text.strip():
                chunks.append(Chunk(chunk_text, start, cut_at))
            start = cut_at
            seg_start_idx = i

        if i < n - 1 and similarities[i] < threshold:
            cut_at = sents[i + 1][1]
            chunk_text = text[start:cut_at]
            if chunk_text.strip():
                chunks.append(Chunk(chunk_text, start, cut_at))
            start = cut_at
            seg_start_idx = i + 1

    last_piece = text[start:]
    if last_piece.strip():
        chunks.append(Chunk(last_piece, start, len(text)))
    return chunks

def semantic_chunking(text, model, method="percentile", amount=10, max_sentence_length=200,
                       max_chunk_chars=None, return_debug=False):
    sents = split_sentences(text, max_sentence_length)
    if len(sents) < 2:
        single = [Chunk(text, 0, len(text))] if text else []
        return (single, [], 0.0, sents) if return_debug else single

    vectors = model.encode([s[0] for s in sents], show_progress_bar=False)
    similarities = calculate_similarities(vectors)
    threshold = calculate_threshold(similarities, method, amount)
    chunks = split_at_boundaries(sents, similarities, threshold, text, max_chunk_chars=max_chunk_chars)

    if return_debug:
        return chunks, similarities, threshold, sents
    return chunks