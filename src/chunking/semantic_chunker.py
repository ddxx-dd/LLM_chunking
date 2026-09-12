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

def _token_len(s, tokenizer):
    """청크 크기를 토큰 수로 잰다 - 글자 수는 언어마다 토큰 밀도가 달라서
    (영어 ~4자/토큰, 한국어 ~1.5~2.5자/토큰) EN/KO 양방향을 다루는 semantic
    청킹엔 안 맞는다. fixed_chunking은 의도적으로 글자 수 그대로 둔다(그게
    '고정 글자수 분할'의 정의이자, 이 방식의 한계를 있는 그대로 드러내는 지점)."""
    return len(tokenizer.encode(s, add_special_tokens=False))

def split_at_boundaries(sents, similarities, threshold, text, tokenizer=None,
                         max_chunk_tokens=None, min_chunk_tokens=None):
    """문장 사이 유사도가 임계값 아래로 떨어지는 지점에서 자른다.

    - max_chunk_tokens: 다음 문장을 포함시키면 상한을 넘을 경우, 넣기 "전"에 끊는다
      (문장 하나 자체가 상한보다 긴 극단적인 경우만 예외 - 단어 중간을 자르지 않기 위함).
    - min_chunk_tokens: 유사도가 낮아 자르고 싶어도, 지금까지 모은 게 이 최소치보다
      작으면 자르지 않고 다음 문장까지 계속 누적한다(짧은 감탄사 한두 마디짜리
      파편 청크가 생기는 걸 막음 - 문헌에서 "너무 짧은 청크는 답변 생성에 필요한
      맥락이 부족하다"고 지적하는 문제).

    둘 다 문장 단위로만 자르므로, 어느 쪽으로 끊기든 항상 문장 경계에서 끊긴다."""
    chunks = []
    start = 0
    seg_start_idx = 0  # 지금 열려 있는 청크에 포함된 첫 문장의 인덱스
    n = len(sents)

    for i in range(n):
        # seg_end는 실제로 자를 때 쓰는 경계(다음 문장 시작 지점, 또는 마지막
        # 문장이면 텍스트 끝)와 정확히 같은 기준이어야 한다 - 문장 자체의 끝
        # (sents[i][2])으로 검사하면 문장 사이 구분자만큼 길이가 어긋난다.
        seg_end = sents[i + 1][1] if i + 1 < n else len(text)
        # (i > seg_start_idx: 이 청크의 "첫" 문장은 그 자체가 상한보다 길어도
        #  일단 받아들인다 - 안 그러면 청크를 하나도 못 만드는 경우가 생긴다)
        if max_chunk_tokens and i > seg_start_idx and _token_len(text[start:seg_end], tokenizer) > max_chunk_tokens:
            cut_at = sents[i][1]  # 문장 i가 시작하는 자리 = 앞 청크의 끝
            chunk_text = text[start:cut_at]
            if chunk_text.strip():
                chunks.append(Chunk(chunk_text, start, cut_at))
            start = cut_at
            seg_start_idx = i

        if i < n - 1 and similarities[i] < threshold:
            cut_at = sents[i + 1][1]
            # 여기서 자르면 청크가 min_chunk_tokens보다 작아지는가? 그러면 이번
            # 경계는 무시하고 다음 문장까지 계속 모은다(최종적으로는 max_chunk_tokens가
            # 안전판 역할을 해서 무한정 커지지는 않는다).
            if min_chunk_tokens and _token_len(text[start:cut_at], tokenizer) < min_chunk_tokens:
                continue
            chunk_text = text[start:cut_at]
            if chunk_text.strip():
                chunks.append(Chunk(chunk_text, start, cut_at))
            start = cut_at
            seg_start_idx = i + 1

    last_piece = text[start:]
    if last_piece.strip():
        chunks.append(Chunk(last_piece, start, len(text)))
    return chunks

def semantic_chunking(text, model, method="percentile", amount=15, max_sentence_length=200,
                       max_chunk_tokens=None, min_chunk_tokens=None):
    sents = split_sentences(text, max_sentence_length)
    if len(sents) < 2:
        return [Chunk(text, 0, len(text))] if text else []

    vectors = model.encode([s[0] for s in sents], show_progress_bar=False)
    similarities = calculate_similarities(vectors)
    threshold = calculate_threshold(similarities, method, amount)
    return split_at_boundaries(sents, similarities, threshold, text, tokenizer=model.tokenizer,
                                max_chunk_tokens=max_chunk_tokens, min_chunk_tokens=min_chunk_tokens)