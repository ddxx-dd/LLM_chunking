import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core import Chunk

def fixed_chunking(text, chunk_size=256, overlap=0):
    if not text or not text.strip():
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap은 chunk_size보다 작아야 함")

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end]
        if piece.strip():
            chunks.append(Chunk(piece, start, end))
        start += step

    return chunks

def fixed_chunking_k(text, k):
    size = max(1, len(text) // max(1, k))
    return fixed_chunking(text, chunk_size=size, overlap=0)

def _token_len(tokenizer, s):
    return len(tokenizer.encode(s, add_special_tokens=False))

def fixed_chunking_tokens(text, tokenizer, max_tokens=256, overlap_tokens=0):
    """글자 수가 아니라 실제 토크나이저(BPE 등) 토큰 수 기준으로 엄격하게 분할.

    주의: byte-level BPE(Qwen 등)는 한글처럼 멀티바이트 문자에서 offset_mapping이
    토큰끼리 서로 겹치는 구간을 보고하는 경우가 있어(예: 한 글자를 여러 바이트-토큰이
    나눠 가리킴), offset_mapping만으로 자르면 원문이 손실될 수 있다.
    그래서 글자 위치 기준 이진 탐색으로 "이 구간을 다시 인코딩했을 때 몇 토큰인가"를
    직접 재확인하며 자른다 -> 원문 무손실(lossless) 재구성이 항상 보장된다."""
    if not text or not text.strip():
        return []
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens은 max_tokens보다 작아야 함")

    n = len(text)
    chunks = []
    start = 0
    while start < n:
        lo, hi = start + 1, n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _token_len(tokenizer, text[start:mid]) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        end = lo
        piece = text[start:end]
        if piece.strip():
            chunks.append(Chunk(piece, start, end))
        elif chunks:
            # 공백만 있는 조각은 새 청크로 만들지 않고 직전 청크에 이어붙여
            # 무손실(lossless) 재구성을 유지한다.
            prev = chunks[-1]
            chunks[-1] = Chunk(prev.text + piece, prev.start, end)

        if overlap_tokens > 0 and end < n:
            lo2, hi2 = start, end
            while lo2 < hi2:
                mid2 = (lo2 + hi2) // 2
                if _token_len(tokenizer, text[mid2:end]) <= overlap_tokens:
                    hi2 = mid2
                else:
                    lo2 = mid2 + 1
            start = lo2 if lo2 > start else end
        else:
            start = end
    return chunks