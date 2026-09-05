# 단순 글자수 분할 베이스라인
# 글자 수 기준으로 기계적으로 자른다.
# 예)  text = "포인터는 변수의 주소를 저장합니다."  (18자), chunk_size=8
#        -> Chunk("포인터는 변수의 ", 0,  8)
#           Chunk("주소를 저장합니", 8, 16)
#           Chunk("다.",            16, 18)
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core import Chunk

def fixed_chunking(text, chunk_size=256, overlap=0):
    if not text.strip():
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
#청크가 k개 가깝게 나오도록 size를 역산해서 자른다.

    size = max(1, len(text) // max(1, k))
    return fixed_chunking(text, chunk_size=size, overlap=0)