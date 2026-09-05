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