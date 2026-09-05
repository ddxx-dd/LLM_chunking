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

def split_at_boundaries(sents, similarities, threshold, text):
    chunks = []
    start = 0
    for i, sim in enumerate(similarities):
        if sim < threshold:
            end = sents[i + 1][1]
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(Chunk(chunk_text, start, end))
            start = end
    last_piece = text[start:]
    if last_piece.strip():
        chunks.append(Chunk(last_piece, start, len(text)))
    return chunks

def semantic_chunking(text, model, method="percentile", amount=10, max_sentence_length=200, return_debug=False):
    sents = split_sentences(text, max_sentence_length)
    if len(sents) < 2:
        single = [Chunk(text, 0, len(text))] if text else []
        return (single, [], 0.0, sents) if return_debug else single

    vectors = model.encode([s[0] for s in sents], show_progress_bar=False)
    similarities = calculate_similarities(vectors)
    threshold = calculate_threshold(similarities, method, amount)
    chunks = split_at_boundaries(sents, similarities, threshold, text)

    if return_debug:
        return chunks, similarities, threshold, sents
    return chunks