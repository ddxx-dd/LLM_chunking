import re
import numpy as np

from schema import Chunk

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
    """청크 크기는 글자수 대신 토큰수로 잰다 - EN/KO 글자당 토큰 밀도가 달라서.
    fixed_chunking은 '고정 글자수 분할'의 정의 자체이므로 의도적으로 글자수 그대로 둔다."""
    return len(tokenizer.encode(s, add_special_tokens=False))

def split_at_boundaries(sents, similarities, threshold, text, tokenizer=None,
                         max_chunk_tokens=None, min_chunk_tokens=None):
    """문장 간 유사도가 임계값 아래로 떨어지는 지점에서 자른다. max_chunk_tokens는
    상한 초과 "직전"에 끊고, min_chunk_tokens는 미달 시 다음 문장까지 계속 모은다
    (짧은 파편 청크 방지). 둘 다 문장 단위로만 잘라 항상 문장 경계에서 끊긴다."""
    chunks = []
    start = 0
    seg_start_idx = 0  # 지금 열려 있는 청크에 포함된 첫 문장의 인덱스
    n = len(sents)

    for i in range(n):
        seg_end = sents[i + 1][1] if i + 1 < n else len(text)
        if max_chunk_tokens and i > seg_start_idx and _token_len(text[start:seg_end], tokenizer) > max_chunk_tokens:
            cut_at = sents[i][1]
            chunk_text = text[start:cut_at]
            if chunk_text.strip():
                chunks.append(Chunk(chunk_text, start, cut_at))
            start = cut_at
            seg_start_idx = i

        if i < n - 1 and similarities[i] < threshold:
            cut_at = sents[i + 1][1]
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


def default_subtitle_chunkers(embed_model):
    """자막 실험 스크립트들이 공유하는 fixed/semantic 기본 설정(실측 검증된 값) -
    스크립트마다 따로 들면 한쪽만 갱신되는 드리프트가 생겨서 여기 하나로 모음."""
    return {
        "fixed": lambda text: fixed_chunking(text, chunk_size=500),
        "semantic": lambda text: semantic_chunking(text, embed_model, method="percentile", amount=15,
                                                     min_chunk_tokens=128, max_chunk_tokens=1024),
    }
