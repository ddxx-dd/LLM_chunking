import re
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

_WORD_RE = re.compile(r"\S+|\s+")

def _words_with_trailing_space(text):
    """공백을 독립된 조각으로 두지 않고 항상 "직전 단어"에 붙여서 반환한다.
    독립된 공백 조각을 그대로 두면, 그 조각 하나가 상한 검사를 통과 못 했을 때
    무손실 유지를 위해 이미 꽉 찬 이전 청크에 검사 없이 그냥 붙게 되어 상한을
    조용히 넘기는 경우가 생긴다(실제로 겪은 버그). 단어+뒤따르는 공백을 항상
    하나의 단위로 묶어두면 그 단위 자체를 포함시킬지 말지를 상한 검사 한 번으로
    결정하게 되어 이 문제가 생기지 않는다."""
    raw = _WORD_RE.findall(text)
    merged = []
    i = 0
    while i < len(raw):
        p = raw[i]
        if not p.strip():
            if merged:
                merged[-1] += p
            elif i + 1 < len(raw):
                merged.append(p + raw[i + 1])
                i += 1
            else:
                merged.append(p)
        else:
            merged.append(p)
        i += 1
    return merged

def fixed_chunking_tokens(text, tokenizer, max_tokens=256, overlap_tokens=0):
    """글자 수가 아니라 실제 토크나이저(BPE 등) 토큰 수 기준으로 엄격하게 분할.

    어절(공백으로 구분되는 덩어리) 단위로 하나씩 누적하면서, 그 어절까지 포함한
    구간을 실제로 토크나이징해 토큰 수를 직접 세고 max_tokens와 비교한다 -
    "토큰 수를 세서 상한과 비교한다"는 핵심 동작을 이진 탐색이나 글자수 추정
    같은 우회 없이 가장 단순한 형태 그대로 구현한 것. 어절 단위라 단어 중간을
    자르는 일도 없고, 무손실(lossless) 재구성도 항상 보장된다."""
    if not text or not text.strip():
        return []
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens은 max_tokens보다 작아야 함")

    pieces = _words_with_trailing_space(text)
    offsets, pos = [], 0
    for p in pieces:
        offsets.append((pos, pos + len(p)))
        pos += len(p)
    n = len(pieces)
    if n == 0:
        return []

    # 1) 겹침 없이, 어절 단위로 max_tokens를 지키는 청크 경계(어절 인덱스 구간)를 구한다
    segments = []
    seg_start_i = 0
    i = 0
    while i < n:
        candidate_end = offsets[i][1]
        # 이 청크의 "첫" 어절은 그 자체가 상한보다 길어도 일단 받아들인다
        # (안 그러면 청크를 하나도 못 만드는 경우가 생긴다)
        if i > seg_start_i and _token_len(tokenizer, text[offsets[seg_start_i][0]:candidate_end]) > max_tokens:
            segments.append((seg_start_i, i))
            seg_start_i = i
            continue
        i += 1
    segments.append((seg_start_i, n))

    # 2) overlap_tokens만큼, 두 번째 청크부터 시작 지점을 직전 청크 쪽으로 당긴다.
    #    당기는 구간 자체의 토큰 수만 직접 세서 overlap_tokens를 넘지 않는 선까지 당긴다.
    chunks = []
    for k, (s_i, e_i) in enumerate(segments):
        actual_s_i = s_i
        if overlap_tokens > 0 and k > 0:
            back_i = s_i
            while back_i > 0:
                overlap_text = text[offsets[back_i - 1][0]:offsets[s_i][0]]
                if _token_len(tokenizer, overlap_text) > overlap_tokens:
                    break
                back_i -= 1
            actual_s_i = back_i
        char_start = offsets[actual_s_i][0]
        char_end = offsets[e_i - 1][1]
        piece = text[char_start:char_end]
        if piece.strip():
            chunks.append(Chunk(piece, char_start, char_end))
        elif chunks:
            # 공백만 있는 조각은 새 청크로 만들지 않고 직전 청크에 이어붙여
            # 무손실(lossless) 재구성을 유지한다.
            prev = chunks[-1]
            chunks[-1] = Chunk(prev.text + piece, prev.start, char_end)
    return chunks