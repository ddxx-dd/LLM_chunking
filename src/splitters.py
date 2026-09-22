"""청킹 베이스라인 - LangChain TextSplitter 확장점(상속)에 우리 알고리즘을 꽂는다.
split_text()가 원문의 리터럴 부분 문자열만 반환하면, 상위 클래스의
create_documents(add_start_index=True)가 정확한 오프셋을 계산해준다(실측 검증됨) -
그래서 어떤 구현도 text[start:end] 슬라이싱만 쓰고 join()으로 재조합하지 않는다."""
import re
import numpy as np
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters.base import TextSplitter


def make_fixed_splitter(chunk_size=256, overlap=0):
    """단순 글자수 분할 - 문장/단어/표/자막 큐 경계를 전혀 보지 않는다.
    separator=""가 핵심: 기본 separator(["\\n\\n","\\n"," ",""])를 쓰는
    RecursiveCharacterTextSplitter는 문단/문장 경계를 먼저 찾으려 해서
    '단순 글자수 분할'의 정의 자체를 깬다."""
    if overlap >= chunk_size:
        raise ValueError("overlap은 chunk_size보다 작아야 함")
    return CharacterTextSplitter(
        separator="", chunk_size=chunk_size, chunk_overlap=overlap,
        keep_separator=False, strip_whitespace=False, add_start_index=True,
    )


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
    fixed는 '고정 글자수 분할'의 정의 자체이므로 의도적으로 글자수 그대로 둔다."""
    return len(tokenizer.encode(s, add_special_tokens=False))


class SemanticTextSplitter(TextSplitter):
    """의미 기반 분할 - 인접 문장 임베딩 코사인 유사도가 임계값 아래로 떨어지는
    지점에서 자른다. LangChain TextSplitter를 상속해 create_documents()/
    split_documents()/add_start_index 등 프레임워크 표준 메커니즘에 그대로 올라탄다
    (langchain_experimental.SemanticChunker는 " ".join() 재조합으로 오프셋이
    깨지는 게 실측 확인되어 쓰지 않음 - 이 클래스는 리터럴 슬라이싱만 써서 피해간다)."""

    def __init__(self, embed_model, method="percentile", amount=15, max_sentence_length=200,
                 max_chunk_tokens=None, min_chunk_tokens=None, **kwargs):
        kwargs.setdefault("chunk_overlap", 0)
        kwargs.setdefault("add_start_index", True)
        super().__init__(**kwargs)
        self._embed_model = embed_model
        self._method = method
        self._amount = amount
        self._max_sentence_length = max_sentence_length
        self._max_chunk_tokens = max_chunk_tokens
        self._min_chunk_tokens = min_chunk_tokens

    def split_text(self, text):
        sents = split_sentences(text, self._max_sentence_length)
        if len(sents) < 2:
            return [text] if text.strip() else []

        vectors = self._embed_model.encode([s[0] for s in sents], show_progress_bar=False)
        sims = calculate_similarities(vectors)
        threshold = calculate_threshold(sims, self._method, self._amount)
        tokenizer = self._embed_model.tokenizer

        chunks = []
        start = 0
        seg_start_idx = 0
        n = len(sents)

        for i in range(n):
            seg_end = sents[i + 1][1] if i + 1 < n else len(text)
            if (self._max_chunk_tokens and i > seg_start_idx
                    and _token_len(text[start:seg_end], tokenizer) > self._max_chunk_tokens):
                cut_at = sents[i][1]
                piece = text[start:cut_at]
                if piece.strip():
                    chunks.append(piece)
                start = cut_at
                seg_start_idx = i

            if i < n - 1 and sims[i] < threshold:
                cut_at = sents[i + 1][1]
                if self._min_chunk_tokens and _token_len(text[start:cut_at], tokenizer) < self._min_chunk_tokens:
                    continue
                piece = text[start:cut_at]
                if piece.strip():
                    chunks.append(piece)
                start = cut_at
                seg_start_idx = i + 1

        last_piece = text[start:]
        if last_piece.strip():
            chunks.append(last_piece)
        return chunks


def default_subtitle_chunkers(embed_model):
    """자막 실험 스크립트들이 공유하는 fixed/semantic 기본 설정(실측 검증된 값) -
    스크립트마다 따로 들면 한쪽만 갱신되는 드리프트가 생겨서 여기 하나로 모음.
    반환값은 TextSplitter 인스턴스 - 호출부는 splitter.split_documents([doc])로 쓴다."""
    return {
        "fixed": make_fixed_splitter(chunk_size=500),
        "semantic": SemanticTextSplitter(embed_model, method="percentile", amount=15,
                                          min_chunk_tokens=128, max_chunk_tokens=1024),
    }
