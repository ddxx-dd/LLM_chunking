"""글자(텍스트) 수준 청커 3종 - LangChain 분할기를 그대로 쓰고 결과를 평문의 연속 구간으로 돌려준다. 임베딩이 필요 없다.

  fixed      CharacterTextSplitter(separator="")  글자수 N으로 균등 분할(구조·문장을 보지 않음)
  recursive  RecursiveCharacterTextSplitter       상한 N자, 구분자(문단 -> 줄 -> 공백 -> 글자) 순으로 재귀 분할
  header     MarkdownHeaderTextSplitter           제목(H1~H3)에서 분할(제목 줄 유지). T를 주면 그보다 긴 조각만 recursive로 재분할
크기: size_chars(글자) 또는 T(목표 토큰, 글자 = T x 이 데이터셋의 글자/토큰 비율). 겹침 0.
header는 LangChain 출력이 줄을 다시 이어 붙인 글(리터럴 슬라이스가 아님)이라, 그대로 실행한 뒤 '공백이 아니고 출력 가능한 글자 열'을
맞춰 원문 구간으로 복원한다(경계는 LangChain과 같고, 글자 열이 어긋나면 오류).
"""
from langchain_text_splitters import CharacterTextSplitter, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from chunking.base import Chunker, register


def fixed_spans(text, chunk_size, strip_whitespace=True, chunk_overlap=0):
    sp = CharacterTextSplitter(separator="", chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                               strip_whitespace=strip_whitespace, add_start_index=True)
    spans = []
    for d in sp.create_documents([text]):
        s = d.metadata["start_index"]
        if s < 0:
            raise ValueError("fixed 청크의 위치를 원문에서 찾지 못했다")
        spans.append((s, s + len(d.page_content)))
    return spans


def recursive_spans(text, size, base=0):
    sp = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=0, length_function=len, add_start_index=True)
    out = []
    for d in sp.create_documents([text]):
        s = d.metadata["start_index"]
        if s < 0:
            raise ValueError("recursive 청크의 위치를 원문에서 찾지 못했다")
        out.append((base + s, base + s + len(d.page_content)))
    return out


def _printable_index(text):
    """공백이 아니고 출력 가능한 글자의 위치(LangChain은 줄마다 비출력 글자를 지우고 공백을 다시 이어 붙인다)."""
    return [i for i, ch in enumerate(text) if ch.isprintable() and not ch.isspace()]


def header_spans(text, max_level=3, size=None, doc_id=""):
    heads = [("#" * k, f"H{k}") for k in range(1, max_level + 1)]
    docs = MarkdownHeaderTextSplitter(heads, strip_headers=False).split_text(text)
    idx = _printable_index(text)
    stream = "".join(text[i] for i in idx)
    segs, pos = [], 0
    for d in docs:
        piece = "".join(ch for ch in d.page_content if ch.isprintable() and not ch.isspace())
        if not piece:
            continue
        if stream[pos:pos + len(piece)] != piece:
            raise ValueError(f"header 조각이 원문 글자 열과 어긋난다 (문서 {doc_id}, 위치 {pos})")
        segs.append((idx[pos], idx[pos + len(piece) - 1] + 1))
        pos += len(piece)
    if pos != len(idx):
        raise ValueError(f"header 조각이 원문을 다 덮지 못했다 ({pos}/{len(idx)}, 문서 {doc_id})")
    out = []
    for s, e in segs:
        if size and e - s > size:
            out.extend(recursive_spans(text[s:e], size, base=s))
        else:
            out.append((s, e))
    return out


@register
class Fixed(Chunker):
    name, knob = "fixed", "T"

    def chunk(self, ctx, T=None, size_chars=None, strip_whitespace=True):
        size = ctx.size_chars(T, size_chars)
        if size is None:
            raise ValueError("fixed에는 T(토큰) 또는 size_chars(글자)가 필요하다")
        return fixed_spans(ctx.text, size, strip_whitespace)


@register
class Recursive(Chunker):
    name, knob = "recursive", "T"

    def chunk(self, ctx, T=None, size_chars=None):
        size = ctx.size_chars(T, size_chars)
        if size is None:
            raise ValueError("recursive에는 T(토큰) 또는 size_chars(글자)가 필요하다")
        return recursive_spans(ctx.text, size)


@register
class Header(Chunker):
    name, knob = "header", "T"

    def chunk(self, ctx, T=None, size_chars=None, max_level=3):
        return header_spans(ctx.text, max_level, ctx.size_chars(T, size_chars), ctx.doc_id)
