"""semantic: LangChain SemanticChunker 기본 구현(langchain_experimental 0.4.2). 문장 분리만 공용 함수(chunking.sentences)로 바꾼다.

원래 알고리즘 그대로: 문장마다 앞뒤 buffer_size 문장을 이어 임베딩 → 이웃 문장 사이 코사인 거리 → 거리가 문서 안 분포의
breakpoint_threshold_amount(기본 백분위 95)보다 큰 곳에서 자름 → 문장을 공백으로 이어 청크. 청크 크기 상한은 없다.
원래 문장 분리('(?<=[.?!])\\s+' 정규식)는 advanced_semantic과 같은 단위를 쓰려고 바꿔 끼운다(2026-10-16 결정).
"""
from chunking.embeddings import as_langchain
from chunking.sentences import split_text


def chunk(text, encode, lang="en", breakpoint_threshold_type="percentile", breakpoint_threshold_amount=None, buffer_size=1):
    """텍스트 → 청크 문자열 목록. encode: 글 목록 → 길이 1 벡터(chunking.embeddings)."""
    from langchain_experimental.text_splitter import SemanticChunker

    class _Chunker(SemanticChunker):
        def _get_single_sentences_list(self, t):
            return split_text(t, lang) or [t]

    sp = _Chunker(as_langchain(encode), buffer_size=buffer_size, breakpoint_threshold_type=breakpoint_threshold_type,
                  breakpoint_threshold_amount=breakpoint_threshold_amount)
    return sp.split_text(text)
