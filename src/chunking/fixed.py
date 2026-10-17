"""fixed: 글자 수 기준 고정 길이 분할(LangChain CharacterTextSplitter).

separator=''(기본)이면 글자 단위로 chunk_size 글자씩 채운다(문장·문단 경계를 보지 않는 기준선).
separator를 주면 LangChain 기본 동작대로 그 구분자로 나눈 조각을 chunk_size 글자까지 묶는다.
크기 값(chunk_size, chunk_overlap)은 실험 전에 정한다(AGENTS.md 2절 4항).
"""


def chunk(text, chunk_size, chunk_overlap=0, separator=""):
    """텍스트 → 청크 문자열 목록."""
    from langchain_text_splitters import CharacterTextSplitter

    sp = CharacterTextSplitter(separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    return sp.split_text(text)
