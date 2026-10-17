"""recursive: LangChain RecursiveCharacterTextSplitter 기본 구현 그대로(구분자 '\\n\\n' → '\\n' → ' ' → '', 길이는 글자 수).

크기 값(chunk_size, chunk_overlap)은 실험 전에 정한다(AGENTS.md 2절 4항).
"""


def chunk(text, chunk_size, chunk_overlap=0):
    """텍스트 → 청크 문자열 목록."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_text(text)
