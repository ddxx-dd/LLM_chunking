"""여러 docx 문서 묶음(큰 주제의 소주제 분할본이든, 서로 다른 주제 묶음이든)을
문서 경계를 넘지 않고 각각 청킹한 뒤, top-k 검색 + 요약에 쓸 프롬프트를 구성한다."""

def chunk_bundle(docs, chunker_fn):
    """docs: Doc 리스트. chunker_fn(text) -> list[Chunk].
    각 문서를 독립적으로 청킹해서(문서 경계를 절대 넘지 않음) 전역 리스트로 반환.
    반환: [{"doc_idx":i, "doc_name":doc.name, "chunk":Chunk}, ...]"""
    out = []
    for doc_idx, doc in enumerate(docs):
        chunks = chunker_fn(doc.text)
        for c in chunks:
            out.append({"doc_idx": doc_idx, "doc_name": docs[doc_idx].name, "chunk": c})
    return out

def build_summary_prompt(query, retrieved):
    """retrieved: retrieve_top_k_bundle()의 반환값.
    각 조각이 어느 문서에서 왔는지 라벨을 붙여, 근거 없는 내용을 지어내지 않도록 지시.
    (라벨을 LLM에서 숨기는 방안도 검토했으나, 이미 top-k로 관련성 필터링된 조각들만
    들어오고 그 내용만 근거로 요약하므로 굳이 숨길 필요가 없다고 판단해 그대로 둠.)"""
    body_parts = []
    for i, item in enumerate(retrieved, 1):
        body_parts.append(f"[출처: {item['doc_name']}]\n{item['chunk'].text.strip()}")
    body = "\n\n".join(body_parts)
    instruction = (
        "다음은 여러 문서에서 검색된 조각들입니다. 아래 조각들에 있는 내용만 근거로 삼아 "
        "질문에 답하는 요약문을 작성하세요. 조각에 없는 내용은 지어내지 마세요. "
        "관련 없는 조각은 무시하세요.\n\n"
        f"질문: {query}\n\n"
        f"검색된 조각들:\n{body}\n\n"
        "답변(요약):"
    )
    return instruction
