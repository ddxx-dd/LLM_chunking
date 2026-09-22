def covered_units(chunk):
    """겹침 계산을 부모 doc 인자 없이 chunk 하나만으로 수행 -
    split_documents()가 부모 metadata(units 포함)를 청크에 전부 복사해두기 때문.
    검색으로 낱개 청크만 들고 있는 상황(docx_compose 등)에서 쓴다."""
    c_start = chunk.metadata["start_index"]
    c_end = c_start + len(chunk.page_content)
    out = []
    for u in chunk.metadata["units"]:
        s, e = max(u["start"], c_start), min(u["end"], c_end)
        if s < e:
            out.append(u)
    return out


def chunk_bundle(docs, splitter):
    """여러 문서 묶음을 문서 경계를 넘지 않고 각각 청킹한다.
    docs: Document 리스트. splitter: TextSplitter 인스턴스(split_documents 사용 -
    부모 metadata가 각 청크에 자동으로 복사되어 doc_name 등을 그대로 들고 감).
    반환: [{"doc_idx":i, "doc_name":doc.metadata["name"], "chunk":Document}, ...]"""
    out = []
    for doc_idx, doc in enumerate(docs):
        chunks = splitter.split_documents([doc])
        for c in chunks:
            out.append({"doc_idx": doc_idx, "doc_name": doc.metadata["name"], "chunk": c})
    return out
