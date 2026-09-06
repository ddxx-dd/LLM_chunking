"""여러 docx 문서 묶음을 문서 경계를 넘지 않고 각각 청킹한다."""

def chunk_bundle(docs, chunker_fn):
    """docs: Doc 리스트. 문서별로 독립 청킹(경계 안 넘음).
    반환: [{"doc_idx":i, "doc_name":doc.name, "chunk":Chunk}, ...]"""
    out = []
    for doc_idx, doc in enumerate(docs):
        chunks = chunker_fn(doc.text)
        for c in chunks:
            out.append({"doc_idx": doc_idx, "doc_name": docs[doc_idx].name, "chunk": c})
    return out
