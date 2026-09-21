import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 문서 묶음(여러 파일) 검색은 indexing.py: build_retriever()로 대체됨
# (InMemoryVectorStore + as_retriever(), 브루트포스 대신 LangChain 표준 메커니즘 사용).

def retrieve_top_k(query, chunks, model, k=3):
    """단일 문서용 - 아직 어디서도 호출 안 함(골든 데이터셋 트랙에서 쓸 예정, 이번 리팩터링
    범위 밖). chunks는 여전히 옛 Chunk 인터페이스(.text) 기준이라, 실제로 쓸 때는
    Document(.page_content)에 맞게 갱신 필요."""
    texts = [c.text for c in chunks]
    qv = model.encode([query])
    cv = model.encode(texts)
    sims = cosine_similarity(qv, cv)[0]
    ranked = np.argsort(sims)[::-1]
    return [{"index": int(i), "score": float(sims[i]), "chunk": chunks[i]} for i in ranked[:k]]