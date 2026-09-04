import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#질문과 가장 유사한 청크 k개를 찾는다.
#비교대상: 질문 - 모든 청크
def retrieve_top_k(query, chunks, model, k = 3):
    
    query_vector = model.encode([query])
    chunk_vectors = model.encode(chunks)
    
    sims = cosine_similarity(query_vector,chunk_vectors)[0]
    
    ranked = np.argsort(sims)[::-1]
    
    results = []
    for idx in ranked[:k]:
        results.append({
            "index": int(idx),
            "score": float(sims[idx]),
            "text": chunks[idx],
        })
    return results


def print_retrieved(query, retrieved, preview = 100):
    print("질문: " + query)
    print("=" * 60)
    
    for rank in range(len(retrieved)):
        item = retrieved[rank]
        text = item["text"].strip()[:preview]
        if len(item["text"].strip()) > preview:
            text = text + "..."
            
        print("[" + str(rank + 1) + "위] 청크" + str(item["index"] + 1)
              + " | 유사도 " + format(item["score"], ".3f"))
        print("  " + text)
        print()