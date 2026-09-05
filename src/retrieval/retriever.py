#질문과 가장 유사한 청크 k개를 찾는다.

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def retrieve_top_k(query, chunks, model, k=3):
#질문 벡터와 모든 청크 벡터의 코사인 유사도 -> 상위 k개.

    texts = [c.text for c in chunks]
    qv = model.encode([query])
    cv = model.encode(texts)
    sims = cosine_similarity(qv, cv)[0]
    ranked = np.argsort(sims)[::-1]                 
    return [{"index": int(i), "score": float(sims[i]), "chunk": chunks[i]}
            for i in ranked[:k]]


def print_retrieved(query, retrieved, preview=90):
    print("질문:", query)
    print("=" * 62)
    for rank, item in enumerate(retrieved, 1):
        t = item["chunk"].text.strip().replace("\n", " / ")
        if len(t) > preview:
            t = t[:preview] + "..."
        print("[{}위] 청크{} | 유사도 {:.3f}".format(rank, item["index"], item["score"]))
        print("  ", t)
        print()