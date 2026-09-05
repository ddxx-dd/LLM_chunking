import sys
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).resolve().parent.parent))

def retrieve_top_k(query, chunks, model, k=3):
    texts = [c.text for c in chunks]
    qv = model.encode([query])
    cv = model.encode(texts)
    sims = cosine_similarity(qv, cv)[0]
    ranked = np.argsort(sims)[::-1]
    return [{"index": int(i), "score": float(sims[i]), "chunk": chunks[i]} for i in ranked[:k]]

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