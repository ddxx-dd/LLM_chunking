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

def retrieve_top_k_bundle(query, bundle_chunks, model, k=5):
    """여러 문서에 걸친 청크 묶음에서 top-k 검색.
    bundle_chunks: [{"doc_name":..., "doc_idx":..., "chunk":Chunk}, ...]
    (문서 경계를 넘어 청킹하지 않고, 검색 단계에서만 전체를 대상으로 비교한다)"""
    texts = [bc["chunk"].text for bc in bundle_chunks]
    qv = model.encode([query])
    cv = model.encode(texts)
    sims = cosine_similarity(qv, cv)[0]
    ranked = np.argsort(sims)[::-1]
    out = []
    for i in ranked[:k]:
        item = dict(bundle_chunks[int(i)])
        item["score"] = float(sims[i])
        out.append(item)
    return out