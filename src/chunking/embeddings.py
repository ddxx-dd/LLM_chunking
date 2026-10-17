"""청커가 쓰는 임베딩: 글 목록 → 길이 1 벡터(np.float32, (n, d)). 기본은 bge-m3 fp16(common.models).

semantic(LangChain SemanticChunker)은 LangChain Embeddings 객체가 필요해서 as_langchain()으로 감싼다.
테스트는 encode 함수를 직접 넘겨 모델 없이 돈다.
"""
import numpy as np


def default_encode():
    from common.models import load_embedder, make_encode

    return make_encode(load_embedder())


def as_langchain(encode):
    """encode(글 목록) → LangChain Embeddings."""
    from langchain_core.embeddings import Embeddings

    class _Emb(Embeddings):
        def embed_documents(self, texts):
            return np.asarray(encode(list(texts)), dtype=np.float32).tolist()

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    return _Emb()
