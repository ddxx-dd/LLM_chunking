"""docx 트랙 검색 - InMemoryVectorStore(추가 설치 불필요) + as_retriever().
브루트포스 sklearn cosine(retriever.retrieve_top_k_bundle)을 대체한다."""
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore


class SentenceTransformerEmbeddings(Embeddings):
    """이미 로드된 SentenceTransformer 인스턴스를 감싸는 얇은 어댑터 - 청킹(semantic)과
    검색(vectorstore)이 bge-m3를 각자 새로 로드하면 GPU 메모리가 중복되므로,
    같은 모델 인스턴스를 공유해서 쓴다."""

    def __init__(self, model):
        self._model = model

    def embed_documents(self, texts):
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self._model.encode([text], show_progress_bar=False)[0].tolist()


def build_retriever(documents, embed_model, k=5):
    """documents: langchain Document 리스트(청크). embed_model: SentenceTransformer 인스턴스.
    반환: retriever.invoke(query) -> 상위 k개 Document."""
    embeddings = SentenceTransformerEmbeddings(embed_model)
    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_documents(documents)
    return vectorstore.as_retriever(search_kwargs={"k": k})
