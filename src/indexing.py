"""docx 트랙 검색 - InMemoryVectorStore(추가 설치 불필요) + as_retriever().
브루트포스 sklearn cosine을 대체한다."""
from langchain_core.vectorstores import InMemoryVectorStore


def build_retriever(documents, embeddings, k=5):
    """documents: langchain Document 리스트(청크). embeddings: LangChain Embeddings
    인터페이스 객체(예: HuggingFaceEmbeddings) - 청킹(semantic)에 쓴 것과 같은
    인스턴스를 넘기면 bge-m3가 GPU에 중복 로드되지 않는다.
    반환: retriever.invoke(query) -> 상위 k개 Document."""
    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_documents(documents)
    return vectorstore.as_retriever(search_kwargs={"k": k})
