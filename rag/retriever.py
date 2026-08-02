from langchain_core.documents import Document
from rag.vector_store import VectorStore
from config import RAG_TOP_K

class Retriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def retrieve(self, query: str, k: int = RAG_TOP_K) -> list[tuple[Document, float]]:
        # TODO: Validate sensibility of RAG via LLM
        # TODO: Rewrite prompt via LLM
        # TODO: Reranking (top 5 from 20)
        # TODO: Context expansion
        return self.vector_store.search(query, k)