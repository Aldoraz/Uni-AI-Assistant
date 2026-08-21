from langchain_core.documents import Document
from context.entities import Message
from context.prompts import SYSTEM_PROMPT_REWRITE
from llm.provider import LLMProvider
from rag.vector_store import VectorStore
from config import RAG_TOP_K

class Retriever:
    def __init__(self, vector_store: VectorStore, llm: LLMProvider):
        self.vector_store = vector_store
        self.llm = llm
    
    def retrieve(
        self,
        query: str,
        history: list[Message],
        k: int = RAG_TOP_K,
    ) -> list[tuple[Document, float]]:
        # TODO: Validate sensibility of RAG via LLM
        # TODO: Reranking (top 5 from 20)
        # TODO: Context expansion
        rewritten_query = self._rewrite_query(query, history)
        return self.vector_store.search(rewritten_query, k)

    def _rewrite_query(self, query: str, history: list[Message]) -> str:
        messages = [
            Message(role="system", content=SYSTEM_PROMPT_REWRITE),
            *history,
            Message(role="user", content=query),
        ]

        try:
            rewritten_query = self.llm.chat(messages).strip()
        except Exception:
            return query

        return rewritten_query or query
