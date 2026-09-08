import json

from langchain_core.documents import Document

from config import (
    RAG_CANDIDATE_K,
    RAG_EXPANSION_RADIUS,
    RAG_MAX_DISTANCE,
    RAG_TOP_K,
)
from context.entities import Message
from context.prompts import SYSTEM_PROMPT_RERANK, SYSTEM_PROMPT_REWRITE
from llm.provider import LLMProvider
from rag.vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore, llm: LLMProvider):
        self.vector_store = vector_store
        self.llm = llm

    def retrieve(
        self,
        query: str,
        history: list[Message],
        k: int = RAG_TOP_K,
    ) -> list[Document]:
        rewritten_query = self._rewrite_query(query, history)

        candidate_k = max(k, RAG_CANDIDATE_K)
        candidates = self.vector_store.search(rewritten_query, candidate_k)
        candidates = [
            candidate
            for candidate in candidates
            if candidate[1] <= RAG_MAX_DISTANCE
        ]

        if not candidates:
            return []

        reranked = self._rerank(rewritten_query, candidates, k)
        return self._expand_context(reranked, RAG_EXPANSION_RADIUS)

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

    def _rerank(
        self,
        query: str,
        candidates: list[tuple[Document, float]],
        k: int,
    ) -> list[tuple[Document, float]]:
        candidate_map = {
            f"doc_{index}": candidate
            for index, candidate in enumerate(candidates)
        }

        prompt_parts: list[str] = []
        for candidate_id, (doc, _) in candidate_map.items():
            prompt_parts.append(
                f"Document ID: {candidate_id}\n"
                f"Source: {doc.metadata.get('filename', 'Unknown')}\n"
                f"Page: {doc.metadata.get('page', 'Unknown')}\n"
                f"Content:\n{doc.page_content}"
            )
        prompt = "\n\n".join(prompt_parts)

        messages = [
            Message(role="system", content=SYSTEM_PROMPT_RERANK),
            Message(
                role="user",
                content=(
                    f"Query:\n{query}\n\n"
                    f"Maximum candidates: {k}\n\n"
                    f"Candidates:\n{prompt}"
                ),
            ),
        ]

        try:
            raw_response = self.llm.chat(messages).strip()
            response = json.loads(raw_response)
        except Exception:
            return candidates[:k]

        if not isinstance(response, list):
            return candidates[:k]

        ranked_ids: list[str] = []
        for item in response:
            if not isinstance(item, str):
                continue

            if item not in candidate_map:
                continue

            if item in ranked_ids:
                continue

            ranked_ids.append(item)

        if response and not ranked_ids:
            return candidates[:k]

        if len(ranked_ids) > k:
            ranked_ids = ranked_ids[:k]

        return [candidate_map[candidate_id] for candidate_id in ranked_ids]

    def _expand_context(
        self,
        reranked: list[tuple[Document, float]],
        radius: int,
    ) -> list[Document]:
        reranked_docs = [doc for doc, _ in reranked]

        if radius <= 0:
            return reranked_docs

        expanded_documents: list[Document] = []
        seen_chunk_ids: set[str] = set()
        for doc in reranked_docs:
            chunk_id = doc.metadata.get("chunk_id")

            if not isinstance(chunk_id, str):
                expanded_documents.append(doc)
                continue

            try:
                source, index_text = chunk_id.rsplit("::", 1)
                index = int(index_text)
            except ValueError:
                expanded_documents.append(doc)
                continue

            start_index = max(0, index - radius)
            end_index = index + radius
            neighbor_ids = [
                f"{source}::{index}"
                for index in range(start_index, end_index + 1)
            ]

            neighbor_documents = self.vector_store.get_by_ids(neighbor_ids)
            documents_by_id: dict[str, Document] = {}
            for neighbor_doc in neighbor_documents:
                neighbor_id = neighbor_doc.metadata.get("chunk_id")
                if isinstance(neighbor_id, str):
                    documents_by_id[neighbor_id] = neighbor_doc

            documents_by_id.setdefault(chunk_id, doc)

            for neighbor_id in neighbor_ids:
                if neighbor_id in seen_chunk_ids:
                    continue

                neighbor_doc = documents_by_id.get(neighbor_id)
                if neighbor_doc is None:
                    continue

                seen_chunk_ids.add(neighbor_id)
                expanded_documents.append(neighbor_doc)

        return expanded_documents
