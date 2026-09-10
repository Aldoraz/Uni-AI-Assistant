import json
import logging
from pathlib import Path

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

logger = logging.getLogger(__name__)


def _document_filename(document: Document) -> str:
    filename = document.metadata.get("filename")
    if not isinstance(filename, str):
        source = document.metadata.get("source")
        filename = Path(source).name if isinstance(source, str) else "Unknown"
    return filename


def _chunk_index(document: Document) -> int | None:
    chunk_id = document.metadata.get("chunk_id")
    if not isinstance(chunk_id, str):
        return None

    try:
        _, chunk_index = chunk_id.rsplit("::", 1)
        return int(chunk_index)
    except ValueError:
        return None


def _document_identifier(document: Document) -> str:
    filename = _document_filename(document)

    chunk_index = _chunk_index(document)
    if chunk_index is not None:
        return f"{filename}::{chunk_index}"

    page = document.metadata.get("page")
    if isinstance(page, int):
        return f"{filename}::page-{page + 1}"

    return filename


def _format_documents(documents: list[Document]) -> str:
    formatted: list[str] = []
    previous_filename: str | None = None

    for document in documents:
        filename = _document_filename(document)
        chunk_index = _chunk_index(document)

        if filename == previous_filename and chunk_index is not None:
            formatted.append(f"::{chunk_index}")
        else:
            formatted.append(_document_identifier(document))

        previous_filename = filename

    return ", ".join(formatted)


def _format_expansion_range(documents: list[Document]) -> str:
    indices = [
        index
        for document in documents
        if (index := _chunk_index(document)) is not None
    ]
    if (
        indices
        and len(indices) == len(documents)
        and indices == list(range(indices[0], indices[-1] + 1))
    ):
        if len(indices) == 1:
            return f"::{indices[0]}"
        return f"::{indices[0]} - ::{indices[-1]}"

    return f"[{_format_documents(documents)}]"


def _format_scored_documents(
    documents: list[tuple[Document, float]],
) -> str:
    return ", ".join(
        f"{_document_identifier(document)} (distance={score:.4f})"
        for document, score in documents
    )


class Retriever:
    def __init__(self, vector_store: VectorStore, llm: LLMProvider) -> None:
        self.vector_store = vector_store
        self.llm = llm

    def retrieve(
        self,
        query: str,
        history: list[Message],
        k: int = RAG_TOP_K,
    ) -> list[Document]:
        logger.info(
            "Retrieval started (history_messages=%d, requested_documents=%d)",
            len(history),
            k,
        )
        rewritten_query = self._rewrite_query(query, history)

        candidate_k = max(k, RAG_CANDIDATE_K)
        candidates = self.vector_store.search(rewritten_query, candidate_k)
        candidate_count = len(candidates)
        logger.info(
            "Vector candidates (count=%d): [%s]",
            candidate_count,
            _format_scored_documents(candidates),
        )
        candidates = [
            candidate
            for candidate in candidates
            if candidate[1] <= RAG_MAX_DISTANCE
        ]
        logger.info(
            "Distance threshold %.4f retained %d/%d candidates",
            RAG_MAX_DISTANCE,
            len(candidates),
            candidate_count,
        )

        if not candidates:
            logger.info("Retrieval completed without relevant documents")
            return []

        reranked = self._rerank(rewritten_query, candidates, k)
        expanded = self._expand_context(reranked, RAG_EXPANSION_RADIUS)
        logger.info(
            "Final retrieval context (count=%d): [%s]",
            len(expanded),
            _format_documents(expanded),
        )
        return expanded

    def _rewrite_query(self, query: str, history: list[Message]) -> str:
        messages = [
            Message(role="system", content=SYSTEM_PROMPT_REWRITE),
            *history,
            Message(role="user", content=query),
        ]

        try:
            rewritten_query = self.llm.chat(messages).strip()
        except Exception as error:
            logger.warning(
                "Query rewriting failed; using original query (error=%s)",
                error,
            )
            return query

        if not rewritten_query:
            logger.warning(
                "Query rewriting returned blank output; using original query"
            )
        else:
            logger.info("Query rewritten to: %r", rewritten_query)

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
        except Exception as error:
            logger.warning(
                "Reranking failed; preserving vector order "
                "(error=%s, documents=[%s])",
                error,
                _format_scored_documents(candidates[:k]),
            )
            return candidates[:k]

        if not isinstance(response, list):
            logger.warning(
                "Reranking returned a non-list; preserving vector order "
                "(documents=[%s])",
                _format_scored_documents(candidates[:k]),
            )
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
            logger.warning(
                "Reranking returned no valid IDs; preserving vector order "
                "(documents=[%s])",
                _format_scored_documents(candidates[:k]),
            )
            return candidates[:k]

        if len(ranked_ids) > k:
            ranked_ids = ranked_ids[:k]

        reranked = [candidate_map[candidate_id] for candidate_id in ranked_ids]
        logger.info(
            "Reranked documents (count=%d): [%s]",
            len(reranked),
            _format_scored_documents(reranked),
        )
        return reranked

    def _expand_context(
        self,
        reranked: list[tuple[Document, float]],
        radius: int,
    ) -> list[Document]:
        reranked_docs = [doc for doc, _ in reranked]

        if radius <= 0:
            logger.info("Context expansion skipped (radius=%d)", radius)
            return reranked_docs

        expanded_documents: list[Document] = []
        seen_chunk_ids: set[str] = set()
        for doc in reranked_docs:
            chunk_id = doc.metadata.get("chunk_id")

            if not isinstance(chunk_id, str):
                logger.warning(
                    "Context expansion skipped for %s: missing chunk ID",
                    _document_identifier(doc),
                )
                expanded_documents.append(doc)
                continue

            try:
                source, index_text = chunk_id.rsplit("::", 1)
                index = int(index_text)
            except ValueError:
                logger.warning(
                    "Context expansion skipped for %s: invalid chunk ID",
                    _document_identifier(doc),
                )
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

            available_neighbors = [
                documents_by_id[neighbor_id]
                for neighbor_id in neighbor_ids
                if neighbor_id in documents_by_id
            ]
            logger.info(
                "Context expanded %s with radius %d to %s",
                _document_identifier(doc),
                radius,
                _format_expansion_range(available_neighbors),
            )

            for neighbor_id in neighbor_ids:
                if neighbor_id in seen_chunk_ids:
                    continue

                neighbor_doc = documents_by_id.get(neighbor_id)
                if neighbor_doc is None:
                    continue

                seen_chunk_ids.add(neighbor_id)
                expanded_documents.append(neighbor_doc)

        return expanded_documents
