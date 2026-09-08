# AI-generated with OpenAI Codex

import unittest
from collections.abc import Iterator

from langchain_core.documents import Document

from config import RAG_CANDIDATE_K, RAG_MAX_DISTANCE
from context.entities import Message
from context.prompts import SYSTEM_PROMPT_RERANK, SYSTEM_PROMPT_REWRITE
from llm.provider import LLMProvider
from rag.retriever import Retriever
from rag.vector_store import VectorStore


class FakeLLM(LLMProvider):
    def __init__(
        self,
        response: str = "rewritten query",
        responses: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.responses = list(responses) if responses is not None else None
        self.error = error
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error

        if self.responses is not None:
            return self.responses.pop(0)

        return self.response

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        yield self.response


class FakeVectorStore(VectorStore):
    def __init__(
        self,
        results: list[tuple[Document, float]] | None = None,
        documents_by_id: dict[str, Document] | None = None,
        search_error: Exception | None = None,
    ) -> None:
        self.results = list(results) if results is not None else []
        self.documents_by_id = documents_by_id or {}
        self.search_error = search_error
        self.calls: list[tuple[str, int]] = []
        self.get_by_ids_calls: list[list[str]] = []

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        self.calls.append((query, k))
        if self.search_error is not None:
            raise self.search_error
        return self.results

    def get_by_ids(self, ids: list[str]) -> list[Document]:
        self.get_by_ids_calls.append(ids)
        return [
            self.documents_by_id[chunk_id]
            for chunk_id in ids
            if chunk_id in self.documents_by_id
        ]


class RetrieverTests(unittest.TestCase):
    def test_rewrites_query_with_history_and_returns_search_results(self):
        document = Document(page_content="result")
        results = [(document, 0.25)]
        llm = FakeLLM(
            responses=[
                "  photosynthesis light reactions  \n",
                '["doc_0"]',
            ]
        )
        vector_store = FakeVectorStore(results)
        retriever = Retriever(vector_store=vector_store, llm=llm)
        history = [
            Message(role="user", content="Explain photosynthesis."),
            Message(
                role="assistant",
                content="It has light-dependent reactions and the Calvin cycle.",
            ),
        ]

        actual = retriever.retrieve("What about the first part?", history, k=7)

        self.assertEqual(actual, [document])
        self.assertEqual(
            vector_store.calls,
            [("photosynthesis light reactions", RAG_CANDIDATE_K)],
        )
        rewrite_messages = llm.calls[0]
        self.assertEqual(
            rewrite_messages[0],
            Message(role="system", content=SYSTEM_PROMPT_REWRITE),
        )
        self.assertEqual(rewrite_messages[1:3], history)
        self.assertEqual(
            rewrite_messages[3],
            Message(role="user", content="What about the first part?"),
        )
        self.assertEqual(
            sum(
                message.content == "What about the first part?"
                for message in rewrite_messages
            ),
            1,
        )
        self.assertEqual(
            llm.calls[1][0],
            Message(role="system", content=SYSTEM_PROMPT_RERANK),
        )

    def test_blank_rewrite_falls_back_to_original_query(self):
        vector_store = FakeVectorStore()
        retriever = Retriever(vector_store=vector_store, llm=FakeLLM(" \n "))

        retriever.retrieve("original query", [])

        self.assertEqual(vector_store.calls[0][0], "original query")

    def test_rewrite_error_falls_back_to_original_query(self):
        vector_store = FakeVectorStore()
        retriever = Retriever(
            vector_store=vector_store,
            llm=FakeLLM(error=RuntimeError("provider unavailable")),
        )

        retriever.retrieve("original query", [])

        self.assertEqual(vector_store.calls[0][0], "original query")

    def test_custom_k_larger_than_candidate_count_is_preserved(self):
        vector_store = FakeVectorStore()
        retriever = Retriever(vector_store=vector_store, llm=FakeLLM())

        retriever.retrieve("query", [], k=20)

        self.assertEqual(vector_store.calls, [("rewritten query", 20)])

    def test_filters_weak_candidates_before_reranking(self):
        relevant = Document(page_content="relevant")
        weak = Document(page_content="weak")
        vector_store = FakeVectorStore(
            [(relevant, 0.2), (weak, RAG_MAX_DISTANCE + 0.01)]
        )
        llm = FakeLLM(responses=["rewritten", '["doc_0"]'])
        retriever = Retriever(vector_store=vector_store, llm=llm)

        actual = retriever.retrieve("query", [])

        self.assertEqual(actual, [relevant])
        rerank_prompt = llm.calls[1][1].content
        self.assertIn("relevant", rerank_prompt)
        self.assertNotIn("weak", rerank_prompt)

    def test_reranks_candidates_and_honors_requested_limit(self):
        documents = [
            Document(page_content="first"),
            Document(page_content="second"),
            Document(page_content="third"),
        ]
        vector_store = FakeVectorStore([(document, 0.2) for document in documents])
        llm = FakeLLM(responses=["rewritten", '["doc_2", "doc_0", "doc_1"]'])
        retriever = Retriever(vector_store=vector_store, llm=llm)

        actual = retriever.retrieve("query", [], k=2)

        self.assertEqual(actual, [documents[2], documents[0]])

    def test_empty_rerank_selection_returns_no_documents(self):
        document = Document(page_content="irrelevant")
        vector_store = FakeVectorStore([(document, 0.2)])
        llm = FakeLLM(responses=["rewritten", "[]"])
        retriever = Retriever(vector_store=vector_store, llm=llm)

        self.assertEqual(retriever.retrieve("query", []), [])

    def test_invalid_rerank_response_falls_back_to_vector_order(self):
        documents = [Document(page_content="first"), Document(page_content="second")]
        vector_store = FakeVectorStore([(document, 0.2) for document in documents])
        llm = FakeLLM(responses=["rewritten", "not JSON"])
        retriever = Retriever(vector_store=vector_store, llm=llm)

        self.assertEqual(retriever.retrieve("query", [], k=1), [documents[0]])

    def test_expands_context_in_chunk_order_and_removes_duplicates(self):
        chunks = {
            f"source::{index}": Document(
                page_content=f"chunk {index}",
                metadata={"chunk_id": f"source::{index}"},
            )
            for index in range(4)
        }
        vector_store = FakeVectorStore(documents_by_id=chunks)
        retriever = Retriever(vector_store=vector_store, llm=FakeLLM())

        actual = retriever._expand_context(
            [(chunks["source::1"], 0.1), (chunks["source::2"], 0.2)],
            radius=1,
        )

        self.assertEqual(
            [document.metadata["chunk_id"] for document in actual],
            ["source::0", "source::1", "source::2", "source::3"],
        )

    def test_malformed_chunk_id_keeps_original_document(self):
        document = Document(
            page_content="content",
            metadata={"chunk_id": "malformed"},
        )
        vector_store = FakeVectorStore()
        retriever = Retriever(vector_store=vector_store, llm=FakeLLM())

        actual = retriever._expand_context([(document, 0.1)], radius=1)

        self.assertEqual(actual, [document])
        self.assertEqual(vector_store.get_by_ids_calls, [])

    def test_vector_store_errors_propagate(self):
        retriever = Retriever(
            vector_store=FakeVectorStore(search_error=RuntimeError("search failed")),
            llm=FakeLLM(),
        )

        with self.assertRaisesRegex(RuntimeError, "search failed"):
            retriever.retrieve("query", [])


if __name__ == "__main__":
    unittest.main()
