# AI-generated with OpenAI Codex

import unittest
from collections.abc import Iterator

from langchain_core.documents import Document

from context.entities import Message
from context.prompts import SYSTEM_PROMPT_REWRITE
from llm.provider import LLMProvider
from rag.retriever import Retriever
from rag.vector_store import VectorStore


class FakeLLM(LLMProvider):
    def __init__(
        self,
        response: str = "rewritten query",
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.messages: list[Message] | None = None

    def chat(self, messages: list[Message]) -> str:
        self.messages = messages
        if self.error is not None:
            raise self.error
        return self.response

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        yield self.response


class FakeVectorStore(VectorStore):
    def __init__(self, results: list[tuple[Document, float]] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        self.calls.append((query, k))
        return self.results


class RetrieverTests(unittest.TestCase):
    def test_rewrites_query_with_history_and_returns_search_results(self):
        results = [(Document(page_content="result"), 0.25)]
        llm = FakeLLM("  photosynthesis light reactions  \n")
        vector_store = FakeVectorStore(results)
        retriever = Retriever(vector_store=vector_store, llm=llm)
        history = [
            Message(role="user", content="Explain photosynthesis."),
            Message(role="assistant", content="It has light-dependent reactions and the Calvin cycle."),
        ]

        actual = retriever.retrieve("What about the first part?", history, k=7)

        self.assertIs(actual, results)
        self.assertEqual(vector_store.calls, [("photosynthesis light reactions", 7)])
        self.assertIsNotNone(llm.messages)
        assert llm.messages is not None
        self.assertEqual(llm.messages[0], Message(role="system", content=SYSTEM_PROMPT_REWRITE))
        self.assertEqual(llm.messages[1:3], history)
        self.assertEqual(
            llm.messages[3],
            Message(role="user", content="What about the first part?"),
        )
        self.assertEqual(
            sum(message.content == "What about the first part?" for message in llm.messages),
            1,
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


if __name__ == "__main__":
    unittest.main()
