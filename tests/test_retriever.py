import unittest

from context.entities import Message
from context.prompts import SYSTEM_PROMPT_REWRITE
from rag.retriever import Retriever


class FakeLLM:
    def __init__(self, response="rewritten query", error=None):
        self.response = response
        self.error = error
        self.messages = None

    def chat(self, messages):
        self.messages = messages
        if self.error is not None:
            raise self.error
        return self.response


class FakeVectorStore:
    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def search(self, query, k):
        self.calls.append((query, k))
        return self.results


class RetrieverTests(unittest.TestCase):
    def test_rewrites_query_with_history_and_returns_search_results(self):
        results = [(object(), 0.25)]
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
