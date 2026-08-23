# AI-generated with OpenAI Codex

import unittest
from collections.abc import Iterator
from unittest.mock import patch

from langchain_core.documents import Document

from assistant.assistant import Assistant
from context.entities import Message
from context.history import HistoryManager
from llm.provider import LLMProvider
from rag.retriever import Retriever
from config import LLM_CONTEXT_SIZE, RAG_CONTEXT_SIZE


class FakeHistory(HistoryManager):
    def __init__(self, messages: list[Message]) -> None:
        self.messages = list(messages)
        self.get_messages_calls: list[tuple[int | None, int | None]] = []

    def add_message(self, msg: Message) -> None:
        self.messages.append(msg)

    def get_messages(
        self,
        conversation_id: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        self.get_messages_calls.append((conversation_id, limit))
        messages = self.messages.copy()
        return messages if limit is None else messages[-limit:]


class FakeRetriever(Retriever):
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[Message], int]] = []

    def retrieve(
        self,
        query: str,
        history: list[Message],
        k: int = 5,
    ) -> list[tuple[Document, float]]:
        self.calls.append((query, history, k))
        return []


class FakeLLM(LLMProvider):
    def __init__(
        self,
        response: str = "",
        chunks: list[str] | None = None,
    ) -> None:
        self.response = response
        self.chunks = chunks or ["answer"]
        self.stream_messages: list[Message] | None = None

    def chat(self, messages: list[Message]) -> str:
        return self.response

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        self.stream_messages = messages
        yield from self.chunks


class AssistantTests(unittest.TestCase):
    def test_chat_delegates_to_llm(self):
        llm = FakeLLM(response="response")
        assistant = Assistant(llm=llm, history=FakeHistory([]), retriever=FakeRetriever())
        messages = [Message(role="user", content="question")]

        self.assertEqual(assistant.chat(messages), "response")

    def test_format_context_returns_empty_string_without_documents(self):
        assistant = Assistant(
            llm=FakeLLM(), history=FakeHistory([]), retriever=FakeRetriever()
        )

        self.assertEqual(assistant._format_context([]), "")

    def test_format_context_includes_document_metadata_content_and_score(self):
        assistant = Assistant(
            llm=FakeLLM(), history=FakeHistory([]), retriever=FakeRetriever()
        )
        document = Document(
            page_content="  Relevant passage.  ",
            metadata={"filename": "notes.pdf", "page": 1},
        )

        context = assistant._format_context([(document, 0.4321)])

        self.assertIn("--- Document 1 ---", context)
        self.assertIn("Source: notes.pdf", context)
        self.assertIn("Page: 2", context)
        self.assertIn("Similarity Score: 0.4321", context)
        self.assertIn("Relevant passage.", context)

    def test_retrieval_uses_prior_history_and_current_prompt_is_added_once(self):
        prior_history = [Message(role="user", content="Earlier question")]
        history = FakeHistory(prior_history)
        retriever = FakeRetriever()
        llm = FakeLLM()
        assistant = Assistant(llm=llm, history=history, retriever=retriever)
        with patch.object(assistant, "_dump_messages"):
            completion = "".join(assistant.stream_chat("Follow-up question"))

        self.assertEqual(completion, "answer")
        self.assertEqual(retriever.calls, [("Follow-up question", prior_history, 5)])
        self.assertEqual(
            history.get_messages_calls,
            [(None, RAG_CONTEXT_SIZE), (None, LLM_CONTEXT_SIZE)],
        )
        user_messages = [
            message
            for message in history.messages
            if message.role == "user" and message.content == "Follow-up question"
        ]
        self.assertEqual(len(user_messages), 1)
        self.assertIsNotNone(llm.stream_messages)
        assert llm.stream_messages is not None
        self.assertIn(user_messages[0], llm.stream_messages)
        self.assertEqual(history.messages[-1], Message(role="assistant", content="answer"))

    def test_stream_chat_yields_all_chunks_and_stores_combined_completion(self):
        history = FakeHistory([])
        llm = FakeLLM(chunks=["one", " ", "two"])
        assistant = Assistant(llm=llm, history=history, retriever=FakeRetriever())
        with patch.object(assistant, "_dump_messages"):
            chunks = list(assistant.stream_chat("question"))

        self.assertEqual(chunks, ["one", " ", "two"])
        self.assertEqual(history.messages[-1], Message(role="assistant", content="one two"))


if __name__ == "__main__":
    unittest.main()
