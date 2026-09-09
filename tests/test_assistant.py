# AI-generated with OpenAI Codex

import unittest
from collections.abc import Iterator
from unittest.mock import mock_open, patch

from langchain_core.documents import Document

from assistant.assistant import Assistant
from config import LLM_CONTEXT_SIZE, RAG_CONTEXT_SIZE
from context.entities import Message, ToolCall
from context.history import HistoryManager
from llm.provider import LLMProvider, LLMResponse
from rag.retriever import Retriever
from tools.tool import Tool, ToolDefinition, ToolResult
from tools.tool_orchestrator import ToolOrchestrator


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
    ) -> list[Document]:
        self.calls.append((query, history, k))
        return []


class FakeLLM(LLMProvider):
    def __init__(
        self,
        response: str = "",
        chunks: list[str] | None = None,
        tool_responses: list[LLMResponse] | None = None,
    ) -> None:
        self.response = response
        self.chunks = chunks or ["answer"]
        self.stream_messages: list[Message] | None = None
        self.tool_responses = list(tool_responses or [LLMResponse("answer", [])])
        self.tool_calls: list[tuple[list[Message], list[ToolDefinition]]] = []

    def chat(self, messages: list[Message]) -> str:
        return self.response

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        self.stream_messages = messages
        yield from self.chunks

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> LLMResponse:
        self.tool_calls.append((messages.copy(), tools.copy()))
        return self.tool_responses.pop(0)

    def stream_chat_with_tools(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> Iterator[LLMResponse]:
        self.tool_calls.append((messages.copy(), tools.copy()))
        response = self.tool_responses.pop(0)

        if response.content:
            yield LLMResponse(content=response.content, tool_calls=[])

        if response.tool_calls:
            yield LLMResponse(content="", tool_calls=response.tool_calls)


class FakeTool(Tool):
    definition = ToolDefinition(
        name="fake_tool",
        description="A fake tool.",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )

    def __init__(self, result: ToolResult | None = None) -> None:
        self.result = result or ToolResult("tool output")
        self.calls: list[dict[str, object]] = []

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        self.calls.append(arguments)
        return self.result


class AssistantTests(unittest.TestCase):
    def test_chat_delegates_to_llm(self):
        llm = FakeLLM(response="response")
        assistant = Assistant(
            llm=llm,
            history=FakeHistory([]),
            retriever=FakeRetriever(),
            tool_orchestrator=ToolOrchestrator([]),
        )
        messages = [Message(role="user", content="question")]

        self.assertEqual(assistant.chat(messages), "response")

    def test_format_context_returns_empty_string_without_documents(self):
        assistant = Assistant(
            llm=FakeLLM(),
            history=FakeHistory([]),
            retriever=FakeRetriever(),
            tool_orchestrator=ToolOrchestrator([]),
        )

        self.assertEqual(assistant._format_context([]), "")

    def test_format_context_includes_document_metadata_and_content(self):
        assistant = Assistant(
            llm=FakeLLM(),
            history=FakeHistory([]),
            retriever=FakeRetriever(),
            tool_orchestrator=ToolOrchestrator([]),
        )
        document = Document(
            page_content="  Relevant passage.  ",
            metadata={"filename": "notes.pdf", "page": 1},
        )

        context = assistant._format_context([document])

        self.assertIn("--- Document 1 ---", context)
        self.assertIn("Source: notes.pdf", context)
        self.assertIn("Page: 2", context)
        self.assertNotIn("Similarity Score", context)
        self.assertIn("Relevant passage.", context)

    def test_prompt_dump_includes_messages_tools_and_tool_calls(self):
        assistant = Assistant(
            llm=FakeLLM(),
            history=FakeHistory([]),
            retriever=FakeRetriever(),
            tool_orchestrator=ToolOrchestrator([]),
        )
        tool_call = ToolCall(
            id="call-1",
            name="fake_tool",
            arguments={"value": 1},
        )
        messages = [
            Message(role="user", content="question"),
            Message(
                role="assistant",
                content="",
                tool_calls=[tool_call],
            ),
            Message(
                role="tool",
                content="tool result",
                tool_call_id="call-1",
            ),
        ]
        mocked_file = mock_open()

        with patch("builtins.open", mocked_file):
            assistant._dump_messages(messages, [FakeTool.definition])

        output = "".join(
            call.args[0]
            for call in mocked_file().write.call_args_list
        )
        self.assertIn("=== AVAILABLE TOOLS ===", output)
        self.assertIn("Name: fake_tool", output)
        self.assertIn("=== USER ===\nquestion", output)
        self.assertIn("Tool call: fake_tool", output)
        self.assertIn('Arguments: {"value": 1}', output)
        self.assertIn("=== TOOL ===\nTool call ID: call-1\ntool result", output)

    def test_retrieval_uses_prior_history_and_current_prompt_is_added_once(self):
        prior_history = [Message(role="user", content="Earlier question")]
        history = FakeHistory(prior_history)
        retriever = FakeRetriever()
        llm = FakeLLM()
        assistant = Assistant(
            llm=llm,
            history=history,
            retriever=retriever,
            tool_orchestrator=ToolOrchestrator([]),
        )
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
        self.assertIn(user_messages[0], llm.tool_calls[0][0])
        self.assertEqual(history.messages[-1], Message(role="assistant", content="answer"))

    def test_tool_call_is_executed_and_only_final_messages_are_persisted(self):
        history = FakeHistory([])
        call = ToolCall(id="call-1", name="fake_tool", arguments={"value": 1})
        llm = FakeLLM(
            tool_responses=[
                LLMResponse(content="", tool_calls=[call]),
                LLMResponse(content="final answer", tool_calls=[]),
            ]
        )
        tool = FakeTool()
        assistant = Assistant(
            llm=llm,
            history=history,
            retriever=FakeRetriever(),
            tool_orchestrator=ToolOrchestrator([tool]),
        )

        with patch.object(assistant, "_dump_messages"):
            chunks = list(assistant.stream_chat("question"))

        self.assertEqual(chunks, ["final answer"])
        self.assertEqual(tool.calls, [{"value": 1}])
        second_request = llm.tool_calls[1][0]
        self.assertEqual(second_request[-2].role, "assistant")
        self.assertEqual(second_request[-2].tool_calls, [call])
        self.assertEqual(second_request[-1].role, "tool")
        self.assertEqual(second_request[-1].tool_call_id, "call-1")
        self.assertEqual(second_request[-1].content, "tool output")
        self.assertEqual(
            history.messages,
            [
                Message(role="user", content="question"),
                Message(role="assistant", content="final answer"),
            ],
        )

    def test_tool_error_is_returned_to_model_without_being_persisted(self):
        history = FakeHistory([])
        call = ToolCall(id="call-1", name="fake_tool", arguments={})
        llm = FakeLLM(
            tool_responses=[
                LLMResponse(content="", tool_calls=[call]),
                LLMResponse(content="fallback answer", tool_calls=[]),
            ]
        )
        assistant = Assistant(
            llm=llm,
            history=history,
            retriever=FakeRetriever(),
            tool_orchestrator=ToolOrchestrator(
                [FakeTool(ToolResult("failed", is_error=True))]
            ),
        )

        with patch.object(assistant, "_dump_messages"):
            list(assistant.stream_chat("question"))

        self.assertEqual(llm.tool_calls[1][0][-1].content, "Tool error: failed")
        self.assertFalse(any(message.role == "tool" for message in history.messages))

    def test_tool_limit_uses_streaming_fallback_and_stores_completion(self):
        history = FakeHistory([])
        calls = [
            ToolCall(id=f"call-{index}", name="fake_tool", arguments={})
            for index in range(2)
        ]
        llm = FakeLLM(
            chunks=["one", " ", "two"],
            tool_responses=[
                LLMResponse(content="", tool_calls=[call]) for call in calls
            ],
        )
        assistant = Assistant(
            llm=llm,
            history=history,
            retriever=FakeRetriever(),
            tool_orchestrator=ToolOrchestrator([FakeTool()]),
        )

        with (
            patch("assistant.assistant.MAX_TOOL_CALLS", 2),
            patch.object(assistant, "_dump_messages"),
        ):
            chunks = list(assistant.stream_chat("question"))

        self.assertEqual(chunks, ["one", " ", "two"])
        self.assertEqual(history.messages[-1], Message(role="assistant", content="one two"))
        self.assertIsNotNone(llm.stream_messages)
        assert llm.stream_messages is not None
        self.assertEqual(llm.stream_messages[-1].role, "system")
        self.assertIn("tool-call limit", llm.stream_messages[-1].content)


if __name__ == "__main__":
    unittest.main()
