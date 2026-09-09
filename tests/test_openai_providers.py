# AI-generated with OpenAI Codex

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)

from context.entities import Message, ToolCall
from embedding.openai import OpenAIEmbeddingProvider
from llm.openai import OpenAIChatProvider
from llm.provider import LLMResponse
from tools.tool import ToolDefinition


class OpenAIChatProviderTests(unittest.TestCase):
    @patch("llm.openai.ChatOpenAI")
    @patch("llm.openai.load_dotenv")
    def test_initializes_model_and_loads_environment(self, load_dotenv, chat_openai):
        provider = OpenAIChatProvider("test-model")

        load_dotenv.assert_called_once_with()
        chat_openai.assert_called_once_with(model="test-model")
        self.assertIs(provider.model, chat_openai.return_value)

    def test_chat_converts_messages_and_returns_content(self):
        provider = OpenAIChatProvider.__new__(OpenAIChatProvider)
        provider.model = Mock()
        provider.model.invoke.return_value = SimpleNamespace(content="answer")
        messages = [Message(role="user", content="question")]

        result = provider.chat(messages)

        self.assertEqual(result, "answer")
        provider.model.invoke.assert_called_once_with(
            [HumanMessage(content="question")]
        )

    def test_stream_chat_skips_empty_chunks(self):
        provider = OpenAIChatProvider.__new__(OpenAIChatProvider)
        provider.model = Mock()
        provider.model.stream.return_value = [
            SimpleNamespace(content="first"),
            SimpleNamespace(content=""),
            SimpleNamespace(content="second"),
        ]

        chunks = list(provider.stream_chat([Message(role="user", content="question")]))

        self.assertEqual(chunks, ["first", "second"])

    def test_converts_assistant_tool_calls_and_tool_results(self):
        provider = OpenAIChatProvider.__new__(OpenAIChatProvider)
        call = ToolCall(
            id="call-1",
            name="lookup",
            arguments={"query": "example"},
        )

        converted = provider._to_langchain_messages(
            [
                Message(role="assistant", content="", tool_calls=[call]),
                Message(role="tool", content="result", tool_call_id="call-1"),
            ]
        )

        self.assertIsInstance(converted[0], AIMessage)
        self.assertEqual(converted[0].tool_calls[0]["id"], "call-1")
        self.assertEqual(converted[0].tool_calls[0]["args"], {"query": "example"})
        self.assertIsInstance(converted[1], ToolMessage)
        self.assertEqual(converted[1].tool_call_id, "call-1")

    def test_tool_message_requires_call_id(self):
        provider = OpenAIChatProvider.__new__(OpenAIChatProvider)

        with self.assertRaisesRegex(ValueError, "tool_call_id"):
            provider._to_langchain_messages([Message(role="tool", content="result")])

    def test_chat_with_tools_binds_schemas_and_returns_structured_calls(self):
        provider = OpenAIChatProvider.__new__(OpenAIChatProvider)
        provider.model = Mock()
        bound_model = provider.model.bind_tools.return_value
        bound_model.invoke.return_value = SimpleNamespace(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "lookup",
                    "args": {"query": "example"},
                }
            ],
        )
        definition = ToolDefinition(
            name="lookup",
            description="Look something up.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        )

        response = provider.chat_with_tools(
            [Message(role="user", content="question")],
            [definition],
        )

        self.assertEqual(
            response,
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="lookup",
                        arguments={"query": "example"},
                    )
                ],
            ),
        )
        provider.model.bind_tools.assert_called_once_with(
            [
                {
                    "name": "lookup",
                    "description": "Look something up.",
                    "parameters": definition.input_schema,
                }
            ],
            strict=True,
            parallel_tool_calls=False,
        )
        bound_model.invoke.assert_called_once_with(
            [HumanMessage(content="question")]
        )

    def test_stream_chat_with_tools_yields_text_and_completed_tool_call(self):
        provider = OpenAIChatProvider.__new__(OpenAIChatProvider)
        provider.model = Mock()
        bound_model = provider.model.bind_tools.return_value
        bound_model.stream.return_value = [
            AIMessageChunk(content="Checking. "),
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "lookup",
                        "args": '{"query":',
                        "id": "call-1",
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            ),
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": None,
                        "args": '"example"}',
                        "id": None,
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            ),
        ]
        definition = ToolDefinition(
            name="lookup",
            description="Look something up.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        )

        events = list(
            provider.stream_chat_with_tools(
                [Message(role="user", content="question")],
                [definition],
            )
        )

        self.assertEqual(
            events,
            [
                LLMResponse(content="Checking. ", tool_calls=[]),
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="lookup",
                            arguments={"query": "example"},
                        )
                    ],
                ),
            ],
        )


class OpenAIEmbeddingProviderTests(unittest.TestCase):
    @patch("embedding.openai.OpenAIEmbeddings")
    @patch("embedding.openai.load_dotenv")
    def test_initializes_model_and_loads_environment(self, load_dotenv, embeddings):
        provider = OpenAIEmbeddingProvider("embedding-model")

        load_dotenv.assert_called_once_with()
        embeddings.assert_called_once_with(model="embedding-model")
        self.assertIs(provider.model, embeddings.return_value)

    def test_delegates_document_and_query_embedding(self):
        provider = OpenAIEmbeddingProvider.__new__(OpenAIEmbeddingProvider)
        provider.model = Mock()
        provider.model.embed_documents.return_value = [[1.0], [2.0]]
        provider.model.embed_query.return_value = [3.0]

        documents = provider.embed_documents(["one", "two"])
        query = provider.embed_query("question")

        self.assertEqual(documents, [[1.0], [2.0]])
        self.assertEqual(query, [3.0])
        provider.model.embed_documents.assert_called_once_with(["one", "two"])
        provider.model.embed_query.assert_called_once_with("question")


if __name__ == "__main__":
    unittest.main()
