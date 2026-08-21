# AI-generated with OpenAI Codex

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from context.entities import Message
from embedding.openai import OpenAIEmbeddingProvider
from llm.openai import OpenAIChatProvider


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
            [{"role": "user", "content": "question"}]
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
