# AI-generated with OpenAI Codex

import unittest
from unittest.mock import patch

import app


class CreateProvidersTests(unittest.TestCase):
    @patch("app.OpenAIEmbeddingProvider")
    @patch("app.OpenAIChatProvider")
    def test_creates_openai_providers(self, chat_provider, embedding_provider):
        with (
            patch.object(app, "CHAT_PROVIDER", "openai"),
            patch.object(app, "CHAT_MODEL", "chat-model"),
            patch.object(app, "EMBEDDING_PROVIDER", "openai"),
            patch.object(app, "EMBEDDING_MODEL", "embedding-model"),
        ):
            llm, embedder = app.create_providers()

        chat_provider.assert_called_once_with(model="chat-model")
        embedding_provider.assert_called_once_with(model="embedding-model")
        self.assertIs(llm, chat_provider.return_value)
        self.assertIs(embedder, embedding_provider.return_value)

    @patch("app.OllamaEmbeddingProvider")
    @patch("app.OllamaChatProvider")
    def test_creates_configured_ollama_provider_objects(
        self, chat_provider, embedding_provider
    ):
        with (
            patch.object(app, "CHAT_PROVIDER", "ollama"),
            patch.object(app, "CHAT_MODEL", "chat-model"),
            patch.object(app, "EMBEDDING_PROVIDER", "ollama"),
            patch.object(app, "EMBEDDING_MODEL", "embedding-model"),
        ):
            llm, embedder = app.create_providers()

        chat_provider.assert_called_once_with(model="chat-model")
        embedding_provider.assert_called_once_with(model="embedding-model")
        self.assertIs(llm, chat_provider.return_value)
        self.assertIs(embedder, embedding_provider.return_value)

    def test_rejects_unsupported_chat_provider(self):
        with patch.object(app, "CHAT_PROVIDER", "unsupported"):
            with self.assertRaisesRegex(ValueError, "Unsupported chat provider"):
                app.create_providers()

    @patch("app.OpenAIChatProvider")
    def test_rejects_unsupported_embedding_provider(self, chat_provider):
        with (
            patch.object(app, "CHAT_PROVIDER", "openai"),
            patch.object(app, "EMBEDDING_PROVIDER", "unsupported"),
        ):
            with self.assertRaisesRegex(ValueError, "Unsupported embedding provider"):
                app.create_providers()


if __name__ == "__main__":
    unittest.main()
