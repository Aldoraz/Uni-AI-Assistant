# AI-generated with OpenAI Codex

import unittest
from unittest.mock import Mock, patch

import app


class CreateProvidersTests(unittest.TestCase):
    @patch("app.ToolOrchestrator")
    @patch("app.WebSearchTool")
    @patch("app.ReadDocumentTool")
    def test_creates_tool_orchestrator_with_available_tools(
        self,
        read_document_tool,
        web_search_tool,
        orchestrator,
    ):
        indexer = Mock(spec=app.Indexer)

        with (
            patch.object(app, "WEB_SEARCH_ENABLED", True),
            patch("app.load_dotenv") as load_dotenv,
            patch("app.os.getenv", return_value="tavily-key"),
        ):
            result = app.create_tool_orchestrator(indexer)

        load_dotenv.assert_called_once_with()
        read_document_tool.assert_called_once_with(indexer)
        web_search_tool.assert_called_once_with()
        orchestrator.assert_called_once_with(
            tools=[
                read_document_tool.return_value,
                web_search_tool.return_value,
            ]
        )
        self.assertIs(result, orchestrator.return_value)

    @patch("app.ToolOrchestrator")
    @patch("app.WebSearchTool")
    @patch("app.ReadDocumentTool")
    def test_omits_web_search_when_disabled(
        self,
        read_document_tool,
        web_search_tool,
        orchestrator,
    ):
        indexer = Mock(spec=app.Indexer)

        with patch.object(app, "WEB_SEARCH_ENABLED", False):
            app.create_tool_orchestrator(indexer)

        web_search_tool.assert_not_called()
        orchestrator.assert_called_once_with(
            tools=[read_document_tool.return_value]
        )

    @patch("app.ToolOrchestrator")
    @patch("app.WebSearchTool")
    @patch("app.ReadDocumentTool")
    def test_omits_web_search_without_api_key(
        self,
        read_document_tool,
        web_search_tool,
        orchestrator,
    ):
        indexer = Mock(spec=app.Indexer)

        with (
            patch.object(app, "WEB_SEARCH_ENABLED", True),
            patch("app.load_dotenv"),
            patch("app.os.getenv", return_value="  "),
        ):
            app.create_tool_orchestrator(indexer)

        web_search_tool.assert_not_called()
        orchestrator.assert_called_once_with(
            tools=[read_document_tool.return_value]
        )

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
            patch.object(app, "OLLAMA_BASE_URL", "http://ollama.test"),
            patch.object(app, "OLLAMA_CONTEXT_WINDOW", 8192),
            patch.object(app, "OLLAMA_EMBEDDING_BATCH_SIZE", 64),
            patch.object(app, "OLLAMA_REASONING_ENABLED", False),
            patch.object(app, "OLLAMA_KEEP_ALIVE", "15m"),
        ):
            llm, embedder = app.create_providers()

        chat_provider.assert_called_once_with(
            model="chat-model",
            base_url="http://ollama.test",
            context_window=8192,
            reasoning=False,
            keep_alive="15m",
        )
        embedding_provider.assert_called_once_with(
            model="embedding-model",
            base_url="http://ollama.test",
            batch_size=64,
        )
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
