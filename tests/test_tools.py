# AI-generated with OpenAI Codex

import unittest
from collections.abc import Iterator
from unittest.mock import patch

from langchain_core.documents import Document

from config import WEB_SEARCH_DEPTH, WEB_SEARCH_MAX_RESULTS
from context.entities import ToolCall
from rag.indexer import Indexer
from tools.read_document import ReadDocumentTool
from tools.tool import Tool, ToolDefinition, ToolResult
from tools.tool_orchestrator import ToolOrchestrator
from tools.web_search import WebSearchTool


class FakeIndexer(Indexer):
    def __init__(
        self,
        documents: list[Document] | None = None,
        error: ValueError | None = None,
    ) -> None:
        self.documents = documents or []
        self.error = error
        self.filenames: list[str] = []

    def load_indexed_document(self, filename: str) -> list[Document]:
        self.filenames.append(filename)
        if self.error is not None:
            raise self.error
        return self.documents


class FakeTool(Tool):
    definition = ToolDefinition(
        name="fake_tool",
        description="A fake tool.",
        input_schema={"type": "object", "properties": {}},
    )

    def __init__(self) -> None:
        self.arguments: list[dict[str, object]] = []

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        self.arguments.append(arguments)
        return ToolResult("result")


class ToolOrchestratorTests(unittest.TestCase):
    def test_exposes_definitions_and_executes_named_tool(self):
        tool = FakeTool()
        orchestrator = ToolOrchestrator([tool])
        call = ToolCall(
            id="call-1",
            name="fake_tool",
            arguments={"value": 1},
        )

        result = orchestrator.execute(call)

        self.assertEqual(orchestrator.get_definitions(), [tool.definition])
        self.assertEqual(result, ToolResult("result"))
        self.assertEqual(tool.arguments, [{"value": 1}])

    def test_unknown_tool_returns_error_result(self):
        orchestrator = ToolOrchestrator([])

        result = orchestrator.execute(
            ToolCall(id="call-1", name="missing", arguments={})
        )

        self.assertTrue(result.is_error)
        self.assertIn("not found", result.content)

    def test_duplicate_tool_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            ToolOrchestrator([FakeTool(), FakeTool()])


class ReadDocumentToolTests(unittest.TestCase):
    def test_rejects_missing_blank_and_non_string_filenames(self):
        indexer = FakeIndexer()
        tool = ReadDocumentTool(indexer)

        invalid_arguments: Iterator[dict[str, object]] = iter(
            [{}, {"filename": ""}, {"filename": "   "}, {"filename": 42}]
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                result = tool.execute(arguments)
                self.assertTrue(result.is_error)
                self.assertIn("non-empty string", result.content)

        self.assertEqual(indexer.filenames, [])

    def test_loads_stripped_filename_and_formats_pages(self):
        indexer = FakeIndexer(
            documents=[
                Document(page_content="First page"),
                Document(page_content="Second page"),
            ]
        )
        tool = ReadDocumentTool(indexer)

        result = tool.execute({"filename": "  notes.pdf  "})

        self.assertFalse(result.is_error)
        self.assertEqual(indexer.filenames, ["notes.pdf"])
        self.assertEqual(
            result.content,
            "--- Page 1 ---\nFirst page\n\n--- Page 2 ---\nSecond page",
        )

    def test_truncates_oversized_document_and_marks_the_result(self):
        indexer = FakeIndexer(documents=[Document(page_content="abcdefghij")])
        tool = ReadDocumentTool(indexer)

        with patch("tools.read_document.TOOL_MAX_CONTENT_CHARS", 10):
            result = tool.execute({"filename": "notes.pdf"})

        self.assertFalse(result.is_error)
        self.assertTrue(result.content.startswith("--- Page 1"))
        self.assertIn("[Document truncated", result.content)

    def test_converts_expected_indexer_errors_to_tool_errors(self):
        tool = ReadDocumentTool(
            FakeIndexer(error=ValueError("No indexed document found"))
        )

        result = tool.execute({"filename": "missing.pdf"})

        self.assertTrue(result.is_error)
        self.assertEqual(result.content, "No indexed document found")


class WebSearchToolTests(unittest.TestCase):
    @patch("tools.web_search.TavilyClient")
    def test_searches_with_configured_options_and_formats_results(
        self,
        client_type,
    ):
        client = client_type.return_value
        client.search.return_value = {
            "results": [
                {
                    "title": "Example",
                    "url": "https://example.com",
                    "content": "Relevant information",
                }
            ]
        }
        tool = WebSearchTool()

        result = tool.execute({"query": "  current topic  "})

        self.assertFalse(result.is_error)
        self.assertIn("Title: Example", result.content)
        self.assertIn("URL: https://example.com", result.content)
        self.assertIn("Content: Relevant information", result.content)
        client.search.assert_called_once_with(
            query="current topic",
            search_depth=WEB_SEARCH_DEPTH,
            max_results=WEB_SEARCH_MAX_RESULTS,
            include_answer=False,
            include_raw_content=False,
        )

    @patch("tools.web_search.TavilyClient")
    def test_rejects_invalid_queries_without_searching(self, client_type):
        client = client_type.return_value
        tool = WebSearchTool()

        for arguments in ({}, {"query": "   "}, {"query": 42}):
            with self.subTest(arguments=arguments):
                result = tool.execute(arguments)
                self.assertTrue(result.is_error)
                self.assertIn("non-empty string", result.content)

        client.search.assert_not_called()

    @patch("tools.web_search.TavilyClient")
    def test_reports_empty_results_and_provider_errors(self, client_type):
        client = client_type.return_value
        tool = WebSearchTool()

        client.search.return_value = {"results": []}
        empty_result = tool.execute({"query": "unknown topic"})

        client.search.side_effect = RuntimeError("provider unavailable")
        error_result = tool.execute({"query": "current topic"})

        self.assertTrue(empty_result.is_error)
        self.assertIn("No web results", empty_result.content)
        self.assertTrue(error_result.is_error)
        self.assertIn("provider unavailable", error_result.content)

    @patch("tools.web_search.TavilyClient")
    def test_truncates_oversized_search_results(self, client_type):
        client_type.return_value.search.return_value = {
            "results": [
                {
                    "title": "Example",
                    "url": "https://example.com",
                    "content": "a" * 100,
                }
            ]
        }
        tool = WebSearchTool()

        with patch("tools.web_search.TOOL_MAX_CONTENT_CHARS", 20):
            result = tool.execute({"query": "topic"})

        self.assertFalse(result.is_error)
        self.assertIn("[Search results truncated", result.content)


if __name__ == "__main__":
    unittest.main()
