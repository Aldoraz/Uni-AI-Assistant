# AI-generated with OpenAI Codex

import unittest
from collections.abc import Iterator
from unittest.mock import patch

from langchain_core.documents import Document

from rag.indexer import Indexer
from tools.read_document import ReadDocumentTool


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


if __name__ == "__main__":
    unittest.main()
