# AI-generated with OpenAI Codex

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from rag.indexer import DocumentLoader, Indexer, IndexingResult


class DocumentLoaderTests(unittest.TestCase):
    def test_rejects_missing_or_non_directory_path(self):
        loader = DocumentLoader(IndexingResult())

        with self.assertRaisesRegex(ValueError, "not a valid directory"):
            loader.load_documents(Path("missing-directory"))

    def test_loads_txt_recursively_and_skips_unsupported_files(self):
        result = IndexingResult()
        loader = DocumentLoader(result)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            text_file = nested / "notes.TXT"
            text_file.write_text("course notes", encoding="utf-8")
            (root / "image.png").write_bytes(b"not an image")

            documents = loader.load_documents(root)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "course notes")
        self.assertEqual(documents[0].metadata["filename"], "notes.TXT")
        self.assertEqual(result.files_found, 2)
        self.assertEqual(result.files_loaded, 1)
        self.assertEqual(result.files_skipped, 1)
        self.assertEqual(result.documents_generated, 1)

    @patch("rag.indexer.PyPDFLoader")
    def test_pdf_loader_adds_filename_metadata(self, pdf_loader):
        result = IndexingResult()
        loader = DocumentLoader(result)
        document = Document(page_content="page", metadata={"page": 0})
        pdf_loader.return_value.load.return_value = [document]

        documents = loader._load_pdf(Path("lecture.pdf"))

        self.assertEqual(documents[0].metadata["filename"], "lecture.pdf")
        self.assertEqual(result.files_loaded, 1)

    @patch("rag.indexer.PyPDFLoader")
    def test_pdf_errors_are_counted_and_skipped(self, pdf_loader):
        result = IndexingResult()
        loader = DocumentLoader(result)
        pdf_loader.return_value.load.side_effect = RuntimeError("broken pdf")

        documents = loader._load_pdf(Path("broken.pdf"))

        self.assertEqual(documents, [])
        self.assertEqual(result.files_failed, 1)


class IndexerTests(unittest.TestCase):
    def test_index_folder_runs_complete_pipeline_and_updates_counts(self):
        embedder = Mock()
        embedder.embed_documents.return_value = [[0.1], [0.2]]
        vector_store = Mock()
        vector_store.store.return_value = 2
        indexer = Indexer(embedder, vector_store)
        documents = [Document(page_content="source")]
        chunks = [Document(page_content="one"), Document(page_content="two")]
        indexer.loader = Mock()
        indexer.loader.load_documents.return_value = documents
        indexer.splitter = Mock()
        indexer.splitter.split_documents.return_value = chunks

        indexer.index_folder(Path("documents"))

        indexer.loader.load_documents.assert_called_once_with(Path("documents"))
        indexer.splitter.split_documents.assert_called_once_with(documents)
        embedder.embed_documents.assert_called_once_with(["one", "two"])
        vector_store.store.assert_called_once_with(chunks, [[0.1], [0.2]])
        self.assertEqual(indexer.result.chunks_generated, 2)
        self.assertEqual(indexer.result.embeddings_generated, 2)
        self.assertEqual(indexer.result.vectors_stored, 2)


if __name__ == "__main__":
    unittest.main()
