# AI-generated with OpenAI Codex

import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, Mock, patch

from langchain_core.documents import Document

from rag.indexer import (
    DocumentLoader,
    DocumentLoadResult,
    IndexCatalog,
    Indexer,
    IndexingResult,
)


class DocumentLoaderTests(unittest.TestCase):
    def test_loads_supported_files_and_reports_unsupported_files(self):
        result = IndexingResult()
        loader = DocumentLoader(result)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text_file = root / "notes.TXT"
            unsupported_file = root / "image.png"
            text_file.write_text("course notes", encoding="utf-8")
            unsupported_file.write_bytes(b"not an image")

            loaded = loader.load_documents(
                {text_file: "text-hash", unsupported_file: "image-hash"}
            )

        self.assertEqual(len(loaded.documents), 1)
        self.assertEqual(loaded.documents[0].page_content, "course notes")
        self.assertEqual(loaded.documents[0].metadata["filename"], "notes.TXT")
        self.assertEqual(loaded.documents[0].metadata["source"], str(text_file))
        self.assertEqual(loaded.succeeded, {text_file})
        self.assertIn(unsupported_file, loaded.skipped)
        self.assertEqual(loaded.failed, {})
        self.assertEqual(result.files_loaded, 1)
        self.assertEqual(result.files_skipped, 1)
        self.assertEqual(result.documents_generated, 1)

    @patch("rag.indexer.PyPDFLoader")
    def test_pdf_loader_adds_source_metadata(self, pdf_loader):
        loader = DocumentLoader(IndexingResult())
        document = Document(page_content="page", metadata={"page": 0})
        pdf_loader.return_value.load.return_value = [document]

        documents = loader._load_pdf(Path("lecture.pdf"))

        self.assertEqual(documents[0].metadata["filename"], "lecture.pdf")
        self.assertEqual(documents[0].metadata["source"], "lecture.pdf")

    @patch("rag.indexer.PyPDFLoader")
    def test_load_errors_are_reported_per_file(self, pdf_loader):
        result = IndexingResult()
        loader = DocumentLoader(result)
        file = Path("broken.pdf")
        pdf_loader.return_value.load.side_effect = RuntimeError("broken pdf")

        loaded = loader.load_documents({file: "hash"})

        self.assertEqual(loaded.documents, [])
        self.assertEqual(loaded.succeeded, set())
        self.assertEqual(loaded.failed, {file: "broken pdf"})
        self.assertEqual(result.files_failed, 1)

    def test_empty_loader_output_is_reported_as_skipped(self):
        result = IndexingResult()
        loader = DocumentLoader(result)
        file = Path("empty.txt")
        loader.SUPPORTED_TYPES[".txt"] = Mock(return_value=[])

        loaded = loader.load_documents({file: "hash"})

        self.assertEqual(loaded.documents, [])
        self.assertEqual(loaded.skipped, {file: "No documents produced"})
        self.assertEqual(result.files_skipped, 1)


class IndexCatalogTests(unittest.TestCase):
    def test_successful_record_is_unchanged_until_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "notes.txt"
            file.write_text("first", encoding="utf-8")
            catalog = IndexCatalog(root / "index.db")

            changed, deleted = catalog.compare_records([file], "signature")
            chunk = Document(page_content="first", metadata={"source": str(file)})
            catalog.update_successful_records(changed, deleted, [chunk], "signature")

            unchanged, deleted = catalog.compare_records([file], "signature")
            file.write_text("second", encoding="utf-8")
            changed_again, _ = catalog.compare_records([file], "signature")
            catalog.conn.close()

        self.assertEqual(unchanged, {})
        self.assertEqual(deleted, set())
        self.assertIn(file, changed_again)

    def test_failed_records_are_retried_but_skipped_records_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failed_file = root / "broken.pdf"
            skipped_file = root / "unsupported.docx"
            failed_file.write_bytes(b"broken")
            skipped_file.write_bytes(b"unsupported")
            catalog = IndexCatalog(root / "index.db")
            changed, _ = catalog.compare_records(
                [failed_file, skipped_file], "signature"
            )
            catalog.update_status_records(
                {skipped_file: "unsupported"},
                {failed_file: "broken"},
                changed,
                "signature",
            )

            changed_again, _ = catalog.compare_records(
                [failed_file, skipped_file], "signature"
            )
            catalog.conn.close()

        self.assertIn(failed_file, changed_again)
        self.assertNotIn(skipped_file, changed_again)

    def test_missing_files_are_returned_as_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "notes.txt"
            file.write_text("notes", encoding="utf-8")
            catalog = IndexCatalog(root / "index.db")
            changed, _ = catalog.compare_records([file], "signature")
            chunk = Document(page_content="notes", metadata={"source": str(file)})
            catalog.update_successful_records(changed, set(), [chunk], "signature")

            _, deleted = catalog.compare_records([], "signature")
            catalog.conn.close()

        self.assertEqual(deleted, {str(file)})


class IndexerTests(unittest.TestCase):
    @patch("rag.indexer.IndexCatalog")
    def test_index_folder_processes_only_indexable_changed_files(self, catalog_type):
        indexed_file = Path("documents/notes.txt")
        skipped_file = Path("documents/image.png")
        deleted_file = "documents/old.txt"
        catalog = catalog_type.return_value
        changed_files = {
            indexed_file: "text-hash",
            skipped_file: "image-hash",
        }
        catalog.compare_records.return_value = (changed_files, {deleted_file})
        catalog.get_chunk_count.return_value = 0

        embedder = Mock()
        embedder.embed_documents.return_value = [[0.1]]
        vector_store = Mock()
        vector_store.store.return_value = 1
        indexer = Indexer(embedder, vector_store)
        indexer._discover_files = Mock(return_value=[indexed_file, skipped_file])
        document = Document(
            page_content="source", metadata={"source": str(indexed_file)}
        )
        indexer.loader = Mock()
        indexer.loader.load_documents.return_value = DocumentLoadResult(
            documents=[document],
            succeeded={indexed_file},
            skipped={skipped_file: "unsupported"},
            failed={},
        )

        indexer.index_folder(Path("documents"))

        stored_chunks = vector_store.store.call_args.args[0]
        self.assertEqual(stored_chunks[0].metadata["chunk_id"], f"{indexed_file}::0")
        successful_files = catalog.update_successful_records.call_args.args[0]
        self.assertEqual(successful_files, {indexed_file: "text-hash"})
        catalog.update_status_records.assert_called_once_with(
            {skipped_file: "unsupported"}, {}, changed_files, ANY
        )


if __name__ == "__main__":
    unittest.main()
