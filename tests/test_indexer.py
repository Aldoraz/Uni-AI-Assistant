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
    IndexRecord,
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

            loaded = loader.load_documents([text_file, unsupported_file])

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

        loaded = loader.load_documents([file])

        self.assertEqual(loaded.documents, [])
        self.assertEqual(loaded.succeeded, set())
        self.assertEqual(loaded.failed, {file: "broken pdf"})
        self.assertEqual(result.files_failed, 1)

    def test_empty_loader_output_is_reported_as_skipped(self):
        result = IndexingResult()
        loader = DocumentLoader(result)
        file = Path("empty.txt")
        loader.SUPPORTED_TYPES[".txt"] = Mock(return_value=[])

        loaded = loader.load_documents([file])

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

    def test_get_records_returns_displayable_index_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "notes.txt"
            file.write_text("notes", encoding="utf-8")
            catalog = IndexCatalog(root / "index.db")
            changed, _ = catalog.compare_records([file], "signature")
            chunk = Document(page_content="notes", metadata={"source": str(file)})
            catalog.update_successful_records(changed, set(), [chunk], "signature")

            records = catalog.get_records()
            catalog.conn.close()

        self.assertEqual(
            records,
            [
                IndexRecord(
                    path=str(file),
                    chunk_count=1,
                    index_status="succeeded",
                    error_message=None,
                )
            ],
        )

    def test_get_records_by_filename_matches_exact_successful_name(self):
        catalog = IndexCatalog.__new__(IndexCatalog)
        records = [
            IndexRecord("documents/Notes.pdf", 2, "succeeded", None),
            IndexRecord("documents/old-notes.pdf", 2, "succeeded", None),
            IndexRecord("archive/notes.pdf", 0, "failed", "broken"),
        ]

        with patch.object(catalog, "get_records", return_value=records):
            actual = catalog.get_records_by_filename("notes.PDF")

        self.assertEqual(actual, [records[0]])


class IndexerTests(unittest.TestCase):
    @patch("rag.indexer.IndexCatalog")
    def test_index_folder_processes_only_indexable_changed_files(self, catalog_type):
        indexed_file = Path("documents/notes.txt")
        unsupported_file = Path("documents/video.mp4")
        deleted_file = "documents/old.txt"
        catalog = catalog_type.return_value
        changed_files = {indexed_file: "text-hash"}
        catalog.compare_records.return_value = (changed_files, {deleted_file})
        catalog.get_chunk_count.return_value = 0

        embedder = Mock()
        embedder.embed_documents.return_value = [[0.1]]
        vector_store = Mock()
        vector_store.store.return_value = 1
        indexer = Indexer(embedder, vector_store)
        document = Document(
            page_content="source", metadata={"source": str(indexed_file)}
        )
        load_result = DocumentLoadResult(
            documents=[document],
            succeeded={indexed_file},
            skipped={},
            failed={},
        )

        with (
            patch.object(
                indexer,
                "_discover_files",
                return_value=[indexed_file, unsupported_file],
            ),
            patch.object(
                indexer.loader,
                "load_documents",
                return_value=load_result,
            )
        ):
            result = indexer.index_folder(Path("documents"))

        catalog.compare_records.assert_called_once_with([indexed_file], ANY)
        stored_chunks = vector_store.store.call_args.args[0]
        self.assertEqual(stored_chunks[0].metadata["chunk_id"], f"{indexed_file}::0")
        successful_files = catalog.update_successful_records.call_args.args[0]
        self.assertEqual(successful_files, {indexed_file: "text-hash"})
        catalog.update_status_records.assert_called_once_with(
            {}, {}, changed_files, ANY
        )
        self.assertIs(result, indexer.result)

    @patch("rag.indexer.IndexCatalog")
    def test_get_index_status_delegates_to_catalog(self, catalog_type):
        expected = [IndexRecord("notes.txt", 2, "succeeded", None)]
        catalog_type.return_value.get_records.return_value = expected
        indexer = Indexer(Mock(), Mock())

        actual = indexer.get_index_status()

        self.assertIs(actual, expected)

    def test_load_indexed_document_rejects_missing_and_ambiguous_names(self):
        indexer = Indexer.__new__(Indexer)
        indexer.index_catalog = Mock()

        indexer.index_catalog.get_records_by_filename.return_value = []
        with self.assertRaisesRegex(ValueError, "No indexed document"):
            indexer.load_indexed_document("missing.pdf")

        indexer.index_catalog.get_records_by_filename.return_value = [
            IndexRecord("one/notes.pdf", 1, "succeeded", None),
            IndexRecord("two/notes.pdf", 1, "succeeded", None),
        ]
        with self.assertRaisesRegex(ValueError, "Multiple indexed documents"):
            indexer.load_indexed_document("notes.pdf")

    @patch("rag.indexer.DocumentLoader")
    def test_load_indexed_document_returns_loaded_documents(self, loader_type):
        path = Path("documents/notes.pdf")
        expected = [Document(page_content="content")]
        indexer = Indexer.__new__(Indexer)
        indexer.index_catalog = Mock()
        indexer.index_catalog.get_records_by_filename.return_value = [
            IndexRecord(str(path), 1, "succeeded", None)
        ]
        loader_type.return_value.load_documents.return_value = DocumentLoadResult(
            documents=expected,
            succeeded={path},
            skipped={},
            failed={},
        )

        actual = indexer.load_indexed_document("notes.pdf")

        self.assertIs(actual, expected)
        loader_type.return_value.load_documents.assert_called_once_with([path])

    @patch("rag.indexer.DocumentLoader")
    def test_load_indexed_document_surfaces_loader_failure(self, loader_type):
        path = Path("documents/broken.pdf")
        indexer = Indexer.__new__(Indexer)
        indexer.index_catalog = Mock()
        indexer.index_catalog.get_records_by_filename.return_value = [
            IndexRecord(str(path), 0, "succeeded", None)
        ]
        loader_type.return_value.load_documents.return_value = DocumentLoadResult(
            documents=[],
            succeeded=set(),
            skipped={},
            failed={path: "broken PDF"},
        )

        with self.assertRaisesRegex(ValueError, "broken PDF"):
            indexer.load_indexed_document("broken.pdf")


if __name__ == "__main__":
    unittest.main()
