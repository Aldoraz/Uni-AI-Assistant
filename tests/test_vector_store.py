# AI-generated with OpenAI Codex

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from embedding.provider import EmbeddingProvider
from rag.vector_store import VectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.model: Embeddings = Mock(spec=Embeddings)
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2]


class VectorStoreTests(unittest.TestCase):
    def test_initializes_without_database_when_index_is_missing(self):
        provider = FakeEmbeddingProvider()
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "missing-index"
            with patch("rag.vector_store.INDEX_PATH", index_path):
                store = VectorStore(provider)

        self.assertIsNone(store.db)

    @patch("rag.vector_store.FAISS")
    def test_loads_existing_index(self, faiss):
        provider = FakeEmbeddingProvider()
        faiss.load_local.return_value.index.ntotal = 3
        with tempfile.TemporaryDirectory() as directory:
            index_root = Path(directory)
            index_path = index_root / "test-provider--test-model"
            index_path.mkdir()
            (index_path / "index.faiss").touch()
            (index_path / "index.pkl").touch()
            with patch.multiple(
                "rag.vector_store",
                INDEX_PATH=index_root,
                EMBEDDING_PROVIDER="test-provider",
                EMBEDDING_MODEL="test-model",
            ):
                store = VectorStore(provider)

        faiss.load_local.assert_called_once_with(
            str(index_path), provider.model, allow_dangerous_deserialization=True
        )
        self.assertIs(store.db, faiss.load_local.return_value)

    @patch("rag.vector_store.FAISS")
    def test_store_builds_and_persists_faiss_database(self, faiss):
        provider = FakeEmbeddingProvider()
        store = VectorStore.__new__(VectorStore)
        store.embedding_provider = provider
        store.embedding_model = provider.model
        store.index_path = Path("index-path")
        store.db = None
        documents = [
            Document(page_content="one", metadata={"file": "a", "chunk_id": "a::0"}),
            Document(page_content="two", metadata={"file": "b", "chunk_id": "b::0"}),
        ]
        database = Mock()
        faiss.from_embeddings.return_value = database

        count = store.store(documents, [[1.0], [2.0]])

        faiss.from_embeddings.assert_called_once_with(
            text_embeddings=[("one", [1.0]), ("two", [2.0])],
            embedding=provider.model,
            metadatas=[
                {"file": "a", "chunk_id": "a::0"},
                {"file": "b", "chunk_id": "b::0"},
            ],
            ids=["a::0", "b::0"],
        )
        database.save_local.assert_called_once_with("index-path")
        self.assertEqual(count, 2)
        self.assertIs(store.db, database)

    def test_rejects_incomplete_index(self):
        provider = FakeEmbeddingProvider()
        with tempfile.TemporaryDirectory() as directory:
            index_root = Path(directory)
            index_path = index_root / "test-provider--test-model"
            index_path.mkdir()
            (index_path / "index.faiss").touch()

            with (
                patch.multiple(
                    "rag.vector_store",
                    INDEX_PATH=index_root,
                    EMBEDDING_PROVIDER="test-provider",
                    EMBEDDING_MODEL="test-model",
                ),
                self.assertRaisesRegex(ValueError, "inconsistent"),
            ):
                VectorStore(provider)

    def test_store_with_no_chunks_is_a_no_op(self):
        store = VectorStore.__new__(VectorStore)
        store.db = None

        self.assertEqual(store.store([], []), 0)
        self.assertIsNone(store.db)

    def test_delete_with_no_ids_is_a_no_op(self):
        store = VectorStore.__new__(VectorStore)
        store.db = None

        store.delete_by_ids([])

        self.assertIsNone(store.db)

    def test_delete_requires_initialized_database_for_nonempty_ids(self):
        store = VectorStore.__new__(VectorStore)
        store.db = None

        with self.assertRaisesRegex(ValueError, "not initialized"):
            store.delete_by_ids(["a::0"])

    def test_get_by_ids_with_no_ids_is_a_no_op(self):
        store = VectorStore.__new__(VectorStore)
        store.db = None

        self.assertEqual(store.get_by_ids([]), [])

    def test_get_by_ids_requires_initialized_database(self):
        store = VectorStore.__new__(VectorStore)
        store.db = None

        with self.assertRaisesRegex(ValueError, "not initialized"):
            store.get_by_ids(["a::0"])

    def test_get_by_ids_delegates_to_database(self):
        store = VectorStore.__new__(VectorStore)
        store.db = Mock()
        expected = [Document(page_content="match")]
        store.db.get_by_ids.return_value = expected

        actual = store.get_by_ids(["a::0"])

        self.assertIs(actual, expected)
        store.db.get_by_ids.assert_called_once_with(["a::0"])

    def test_search_requires_initialized_database(self):
        store = VectorStore.__new__(VectorStore)
        store.db = None

        with self.assertRaisesRegex(ValueError, "not initialized"):
            store.search("query", 5)

    def test_search_embeds_query_and_delegates_to_database(self):
        provider = FakeEmbeddingProvider()
        store = VectorStore.__new__(VectorStore)
        store.embedding_provider = provider
        store.db = Mock()
        expected = [(Document(page_content="match"), 0.2)]
        store.db.similarity_search_with_score_by_vector.return_value = expected

        actual = store.search("query", 4)

        self.assertIs(actual, expected)
        self.assertEqual(provider.queries, ["query"])
        store.db.similarity_search_with_score_by_vector.assert_called_once_with(
            [0.1, 0.2], 4
        )


if __name__ == "__main__":
    unittest.main()
