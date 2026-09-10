import logging

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import EMBEDDING_MODEL, EMBEDDING_PROVIDER, INDEX_PATH
from embedding.provider import EmbeddingProvider
from rag.index_paths import get_index_directory

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_provider.model

        self.index_path = get_index_directory(
            INDEX_PATH,
            EMBEDDING_PROVIDER,
            EMBEDDING_MODEL,
        )
        faiss_path = self.index_path / "index.faiss"
        metadata_path = self.index_path / "index.pkl"

        if faiss_path.exists() and metadata_path.exists():
            self.db = FAISS.load_local(
                str(self.index_path),
                self.embedding_model,
                allow_dangerous_deserialization=True,
            )
            logger.info("Loaded vector index (vectors=%d)", self.db.index.ntotal)
        elif not faiss_path.exists() and not metadata_path.exists():
            self.db = None
            logger.info("No existing vector index found")
        else:
            raise ValueError(
                "Vector index files are inconsistent. Both index.faiss and "
                "index.pkl must exist or neither."
            )
    def store(self, chunks: list[Document], embeddings: list[list[float]]) -> int:
        if not chunks:
            return 0

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        text_embeddings = list(zip(texts, embeddings))

        if self.db is None:
            self.db = FAISS.from_embeddings(
                text_embeddings=text_embeddings,
                embedding=self.embedding_model,
                metadatas=metadatas,
                ids=chunk_ids,
            )
        else:
            self.db.add_embeddings(
                text_embeddings=text_embeddings,
                embedding=self.embedding_model,
                metadatas=metadatas,
                ids=chunk_ids,
            )

        self.db.save_local(str(self.index_path))
        logger.info(
            "Stored vectors (added=%d, total=%d)",
            len(chunks),
            self.db.index.ntotal,
        )
        return len(chunks)
    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        if self.db is None:
            raise ValueError("Vector store is not initialized")

        embedding = self.embedding_provider.embed_query(query)

        return self.db.similarity_search_with_score_by_vector(embedding, k)

    def delete_by_ids(self, ids: list[str]) -> None:
        if not ids:
            return

        if self.db is None:
            raise ValueError("Vector store is not initialized")

        self.db.delete(ids)
        self.db.save_local(str(self.index_path))
        logger.info("Deleted vectors (count=%d)", len(ids))

    def get_by_ids(self, ids: list[str]) -> list[Document]:
        if not ids:
            return []

        if self.db is None:
            raise ValueError("Vector store is not initialized")

        return self.db.get_by_ids(ids)
