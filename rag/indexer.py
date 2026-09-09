import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedding.provider import EmbeddingProvider
from rag.index_paths import get_index_directory
from rag.vector_store import VectorStore
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_PROVIDER, EMBEDDING_MODEL, INDEX_PATH

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    files_found: int = 0
    files_loaded: int = 0
    files_skipped: int = 0
    files_failed: int = 0

    documents_generated: int = 0
    chunks_generated: int = 0
    embeddings_generated: int = 0
    vectors_stored: int = 0

@dataclass
class DocumentLoadResult:
    documents: list[Document]
    succeeded: set[Path]
    skipped: dict[Path, str]
    failed: dict[Path, str]

@dataclass
class IndexRecord:
    path: str
    chunk_count: int
    index_status: str
    error_message: str | None


class Indexer:
    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: VectorStore):
        self.result = IndexingResult()
        self.loader = DocumentLoader(self.result)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP)
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.index_catalog = IndexCatalog()

    def index_folder(self, folder: Path) -> IndexingResult:
        logger.info("Index update started (folder=%s)", folder)
        # Find all files in the folder and subfolders
        files = self._discover_files(folder)

        supported_files = [
            file
            for file in files
            if file.suffix.lower() in self.loader.SUPPORTED_TYPES
        ]
        logger.info(
            "Files discovered (count=%d): [%s]",
            len(files),
            ", ".join(sorted(file.name for file in files)),
        )
        logger.info(
            "Supported documents (count=%d): [%s]",
            len(supported_files),
            ", ".join(sorted(file.name for file in supported_files)),
        )

        # Check which files have changed since last indexing
        signature = json.dumps({
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_provider": EMBEDDING_PROVIDER,
            "embedding_model": EMBEDDING_MODEL
        }, sort_keys=True)
        changed_files, deleted_files = self.index_catalog.compare_records(supported_files, signature)
        logger.info(
            "Index changes: changed=[%s], deleted=[%s]",
            ", ".join(sorted(file.name for file in changed_files)),
            ", ".join(sorted(Path(file).name for file in deleted_files)),
        )

        # Recursively find and turn all files into langchain Documents
        load_results = self.loader.load_documents(list(changed_files))

        # Split documents into chunks with certain overlap
        chunks = self._chunk_documents(load_results.documents)
        chunk_counts: dict[str, int] = {}
        for chunk in chunks:
            source = str(chunk.metadata.get("source", "Unknown"))
            chunk_counts[source] = chunk_counts.get(source, 0) + 1
        logger.info(
            "Document chunks generated (total=%d): [%s]",
            len(chunks),
            ", ".join(
                f"{Path(source).name}={count}"
                for source, count in chunk_counts.items()
            ),
        )

        # Determine which files are being indexed (for deletion of old records)
        indexed_sources = {chunk.metadata["source"] for chunk in chunks}
        indexed_files = {file: content_hash for file, content_hash in changed_files.items() if str(file) in indexed_sources}

        # Turn chunks into embeddings (vectors)
        embeddings = self._embed_chunks(chunks)

        # Delete old and changed records from the vector store
        for file in indexed_files:
            chunk_count = self.index_catalog.get_chunk_count(str(file))
            logger.info(
                "Replacing document vectors (document=%s, previous_vectors=%d)",
                file.name,
                chunk_count,
            )
            self.vector_store.delete_by_ids([f"{file}::{i}" for i in range(chunk_count)])
        for file in deleted_files:
            chunk_count = self.index_catalog.get_chunk_count(file)
            logger.info(
                "Removing deleted document vectors (document=%s, vectors=%d)",
                Path(file).name,
                chunk_count,
            )
            self.vector_store.delete_by_ids([f"{file}::{i}" for i in range(chunk_count)])
        for file in load_results.skipped:
            chunk_count = self.index_catalog.get_chunk_count(str(file))
            logger.info(
                "Removing vectors for skipped document "
                "(document=%s, vectors=%d)",
                file.name,
                chunk_count,
            )
            self.vector_store.delete_by_ids([f"{file}::{i}" for i in range(chunk_count)])

        # Store new/updated chunks (data) and embeddings (vectors)
        self.result.vectors_stored = self.vector_store.store(chunks, embeddings)

        # Update the index catalog with the new records and delete the old ones
        self.index_catalog.update_successful_records(indexed_files, deleted_files, chunks, signature)

        # Update the index catalog with skipped and failed records
        self.index_catalog.update_status_records(load_results.skipped, load_results.failed, changed_files, signature)

        logger.info(
            "Index catalog updated: succeeded=[%s], skipped=[%s], "
            "failed=[%s], deleted=[%s]",
            ", ".join(sorted(file.name for file in indexed_files)),
            ", ".join(sorted(file.name for file in load_results.skipped)),
            ", ".join(sorted(file.name for file in load_results.failed)),
            ", ".join(sorted(Path(file).name for file in deleted_files)),
        )

        logger.info(
            "Index update completed (loaded=%d, skipped=%d, failed=%d, vectors=%d)",
            self.result.files_loaded,
            self.result.files_skipped,
            self.result.files_failed,
            self.result.vectors_stored,
        )
        return self.result

    def _discover_files(self, folder: Path) -> list[Path]:
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Provided path {folder} is not a valid directory.")

        files: list[Path] = []
        for file in folder.rglob("*"):
            if file.is_file():
                files.append(file)
                self.result.files_found += 1

        return files

    def _chunk_documents(self, documents: list[Document]) -> list[Document]:
        chunks = self.splitter.split_documents(documents)
        self.result.chunks_generated = len(chunks)

        count = 0
        current_source = None
        for chunk in chunks:
            source = chunk.metadata["source"]
            if source != current_source:
                current_source = source
                count = 0
            chunk.metadata["chunk_id"] = f"{source}::{count}"
            count += 1
        return chunks

    def _embed_chunks(self, chunks: list[Document]) -> list[list[float]]:
        if not chunks:
            logger.info("Embedding generation skipped: no chunks")
            return []

        vectors = self.embedding_provider.embed_documents(
            [chunk.page_content for chunk in chunks]
        )
        self.result.embeddings_generated = len(vectors)
        logger.info("Embeddings generated (count=%d)", len(vectors))
        return vectors

    def get_index_status(self) -> list[IndexRecord]:
        return self.index_catalog.get_records()

    def load_indexed_document(self, filename: str) -> list[Document]:
        indexed_files = self.index_catalog.get_records_by_filename(filename)

        if not indexed_files:
            raise ValueError(f"No indexed document found with filename: {filename}")

        if len(indexed_files) > 1:
            raise ValueError(
                f"Multiple indexed documents found with filename: {filename}."
            )

        path = Path(indexed_files[0].path)
        loader = DocumentLoader(IndexingResult())
        load_result = loader.load_documents([path])

        if path in load_result.failed:
            raise ValueError(
                f"Could not load '{filename}': {load_result.failed[path]}"
            )

        if path in load_result.skipped:
            raise ValueError(
                f"Could not read '{filename}': {load_result.skipped[path]}"
            )

        if not load_result.documents:
            raise ValueError(
                f"Loading '{filename}' produced no content."
            )

        return load_result.documents


class DocumentLoader:
    def __init__(self, result: IndexingResult):
        self.result = result
        self.SUPPORTED_TYPES = {
            ".pdf": self._load_pdf,
            ".txt": self._load_txt,
            # TODO: MP4 -> Whisper Transcript -> Index as .txt
            # TODO images??
        }

    def load_documents(self, files: list[Path]) -> DocumentLoadResult:
        load_results: DocumentLoadResult = DocumentLoadResult(
            documents=[],
            succeeded=set(),
            skipped={},
            failed={}
        )
        for file in files:
            loader = self.SUPPORTED_TYPES.get(file.suffix.lower())
            if loader is None:
                logger.warning("Skipping unsupported file: %s", file.name)
                load_results.skipped[file] = f"Unsupported file type {file.suffix.lower()}"
                self.result.files_skipped += 1
                continue
            try:
                documents = loader(file)
                if not documents:
                    logger.warning("Skipping empty document: %s", file.name)
                    load_results.skipped[file] = "No documents produced"
                    self.result.files_skipped += 1
                    continue
                load_results.documents.extend(documents)
                load_results.succeeded.add(file)
            except Exception as error:
                logger.error("Document loading failed (%s): %s", file.name, error)
                load_results.failed[file] = str(error)
                self.result.files_failed += 1

        self.result.documents_generated = len(load_results.documents)
        return load_results

    def _load_pdf(self, file: Path) -> list[Document]:
        logger.info("Loading PDF document: %s", file.name)
        # TODO: Visual Search
        documents = PyPDFLoader(str(file)).load()

        for doc in documents:
            doc.metadata["filename"] = file.name
            doc.metadata["source"] = str(file)
        self.result.files_loaded += 1
        return documents

    def _load_txt(self, file: Path) -> list[Document]:
        logger.info("Loading TXT document: %s", file.name)
        with open(file, encoding="utf-8") as f:
            text = f.read()
        self.result.files_loaded += 1
        return [
            Document(
                page_content=text,
                metadata={
                    "filename": file.name,
                    "source": str(file)
                }
            )
        ]


class IndexCatalog:
    def __init__(self, index_path: Path | None = None):
        if index_path is None:
            self.index_path = get_index_directory(
                INDEX_PATH,
                EMBEDDING_PROVIDER,
                EMBEDDING_MODEL,
            ) / "catalog.db"
        else:
            self.index_path = index_path
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.index_path)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS index_metadata (
                path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                index_status TEXT NOT NULL,
                error_message TEXT,
                config_signature TEXT NOT NULL
            );""")

    def compare_records(self, files: list[Path], config_signature: str) -> tuple[dict[Path, str], set[str]]:
        cursor = self.conn.cursor()

        changed_files: dict[Path, str] = {}
        for file in files:
            path = str(file)
            content_hash = self._calculate_content_hash(file)

            cursor.execute("SELECT content_hash, config_signature, index_status FROM index_metadata WHERE path = ?", (path,))
            record = cursor.fetchone()

            if record is None or \
                record[0] !=  content_hash or \
                record[1] != config_signature or \
                record[2] == "failed":

                changed_files[file] = content_hash

        # Remove records for files that no longer exist
        cursor.execute("SELECT path FROM index_metadata")
        all_records = {row[0] for row in cursor.fetchall()}
        deleted_files = all_records - {str(file) for file in files}

        return changed_files, deleted_files

    def delete_record(self, path: str) -> None:
        self.conn.execute("DELETE FROM index_metadata WHERE path = ?", (path,))
        self.conn.commit()

    def _calculate_content_hash(self, file: Path) -> str:
        hasher = hashlib.sha256()

        with open(file, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)

        return hasher.hexdigest()

    def update_successful_records(self, changed_files: dict[Path, str], deleted_files: set[str], chunks: list[Document], config_signature: str) -> None:
        cursor = self.conn.cursor()

        for file, content_hash in changed_files.items():
            path = str(file)
            chunk_count = len([chunk for chunk in chunks if chunk.metadata["source"] == path])
            index_status = "succeeded"
            error_message = None

            cursor.execute("""
                INSERT INTO index_metadata (path, content_hash, chunk_count, index_status, error_message, config_signature)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    chunk_count=excluded.chunk_count,
                    index_status=excluded.index_status,
                    error_message=excluded.error_message,
                    config_signature=excluded.config_signature;
            """, (path, content_hash, chunk_count, index_status, error_message, config_signature))

        for path in deleted_files:
            cursor.execute("DELETE FROM index_metadata WHERE path = ?", (path,))

        self.conn.commit()

    def update_status_records(self, skipped: dict[Path, str], failed: dict[Path, str], changed_files: dict[Path, str], config_signature: str,) -> None:
        cursor = self.conn.cursor()

        for file, error_message in skipped.items():
            path = str(file)
            content_hash = changed_files[file]
            chunk_count = 0
            index_status = "skipped"

            cursor.execute("""
                INSERT INTO index_metadata (path, content_hash, chunk_count, index_status, error_message, config_signature)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    chunk_count=excluded.chunk_count,
                    index_status=excluded.index_status,
                    error_message=excluded.error_message,
                    config_signature=excluded.config_signature;
            """, (path, content_hash, chunk_count, index_status, error_message, config_signature))

        for file, error_message in failed.items():
            path = str(file)
            content_hash = changed_files[file]
            chunk_count = 0
            index_status = "failed"

            cursor.execute("""
                INSERT INTO index_metadata (path, content_hash, chunk_count, index_status, error_message, config_signature)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    index_status=excluded.index_status,
                    error_message=excluded.error_message;
            """, (path, content_hash, chunk_count, index_status, error_message, config_signature))

        self.conn.commit()

    def get_chunk_count(self, path: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT chunk_count FROM index_metadata WHERE path = ?", (path,))
        record = cursor.fetchone()
        return record[0] if record else 0

    def get_records(self) -> list[IndexRecord]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT path, chunk_count, index_status, error_message FROM index_metadata ORDER BY path ASC")
        records = cursor.fetchall()
        return [IndexRecord(path=row[0], chunk_count=row[1], index_status=row[2], error_message=row[3]) for row in records]

    def get_records_by_filename(self, filename: str) -> list[IndexRecord]:
        return [
            record
            for record in self.get_records()
            if record.index_status == "succeeded"
            and Path(record.path).name.casefold() == filename.casefold()
        ]
