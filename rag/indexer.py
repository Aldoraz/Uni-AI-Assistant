from pathlib import Path
from dataclasses import dataclass
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

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

class Indexer:
    def __init__(self):
        self.result = IndexingResult()
        self.loader = DocumentLoader(self.result)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP)
        
    def index_folder(self, folder: Path) -> None:
        # Recursively find and turn all files into langchain Documents
        documents = self.loader.load_documents(folder)
        
        chunks = self._chunk_documents(documents)
        
        embeddings = self._embed_chunks(chunks)
        
        self._store_embeddings(embeddings)
        
        print(self.result)
        
    def _chunk_documents(self, documents: list[Document]) -> list[Document]:
        chunks = self.splitter.split_documents(documents)
        self.result.chunks_generated = len(chunks)
        return chunks

    def _embed_chunks(self, chunks: list[Document]) -> list[dict]:
        ... # TODO: implement
        
    def _store_embeddings(self, embeddings: list[dict]) -> None:
        ... # TODO: implement
        
class DocumentLoader:
    def __init__(self, result: IndexingResult):
        self.result = result
        self.SUPPORTED_TYPES = {
            ".pdf": self._load_pdf,
            ".txt": self._load_txt,
            # TODO: MP4 -> Whisper Transcript -> Index as .txt
            # TODO images??
        }
    
    def load_documents(self, folder: Path) -> list[Document]:
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Provided path {folder} is not a valid directory.")
        
        documents: list[Document] = []
        for file in folder.rglob("*"):
            if not file.is_file():
                continue
            
            self.result.files_found += 1
            loader = self.SUPPORTED_TYPES.get(file.suffix.lower())
            if loader is None:
                print(f"Skipping unsupported file: {file.name}")
                self.result.files_skipped += 1
                continue

            documents.extend(loader(file))
            
        self.result.documents_generated = len(documents)
        return documents
    
    def _load_pdf(self, file: Path) -> list[Document]:
        print(f"Loading PDF document: {file.name}")
        
        try:
            documents = PyPDFLoader(str(file)).load()
            
            for doc in documents:
                # Optionally add metadata about the source file
                doc.metadata["filename"] = file.name
            self.result.files_loaded += 1
            return documents
        except Exception as e:
            print(f"Error loading PDF {file.name}: {e}")
            self.result.files_failed += 1
            return []
            
    def _load_txt(self, file: Path) -> list[Document]:
        print(f"Loading TXT document: {file.name}")
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
    
    
    
from config import DOCUMENTS_PATH
if __name__ == "__main__":
    indexer = Indexer()
    indexer.index_folder(DOCUMENTS_PATH)
        
        
        