from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from embedding.provider import EmbeddingProvider
from config import INDEX_PATH

class VectorStore:
    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_provider.model
        
        if Path(INDEX_PATH).exists():
            self.db = FAISS.load_local(
                str(INDEX_PATH),
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            print(f"Loaded {self.db.index.ntotal} vectors.")
        else:
            self.db = None
            
        
    def store(self, chunks: list[Document], embeddings: list[list[float]]) -> int:
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        text_embeddings = list(zip(texts, embeddings))
        
        # if self.db is None:
        self.db = FAISS.from_embeddings(
                text_embeddings=text_embeddings,
                embedding=self.embedding_model,
                metadatas=metadatas,
            )
        # else: 
        #     self.db.add_embeddings(
        #         text_embeddings=text_embeddings,
        #         embedding=self.embedding_model,
        #         metadatas=metadatas,
        #     )
            
        self.db.save_local(str(INDEX_PATH))
        return len(chunks)


    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        if self.db is None:
            raise ValueError("Vector store is not initialized")
        
        embedding = self.embedding_provider.embed_query(query)
        
        return self.db.similarity_search_with_score_by_vector(embedding, k)