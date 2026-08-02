from abc import ABC, abstractmethod
from langchain_core.embeddings import Embeddings

class EmbeddingProvider(ABC):
    model: Embeddings 
    
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...
        
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        ...
