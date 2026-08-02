from embedding.provider import EmbeddingProvider
from langchain_ollama import OllamaEmbeddings

class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model):
        self.model = OllamaEmbeddings(
            model=model
        )
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...
        
    def embed_query(self, text: str) -> list[float]:
        ...
