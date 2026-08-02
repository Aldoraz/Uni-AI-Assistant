from embedding.provider import EmbeddingProvider

class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model):
        ...

    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...
        
    def embed_query(self, text: str) -> list[float]:
        ...
