from embedding.provider import EmbeddingProvider
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str):
        load_dotenv()
        self.model = OpenAIEmbeddings(model=model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)
        
    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text)
