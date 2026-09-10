import logging

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from embedding.provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str) -> None:
        load_dotenv()
        self.model = OpenAIEmbeddings(model=model)
        logger.info("Initialized OpenAI embedding provider (model=%s)", model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text)
