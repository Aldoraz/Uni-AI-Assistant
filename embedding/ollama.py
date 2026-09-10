import logging

from langchain_ollama import OllamaEmbeddings

from embedding.provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str, base_url: str, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("Embedding batch size must be positive")

        self.model = OllamaEmbeddings(
            model=model,
            base_url=base_url,
            validate_model_on_init=True,
        )
        self.batch_size = batch_size
        logger.info("Initialized Ollama embedding provider (model=%s)", model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self.model.embed_documents(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text)
