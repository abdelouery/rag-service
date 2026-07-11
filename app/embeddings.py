import logging

from openai import AsyncOpenAI

from .config import Settings

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Wraps OpenAI's embedding API (text-embedding-3-small by default)."""

    def __init__(self, settings: Settings) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(model=self._model, input=texts)
        logger.info(
            "embeddings.create model=%s inputs=%d usage=%s",
            response.model,
            len(texts),
            response.usage.model_dump() if response.usage else None,
        )
        logger.debug("embeddings.create response=%s", response.model_dump_json())
        return [item.embedding for item in response.data]
