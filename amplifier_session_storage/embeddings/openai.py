"""
OpenAI direct embedding provider implementation.

Uses the official OpenAI Python SDK for embedding generation.
Works with OpenAI API directly (not through Azure).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import AsyncOpenAI
else:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        AsyncOpenAI = None  # type: ignore

from .base import EmbeddingProvider
from .cache import EmbeddingCache

logger = logging.getLogger(__name__)


class OpenAIEmbeddings(EmbeddingProvider):
    """
    OpenAI direct embedding provider with caching.

    Supports models:
    - text-embedding-3-small (1536 dimensions)
    - text-embedding-3-large (3072 dimensions)
    - text-embedding-ada-002 (1536 dimensions)
    """

    # Model dimension mappings
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        dimensions: int | None = None,
        cache_size: int = 1000,
        base_url: str | None = None,
    ):
        """
        Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key
            model: Model name (default: text-embedding-3-large)
            dimensions: Vector dimensions (auto-detected if None)
            cache_size: Max cached embeddings (0 to disable)
            base_url: Optional base URL for custom endpoints
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

        # Auto-detect dimensions if not provided
        if dimensions is None:
            if model in self.MODEL_DIMENSIONS:
                self._dimensions = self.MODEL_DIMENSIONS[model]
            else:
                logger.warning(
                    f"Unknown model '{model}', defaulting to 3072 dimensions. "
                    f"Pass explicit dimensions parameter if different."
                )
                self._dimensions = 3072
        else:
            self._dimensions = dimensions

        self._client: Any = None  # AsyncOpenAI (optional dependency)
        self._cache = EmbeddingCache(max_entries=cache_size) if cache_size > 0 else None

        logger.info(
            f"OpenAI embeddings initialized: model={model}, "
            f"dimensions={self._dimensions}, cache_size={cache_size}"
        )

    @classmethod
    def from_env(cls) -> OpenAIEmbeddings:
        """
        Create provider from environment variables.

        Required env vars:
            OPENAI_API_KEY: OpenAI API key

        Optional env vars:
            OPENAI_EMBEDDING_MODEL: Model name (default: text-embedding-3-large)
            OPENAI_EMBEDDING_DIMENSIONS: Vector dimensions (auto-detected)
            OPENAI_EMBEDDING_CACHE_SIZE: Cache size (default: 1000)
            OPENAI_BASE_URL: Custom base URL (optional)

        Examples:
            export OPENAI_API_KEY="sk-..."
            export OPENAI_EMBEDDING_MODEL="text-embedding-3-large"
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        dimensions_str = os.environ.get("OPENAI_EMBEDDING_DIMENSIONS")
        cache_size_str = os.environ.get("OPENAI_EMBEDDING_CACHE_SIZE", "1000")
        base_url = os.environ.get("OPENAI_BASE_URL")

        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")

        dimensions = int(dimensions_str) if dimensions_str else None
        cache_size = int(cache_size_str)

        return cls(
            api_key=api_key,
            model=model,
            dimensions=dimensions,
            cache_size=cache_size,
            base_url=base_url,
        )

    @property
    def dimensions(self) -> int:
        """Number of dimensions in embedding vectors."""
        return self._dimensions

    @property
    def model_name(self) -> str:
        """Model identifier."""
        return self.model

    async def _ensure_client(self) -> Any:
        """Lazy initialize the OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text with caching.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Check cache first
        if self._cache:
            cached = self._cache.get(text, self.model)
            if cached is not None:
                logger.debug(f"Cache hit for text (len={len(text)})")
                return cached

        # Generate embedding
        client = await self._ensure_client()

        response = await client.embeddings.create(
            input=text,
            model=self.model,
            dimensions=self._dimensions,
        )

        # Extract embedding
        embedding = response.data[0].embedding

        # Cache the result
        if self._cache:
            self._cache.put(text, self.model, embedding)
            logger.debug(f"Cached embedding for text (len={len(text)})")

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts with partial caching.

        Checks cache for each text individually, only requests embeddings
        for cache misses, then caches new results.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (same order as input)
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        texts_to_embed: list[tuple[int, str]] = []

        # Check cache for each text
        if self._cache:
            for i, text in enumerate(texts):
                cached = self._cache.get(text, self.model)
                if cached is not None:
                    results[i] = cached
                else:
                    texts_to_embed.append((i, text))
        else:
            texts_to_embed = list(enumerate(texts))

        # Generate embeddings for cache misses
        if texts_to_embed:
            client = await self._ensure_client()

            # Extract just the texts for API call
            texts_for_api = [text for _, text in texts_to_embed]

            response = await client.embeddings.create(
                input=texts_for_api,
                model=self.model,
                dimensions=self._dimensions,
            )

            # Store results and cache
            for (original_idx, text), embedding_data in zip(
                texts_to_embed, response.data, strict=True
            ):
                embedding = embedding_data.embedding
                results[original_idx] = embedding

                if self._cache:
                    self._cache.put(text, self.model, embedding)

            logger.info(
                f"Generated {len(texts_to_embed)} embeddings, "
                f"{len(texts) - len(texts_to_embed)} from cache"
            )

        # Verify all results are populated (explicit check — assert is stripped with -O)
        missing = [i for i, r in enumerate(results) if r is None]
        if missing:
            raise ValueError(
                f"Embedding generation incomplete: {len(missing)}/{len(results)} "
                f"results missing at indices {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        return results  # type: ignore

    async def close(self) -> None:
        """Close the OpenAI client."""
        if self._client:
            await self._client.close()
            self._client = None

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if self._cache:
            return self._cache.stats()
        return {"cache_enabled": False}
