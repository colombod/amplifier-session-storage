"""
Abstract base class for embedding providers.

Allows pluggable embedding generation from different sources:
- Azure OpenAI
- OpenAI Direct
- Local models (sentence-transformers, etc.)
- Custom implementations
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base for embedding generation."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Number of dimensions in the embedding vector."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name/identifier of the embedding model."""
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            Exception: If embedding generation fails
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts (batch operation).

        Batch operations are more efficient than individual calls.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (same order as input texts)

        Raises:
            Exception: If batch embedding fails
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Cleanup resources (close connections, release memory)."""
        pass

    async def __aenter__(self) -> EmbeddingProvider:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        """Async context manager exit."""
        await self.close()
