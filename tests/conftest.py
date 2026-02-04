"""
Shared test configuration and fixtures.

Provides embedding provider factory that uses real embedding providers when configured,
or falls back to mock provider for local testing.

Provider priority:
1. OpenAI direct (if OPENAI_API_KEY set)
2. Azure OpenAI (if AZURE_OPENAI_ENDPOINT set)
3. Mock (fallback)
"""

import logging
import os

import pytest

from amplifier_session_storage.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Mock embedding provider for testing without API costs.

    Generates deterministic embeddings based on text content.
    """

    def __init__(self, dimensions: int = 3072):
        self._dimensions = dimensions
        self._model_name = "mock-embeddings"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_text(self, text: str) -> list[float]:
        """Generate deterministic embedding based on text hash."""
        # Use hash for deterministic but varied vectors
        hash_val = hash(text) % 1000
        base_value = float(hash_val) / 1000.0

        # Create vector with some variation
        return [base_value + (i % 10) / 1000.0 for i in range(self._dimensions)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for batch."""
        return [await self.embed_text(text) for text in texts]

    async def close(self) -> None:
        """No cleanup needed for mock."""
        pass


def try_create_real_embedding_provider() -> EmbeddingProvider | None:
    """
    Try to create real Azure OpenAI embedding provider from environment.

    Returns None if not configured (tests should use mock instead).
    """
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")

    if not endpoint:
        return None

    try:
        from amplifier_session_storage.embeddings.azure_openai import AzureOpenAIEmbeddings

        provider = AzureOpenAIEmbeddings.from_env()
        logger.info("Using real Azure OpenAI embeddings for tests")
        return provider
    except Exception as e:
        logger.warning(f"Could not create Azure OpenAI provider: {e}")
        return None


@pytest.fixture
async def embedding_provider():
    """
    Fixture providing embedding provider.

    Uses real Azure OpenAI if AZURE_OPENAI_ENDPOINT is set,
    otherwise uses mock provider.
    """
    # Try real provider first
    provider = try_create_real_embedding_provider()

    if provider:
        yield provider
        await provider.close()
    else:
        # Fallback to mock
        provider = MockEmbeddingProvider(dimensions=3072)
        yield provider
        await provider.close()


@pytest.fixture
async def embedding_provider_small():
    """
    Fixture providing small-dimension embedding provider for faster tests.

    Always uses mock with 3 dimensions for speed.
    """
    provider = MockEmbeddingProvider(dimensions=3)
    yield provider
    await provider.close()
