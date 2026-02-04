"""
Tests for Azure OpenAI embedding provider.

Uses mocking to avoid actual API calls during tests.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_session_storage.embeddings.azure_openai import AzureOpenAIEmbeddings


class MockEmbeddingResponse:
    """Mock response from Azure OpenAI embeddings API."""

    def __init__(self, embeddings: list[list[float]]):
        self.data = [MagicMock(embedding=emb) for emb in embeddings]


class TestAzureOpenAIEmbeddings:
    """Tests for Azure OpenAI embedding provider."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Provider initializes with correct configuration."""
        provider = AzureOpenAIEmbeddings(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            model="text-embedding-3-large",
            cache_size=100,
        )

        assert provider.endpoint == "https://test.openai.azure.com"
        assert provider.model == "text-embedding-3-large"
        assert provider.dimensions == 3072  # Auto-detected
        assert provider._cache is not None

    @pytest.mark.asyncio
    async def test_dimension_autodetection(self):
        """Dimensions are auto-detected based on model."""
        provider_small = AzureOpenAIEmbeddings(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            model="text-embedding-3-small",
        )
        assert provider_small.dimensions == 1536

        provider_large = AzureOpenAIEmbeddings(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            model="text-embedding-3-large",
        )
        assert provider_large.dimensions == 3072

        provider_ada = AzureOpenAIEmbeddings(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            model="text-embedding-ada-002",
        )
        assert provider_ada.dimensions == 1536

    @pytest.mark.asyncio
    async def test_explicit_dimensions(self):
        """Explicit dimensions override auto-detection."""
        provider = AzureOpenAIEmbeddings(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            model="custom-model",
            dimensions=2048,
        )
        assert provider.dimensions == 2048

    @pytest.mark.asyncio
    async def test_cache_disabled(self):
        """Cache can be disabled."""
        provider = AzureOpenAIEmbeddings(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            cache_size=0,
        )
        assert provider._cache is None

    @pytest.mark.asyncio
    async def test_embed_text_with_cache(self):
        """Single text embedding with cache hit."""
        provider = AzureOpenAIEmbeddings(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            cache_size=10,
        )

        mock_embedding = [0.1, 0.2, 0.3] * 1024  # 3072-d

        # Mock the client
        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(return_value=MockEmbeddingResponse([mock_embedding]))

        with patch.object(provider, "_ensure_client", return_value=mock_client):
            # First call - should hit API
            result1 = await provider.embed_text("test text")
            assert result1 == mock_embedding
            assert mock_client.embed.call_count == 1

            # Second call - should hit cache
            result2 = await provider.embed_text("test text")
            assert result2 == mock_embedding
            assert mock_client.embed.call_count == 1  # No additional call

    @pytest.mark.asyncio
    async def test_embed_text_without_cache(self):
        """Single text embedding without cache."""
        provider = AzureOpenAIEmbeddings(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            cache_size=0,  # Disabled
        )

        mock_embedding = [0.1, 0.2, 0.3] * 1024

        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(return_value=MockEmbeddingResponse([mock_embedding]))

        with patch.object(provider, "_ensure_client", return_value=mock_client):
            await provider.embed_text("test text")
            await provider.embed_text("test text")

            # Both should hit API (no cache)
            assert mock_client.embed.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """Batch embedding generation."""
        provider = AzureOpenAIEmbeddings(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            cache_size=10,
        )

        texts = ["text 1", "text 2", "text 3"]
        mock_embeddings = [
            [0.1] * 3072,
            [0.2] * 3072,
            [0.3] * 3072,
        ]

        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(return_value=MockEmbeddingResponse(mock_embeddings))

        with patch.object(provider, "_ensure_client", return_value=mock_client):
            results = await provider.embed_batch(texts)

            assert len(results) == 3
            assert results == mock_embeddings
            assert mock_client.embed.call_count == 1  # Single batch call

    @pytest.mark.asyncio
    async def test_embed_batch_with_partial_cache(self):
        """Batch embedding with some results cached."""
        provider = AzureOpenAIEmbeddings(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            cache_size=10,
        )

        # Pre-populate cache
        assert provider._cache is not None
        provider._cache.put("text 1", provider.model, [0.1] * 3072)
        provider._cache.put("text 3", provider.model, [0.3] * 3072)

        texts = ["text 1", "text 2", "text 3"]
        mock_embedding_2 = [0.2] * 3072

        mock_client = AsyncMock()
        # Only text 2 needs to be embedded
        mock_client.embed = AsyncMock(return_value=MockEmbeddingResponse([mock_embedding_2]))

        with patch.object(provider, "_ensure_client", return_value=mock_client):
            results = await provider.embed_batch(texts)

            assert len(results) == 3
            assert results[0] == [0.1] * 3072  # From cache
            assert results[1] == [0.2] * 3072  # From API
            assert results[2] == [0.3] * 3072  # From cache

            # Should only call API for text 2
            mock_client.embed.assert_called_once()
            call_args = mock_client.embed.call_args
            assert call_args[1]["input"] == ["text 2"]

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self):
        """Empty batch returns empty list."""
        provider = AzureOpenAIEmbeddings(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
        )

        results = await provider.embed_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Cache statistics are accurate."""
        provider = AzureOpenAIEmbeddings(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            cache_size=10,
        )

        stats = provider.get_cache_stats()
        assert stats["size"] == 0
        assert stats["max_entries"] == 10

        # Add to cache
        assert provider._cache is not None
        provider._cache.put("text 1", provider.model, [0.1] * 3072)

        stats = provider.get_cache_stats()
        assert stats["size"] == 1

    @pytest.mark.asyncio
    async def test_cache_stats_disabled(self):
        """Cache stats when cache is disabled."""
        provider = AzureOpenAIEmbeddings(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            cache_size=0,
        )

        stats = provider.get_cache_stats()
        assert stats["cache_enabled"] is False

    @pytest.mark.asyncio
    async def test_close(self):
        """Close cleans up client."""
        provider = AzureOpenAIEmbeddings(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
        )

        mock_client = AsyncMock()
        provider._client = mock_client

        await provider.close()

        mock_client.close.assert_called_once()
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Provider works as async context manager."""
        provider = AzureOpenAIEmbeddings(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
        )

        mock_client = AsyncMock()
        provider._client = mock_client  # Set client directly

        async with provider as p:
            assert p is provider

        # Should have closed
        mock_client.close.assert_called_once()

    def test_from_env_missing_endpoint(self):
        """from_env raises when AZURE_OPENAI_ENDPOINT missing."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
                AzureOpenAIEmbeddings.from_env()

    def test_from_env_missing_api_key(self):
        """from_env raises when AZURE_OPENAI_API_KEY missing."""
        with patch.dict(
            "os.environ",
            {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com"},
            clear=True,
        ):
            with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
                AzureOpenAIEmbeddings.from_env()

    def test_from_env_with_defaults(self):
        """from_env uses defaults when optional vars not set."""
        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
                "AZURE_OPENAI_API_KEY": "test-key",
            },
            clear=True,
        ):
            provider = AzureOpenAIEmbeddings.from_env()

            assert provider.model == "text-embedding-3-large"  # Default
            assert provider.dimensions == 3072
            assert provider._cache is not None
            assert provider._cache.max_entries == 1000  # Default

    def test_from_env_with_custom_config(self):
        """from_env respects custom configuration."""
        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
                "AZURE_OPENAI_EMBEDDING_DIMENSIONS": "1536",
                "AZURE_OPENAI_EMBEDDING_CACHE_SIZE": "500",
            },
            clear=True,
        ):
            provider = AzureOpenAIEmbeddings.from_env()

            assert provider.model == "text-embedding-3-small"
            assert provider.dimensions == 1536
            assert provider._cache.max_entries == 500
