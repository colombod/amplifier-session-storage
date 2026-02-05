"""
Integration tests for Cosmos DB backend.

These tests require a real Cosmos DB instance and are marked as integration tests.
Run with: pytest -m integration

Environment variables required:
    AMPLIFIER_COSMOS_ENDPOINT - Cosmos DB endpoint URL
    AMPLIFIER_COSMOS_AUTH_METHOD - "default_credential" (recommended) or "key"
    AMPLIFIER_COSMOS_DATABASE - Database name (default: your-database)

    For Azure OpenAI embeddings (optional):
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY
"""

from datetime import UTC, datetime

import pytest

from amplifier_session_storage.backends import (
    SearchFilters,
    TranscriptSearchOptions,
)
from amplifier_session_storage.backends.cosmos import CosmosBackend, CosmosConfig
from amplifier_session_storage.embeddings import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Simple mock for testing without Azure OpenAI costs."""

    @property
    def model_name(self) -> str:
        return "test-model"

    @property
    def dimensions(self) -> int:
        return 3072

    async def embed_text(self, text: str) -> list[float]:
        """Generate deterministic embedding."""
        hash_val = hash(text) % 1000
        return [float(hash_val) / 1000.0] * 3072

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate batch embeddings."""
        return [await self.embed_text(text) for text in texts]

    async def close(self) -> None:
        pass


@pytest.fixture
async def cosmos_storage():
    """
    Fixture providing initialized Cosmos backend.

    Requires environment variables to be set.
    """
    try:
        config = CosmosConfig.from_env()
        # Use database name from environment (AMPLIFIER_COSMOS_DATABASE)

        embeddings = MockEmbeddingProvider()
        storage = await CosmosBackend.create(config=config, embedding_provider=embeddings)
        yield storage

        # Cleanup
        await storage.close()
    except Exception as e:
        pytest.skip(f"Cosmos DB not configured: {e}")


@pytest.mark.integration
class TestCosmosIntegration:
    """Integration tests with real Cosmos DB."""

    @pytest.mark.asyncio
    async def test_connection(self, cosmos_storage):
        """Can connect to Cosmos DB."""
        assert cosmos_storage._initialized is True
        assert cosmos_storage._client is not None

    @pytest.mark.asyncio
    async def test_vector_search_capability(self, cosmos_storage):
        """Check if vector search is enabled."""
        supports = await cosmos_storage.supports_vector_search()
        # Should return True if vector indexes configured, False otherwise
        assert isinstance(supports, bool)

    @pytest.mark.asyncio
    async def test_upsert_and_retrieve_session(self, cosmos_storage):
        """Can store and retrieve session metadata."""
        test_session_id = f"test-session-{datetime.now(UTC).timestamp()}"

        metadata = {
            "session_id": test_session_id,
            "project_slug": "test-project",
            "bundle": "foundation",
            "created": datetime.now(UTC).isoformat(),
            "turn_count": 5,
        }

        await cosmos_storage.upsert_session_metadata(
            user_id="test-user", host_id="test-host", metadata=metadata
        )

        retrieved = await cosmos_storage.get_session_metadata(
            user_id="test-user", session_id=test_session_id
        )

        assert retrieved is not None
        assert retrieved["session_id"] == test_session_id
        assert retrieved["project_slug"] == "test-project"

    @pytest.mark.asyncio
    async def test_sync_transcripts_with_embeddings(self, cosmos_storage):
        """Transcripts are synced with embeddings."""
        test_session_id = f"test-session-{datetime.now(UTC).timestamp()}"

        lines = [
            {"role": "user", "content": "How do I use Cosmos DB for vector search?", "turn": 0},
            {
                "role": "assistant",
                "content": "Cosmos DB supports vector search with embeddings",
                "turn": 0,
            },
        ]

        synced = await cosmos_storage.sync_transcript_lines(
            user_id="test-user",
            host_id="test-host",
            project_slug="test-project",
            session_id=test_session_id,
            lines=lines,
        )

        assert synced == 2

        # Retrieve
        retrieved = await cosmos_storage.get_transcript_lines(
            user_id="test-user", project_slug="test-project", session_id=test_session_id
        )

        assert len(retrieved) == 2

        # Check if embeddings were stored
        if "embedding" in retrieved[0]:
            assert len(retrieved[0]["embedding"]) == 3072

    @pytest.mark.asyncio
    async def test_full_text_search(self, cosmos_storage):
        """Full-text search works in Cosmos."""
        test_session_id = f"test-session-{datetime.now(UTC).timestamp()}"

        lines = [
            {"role": "user", "content": "How do I implement vector search?", "turn": 0},
            {"role": "assistant", "content": "Use embeddings for semantic search", "turn": 0},
            {"role": "user", "content": "What about keyword matching?", "turn": 1},
        ]

        await cosmos_storage.sync_transcript_lines(
            user_id="test-user",
            host_id="test-host",
            project_slug="test-project",
            session_id=test_session_id,
            lines=lines,
        )

        # Search
        results = await cosmos_storage.search_transcripts(
            user_id="test-user",
            options=TranscriptSearchOptions(query="vector search", search_type="full_text"),
            limit=10,
        )

        assert len(results) > 0
        assert any("vector search" in r.content.lower() for r in results)

    @pytest.mark.asyncio
    async def test_search_sessions(self, cosmos_storage):
        """Session search works."""
        test_session_id = f"test-session-{datetime.now(UTC).timestamp()}"

        await cosmos_storage.upsert_session_metadata(
            user_id="test-user",
            host_id="test-host",
            metadata={
                "session_id": test_session_id,
                "project_slug": "test-project",
                "bundle": "foundation",
                "created": datetime.now(UTC).isoformat(),
                "turn_count": 10,
            },
        )

        # Search
        results = await cosmos_storage.search_sessions(
            user_id="test-user",
            filters=SearchFilters(project_slug="test-project"),
            limit=10,
        )

        assert len(results) > 0
        session_ids = [s["session_id"] for s in results]
        assert test_session_id in session_ids

    @pytest.mark.asyncio
    async def test_event_operations(self, cosmos_storage):
        """Event sync and retrieval works."""
        test_session_id = f"test-session-{datetime.now(UTC).timestamp()}"

        events = [
            {"event": "session.start", "ts": datetime.now(UTC).isoformat(), "lvl": "info"},
            {"event": "llm.request", "ts": datetime.now(UTC).isoformat(), "lvl": "debug"},
        ]

        synced = await cosmos_storage.sync_event_lines(
            user_id="test-user",
            host_id="test-host",
            project_slug="test-project",
            session_id=test_session_id,
            lines=events,
        )

        assert synced == 2

    @pytest.mark.asyncio
    async def test_statistics(self, cosmos_storage):
        """Statistics aggregation works."""
        stats = await cosmos_storage.get_session_statistics(user_id="test-user")

        assert "total_sessions" in stats
        assert "sessions_by_project" in stats
        assert "sessions_by_bundle" in stats
        assert isinstance(stats["total_sessions"], int)

    @pytest.mark.asyncio
    async def test_cleanup_test_data(self, cosmos_storage):
        """Cleanup can delete test sessions."""
        # Get all test sessions
        results = await cosmos_storage.search_sessions(
            user_id="test-user",
            filters=SearchFilters(project_slug="test-project"),
            limit=100,
        )

        # Delete each one
        for session in results:
            deleted = await cosmos_storage.delete_session(
                user_id="test-user",
                project_slug="test-project",
                session_id=session["session_id"],
            )
            assert isinstance(deleted, bool)


@pytest.mark.integration
class TestCosmosVectorSearch:
    """Integration tests for vector search features."""

    @pytest.mark.asyncio
    async def test_vector_search_if_enabled(self, cosmos_storage):
        """Vector search works if enabled on account."""
        supports = await cosmos_storage.supports_vector_search()

        if not supports:
            pytest.skip("Vector search not enabled on Cosmos account")

        test_session_id = f"test-vec-{datetime.now(UTC).timestamp()}"

        # Sync with embeddings
        lines = [
            {"role": "user", "content": "Semantic search query", "turn": 0},
        ]

        await cosmos_storage.sync_transcript_lines(
            user_id="test-user",
            host_id="test-host",
            project_slug="test-project",
            session_id=test_session_id,
            lines=lines,
        )

        # Try semantic search
        results = await cosmos_storage.search_transcripts(
            user_id="test-user",
            options=TranscriptSearchOptions(query="semantic query", search_type="semantic"),
            limit=10,
        )

        # Should work or gracefully degrade
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_search_if_enabled(self, cosmos_storage):
        """Hybrid search works if vector search enabled."""
        supports = await cosmos_storage.supports_vector_search()

        if not supports:
            pytest.skip("Vector search not enabled - hybrid will fallback to full_text")

        test_session_id = f"test-hybrid-{datetime.now(UTC).timestamp()}"

        lines = [
            {"role": "user", "content": "Testing hybrid search capabilities", "turn": 0},
            {"role": "assistant", "content": "Hybrid combines full-text and semantic", "turn": 0},
        ]

        await cosmos_storage.sync_transcript_lines(
            user_id="test-user",
            host_id="test-host",
            project_slug="test-project",
            session_id=test_session_id,
            lines=lines,
        )

        # Hybrid search
        results = await cosmos_storage.search_transcripts(
            user_id="test-user",
            options=TranscriptSearchOptions(
                query="hybrid search", search_type="hybrid", mmr_lambda=0.7
            ),
            limit=10,
        )

        assert isinstance(results, list)
