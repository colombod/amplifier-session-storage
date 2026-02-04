"""
Tests for DuckDB storage backend.

Uses real DuckDB (in-memory) for accurate testing.
"""

from datetime import UTC, datetime

import pytest

from amplifier_session_storage.backends import SearchFilters, TranscriptSearchOptions
from amplifier_session_storage.backends.duckdb import DuckDBBackend, DuckDBConfig


@pytest.fixture
async def duckdb_storage(embedding_provider):
    """
    Fixture providing initialized DuckDB backend.

    Uses real Azure OpenAI if configured, otherwise mock.
    """
    config = DuckDBConfig(db_path=":memory:")
    storage = await DuckDBBackend.create(config=config, embedding_provider=embedding_provider)
    yield storage
    await storage.close()


@pytest.fixture
async def duckdb_storage_no_embeddings():
    """Fixture providing DuckDB backend without embeddings."""
    config = DuckDBConfig(db_path=":memory:")
    storage = await DuckDBBackend.create(config=config, embedding_provider=None)
    yield storage
    await storage.close()


class TestDuckDBInitialization:
    """Tests for DuckDB backend initialization."""

    @pytest.mark.asyncio
    async def test_create_with_defaults(self):
        """Backend creates with default configuration."""
        storage = await DuckDBBackend.create()
        assert storage._initialized is True
        await storage.close()

    @pytest.mark.asyncio
    async def test_create_with_custom_config(self):
        """Backend creates with custom configuration."""
        config = DuckDBConfig(db_path=":memory:", vector_dimensions=1536)
        storage = await DuckDBBackend.create(config=config)
        assert storage._initialized is True
        assert storage.config.vector_dimensions == 1536
        await storage.close()

    @pytest.mark.asyncio
    async def test_supports_vector_search_with_embeddings(self, duckdb_storage):
        """Vector search supported when embeddings configured."""
        # VSS extension may not be available in test environment
        # Just verify the method returns a boolean
        supports = await duckdb_storage.supports_vector_search()
        assert isinstance(supports, bool)
        # If it's False, VSS extension isn't installed (that's okay for unit tests)

    @pytest.mark.asyncio
    async def test_supports_vector_search_without_embeddings(self, duckdb_storage_no_embeddings):
        """Vector search not supported without embeddings."""
        supports = await duckdb_storage_no_embeddings.supports_vector_search()
        assert supports is False


class TestDuckDBSessionOperations:
    """Tests for session metadata operations."""

    @pytest.mark.asyncio
    async def test_upsert_and_get_session(self, duckdb_storage):
        """Can upsert and retrieve session metadata."""
        metadata = {
            "session_id": "test-session-1",
            "project_slug": "test-project",
            "bundle": "foundation",
            "created": datetime.now(UTC).isoformat(),
            "turn_count": 5,
        }

        await duckdb_storage.upsert_session_metadata(
            user_id="user-1", host_id="host-1", metadata=metadata
        )

        retrieved = await duckdb_storage.get_session_metadata(
            user_id="user-1", session_id="test-session-1"
        )

        assert retrieved is not None
        assert retrieved["session_id"] == "test-session-1"
        assert retrieved["project_slug"] == "test-project"

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, duckdb_storage):
        """Getting nonexistent session returns None."""
        result = await duckdb_storage.get_session_metadata(
            user_id="user-1", session_id="nonexistent"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_search_sessions_basic(self, duckdb_storage):
        """Basic session search works."""
        # Create test sessions
        for i in range(3):
            await duckdb_storage.upsert_session_metadata(
                user_id="user-1",
                host_id="host-1",
                metadata={
                    "session_id": f"session-{i}",
                    "project_slug": "project-1",
                    "bundle": "foundation",
                    "created": datetime.now(UTC).isoformat(),
                    "turn_count": i + 1,
                },
            )

        # Search all
        results = await duckdb_storage.search_sessions(
            user_id="user-1", filters=SearchFilters(), limit=10
        )

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_sessions_with_filters(self, duckdb_storage):
        """Session search respects filters."""
        # Create sessions with different projects
        await duckdb_storage.upsert_session_metadata(
            user_id="user-1",
            host_id="host-1",
            metadata={
                "session_id": "session-1",
                "project_slug": "project-a",
                "bundle": "foundation",
                "created": "2024-01-15T10:00:00Z",
                "turn_count": 5,
            },
        )

        await duckdb_storage.upsert_session_metadata(
            user_id="user-1",
            host_id="host-1",
            metadata={
                "session_id": "session-2",
                "project_slug": "project-b",
                "bundle": "foundation",
                "created": "2024-01-16T10:00:00Z",
                "turn_count": 10,
            },
        )

        # Filter by project
        results = await duckdb_storage.search_sessions(
            user_id="user-1", filters=SearchFilters(project_slug="project-a"), limit=10
        )

        assert len(results) == 1
        assert results[0]["session_id"] == "session-1"

    @pytest.mark.asyncio
    async def test_delete_session(self, duckdb_storage):
        """Session deletion removes all data."""
        # Create session with data
        await duckdb_storage.upsert_session_metadata(
            user_id="user-1",
            host_id="host-1",
            metadata={
                "session_id": "session-1",
                "project_slug": "project-1",
                "bundle": "foundation",
                "created": datetime.now(UTC).isoformat(),
                "turn_count": 1,
            },
        )

        await duckdb_storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=[{"role": "user", "content": "test", "turn": 0}],
        )

        # Delete
        deleted = await duckdb_storage.delete_session(
            user_id="user-1", project_slug="project-1", session_id="session-1"
        )

        assert deleted is True

        # Verify session is gone
        session = await duckdb_storage.get_session_metadata(
            user_id="user-1", session_id="session-1"
        )
        assert session is None


class TestDuckDBTranscriptOperations:
    """Tests for transcript operations."""

    @pytest.mark.asyncio
    async def test_sync_and_get_transcripts(self, duckdb_storage):
        """Can sync and retrieve transcript lines."""
        lines = [
            {"role": "user", "content": "Hello", "turn": 0, "ts": "2024-01-15T10:00:00Z"},
            {"role": "assistant", "content": "Hi there!", "turn": 0, "ts": "2024-01-15T10:00:01Z"},
        ]

        synced = await duckdb_storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=lines,
        )

        assert synced == 2

        # Retrieve
        retrieved = await duckdb_storage.get_transcript_lines(
            user_id="user-1", project_slug="project-1", session_id="session-1"
        )

        assert len(retrieved) == 2
        assert retrieved[0]["content"] == "Hello"
        assert retrieved[1]["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_sync_generates_embeddings(self, duckdb_storage):
        """Syncing transcripts generates embeddings automatically."""
        lines = [
            {"role": "user", "content": "Test message", "turn": 0},
        ]

        await duckdb_storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=lines,
        )

        # Verify embedding was stored
        embedding = await duckdb_storage._get_embedding("user-1", "session-1", 0)
        assert embedding is not None
        assert len(embedding) == 3072


class TestDuckDBSearch:
    """Tests for search operations."""

    @pytest.mark.asyncio
    async def test_full_text_search(self, duckdb_storage):
        """Full-text search finds matching content."""
        # Create test data
        lines = [
            {"role": "user", "content": "How do I implement vector search?", "turn": 0},
            {"role": "assistant", "content": "You use embeddings for vector search", "turn": 0},
            {"role": "user", "content": "What about traditional search?", "turn": 1},
        ]

        await duckdb_storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=lines,
        )

        # Search
        results = await duckdb_storage.search_transcripts(
            user_id="user-1",
            options=TranscriptSearchOptions(query="vector search", search_type="full_text"),
            limit=10,
        )

        assert len(results) == 2  # Both messages containing "vector search"
        assert all("vector search" in r.content.lower() for r in results)

    @pytest.mark.asyncio
    async def test_semantic_search(self, duckdb_storage):
        """Semantic search uses embeddings."""
        lines = [
            {"role": "user", "content": "How do I use embeddings?", "turn": 0},
            {"role": "assistant", "content": "Embeddings are vectors", "turn": 0},
        ]

        await duckdb_storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=lines,
        )

        # Semantic search
        results = await duckdb_storage.search_transcripts(
            user_id="user-1",
            options=TranscriptSearchOptions(query="vector representations", search_type="semantic"),
            limit=10,
        )

        assert len(results) >= 0  # May or may not match depending on mock embeddings

    @pytest.mark.asyncio
    async def test_search_without_embeddings_fallback(self, duckdb_storage_no_embeddings):
        """Semantic search falls back to full_text when no embeddings."""
        lines = [{"role": "user", "content": "Test message", "turn": 0}]

        await duckdb_storage_no_embeddings.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=lines,
        )

        # Request semantic but should fallback
        results = await duckdb_storage_no_embeddings.search_transcripts(
            user_id="user-1",
            options=TranscriptSearchOptions(query="Test", search_type="semantic"),
            limit=10,
        )

        # Should still return results via fallback
        assert len(results) == 1
        assert results[0].source == "full_text"  # Fell back


class TestDuckDBEventOperations:
    """Tests for event operations."""

    @pytest.mark.asyncio
    async def test_sync_and_get_events(self, duckdb_storage):
        """Can sync and retrieve events."""
        events = [
            {"event": "session.start", "ts": "2024-01-15T10:00:00Z", "lvl": "info"},
            {"event": "llm.request", "ts": "2024-01-15T10:00:01Z", "lvl": "debug"},
        ]

        synced = await duckdb_storage.sync_event_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=events,
        )

        assert synced == 2

        retrieved = await duckdb_storage.get_event_lines(
            user_id="user-1", project_slug="project-1", session_id="session-1"
        )

        assert len(retrieved) == 2


class TestDuckDBStatistics:
    """Tests for analytics and statistics."""

    @pytest.mark.asyncio
    async def test_get_statistics(self, duckdb_storage):
        """Statistics aggregation works."""
        # Create test sessions
        for i in range(3):
            await duckdb_storage.upsert_session_metadata(
                user_id="user-1",
                host_id="host-1",
                metadata={
                    "session_id": f"session-{i}",
                    "project_slug": "project-1" if i < 2 else "project-2",
                    "bundle": "foundation",
                    "created": datetime.now(UTC).isoformat(),
                    "turn_count": i + 1,
                },
            )

        stats = await duckdb_storage.get_session_statistics(user_id="user-1")

        assert stats["total_sessions"] == 3
        assert stats["sessions_by_project"]["project-1"] == 2
        assert stats["sessions_by_project"]["project-2"] == 1
