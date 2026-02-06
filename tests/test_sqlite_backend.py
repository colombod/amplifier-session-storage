"""
Tests for SQLite storage backend.

Uses real SQLite (in-memory) for accurate testing.
"""

from datetime import UTC, datetime

import pytest

from amplifier_session_storage.backends import (
    SearchFilters,
    TranscriptSearchOptions,
)
from amplifier_session_storage.backends.sqlite import SQLiteBackend, SQLiteConfig


@pytest.fixture
async def sqlite_storage(embedding_provider):
    """
    Fixture providing initialized SQLite backend.

    Uses real Azure OpenAI if configured, otherwise mock.
    """
    config = SQLiteConfig(db_path=":memory:")
    storage = await SQLiteBackend.create(config=config, embedding_provider=embedding_provider)
    yield storage
    await storage.close()


@pytest.fixture
async def sqlite_storage_no_embeddings():
    """Fixture providing SQLite backend without embeddings."""
    config = SQLiteConfig(db_path=":memory:")
    storage = await SQLiteBackend.create(config=config, embedding_provider=None)
    yield storage
    await storage.close()


class TestSQLiteInitialization:
    """Tests for SQLite backend initialization."""

    @pytest.mark.asyncio
    async def test_create_with_defaults(self):
        """Backend creates with default configuration."""
        storage = await SQLiteBackend.create()
        assert storage._initialized is True
        await storage.close()

    @pytest.mark.asyncio
    async def test_create_with_custom_config(self):
        """Backend creates with custom configuration."""
        config = SQLiteConfig(db_path=":memory:", vector_dimensions=1536)
        storage = await SQLiteBackend.create(config=config)
        assert storage._initialized is True
        assert storage.config.vector_dimensions == 1536
        await storage.close()

    @pytest.mark.asyncio
    async def test_supports_vector_search_with_embeddings(self, sqlite_storage):
        """Vector search availability depends on VSS extension."""
        # VSS extension may not be available
        supports = await sqlite_storage.supports_vector_search()
        assert isinstance(supports, bool)

    @pytest.mark.asyncio
    async def test_supports_vector_search_without_embeddings(self, sqlite_storage_no_embeddings):
        """Vector search requires embeddings."""
        supports = await sqlite_storage_no_embeddings.supports_vector_search()
        assert supports is False


class TestSQLiteSessionOperations:
    """Tests for session metadata operations."""

    @pytest.mark.asyncio
    async def test_upsert_and_get_session(self, sqlite_storage):
        """Can upsert and retrieve session metadata."""
        metadata = {
            "session_id": "test-session-1",
            "project_slug": "test-project",
            "bundle": "foundation",
            "created": datetime.now(UTC).isoformat(),
            "turn_count": 5,
        }

        await sqlite_storage.upsert_session_metadata(
            user_id="user-1", host_id="host-1", metadata=metadata
        )

        retrieved = await sqlite_storage.get_session_metadata(
            user_id="user-1", session_id="test-session-1"
        )

        assert retrieved is not None
        assert retrieved["session_id"] == "test-session-1"
        assert retrieved["project_slug"] == "test-project"

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, sqlite_storage):
        """Getting nonexistent session returns None."""
        result = await sqlite_storage.get_session_metadata(
            user_id="user-1", session_id="nonexistent"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_search_sessions_basic(self, sqlite_storage):
        """Basic session search works."""
        for i in range(3):
            await sqlite_storage.upsert_session_metadata(
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

        results = await sqlite_storage.search_sessions(
            user_id="user-1", filters=SearchFilters(), limit=10
        )

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_sessions_with_project_filter(self, sqlite_storage):
        """Session search filters by project."""
        await sqlite_storage.upsert_session_metadata(
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

        await sqlite_storage.upsert_session_metadata(
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

        results = await sqlite_storage.search_sessions(
            user_id="user-1", filters=SearchFilters(project_slug="project-a"), limit=10
        )

        assert len(results) == 1
        assert results[0]["session_id"] == "session-1"

    @pytest.mark.asyncio
    async def test_search_sessions_with_turn_count_filters(self, sqlite_storage):
        """Session search filters by turn count range."""
        for i in range(5):
            await sqlite_storage.upsert_session_metadata(
                user_id="user-1",
                host_id="host-1",
                metadata={
                    "session_id": f"session-{i}",
                    "project_slug": "project-1",
                    "bundle": "foundation",
                    "created": datetime.now(UTC).isoformat(),
                    "turn_count": (i + 1) * 10,  # 10, 20, 30, 40, 50
                },
            )

        # Filter for sessions with 20-40 turns
        results = await sqlite_storage.search_sessions(
            user_id="user-1",
            filters=SearchFilters(min_turn_count=20, max_turn_count=40),
            limit=10,
        )

        assert len(results) == 3  # Should get sessions with 20, 30, 40 turns

    @pytest.mark.asyncio
    async def test_delete_session(self, sqlite_storage):
        """Session deletion removes all data."""
        await sqlite_storage.upsert_session_metadata(
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

        await sqlite_storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=[{"role": "user", "content": "test", "turn": 0}],
        )

        # Delete
        deleted = await sqlite_storage.delete_session(
            user_id="user-1", project_slug="project-1", session_id="session-1"
        )

        assert deleted is True

        # Verify gone
        session = await sqlite_storage.get_session_metadata(
            user_id="user-1", session_id="session-1"
        )
        assert session is None


class TestSQLiteTranscriptOperations:
    """Tests for transcript operations."""

    @pytest.mark.asyncio
    async def test_sync_and_get_transcripts(self, sqlite_storage):
        """Can sync and retrieve transcript lines."""
        lines = [
            {"role": "user", "content": "Hello", "turn": 0, "ts": "2024-01-15T10:00:00Z"},
            {
                "role": "assistant",
                "content": "Hi there!",
                "turn": 0,
                "ts": "2024-01-15T10:00:01Z",
            },
        ]

        synced = await sqlite_storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=lines,
        )

        assert synced == 2

        retrieved = await sqlite_storage.get_transcript_lines(
            user_id="user-1", project_slug="project-1", session_id="session-1"
        )

        assert len(retrieved) == 2
        assert retrieved[0]["content"] == "Hello"
        assert retrieved[1]["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_sync_generates_embeddings(self, sqlite_storage):
        """Syncing transcripts generates embeddings automatically."""
        lines = [
            {"role": "user", "content": "Test message", "turn": 0},
        ]

        await sqlite_storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=lines,
        )

        # Verify embedding was stored
        embedding = await sqlite_storage._get_embedding("user-1", "session-1", 0)
        assert embedding is not None
        assert len(embedding) == 3072

    @pytest.mark.asyncio
    async def test_sync_incremental(self, sqlite_storage):
        """Incremental sync works correctly."""
        # First batch
        lines1 = [
            {"role": "user", "content": "Message 1", "turn": 0},
            {"role": "assistant", "content": "Response 1", "turn": 0},
        ]

        synced1 = await sqlite_storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=lines1,
            start_sequence=0,
        )

        assert synced1 == 2

        # Second batch
        lines2 = [
            {"role": "user", "content": "Message 2", "turn": 1},
        ]

        synced2 = await sqlite_storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=lines2,
            start_sequence=2,  # Continue from where we left off
        )

        assert synced2 == 1

        # Get all
        all_messages = await sqlite_storage.get_transcript_lines(
            user_id="user-1", project_slug="project-1", session_id="session-1"
        )

        assert len(all_messages) == 3


class TestSQLiteSearch:
    """Tests for search operations."""

    @pytest.mark.asyncio
    async def test_full_text_search(self, sqlite_storage):
        """Full-text search finds matching content."""
        lines = [
            {"role": "user", "content": "How do I implement vector search?", "turn": 0},
            {"role": "assistant", "content": "You use embeddings for vector search", "turn": 0},
            {"role": "user", "content": "What about traditional search?", "turn": 1},
        ]

        await sqlite_storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=lines,
        )

        results = await sqlite_storage.search_transcripts(
            user_id="user-1",
            options=TranscriptSearchOptions(query="vector search", search_type="full_text"),
            limit=10,
        )

        assert len(results) == 2
        assert all("vector search" in r.content.lower() for r in results)

    @pytest.mark.asyncio
    async def test_search_role_filters(self, sqlite_storage):
        """Search respects role filters."""
        lines = [
            {"role": "user", "content": "User asks about search", "turn": 0},
            {"role": "assistant", "content": "Assistant explains search", "turn": 0},
        ]

        await sqlite_storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=lines,
        )

        # Search only user messages
        results = await sqlite_storage.search_transcripts(
            user_id="user-1",
            options=TranscriptSearchOptions(
                query="search",
                search_type="full_text",
                search_in_user=True,
                search_in_assistant=False,
            ),
            limit=10,
        )

        assert len(results) == 1
        assert results[0].metadata["role"] == "user"

    @pytest.mark.asyncio
    async def test_semantic_search_fallback(self, sqlite_storage_no_embeddings):
        """Semantic search falls back to full_text without embeddings."""
        lines = [{"role": "user", "content": "Test message", "turn": 0}]

        await sqlite_storage_no_embeddings.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=lines,
        )

        # Request semantic but should fallback
        results = await sqlite_storage_no_embeddings.search_transcripts(
            user_id="user-1",
            options=TranscriptSearchOptions(query="Test", search_type="semantic"),
            limit=10,
        )

        assert len(results) == 1
        assert results[0].source == "full_text"  # Fell back

    @pytest.mark.asyncio
    async def test_search_with_date_filters(self, sqlite_storage):
        """Search filters by date range."""
        lines = [
            {"role": "user", "content": "Old message", "turn": 0, "ts": "2024-01-01T10:00:00Z"},
            {"role": "user", "content": "New message", "turn": 1, "ts": "2024-02-01T10:00:00Z"},
        ]

        await sqlite_storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=lines,
        )

        # Search only messages after 2024-01-15
        results = await sqlite_storage.search_transcripts(
            user_id="user-1",
            options=TranscriptSearchOptions(
                query="message",
                search_type="full_text",
                filters=SearchFilters(start_date="2024-01-15T00:00:00Z"),
            ),
            limit=10,
        )

        assert len(results) == 1
        assert "New message" in results[0].content


class TestSQLiteEventOperations:
    """Tests for event operations."""

    @pytest.mark.asyncio
    async def test_sync_and_get_events(self, sqlite_storage):
        """Can sync and retrieve events."""
        events = [
            {"event": "session.start", "ts": "2024-01-15T10:00:00Z", "lvl": "info"},
            {"event": "llm.request", "ts": "2024-01-15T10:00:01Z", "lvl": "debug"},
            {"event": "tool.call", "ts": "2024-01-15T10:00:02Z", "lvl": "debug"},
        ]

        synced = await sqlite_storage.sync_event_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=events,
        )

        assert synced == 3

        retrieved = await sqlite_storage.get_event_lines(
            user_id="user-1", project_slug="project-1", session_id="session-1"
        )

        assert len(retrieved) == 3

    @pytest.mark.asyncio
    async def test_search_events_by_type(self, sqlite_storage):
        """Can search events by type."""
        events = [
            {"event": "session.start", "ts": "2024-01-15T10:00:00Z", "lvl": "info"},
            {"event": "llm.request", "ts": "2024-01-15T10:00:01Z", "lvl": "debug"},
            {"event": "llm.request", "ts": "2024-01-15T10:00:02Z", "lvl": "debug"},
            {"event": "tool.call", "ts": "2024-01-15T10:00:03Z", "lvl": "debug"},
        ]

        await sqlite_storage.sync_event_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=events,
        )

        # Search for llm.request events
        results = await sqlite_storage.search_events(
            user_id="user-1",
            event_type="llm.request",
            limit=10,
        )

        assert len(results) == 2
        assert all(r.metadata["event"] == "llm.request" for r in results)

    @pytest.mark.asyncio
    async def test_search_events_by_level(self, sqlite_storage):
        """Can search events by log level."""
        events = [
            {"event": "session.start", "ts": "2024-01-15T10:00:00Z", "lvl": "info"},
            {"event": "tool.error", "ts": "2024-01-15T10:00:01Z", "lvl": "error"},
            {"event": "llm.request", "ts": "2024-01-15T10:00:02Z", "lvl": "debug"},
        ]

        await sqlite_storage.sync_event_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=events,
        )

        # Search for error level events
        results = await sqlite_storage.search_events(
            user_id="user-1",
            level="error",
            limit=10,
        )

        assert len(results) == 1
        assert results[0].metadata["event"] == "tool.error"

    @pytest.mark.asyncio
    async def test_large_event_truncation(self, sqlite_storage):
        """Large events are truncated with metadata preserved."""
        # Create a large event (>400KB)
        large_data = {"data": "x" * 500000}  # ~500KB
        events = [
            {"event": "large.event", "ts": "2024-01-15T10:00:00Z", "lvl": "info", **large_data}
        ]

        synced = await sqlite_storage.sync_event_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=events,
        )

        assert synced == 1

        # Retrieve - should have truncation flag
        retrieved = await sqlite_storage.get_event_lines(
            user_id="user-1", project_slug="project-1", session_id="session-1"
        )

        assert len(retrieved) == 1
        assert retrieved[0]["data_truncated"] is True
        assert retrieved[0]["data_size_bytes"] > 400 * 1024


class TestSQLiteStatistics:
    """Tests for analytics and statistics."""

    @pytest.mark.asyncio
    async def test_get_statistics(self, sqlite_storage):
        """Statistics aggregation works."""
        # Create test sessions
        for i in range(5):
            await sqlite_storage.upsert_session_metadata(
                user_id="user-1",
                host_id="host-1",
                metadata={
                    "session_id": f"session-{i}",
                    "project_slug": "project-1" if i < 3 else "project-2",
                    "bundle": "foundation" if i % 2 == 0 else "custom",
                    "created": datetime.now(UTC).isoformat(),
                    "turn_count": i + 1,
                },
            )

        stats = await sqlite_storage.get_session_statistics(user_id="user-1")

        assert stats["total_sessions"] == 5
        assert stats["sessions_by_project"]["project-1"] == 3
        assert stats["sessions_by_project"]["project-2"] == 2
        assert stats["sessions_by_bundle"]["foundation"] == 3
        assert stats["sessions_by_bundle"]["custom"] == 2

    @pytest.mark.asyncio
    async def test_get_statistics_with_filters(self, sqlite_storage):
        """Filtered statistics work correctly."""
        for i in range(3):
            await sqlite_storage.upsert_session_metadata(
                user_id="user-1",
                host_id="host-1",
                metadata={
                    "session_id": f"session-{i}",
                    "project_slug": f"project-{i}",
                    "bundle": "foundation",
                    "created": datetime.now(UTC).isoformat(),
                    "turn_count": i + 1,
                },
            )

        # Filter by specific project
        stats = await sqlite_storage.get_session_statistics(
            user_id="user-1",
            filters=SearchFilters(project_slug="project-1"),
        )

        assert stats["total_sessions"] == 1
        assert "project-1" in stats["sessions_by_project"]


class TestSQLiteEmbeddingOperations:
    """Tests for embedding operations."""

    @pytest.mark.asyncio
    async def test_upsert_embeddings(self, sqlite_storage):
        """Can backfill embeddings for existing transcripts."""
        # Create transcript without embeddings
        await sqlite_storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=[{"role": "user", "content": "Test", "turn": 0}],
            embeddings=None,  # Explicitly no embeddings
        )

        # Backfill embeddings
        new_embeddings = [
            {"sequence": 0, "vector": [0.5] * 3072, "metadata": {"model": "test-model"}}
        ]

        updated = await sqlite_storage.upsert_embeddings(
            user_id="user-1",
            project_slug="project-1",
            session_id="session-1",
            embeddings=new_embeddings,
        )

        assert updated == 1

        # Verify embedding was added
        embedding = await sqlite_storage._get_embedding("user-1", "session-1", 0)
        assert embedding is not None
        assert embedding == [0.5] * 3072

    @pytest.mark.asyncio
    async def test_vector_search_numpy_fallback(self, sqlite_storage):
        """Vector search works with numpy fallback."""
        lines = [
            {"role": "user", "content": "First message", "turn": 0},
            {"role": "user", "content": "Second message", "turn": 1},
            {"role": "user", "content": "Third message", "turn": 2},
        ]

        await sqlite_storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=lines,
        )

        # Generate query vector
        query_vector = await sqlite_storage.embedding_provider.embed_text("search query")

        # Perform vector search
        results = await sqlite_storage.vector_search(
            user_id="user-1", query_vector=query_vector, filters=None, top_k=2
        )

        assert len(results) <= 2
        assert all(hasattr(r, "score") for r in results)
        assert all(r.source == "semantic" for r in results)


class TestSQLiteHybridSearch:
    """Tests for hybrid search functionality."""

    @pytest.mark.asyncio
    async def test_hybrid_search(self, sqlite_storage):
        """Hybrid search combines full-text and semantic."""
        lines = [
            {"role": "user", "content": "How do I use vector search in databases?", "turn": 0},
            {"role": "assistant", "content": "Vector search uses embeddings", "turn": 0},
            {"role": "user", "content": "What about keyword matching?", "turn": 1},
        ]

        await sqlite_storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=lines,
        )

        # Hybrid search
        results = await sqlite_storage.search_transcripts(
            user_id="user-1",
            options=TranscriptSearchOptions(
                query="vector search", search_type="hybrid", mmr_lambda=0.7
            ),
            limit=10,
        )

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_hybrid_fallback_without_embeddings(self, sqlite_storage_no_embeddings):
        """Hybrid search falls back to full_text without embeddings."""
        lines = [{"role": "user", "content": "Test message", "turn": 0}]

        await sqlite_storage_no_embeddings.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=lines,
        )

        # Request hybrid but should fallback
        results = await sqlite_storage_no_embeddings.search_transcripts(
            user_id="user-1",
            options=TranscriptSearchOptions(query="Test", search_type="hybrid"),
            limit=10,
        )

        assert len(results) == 1
        assert results[0].source == "full_text"


class TestSQLiteSyncStats:
    """Tests for get_session_sync_stats."""

    @pytest.mark.asyncio
    async def test_empty_session(self, sqlite_storage_no_embeddings):
        """Stats for a session with no data returns zeros and None ranges."""
        stats = await sqlite_storage_no_embeddings.get_session_sync_stats(
            user_id="user-1",
            project_slug="project-1",
            session_id="nonexistent-session",
        )

        assert stats.event_count == 0
        assert stats.transcript_count == 0
        assert stats.event_ts_range == (None, None)
        assert stats.transcript_ts_range == (None, None)

    @pytest.mark.asyncio
    async def test_with_data(self, sqlite_storage_no_embeddings):
        """Stats return correct counts and timestamp ranges."""
        storage = sqlite_storage_no_embeddings

        # Sync events
        events = [
            {"event": "session.start", "ts": "2024-01-15T10:00:00Z", "lvl": "info"},
            {"event": "llm.request", "ts": "2024-01-15T10:00:05Z", "lvl": "debug"},
            {"event": "session.end", "ts": "2024-01-15T10:00:10Z", "lvl": "info"},
        ]
        await storage.sync_event_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=events,
        )

        # Sync transcripts
        transcripts = [
            {"role": "user", "content": "Hello", "turn": 1, "ts": "2024-01-15T10:00:01Z"},
            {"role": "assistant", "content": "Hi!", "turn": 1, "ts": "2024-01-15T10:00:02Z"},
        ]
        await storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=transcripts,
        )

        stats = await storage.get_session_sync_stats(
            user_id="user-1",
            project_slug="project-1",
            session_id="session-1",
        )

        assert stats.event_count == 3
        assert stats.transcript_count == 2
        assert stats.event_ts_range == ("2024-01-15T10:00:00Z", "2024-01-15T10:00:10Z")
        assert stats.transcript_ts_range == ("2024-01-15T10:00:01Z", "2024-01-15T10:00:02Z")

    @pytest.mark.asyncio
    async def test_session_isolation(self, sqlite_storage_no_embeddings):
        """Stats only include data for the queried session."""
        storage = sqlite_storage_no_embeddings

        # Session 1: 2 events, 1 transcript
        await storage.sync_event_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=[
                {"event": "a", "ts": "2024-01-15T10:00:00Z", "lvl": "info"},
                {"event": "b", "ts": "2024-01-15T10:00:01Z", "lvl": "info"},
            ],
        )
        await storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=[{"role": "user", "content": "Hello", "turn": 1, "ts": "2024-01-15T10:00:00Z"}],
        )

        # Session 2: 5 events, 3 transcripts
        await storage.sync_event_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-2",
            lines=[
                {"event": "c", "ts": "2024-01-16T10:00:00Z", "lvl": "info"},
                {"event": "d", "ts": "2024-01-16T10:00:01Z", "lvl": "info"},
                {"event": "e", "ts": "2024-01-16T10:00:02Z", "lvl": "info"},
                {"event": "f", "ts": "2024-01-16T10:00:03Z", "lvl": "info"},
                {"event": "g", "ts": "2024-01-16T10:00:04Z", "lvl": "info"},
            ],
        )
        await storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-2",
            lines=[
                {"role": "user", "content": "A", "turn": 1, "ts": "2024-01-16T10:00:00Z"},
                {"role": "assistant", "content": "B", "turn": 1, "ts": "2024-01-16T10:00:01Z"},
                {"role": "user", "content": "C", "turn": 2, "ts": "2024-01-16T10:00:02Z"},
            ],
        )

        # Query session-1 only
        stats1 = await storage.get_session_sync_stats(
            user_id="user-1",
            project_slug="project-1",
            session_id="session-1",
        )
        assert stats1.event_count == 2
        assert stats1.transcript_count == 1

        # Query session-2 only
        stats2 = await storage.get_session_sync_stats(
            user_id="user-1",
            project_slug="project-1",
            session_id="session-2",
        )
        assert stats2.event_count == 5
        assert stats2.transcript_count == 3


class TestSQLiteContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Backend works as async context manager."""
        config = SQLiteConfig(db_path=":memory:")

        async with SQLiteBackend(config=config) as storage:
            assert storage._initialized is True

            # Use storage
            await storage.upsert_session_metadata(
                user_id="user-1",
                host_id="host-1",
                metadata={
                    "session_id": "test",
                    "project_slug": "test",
                    "bundle": "test",
                    "created": datetime.now(UTC).isoformat(),
                    "turn_count": 1,
                },
            )

        # Should be closed after context
        assert storage._initialized is False
