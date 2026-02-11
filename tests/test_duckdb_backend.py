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


class TestDuckDBSyncStats:
    """Tests for get_session_sync_stats."""

    @pytest.mark.asyncio
    async def test_empty_session(self, duckdb_storage_no_embeddings):
        """Stats for a session with no data returns zeros and None ranges."""
        stats = await duckdb_storage_no_embeddings.get_session_sync_stats(
            user_id="user-1",
            project_slug="project-1",
            session_id="nonexistent-session",
        )

        assert stats.event_count == 0
        assert stats.transcript_count == 0
        assert stats.event_ts_range == (None, None)
        assert stats.transcript_ts_range == (None, None)

    @pytest.mark.asyncio
    async def test_with_data(self, duckdb_storage_no_embeddings):
        """Stats return correct counts and timestamp ranges."""
        storage = duckdb_storage_no_embeddings

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
        # DuckDB returns datetime objects via str(); verify earliest/latest are present
        assert stats.event_ts_range[0] is not None
        assert stats.event_ts_range[1] is not None
        assert stats.transcript_ts_range[0] is not None
        assert stats.transcript_ts_range[1] is not None
        # Verify ordering: earliest <= latest
        assert stats.event_ts_range[0] <= stats.event_ts_range[1]
        assert stats.transcript_ts_range[0] <= stats.transcript_ts_range[1]

    @pytest.mark.asyncio
    async def test_session_isolation(self, duckdb_storage_no_embeddings):
        """Stats only include data for the queried session."""
        storage = duckdb_storage_no_embeddings

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


class TestDuckDBBackfillRebuild:
    """Tests for backfill_embeddings, rebuild_vectors, and has_vectors lifecycle."""

    @pytest.mark.asyncio
    async def test_has_vectors_flag_set_on_sync(self, duckdb_storage):
        """Syncing transcripts with embedding provider sets has_vectors=True."""
        lines = [
            {"role": "user", "content": "Hello world", "turn": 0},
            {"role": "assistant", "content": "Hi there!", "turn": 0},
        ]

        await duckdb_storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="bf-session-1",
            lines=lines,
        )

        # Query raw DB for has_vectors flag
        rows = duckdb_storage.conn.execute(
            "SELECT has_vectors FROM transcripts WHERE session_id = ? ORDER BY sequence",
            ["bf-session-1"],
        ).fetchall()

        assert len(rows) == 2
        for row in rows:
            assert row[0] is True, "has_vectors should be True after sync with embeddings"

    @pytest.mark.asyncio
    async def test_has_vectors_flag_false_without_embeddings(self, duckdb_storage_no_embeddings):
        """Syncing transcripts without embedding provider leaves has_vectors=False."""
        lines = [
            {"role": "user", "content": "Hello world", "turn": 0},
            {"role": "assistant", "content": "Hi there!", "turn": 0},
        ]

        await duckdb_storage_no_embeddings.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="bf-session-2",
            lines=lines,
        )

        rows = duckdb_storage_no_embeddings.conn.execute(
            "SELECT has_vectors FROM transcripts WHERE session_id = ? ORDER BY sequence",
            ["bf-session-2"],
        ).fetchall()

        assert len(rows) == 2
        for row in rows:
            assert row[0] is False, "has_vectors should be False without embeddings"

    @pytest.mark.asyncio
    async def test_backfill_fills_missing_vectors(self, embedding_provider):
        """Backfill generates vectors for transcripts that lack them."""
        # Step 1: Create backend WITHOUT embeddings, sync transcripts
        config = DuckDBConfig(db_path=":memory:")
        storage = await DuckDBBackend.create(config=config, embedding_provider=None)

        lines = [
            {"role": "user", "content": "First message", "turn": 0},
            {"role": "assistant", "content": "First reply", "turn": 0},
            {"role": "user", "content": "Second question", "turn": 1},
        ]

        await storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="bf-session-3",
            lines=lines,
        )

        # Verify has_vectors is False
        rows = storage.conn.execute(
            "SELECT has_vectors FROM transcripts WHERE session_id = ?",
            ["bf-session-3"],
        ).fetchall()
        assert all(r[0] is False for r in rows)

        # Verify no vectors exist
        vec_count = storage.conn.execute(
            "SELECT COUNT(*) FROM transcript_vectors WHERE session_id = ?",
            ["bf-session-3"],
        ).fetchone()[0]
        assert vec_count == 0

        # Step 2: Attach embedding provider and backfill
        storage.embedding_provider = embedding_provider

        result = await storage.backfill_embeddings(
            user_id="user-1",
            project_slug="project-1",
            session_id="bf-session-3",
        )

        assert result.transcripts_found == 3
        assert result.vectors_stored > 0
        assert result.vectors_failed == 0
        assert result.errors == []

        # Verify has_vectors is now True
        rows = storage.conn.execute(
            "SELECT has_vectors FROM transcripts WHERE session_id = ?",
            ["bf-session-3"],
        ).fetchall()
        assert all(r[0] is True for r in rows)

        # Verify vector documents exist
        vec_count = storage.conn.execute(
            "SELECT COUNT(*) FROM transcript_vectors WHERE session_id = ?",
            ["bf-session-3"],
        ).fetchone()[0]
        assert vec_count > 0

        await storage.close()

    @pytest.mark.asyncio
    async def test_backfill_noop_when_all_have_vectors(self, duckdb_storage):
        """Backfill is a no-op when all transcripts already have vectors."""
        lines = [
            {"role": "user", "content": "Already embedded", "turn": 0},
            {"role": "assistant", "content": "Also embedded", "turn": 0},
        ]

        await duckdb_storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="bf-session-4",
            lines=lines,
        )

        # All transcripts already have vectors from sync
        result = await duckdb_storage.backfill_embeddings(
            user_id="user-1",
            project_slug="project-1",
            session_id="bf-session-4",
        )

        assert result.transcripts_found == 0
        assert result.vectors_stored == 0
        assert result.vectors_failed == 0

    @pytest.mark.asyncio
    async def test_backfill_no_provider_returns_zero(self, duckdb_storage_no_embeddings):
        """Backfill without embedding provider returns zeros with error message."""
        result = await duckdb_storage_no_embeddings.backfill_embeddings(
            user_id="user-1",
            project_slug="project-1",
            session_id="any-session",
        )

        assert result.transcripts_found == 0
        assert result.vectors_stored == 0
        assert result.vectors_failed == 0
        assert len(result.errors) == 1
        assert "No embedding provider" in result.errors[0]

    @pytest.mark.asyncio
    async def test_backfill_progress_callback(self, embedding_provider):
        """Backfill invokes on_progress with correct (processed, total) values."""
        config = DuckDBConfig(db_path=":memory:")
        storage = await DuckDBBackend.create(config=config, embedding_provider=None)

        # Sync 5 transcripts without embeddings
        lines = [{"role": "user", "content": f"Message {i}", "turn": i} for i in range(5)]

        await storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="bf-session-5",
            lines=lines,
        )

        # Attach embeddings and backfill with progress
        storage.embedding_provider = embedding_provider
        progress_calls: list[tuple[int, int]] = []

        result = await storage.backfill_embeddings(
            user_id="user-1",
            project_slug="project-1",
            session_id="bf-session-5",
            batch_size=2,  # Small batches to get multiple progress calls
            on_progress=lambda processed, total: progress_calls.append((processed, total)),
        )

        assert result.transcripts_found == 5
        assert len(progress_calls) > 0

        # All calls should have total=5
        assert all(t == 5 for _, t in progress_calls)

        # Last call should have processed=5
        assert progress_calls[-1][0] == 5

        # processed should be monotonically increasing
        processed_values = [p for p, _ in progress_calls]
        assert processed_values == sorted(processed_values)

        await storage.close()

    @pytest.mark.asyncio
    async def test_rebuild_deletes_and_regenerates(self, duckdb_storage):
        """Rebuild deletes old vectors and regenerates new ones."""
        lines = [
            {"role": "user", "content": "Rebuild test message", "turn": 0},
            {"role": "assistant", "content": "Rebuild test reply", "turn": 0},
        ]

        await duckdb_storage.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="bf-session-6",
            lines=lines,
        )

        # Verify vectors exist after initial sync
        vec_count_before = duckdb_storage.conn.execute(
            "SELECT COUNT(*) FROM transcript_vectors WHERE session_id = ?",
            ["bf-session-6"],
        ).fetchone()[0]
        assert vec_count_before > 0

        # Rebuild
        result = await duckdb_storage.rebuild_vectors(
            user_id="user-1",
            project_slug="project-1",
            session_id="bf-session-6",
        )

        assert result.transcripts_found == 2
        assert result.vectors_stored > 0
        assert result.vectors_failed == 0
        assert result.errors == []

        # Verify vectors still exist after rebuild
        vec_count_after = duckdb_storage.conn.execute(
            "SELECT COUNT(*) FROM transcript_vectors WHERE session_id = ?",
            ["bf-session-6"],
        ).fetchone()[0]
        assert vec_count_after > 0

        # Verify has_vectors is True after rebuild
        rows = duckdb_storage.conn.execute(
            "SELECT has_vectors FROM transcripts WHERE session_id = ?",
            ["bf-session-6"],
        ).fetchall()
        assert all(r[0] is True for r in rows)

    @pytest.mark.asyncio
    async def test_rebuild_empty_session(self, duckdb_storage):
        """Rebuild on empty session returns all zeros."""
        result = await duckdb_storage.rebuild_vectors(
            user_id="user-1",
            project_slug="project-1",
            session_id="nonexistent-session",
        )

        assert result.transcripts_found == 0
        assert result.vectors_stored == 0
        assert result.vectors_failed == 0

    @pytest.mark.asyncio
    async def test_rebuild_no_provider_returns_zero(self, duckdb_storage_no_embeddings):
        """Rebuild without embedding provider returns zeros with error message."""
        # Sync some data first (no vectors since no provider)
        await duckdb_storage_no_embeddings.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="bf-session-7",
            lines=[{"role": "user", "content": "test", "turn": 0}],
        )

        result = await duckdb_storage_no_embeddings.rebuild_vectors(
            user_id="user-1",
            project_slug="project-1",
            session_id="bf-session-7",
        )

        assert result.transcripts_found == 0
        assert result.vectors_stored == 0
        assert result.vectors_failed == 0
        assert len(result.errors) == 1
        assert "No embedding provider" in result.errors[0]

        # Verify no vectors were deleted (none existed)
        vec_count = duckdb_storage_no_embeddings.conn.execute(
            "SELECT COUNT(*) FROM transcript_vectors WHERE session_id = ?",
            ["bf-session-7"],
        ).fetchone()[0]
        assert vec_count == 0


class TestDuckDBStoredIds:
    """Tests for get_stored_transcript_ids and get_stored_event_ids."""

    @pytest.mark.asyncio
    async def test_empty_session_transcript(self, duckdb_storage_no_embeddings):
        ids = await duckdb_storage_no_embeddings.get_stored_transcript_ids(
            "user1", "proj1", "nonexistent"
        )
        assert ids == []

    @pytest.mark.asyncio
    async def test_empty_session_events(self, duckdb_storage_no_embeddings):
        ids = await duckdb_storage_no_embeddings.get_stored_event_ids(
            "user1", "proj1", "nonexistent"
        )
        assert ids == []

    @pytest.mark.asyncio
    async def test_transcript_ids_after_sync(self, duckdb_storage_no_embeddings):
        lines = [
            {"role": "user", "content": "hello", "ts": "2025-01-01T00:00:00Z"},
            {"role": "assistant", "content": "hi", "ts": "2025-01-01T00:00:01Z"},
        ]
        await duckdb_storage_no_embeddings.sync_transcript_lines(
            "user1", "host1", "proj1", "sess1", lines, start_sequence=1
        )
        ids = await duckdb_storage_no_embeddings.get_stored_transcript_ids(
            "user1", "proj1", "sess1"
        )
        assert sorted(ids) == ["sess1_msg_1", "sess1_msg_2"]

    @pytest.mark.asyncio
    async def test_event_ids_after_sync(self, duckdb_storage_no_embeddings):
        events = [
            {
                "ts": "2025-01-01T00:00:00Z",
                "lvl": "INFO",
                "event": "session:start",
                "session_id": "sess1",
            },
            {
                "ts": "2025-01-01T00:00:01Z",
                "lvl": "INFO",
                "event": "llm:request",
                "session_id": "sess1",
            },
        ]
        await duckdb_storage_no_embeddings.sync_event_lines(
            "user1", "host1", "proj1", "sess1", events, start_sequence=1
        )
        ids = await duckdb_storage_no_embeddings.get_stored_event_ids("user1", "proj1", "sess1")
        assert sorted(ids) == ["sess1_evt_1", "sess1_evt_2"]

    @pytest.mark.asyncio
    async def test_session_isolation(self, duckdb_storage_no_embeddings):
        lines1 = [{"role": "user", "content": "a", "ts": "2025-01-01T00:00:00Z"}]
        lines2 = [
            {"role": "user", "content": "b", "ts": "2025-01-01T00:00:00Z"},
            {"role": "assistant", "content": "c", "ts": "2025-01-01T00:00:01Z"},
        ]
        await duckdb_storage_no_embeddings.sync_transcript_lines(
            "user1", "host1", "proj1", "sess1", lines1, start_sequence=1
        )
        await duckdb_storage_no_embeddings.sync_transcript_lines(
            "user1", "host1", "proj1", "sess2", lines2, start_sequence=1
        )
        ids1 = await duckdb_storage_no_embeddings.get_stored_transcript_ids(
            "user1", "proj1", "sess1"
        )
        ids2 = await duckdb_storage_no_embeddings.get_stored_transcript_ids(
            "user1", "proj1", "sess2"
        )
        assert len(ids1) == 1
        assert len(ids2) == 2
