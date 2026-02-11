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

        # Verify externalized vector documents were created
        container = cosmos_storage._get_container("session_data")
        pk = cosmos_storage.make_partition_key("test-user", "test-project", test_session_id)

        # Query for vector documents
        vector_docs = []
        async for doc in container.query_items(
            query="SELECT * FROM c WHERE c.partition_key = @pk AND c.type = 'transcript_vector'",
            parameters=[{"name": "@pk", "value": pk}],
        ):
            vector_docs.append(doc)

        assert len(vector_docs) > 0, "No transcript_vector documents created"
        for vd in vector_docs:
            assert vd["type"] == "transcript_vector"
            assert "parentId" in vd or "parent_id" in vd
            assert "contentType" in vd or "content_type" in vd
            assert "vector" in vd
            assert isinstance(vd["vector"], list)
            assert len(vd["vector"]) == 3072

        # Verify transcript docs don't have old inline vectors
        transcript_docs = []
        async for doc in container.query_items(
            query="SELECT * FROM c WHERE c.partition_key = @pk AND c.type = 'transcript'",
            parameters=[{"name": "@pk", "value": pk}],
        ):
            transcript_docs.append(doc)

        for td in transcript_docs:
            assert "user_query_vector" not in td, "Old inline vector found on transcript"
            assert "assistant_response_vector" not in td
            assert "assistant_thinking_vector" not in td
            assert "tool_output_vector" not in td

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
    async def test_session_sync_stats(self, cosmos_storage):
        """Sync stats aggregation works without GROUP BY (regression test).

        Validates that get_session_sync_stats uses Cosmos-compatible queries
        (separate single-aggregate queries instead of GROUP BY with multiple
        aggregates, which Cosmos DB cross-partition queries don't support).
        """
        test_session_id = f"test-syncstats-{datetime.now(UTC).timestamp()}"
        test_user = "test-user"
        test_project = "test-project"

        # Seed transcripts (include ts — real sync data always has timestamps)
        lines = [
            {
                "role": "user",
                "content": "First message",
                "turn": 0,
                "ts": datetime.now(UTC).isoformat(),
            },
            {
                "role": "assistant",
                "content": "First reply",
                "turn": 0,
                "ts": datetime.now(UTC).isoformat(),
            },
            {
                "role": "user",
                "content": "Second message",
                "turn": 1,
                "ts": datetime.now(UTC).isoformat(),
            },
        ]
        synced_transcripts = await cosmos_storage.sync_transcript_lines(
            user_id=test_user,
            host_id="test-host",
            project_slug=test_project,
            session_id=test_session_id,
            lines=lines,
        )
        assert synced_transcripts == 3

        # Seed events
        events = [
            {"event": "session.start", "ts": datetime.now(UTC).isoformat(), "lvl": "info"},
            {"event": "llm.request", "ts": datetime.now(UTC).isoformat(), "lvl": "debug"},
        ]
        synced_events = await cosmos_storage.sync_event_lines(
            user_id=test_user,
            host_id="test-host",
            project_slug=test_project,
            session_id=test_session_id,
            lines=events,
        )
        assert synced_events == 2

        # Call the method that previously used unsupported GROUP BY
        stats = await cosmos_storage.get_session_sync_stats(
            user_id=test_user,
            project_slug=test_project,
            session_id=test_session_id,
        )

        # Verify counts
        assert stats.event_count == 2, f"Expected 2 events, got {stats.event_count}"
        assert stats.transcript_count == 3, f"Expected 3 transcripts, got {stats.transcript_count}"

        # Verify timestamp ranges are populated
        assert stats.event_ts_range[0] is not None, "event earliest_ts should not be None"
        assert stats.event_ts_range[1] is not None, "event latest_ts should not be None"
        assert stats.transcript_ts_range[0] is not None, "transcript earliest_ts should not be None"
        assert stats.transcript_ts_range[1] is not None, "transcript latest_ts should not be None"

        # earliest <= latest
        assert stats.event_ts_range[0] <= stats.event_ts_range[1]
        assert stats.transcript_ts_range[0] <= stats.transcript_ts_range[1]

    @pytest.mark.asyncio
    async def test_session_sync_stats_empty_session(self, cosmos_storage):
        """Sync stats returns zeros for a session with no data."""
        stats = await cosmos_storage.get_session_sync_stats(
            user_id="test-user",
            project_slug="test-project",
            session_id=f"nonexistent-{datetime.now(UTC).timestamp()}",
        )

        assert stats.event_count == 0
        assert stats.transcript_count == 0
        assert stats.event_ts_range == (None, None)
        assert stats.transcript_ts_range == (None, None)

    @pytest.mark.asyncio
    async def test_backfill_embeddings(self, cosmos_storage):
        """Backfill generates vectors for transcripts that lack them."""
        test_session_id = f"test-backfill-{datetime.now(UTC).timestamp()}"
        test_user = "test-user"
        test_project = "test-project"

        # Step 1: Temporarily remove embedding provider so sync doesn't generate vectors
        original_provider = cosmos_storage.embedding_provider
        cosmos_storage.embedding_provider = None

        lines = [
            {
                "role": "user",
                "content": "Backfill test message one",
                "turn": 0,
                "ts": datetime.now(UTC).isoformat(),
            },
            {
                "role": "assistant",
                "content": "Backfill test reply one",
                "turn": 0,
                "ts": datetime.now(UTC).isoformat(),
            },
            {
                "role": "user",
                "content": "Backfill test message two",
                "turn": 1,
                "ts": datetime.now(UTC).isoformat(),
            },
        ]

        synced = await cosmos_storage.sync_transcript_lines(
            user_id=test_user,
            host_id="test-host",
            project_slug=test_project,
            session_id=test_session_id,
            lines=lines,
        )
        assert synced == 3

        # Verify no vector documents were created
        container = cosmos_storage._get_container("session_data")
        pk = cosmos_storage.make_partition_key(test_user, test_project, test_session_id)
        vector_docs_before = []
        async for doc in container.query_items(
            query="SELECT * FROM c WHERE c.partition_key = @pk AND c.type = 'transcript_vector'",
            parameters=[{"name": "@pk", "value": pk}],
        ):
            vector_docs_before.append(doc)
        assert len(vector_docs_before) == 0, "No vectors should exist before backfill"

        # Step 2: Restore embedding provider and backfill
        cosmos_storage.embedding_provider = original_provider

        result = await cosmos_storage.backfill_embeddings(
            user_id=test_user,
            project_slug=test_project,
            session_id=test_session_id,
        )

        assert result.transcripts_found == 3, f"Expected 3, got {result.transcripts_found}"
        assert result.vectors_stored > 0, f"Expected >0, got {result.vectors_stored}"
        assert result.vectors_failed == 0, f"Expected 0 failures, got {result.vectors_failed}"
        assert result.errors == []

        # Verify vector documents were created
        vector_docs_after = []
        async for doc in container.query_items(
            query="SELECT * FROM c WHERE c.partition_key = @pk AND c.type = 'transcript_vector'",
            parameters=[{"name": "@pk", "value": pk}],
        ):
            vector_docs_after.append(doc)
        assert len(vector_docs_after) > 0, "Vectors should exist after backfill"

        # Verify backfill is now a no-op (all transcripts have vectors)
        result2 = await cosmos_storage.backfill_embeddings(
            user_id=test_user,
            project_slug=test_project,
            session_id=test_session_id,
        )
        assert result2.transcripts_found == 0, "Second backfill should find 0 missing"

    @pytest.mark.asyncio
    async def test_rebuild_vectors(self, cosmos_storage):
        """Rebuild deletes all vectors and regenerates them."""
        test_session_id = f"test-rebuild-{datetime.now(UTC).timestamp()}"
        test_user = "test-user"
        test_project = "test-project"

        # Sync transcripts WITH embeddings
        lines = [
            {
                "role": "user",
                "content": "Rebuild test message",
                "turn": 0,
                "ts": datetime.now(UTC).isoformat(),
            },
            {
                "role": "assistant",
                "content": "Rebuild test reply",
                "turn": 0,
                "ts": datetime.now(UTC).isoformat(),
            },
        ]

        synced = await cosmos_storage.sync_transcript_lines(
            user_id=test_user,
            host_id="test-host",
            project_slug=test_project,
            session_id=test_session_id,
            lines=lines,
        )
        assert synced == 2

        # Verify vectors exist
        container = cosmos_storage._get_container("session_data")
        pk = cosmos_storage.make_partition_key(test_user, test_project, test_session_id)
        vector_docs_before = []
        async for doc in container.query_items(
            query="SELECT * FROM c WHERE c.partition_key = @pk AND c.type = 'transcript_vector'",
            parameters=[{"name": "@pk", "value": pk}],
        ):
            vector_docs_before.append(doc)
        assert len(vector_docs_before) > 0, "Vectors should exist after sync"

        # Rebuild
        result = await cosmos_storage.rebuild_vectors(
            user_id=test_user,
            project_slug=test_project,
            session_id=test_session_id,
        )

        assert result.transcripts_found == 2, f"Expected 2, got {result.transcripts_found}"
        assert result.vectors_stored > 0, f"Expected >0, got {result.vectors_stored}"
        assert result.vectors_failed == 0, f"Expected 0 failures, got {result.vectors_failed}"
        assert result.errors == []

        # Verify vectors exist after rebuild
        vector_docs_after = []
        async for doc in container.query_items(
            query="SELECT * FROM c WHERE c.partition_key = @pk AND c.type = 'transcript_vector'",
            parameters=[{"name": "@pk", "value": pk}],
        ):
            vector_docs_after.append(doc)
        assert len(vector_docs_after) > 0, "Vectors should exist after rebuild"

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

        # Try semantic search - filter to this session's project to avoid stale data
        results = await cosmos_storage.search_transcripts(
            user_id="test-user",
            options=TranscriptSearchOptions(
                query="semantic query",
                search_type="semantic",
                filters=SearchFilters(project_slug="test-project"),
            ),
            limit=10,
        )

        # Should return actual results
        assert isinstance(results, list)
        assert len(results) > 0, "Vector search returned no results"
        assert results[0].score > 0, "Result has no score"

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
        assert len(results) > 0, "Hybrid search returned no results"


@pytest.mark.integration
class TestCosmosStoredIds:
    """Tests for get_stored_transcript_ids and get_stored_event_ids."""

    @pytest.mark.asyncio
    async def test_empty_session(self, cosmos_storage):
        ids = await cosmos_storage.get_stored_transcript_ids(
            "test-user", "test-project", "nonexistent-session"
        )
        assert ids == []
        ids = await cosmos_storage.get_stored_event_ids(
            "test-user", "test-project", "nonexistent-session"
        )
        assert ids == []

    @pytest.mark.asyncio
    async def test_transcript_ids_after_sync(self, cosmos_storage):
        test_session_id = f"test-ids-t-{datetime.now(UTC).timestamp()}"
        lines = [
            {"role": "user", "content": "hello", "ts": datetime.now(UTC).isoformat()},
            {"role": "assistant", "content": "hi", "ts": datetime.now(UTC).isoformat()},
        ]
        await cosmos_storage.sync_transcript_lines(
            "test-user", "test-host", "test-project", test_session_id, lines, start_sequence=1
        )
        ids = await cosmos_storage.get_stored_transcript_ids(
            "test-user", "test-project", test_session_id
        )
        assert sorted(ids) == [
            f"{test_session_id}_msg_1",
            f"{test_session_id}_msg_2",
        ]
        await cosmos_storage.delete_session("test-user", "test-project", test_session_id)

    @pytest.mark.asyncio
    async def test_event_ids_after_sync(self, cosmos_storage):
        test_session_id = f"test-ids-e-{datetime.now(UTC).timestamp()}"
        events = [
            {
                "ts": datetime.now(UTC).isoformat(),
                "lvl": "INFO",
                "event": "session:start",
                "session_id": test_session_id,
            },
            {
                "ts": datetime.now(UTC).isoformat(),
                "lvl": "INFO",
                "event": "llm:request",
                "session_id": test_session_id,
            },
        ]
        await cosmos_storage.sync_event_lines(
            "test-user", "test-host", "test-project", test_session_id, events, start_sequence=1
        )
        ids = await cosmos_storage.get_stored_event_ids(
            "test-user", "test-project", test_session_id
        )
        assert sorted(ids) == [
            f"{test_session_id}_evt_1",
            f"{test_session_id}_evt_2",
        ]
        await cosmos_storage.delete_session("test-user", "test-project", test_session_id)
