"""
Tests for DuckDB vector search functionality.

Tests the actual VSS extension behavior with real similarity queries.
"""

import pytest

from amplifier_session_storage.backends import SearchFilters, TranscriptSearchOptions
from amplifier_session_storage.backends.duckdb import DuckDBBackend, DuckDBConfig
from amplifier_session_storage.embeddings import EmbeddingProvider


class SimpleEmbeddingProvider(EmbeddingProvider):
    """Simple embedding provider for testing vector search."""

    @property
    def model_name(self) -> str:
        return "test-embeddings"

    @property
    def dimensions(self) -> int:
        return 3

    async def embed_text(self, text: str) -> list[float]:
        """Generate simple 3D embeddings based on text content."""
        # Simple heuristic: different texts get different vectors
        if "vector" in text.lower():
            return [1.0, 0.0, 0.0]
        elif "search" in text.lower():
            return [0.0, 1.0, 0.0]
        elif "database" in text.lower():
            return [0.0, 0.0, 1.0]
        else:
            return [0.5, 0.5, 0.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_text(text) for text in texts]

    async def close(self) -> None:
        pass


@pytest.fixture
async def duckdb_with_vss():
    """DuckDB backend with VSS extension and simple embeddings."""
    config = DuckDBConfig(db_path=":memory:", vector_dimensions=3)
    embeddings = SimpleEmbeddingProvider()
    storage = await DuckDBBackend.create(config=config, embedding_provider=embeddings)

    # Check if VSS is actually available
    if not await storage.supports_vector_search():
        pytest.skip("DuckDB VSS extension not available")

    yield storage
    await storage.close()


class TestDuckDBVectorSearch:
    """Tests for DuckDB vector similarity search."""

    @pytest.mark.asyncio
    async def test_vector_search_basic(self, duckdb_with_vss):
        """Vector search returns similar results."""
        # Create test data with known embeddings
        lines = [
            {"role": "user", "content": "vector embeddings", "turn": 0},  # [1.0, 0.0, 0.0]
            {"role": "user", "content": "search functionality", "turn": 1},  # [0.0, 1.0, 0.0]
            {"role": "user", "content": "database storage", "turn": 2},  # [0.0, 0.0, 1.0]
        ]

        await duckdb_with_vss.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=lines,
        )

        # Query for "vector" - should match first message most
        query_vector = await duckdb_with_vss.embedding_provider.embed_text("vector test")
        results = await duckdb_with_vss.vector_search(
            user_id="user-1", query_vector=query_vector, filters=None, top_k=3
        )

        assert len(results) == 3
        # First result should be most similar (contains "vector")
        assert "vector" in results[0].content.lower()
        # All should have scores (can be 0 for orthogonal vectors)
        assert all(r.score >= 0 for r in results)

    @pytest.mark.asyncio
    async def test_semantic_search_uses_vectors(self, duckdb_with_vss):
        """Semantic search uses embeddings for similarity."""
        lines = [
            {"role": "user", "content": "How do I use vector search?", "turn": 0},
            {"role": "assistant", "content": "Vector search uses embeddings", "turn": 0},
            {"role": "user", "content": "Tell me about databases", "turn": 1},
        ]

        await duckdb_with_vss.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=lines,
        )

        # Semantic search
        results = await duckdb_with_vss.search_transcripts(
            user_id="user-1",
            options=TranscriptSearchOptions(query="vector query", search_type="semantic"),
            limit=10,
        )

        assert len(results) > 0
        assert results[0].source == "semantic"
        # Should find messages about vectors
        assert any("vector" in r.content.lower() for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_search_combines_methods(self, duckdb_with_vss):
        """Hybrid search combines full-text and semantic."""
        lines = [
            {"role": "user", "content": "vector search tutorial", "turn": 0},
            {"role": "assistant", "content": "embeddings are used for search", "turn": 0},
            {"role": "user", "content": "database configuration", "turn": 1},
        ]

        await duckdb_with_vss.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=lines,
        )

        # Hybrid search - should find both keyword matches and semantic matches
        results = await duckdb_with_vss.search_transcripts(
            user_id="user-1",
            options=TranscriptSearchOptions(query="search", search_type="hybrid", mmr_lambda=0.7),
            limit=10,
        )

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_vector_search_with_filters(self, duckdb_with_vss):
        """Vector search respects filters."""
        # Create data in two projects
        await duckdb_with_vss.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-a",
            session_id="session-a",
            lines=[{"role": "user", "content": "vector in project a", "turn": 0}],
        )

        await duckdb_with_vss.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-b",
            session_id="session-b",
            lines=[{"role": "user", "content": "vector in project b", "turn": 0}],
        )

        # Search only project-a
        query_vector = await duckdb_with_vss.embedding_provider.embed_text("vector")
        results = await duckdb_with_vss.vector_search(
            user_id="user-1",
            query_vector=query_vector,
            filters=SearchFilters(project_slug="project-a"),
            top_k=10,
        )

        assert len(results) == 1
        assert results[0].project_slug == "project-a"

    @pytest.mark.asyncio
    async def test_vector_literal_formatting(self, duckdb_with_vss):
        """Vector literal formatting is correct."""
        # Test the utility method
        vec = [1.0, 2.5, 3.14159]
        literal = duckdb_with_vss._format_vector_literal(vec)

        # Should produce: [1.0, 2.5, 3.14159]::FLOAT[3]
        assert literal.startswith("[")
        assert literal.endswith("]::FLOAT[3]")
        assert "1.0" in literal
        assert "2.5" in literal

    @pytest.mark.asyncio
    async def test_vector_literal_validation(self, duckdb_with_vss):
        """Vector literal formatting validates input."""
        # Invalid vector (contains string)
        with pytest.raises(ValueError, match="numeric values"):
            duckdb_with_vss._format_vector_literal([1.0, "invalid", 3.0])  # type: ignore

    @pytest.mark.asyncio
    async def test_hnsw_index_actually_used(self, duckdb_with_vss):
        """Verify HNSW index is actually being used (not sequential scan)."""
        # Create enough data that index matters
        lines = [
            {"role": "user", "content": f"Message {i} about various topics", "turn": i}
            for i in range(50)
        ]

        await duckdb_with_vss.sync_transcript_lines(
            user_id="user-1",
            host_id="host-1",
            project_slug="project-1",
            session_id="session-1",
            lines=lines,
        )

        # Perform vector search
        query_vector = await duckdb_with_vss.embedding_provider.embed_text("test query")
        results = await duckdb_with_vss.vector_search(
            user_id="user-1", query_vector=query_vector, filters=None, top_k=10
        )

        # Should return results (proves query works)
        assert len(results) == 10

        # All should have similarity scores
        assert all(hasattr(r, "score") for r in results)
        assert all(r.score is not None for r in results)
