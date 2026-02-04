"""
DuckDB storage backend with vector search support.

Uses DuckDB's VSS (Vector Similarity Search) extension for efficient local vector search.
Ideal for development, testing, and single-machine deployments.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from ..content_extraction import count_embeddable_content_types, extract_all_embeddable_content
from ..embeddings import EmbeddingProvider
from ..exceptions import StorageConnectionError, StorageIOError
from ..search.mmr import compute_mmr
from .base import (
    EventSearchOptions,
    SearchFilters,
    SearchResult,
    StorageBackend,
    TranscriptSearchOptions,
)

logger = logging.getLogger(__name__)


@dataclass
class DuckDBConfig:
    """Configuration for DuckDB storage."""

    db_path: str | Path = ":memory:"  # Use :memory: for in-memory database
    vector_dimensions: int = 3072  # text-embedding-3-large

    @classmethod
    def from_env(cls) -> DuckDBConfig:
        """Create config from environment variables."""
        import os

        db_path = os.environ.get("AMPLIFIER_DUCKDB_PATH", ":memory:")
        dimensions_str = os.environ.get("AMPLIFIER_DUCKDB_VECTOR_DIMENSIONS", "3072")

        return cls(
            db_path=db_path,
            vector_dimensions=int(dimensions_str),
        )


class DuckDBBackend(StorageBackend):
    """
    DuckDB storage backend with vector similarity search.

    Features:
    - Local database (file-based or in-memory)
    - VSS extension for vector search
    - HNSW indexes for efficient similarity search
    - Full SQL query capabilities
    - Perfect for development and single-machine use
    """

    def __init__(
        self,
        config: DuckDBConfig,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        """
        Initialize DuckDB backend.

        Args:
            config: DuckDB configuration
            embedding_provider: Optional embedding provider for semantic search
        """
        self.config = config
        self.embedding_provider = embedding_provider
        self.conn: Any = None  # DuckDB connection (using Any due to type stub limitations)
        self._initialized = False

    @classmethod
    async def create(
        cls,
        config: DuckDBConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> DuckDBBackend:
        """Create and initialize DuckDB backend."""
        if config is None:
            config = DuckDBConfig.from_env()

        backend = cls(config, embedding_provider)
        await backend.initialize()
        return backend

    @staticmethod
    def _format_vector_literal(vector: list[float]) -> str:
        """
        Format Python list as DuckDB array literal.

        CRITICAL: DuckDB-VSS requires constant array expressions for HNSW index usage.
        Parameter binding (?::FLOAT[N]) creates VALUE_PARAMETER, not VALUE_CONSTANT,
        which causes the optimizer to fall back to sequential scan instead of using the index.

        Args:
            vector: Python list of floats

        Returns:
            DuckDB array literal string: "[1.0, 2.0, 3.0]::FLOAT[N]"
        """
        # Validate all elements are numeric
        if not all(isinstance(x, (int, float)) for x in vector):
            raise ValueError("Vector must contain only numeric values")

        vec_str = "[" + ", ".join(str(float(x)) for x in vector) + "]"
        dimension = len(vector)
        return f"{vec_str}::FLOAT[{dimension}]"

    async def _embed_non_none(self, texts: list[str | None]) -> list[list[float] | None]:
        """
        Generate embeddings only for non-None texts.

        Args:
            texts: List of text strings (some may be None)

        Returns:
            List of embeddings with None preserved at same indices
        """
        if not self.embedding_provider:
            return [None] * len(texts)

        # Find non-None texts and their positions
        texts_to_embed: list[tuple[int, str]] = [
            (i, text) for i, text in enumerate(texts) if text is not None
        ]

        if not texts_to_embed:
            return [None] * len(texts)

        # Extract just the text strings
        just_texts = [text for _, text in texts_to_embed]

        # Generate embeddings
        embeddings = await self.embedding_provider.embed_batch(just_texts)

        # Place embeddings back at original indices
        result: list[list[float] | None] = [None] * len(texts)
        for (original_idx, _), embedding in zip(texts_to_embed, embeddings, strict=True):
            result[original_idx] = embedding

        return result

    async def _generate_multi_vector_embeddings(
        self, lines: list[dict[str, Any]]
    ) -> dict[str, list[list[float] | None]]:
        """
        Generate embeddings for all content types in a batch.

        Extracts all embeddable content from messages and generates
        embeddings for each content type separately.

        Args:
            lines: Transcript lines from Amplifier

        Returns:
            Dict with keys: user_query, assistant_response, assistant_thinking, tool_output
            Each value is a list of embeddings (or None) matching the input lines
        """
        if not self.embedding_provider:
            none_list: list[list[float] | None] = [None] * len(lines)
            return {
                "user_query": none_list,
                "assistant_response": none_list,
                "assistant_thinking": none_list,
                "tool_output": none_list,
            }

        # Extract all content types
        all_content = [extract_all_embeddable_content(line) for line in lines]

        # Separate by content type
        user_queries = [c["user_query"] for c in all_content]
        assistant_responses = [c["assistant_response"] for c in all_content]
        assistant_thinkings = [c["assistant_thinking"] for c in all_content]
        tool_outputs = [c["tool_output"] for c in all_content]

        # Generate embeddings for each type
        return {
            "user_query": await self._embed_non_none(user_queries),
            "assistant_response": await self._embed_non_none(assistant_responses),
            "assistant_thinking": await self._embed_non_none(assistant_thinkings),
            "tool_output": await self._embed_non_none(tool_outputs),
        }

    async def initialize(self) -> None:
        """Initialize DuckDB connection and schema."""
        if self._initialized:
            return

        def _init() -> None:
            """Run sync initialization in thread."""
            self.conn = duckdb.connect(str(self.config.db_path))

            # Enable experimental HNSW persistence for disk-based databases
            if str(self.config.db_path) != ":memory:":
                self.conn.execute("SET hnsw_enable_experimental_persistence = true")
                logger.info("Enabled HNSW experimental persistence for disk storage")

            # Install and load VSS extension for vector search
            try:
                self.conn.execute("INSTALL vss")
                self.conn.execute("LOAD vss")
                logger.info("VSS extension loaded for vector search")
            except Exception as e:
                logger.warning(f"VSS extension not available: {e}")

            # Create sessions table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    user_id VARCHAR NOT NULL,
                    session_id VARCHAR NOT NULL PRIMARY KEY,
                    host_id VARCHAR NOT NULL,
                    project_slug VARCHAR,
                    bundle VARCHAR,
                    created TIMESTAMP,
                    updated TIMESTAMP,
                    turn_count INTEGER,
                    metadata JSON,
                    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create transcripts table with multi-vector support
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS transcripts (
                    id VARCHAR NOT NULL PRIMARY KEY,
                    user_id VARCHAR NOT NULL,
                    host_id VARCHAR NOT NULL,
                    project_slug VARCHAR,
                    session_id VARCHAR NOT NULL,
                    sequence INTEGER NOT NULL,
                    role VARCHAR,
                    content JSON,
                    turn INTEGER,
                    ts TIMESTAMP,
                    user_query_vector FLOAT[{self.config.vector_dimensions}],
                    assistant_response_vector FLOAT[{self.config.vector_dimensions}],
                    assistant_thinking_vector FLOAT[{self.config.vector_dimensions}],
                    tool_output_vector FLOAT[{self.config.vector_dimensions}],
                    embedding_model VARCHAR,
                    vector_metadata JSON,
                    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create events table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id VARCHAR NOT NULL PRIMARY KEY,
                    user_id VARCHAR NOT NULL,
                    host_id VARCHAR NOT NULL,
                    project_slug VARCHAR,
                    session_id VARCHAR NOT NULL,
                    sequence INTEGER NOT NULL,
                    ts TIMESTAMP,
                    lvl VARCHAR,
                    event VARCHAR,
                    turn INTEGER,
                    data JSON,
                    data_truncated BOOLEAN DEFAULT FALSE,
                    data_size_bytes INTEGER,
                    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, created DESC)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_transcripts_session "
                "ON transcripts(user_id, project_slug, session_id, sequence)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_session "
                "ON events(user_id, project_slug, session_id, sequence)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_type ON events(user_id, event, ts)"
            )

            # Create HNSW vector indexes for each content type
            try:
                self.conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_query_vector
                    ON transcripts USING HNSW (user_query_vector)
                    WITH (metric = 'cosine')
                """)
                self.conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_assistant_response_vector
                    ON transcripts USING HNSW (assistant_response_vector)
                    WITH (metric = 'cosine')
                """)
                self.conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_assistant_thinking_vector
                    ON transcripts USING HNSW (assistant_thinking_vector)
                    WITH (metric = 'cosine')
                """)
                self.conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tool_output_vector
                    ON transcripts USING HNSW (tool_output_vector)
                    WITH (metric = 'cosine')
                """)
                logger.info("HNSW vector indexes created for all content types")
            except Exception as e:
                logger.warning(f"Failed to create HNSW indexes: {e}")

        try:
            await asyncio.to_thread(_init)
            self._initialized = True
            logger.info(f"DuckDB backend initialized: {self.config.db_path}")
        except Exception as e:
            raise StorageConnectionError(str(self.config.db_path), e) from e

    async def close(self) -> None:
        """Close DuckDB connection."""
        if self.conn:
            await asyncio.to_thread(self.conn.close)
            self.conn = None

        if self.embedding_provider:
            await self.embedding_provider.close()

        self._initialized = False

    # =========================================================================
    # Session Metadata Operations
    # =========================================================================

    async def upsert_session_metadata(
        self,
        user_id: str,
        host_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """Upsert session metadata."""

        def _upsert() -> None:
            if self.conn is None:
                raise StorageIOError("upsert_session", cause=RuntimeError("Not initialized"))

            self.conn.execute(
                """
                INSERT INTO sessions (
                    user_id, session_id, host_id, project_slug, bundle,
                    created, updated, turn_count, metadata, synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (session_id) DO UPDATE SET
                    host_id = EXCLUDED.host_id,
                    project_slug = EXCLUDED.project_slug,
                    bundle = EXCLUDED.bundle,
                    updated = EXCLUDED.updated,
                    turn_count = EXCLUDED.turn_count,
                    metadata = EXCLUDED.metadata,
                    synced_at = EXCLUDED.synced_at
                """,
                [
                    user_id,
                    metadata.get("session_id"),
                    host_id,
                    metadata.get("project_slug"),
                    metadata.get("bundle"),
                    metadata.get("created"),
                    metadata.get("updated"),
                    metadata.get("turn_count", 0),
                    json.dumps(metadata),
                    datetime.now(UTC).isoformat(),
                ],
            )

        await asyncio.to_thread(_upsert)

    async def get_session_metadata(
        self,
        user_id: str,
        session_id: str,
    ) -> dict[str, Any] | None:
        """Get session metadata by ID."""

        def _get() -> dict[str, Any] | None:
            if self.conn is None:
                raise StorageIOError("get_session", cause=RuntimeError("Not initialized"))

            result = self.conn.execute(
                """
                SELECT user_id, session_id, host_id, project_slug, bundle,
                       created, updated, turn_count, metadata
                FROM sessions
                WHERE user_id = ? AND session_id = ?
                """,
                [user_id, session_id],
            ).fetchone()

            if result is None:
                return None

            # Parse metadata JSON
            return json.loads(result[8]) if result[8] else {}

        return await asyncio.to_thread(_get)

    async def search_sessions(
        self,
        user_id: str,
        filters: SearchFilters,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Search sessions with advanced filters."""

        def _search() -> list[dict[str, Any]]:
            if self.conn is None:
                raise StorageIOError("search_sessions", cause=RuntimeError("Not initialized"))

            # Build WHERE clause
            where_parts = ["user_id = ?"]
            params: list[Any] = [user_id]

            if filters.project_slug:
                where_parts.append("project_slug = ?")
                params.append(filters.project_slug)

            if filters.bundle:
                where_parts.append("bundle = ?")
                params.append(filters.bundle)

            if filters.start_date:
                where_parts.append("created >= ?")
                params.append(filters.start_date)

            if filters.end_date:
                where_parts.append("created <= ?")
                params.append(filters.end_date)

            if filters.min_turn_count is not None:
                where_parts.append("turn_count >= ?")
                params.append(filters.min_turn_count)

            if filters.max_turn_count is not None:
                where_parts.append("turn_count <= ?")
                params.append(filters.max_turn_count)

            where_clause = " AND ".join(where_parts)

            query = f"""
                SELECT user_id, session_id, host_id, project_slug, bundle,
                       created, updated, turn_count, metadata
                FROM sessions
                WHERE {where_clause}
                ORDER BY created DESC
                LIMIT ?
            """

            params.append(limit)

            results = self.conn.execute(query, params).fetchall()

            # Parse metadata JSON for each result
            return [json.loads(row[8]) if row[8] else {} for row in results]

        return await asyncio.to_thread(_search)

    async def delete_session(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
    ) -> bool:
        """Delete session and all its data."""

        def _delete() -> bool:
            if self.conn is None:
                raise StorageIOError("delete_session", cause=RuntimeError("Not initialized"))

            # Delete in transaction
            self.conn.begin()
            try:
                # Delete transcripts
                self.conn.execute(
                    "DELETE FROM transcripts WHERE user_id = ? AND session_id = ?",
                    [user_id, session_id],
                )

                # Delete events
                self.conn.execute(
                    "DELETE FROM events WHERE user_id = ? AND session_id = ?",
                    [user_id, session_id],
                )

                # Delete session - check if it existed
                cursor = self.conn.execute(
                    "SELECT session_id FROM sessions WHERE user_id = ? AND session_id = ?",
                    [user_id, session_id],
                )
                existed = cursor.fetchone() is not None

                if existed:
                    self.conn.execute(
                        "DELETE FROM sessions WHERE user_id = ? AND session_id = ?",
                        [user_id, session_id],
                    )

                self.conn.commit()
                return existed

            except Exception:
                self.conn.rollback()
                raise

        return await asyncio.to_thread(_delete)

    # =========================================================================
    # Transcript Operations
    # =========================================================================

    async def sync_transcript_lines(
        self,
        user_id: str,
        host_id: str,
        project_slug: str,
        session_id: str,
        lines: list[dict[str, Any]],
        start_sequence: int = 0,
        embeddings: dict[str, list[list[float] | None]] | None = None,
    ) -> int:
        """Sync transcript lines with optional multi-vector embeddings."""
        if not lines:
            return 0

        # Generate embeddings if not provided and we have a provider
        if embeddings is None and self.embedding_provider:
            embeddings = await self._generate_multi_vector_embeddings(lines)
            counts = count_embeddable_content_types(lines)
            logger.info(
                f"Generated embeddings: {counts['user_query']} user, "
                f"{counts['assistant_response']} responses, "
                f"{counts['assistant_thinking']} thinking, "
                f"{counts['tool_output']} tool"
            )

        def _sync() -> int:
            if self.conn is None:
                raise StorageIOError("sync_transcript", cause=RuntimeError("Not initialized"))

            synced = 0
            for i, line in enumerate(lines):
                sequence = start_sequence + i
                doc_id = f"{session_id}_msg_{sequence}"

                # Extract vectors for each content type
                user_query_vec = None
                assistant_response_vec = None
                assistant_thinking_vec = None
                tool_output_vec = None

                if embeddings:
                    user_query_vec = embeddings.get("user_query", [None] * len(lines))[i]
                    assistant_response_vec = embeddings.get(
                        "assistant_response", [None] * len(lines)
                    )[i]
                    assistant_thinking_vec = embeddings.get(
                        "assistant_thinking", [None] * len(lines)
                    )[i]
                    tool_output_vec = embeddings.get("tool_output", [None] * len(lines))[i]

                # Build vector metadata
                vector_metadata = {
                    "has_user_query": user_query_vec is not None,
                    "has_assistant_response": assistant_response_vec is not None,
                    "has_assistant_thinking": assistant_thinking_vec is not None,
                    "has_tool_output": tool_output_vec is not None,
                }

                # Store content as JSON (handles both string and array formats)
                content = line.get("content")
                content_json = json.dumps(content) if content is not None else None

                self.conn.execute(
                    """
                    INSERT INTO transcripts (
                        id, user_id, host_id, project_slug, session_id, sequence,
                        role, content, turn, ts,
                        user_query_vector, assistant_response_vector,
                        assistant_thinking_vector, tool_output_vector,
                        embedding_model, vector_metadata, synced_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        user_query_vector = EXCLUDED.user_query_vector,
                        assistant_response_vector = EXCLUDED.assistant_response_vector,
                        assistant_thinking_vector = EXCLUDED.assistant_thinking_vector,
                        tool_output_vector = EXCLUDED.tool_output_vector,
                        embedding_model = EXCLUDED.embedding_model,
                        vector_metadata = EXCLUDED.vector_metadata,
                        synced_at = EXCLUDED.synced_at
                    """,
                    [
                        doc_id,
                        user_id,
                        host_id,
                        project_slug,
                        session_id,
                        sequence,
                        line.get("role"),
                        content_json,
                        line.get("turn"),
                        line.get("ts") or line.get("timestamp"),
                        user_query_vec,
                        assistant_response_vec,
                        assistant_thinking_vec,
                        tool_output_vec,
                        self.embedding_provider.model_name if self.embedding_provider else None,
                        json.dumps(vector_metadata),
                        datetime.now(UTC).isoformat(),
                    ],
                )
                synced += 1

            return synced

        return await asyncio.to_thread(_sync)

    async def get_transcript_lines(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
        after_sequence: int = -1,
    ) -> list[dict[str, Any]]:
        """Get transcript lines for a session."""

        def _get() -> list[dict[str, Any]]:
            if self.conn is None:
                raise StorageIOError("get_transcript", cause=RuntimeError("Not initialized"))

            result = self.conn.execute(
                """
                SELECT id, sequence, role, content, turn, ts
                FROM transcripts
                WHERE user_id = ? AND project_slug = ? AND session_id = ?
                AND sequence > ?
                ORDER BY sequence
                """,
                [user_id, project_slug, session_id, after_sequence],
            ).fetchall()

            return [
                {
                    "id": row[0],
                    "sequence": row[1],
                    "role": row[2],
                    "content": json.loads(row[3]) if row[3] else None,
                    "turn": row[4],
                    "ts": row[5],
                }
                for row in result
            ]

        return await asyncio.to_thread(_get)

    async def search_transcripts(
        self,
        user_id: str,
        options: TranscriptSearchOptions,
        limit: int = 100,
    ) -> list[SearchResult]:
        """Search transcripts with hybrid support."""
        if options.search_type == "semantic" or options.search_type == "hybrid":
            if not await self.supports_vector_search():
                logger.warning(
                    "Vector search requested but not available, falling back to full_text"
                )
                options.search_type = "full_text"

        if options.search_type == "full_text":
            return await self._full_text_search_transcripts(user_id, options, limit)
        elif options.search_type == "semantic":
            return await self._semantic_search_transcripts(user_id, options, limit)
        elif options.search_type == "hybrid":
            return await self._hybrid_search_transcripts(user_id, options, limit)
        else:
            raise ValueError(f"Unknown search_type: {options.search_type}")

    async def _full_text_search_transcripts(
        self,
        user_id: str,
        options: TranscriptSearchOptions,
        limit: int,
    ) -> list[SearchResult]:
        """Full-text search using SQL LIKE."""

        def _search() -> list[SearchResult]:
            if self.conn is None:
                raise StorageIOError("search_transcripts", cause=RuntimeError("Not initialized"))

            # Build WHERE clause
            where_parts = ["user_id = ?"]
            params: list[Any] = [user_id]

            # Role filters
            role_conditions = []
            if options.search_in_user:
                role_conditions.append("role = 'user'")
            if options.search_in_assistant:
                role_conditions.append("role = 'assistant'")

            if role_conditions:
                where_parts.append(f"({' OR '.join(role_conditions)})")

            # Content search
            where_parts.append("content LIKE ?")
            params.append(f"%{options.query}%")

            # Apply filters
            if options.filters:
                if options.filters.project_slug:
                    where_parts.append("project_slug = ?")
                    params.append(options.filters.project_slug)

                if options.filters.start_date:
                    where_parts.append("ts >= ?")
                    params.append(options.filters.start_date)

                if options.filters.end_date:
                    where_parts.append("ts <= ?")
                    params.append(options.filters.end_date)

            where_clause = " AND ".join(where_parts)

            query = f"""
                SELECT id, session_id, project_slug, sequence, content, role, turn, ts
                FROM transcripts
                WHERE {where_clause}
                ORDER BY ts DESC
                LIMIT ?
            """
            params.append(limit)

            results_raw = self.conn.execute(query, params).fetchall()

            return [
                SearchResult(
                    session_id=row[1],
                    project_slug=row[2],
                    sequence=row[3],
                    content=row[4] or "",
                    metadata={
                        "id": row[0],
                        "role": row[5],
                        "turn": row[6],
                        "ts": row[7],
                    },
                    score=1.0,
                    source="full_text",
                )
                for row in results_raw
            ]

        return await asyncio.to_thread(_search)

    async def _semantic_search_transcripts(
        self,
        user_id: str,
        options: TranscriptSearchOptions,
        limit: int,
    ) -> list[SearchResult]:
        """Semantic search using vector similarity."""
        if not self.embedding_provider:
            raise ValueError("Embedding provider required for semantic search")

        query_vector = await self.embedding_provider.embed_text(options.query)
        return await self.vector_search(user_id, query_vector, options.filters, limit)

    async def _hybrid_search_transcripts(
        self,
        user_id: str,
        options: TranscriptSearchOptions,
        limit: int,
    ) -> list[SearchResult]:
        """Hybrid search with MMR re-ranking."""
        if not self.embedding_provider:
            logger.warning("No embedding provider, falling back to full_text")
            return await self._full_text_search_transcripts(user_id, options, limit)

        # Get more candidates
        candidate_limit = limit * 3

        # Get both result sets
        text_results = await self._full_text_search_transcripts(user_id, options, candidate_limit)
        semantic_results = await self._semantic_search_transcripts(
            user_id, options, candidate_limit
        )

        # Merge and deduplicate
        seen = set()
        combined: list[SearchResult] = []
        for result in text_results + semantic_results:
            key = (result.session_id, result.sequence)
            if key not in seen:
                seen.add(key)
                combined.append(result)

        if len(combined) <= limit:
            return combined[:limit]

        # Apply MMR re-ranking
        query_vector = await self.embedding_provider.embed_text(options.query)
        query_np = np.array(query_vector)

        # Extract embeddings (fetch from DB if not in metadata)
        vectors = []
        for result in combined:
            if "embedding" in result.metadata and result.metadata["embedding"]:
                vectors.append(np.array(result.metadata["embedding"]))
            else:
                # Fetch embedding from DB
                embedding = await self._get_embedding(user_id, result.session_id, result.sequence)
                if embedding:
                    vectors.append(np.array(embedding))
                else:
                    vectors.append(np.zeros(len(query_vector)))

        # Apply MMR
        mmr_results = compute_mmr(
            vectors=vectors,
            query=query_np,
            lambda_param=options.mmr_lambda,
            top_k=limit,
        )

        return [combined[idx] for idx, _ in mmr_results]

    async def _get_embedding(
        self, user_id: str, session_id: str, sequence: int
    ) -> list[float] | None:
        """
        Fetch first available embedding for a transcript message.

        Returns the first non-null vector from any of the 4 vector columns.
        Used primarily for testing.
        """

        def _get() -> list[float] | None:
            if self.conn is None:
                return None

            result = self.conn.execute(
                """
                SELECT user_query_vector, assistant_response_vector,
                       assistant_thinking_vector, tool_output_vector
                FROM transcripts
                WHERE user_id = ? AND session_id = ? AND sequence = ?
                """,
                [user_id, session_id, sequence],
            ).fetchone()

            if result:
                # Return first non-null vector
                for vec in result:
                    if vec is not None:
                        return vec
            return None

        return await asyncio.to_thread(_get)

    # =========================================================================
    # Event Operations
    # =========================================================================

    async def sync_event_lines(
        self,
        user_id: str,
        host_id: str,
        project_slug: str,
        session_id: str,
        lines: list[dict[str, Any]],
        start_sequence: int = 0,
    ) -> int:
        """Sync event lines."""
        if not lines:
            return 0

        def _sync() -> int:
            if self.conn is None:
                raise StorageIOError("sync_events", cause=RuntimeError("Not initialized"))

            synced = 0
            for i, line in enumerate(lines):
                sequence = start_sequence + i
                doc_id = f"{session_id}_evt_{sequence}"

                # Check size
                line_json = json.dumps(line)
                data_size = len(line_json.encode("utf-8"))

                # Store full data if under threshold
                data_to_store = line if data_size <= 400 * 1024 else {}
                is_truncated = data_size > 400 * 1024

                self.conn.execute(
                    """
                    INSERT INTO events (
                        id, user_id, host_id, project_slug, session_id, sequence,
                        ts, lvl, event, turn, data, data_truncated, data_size_bytes, synced_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (id) DO UPDATE SET
                        data = EXCLUDED.data,
                        data_truncated = EXCLUDED.data_truncated,
                        synced_at = EXCLUDED.synced_at
                    """,
                    [
                        doc_id,
                        user_id,
                        host_id,
                        project_slug,
                        session_id,
                        sequence,
                        line.get("ts"),
                        line.get("lvl"),
                        line.get("event"),
                        line.get("turn"),
                        json.dumps(data_to_store),
                        is_truncated,
                        data_size,
                        datetime.now(UTC).isoformat(),
                    ],
                )
                synced += 1

            return synced

        return await asyncio.to_thread(_sync)

    async def get_event_lines(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
        after_sequence: int = -1,
    ) -> list[dict[str, Any]]:
        """Get event lines for a session."""

        def _get() -> list[dict[str, Any]]:
            if self.conn is None:
                raise StorageIOError("get_events", cause=RuntimeError("Not initialized"))

            result = self.conn.execute(
                """
                SELECT id, sequence, ts, lvl, event, turn,
                       data_truncated, data_size_bytes
                FROM events
                WHERE user_id = ? AND project_slug = ? AND session_id = ?
                AND sequence > ?
                ORDER BY sequence
                """,
                [user_id, project_slug, session_id, after_sequence],
            ).fetchall()

            return [
                {
                    "id": row[0],
                    "sequence": row[1],
                    "ts": row[2],
                    "lvl": row[3],
                    "event": row[4],
                    "turn": row[5],
                    "data_truncated": row[6],
                    "data_size_bytes": row[7],
                }
                for row in result
            ]

        return await asyncio.to_thread(_get)

    async def search_events(
        self,
        user_id: str,
        options: EventSearchOptions,
        limit: int = 100,
    ) -> list[SearchResult]:
        """Search events by type, tool, and filters."""

        def _search() -> list[SearchResult]:
            if self.conn is None:
                raise StorageIOError("search_events", cause=RuntimeError("Not initialized"))

            # Build WHERE clause
            where_parts = ["user_id = ?"]
            params: list[Any] = [user_id]

            if options.event_type:
                where_parts.append("event = ?")
                params.append(options.event_type)

            if options.tool_name:
                # Search in JSON data field
                where_parts.append("json_extract(data, '$.tool') = ?")
                params.append(options.tool_name)

            if options.level:
                where_parts.append("lvl = ?")
                params.append(options.level)

            # Apply filters
            if options.filters:
                if options.filters.project_slug:
                    where_parts.append("project_slug = ?")
                    params.append(options.filters.project_slug)

                if options.filters.start_date:
                    where_parts.append("ts >= ?")
                    params.append(options.filters.start_date)

                if options.filters.end_date:
                    where_parts.append("ts <= ?")
                    params.append(options.filters.end_date)

            where_clause = " AND ".join(where_parts)

            query = f"""
                SELECT id, session_id, project_slug, sequence, event, ts, data
                FROM events
                WHERE {where_clause}
                ORDER BY ts DESC
                LIMIT ?
            """
            params.append(limit)

            results_raw = self.conn.execute(query, params).fetchall()

            return [
                SearchResult(
                    session_id=row[1],
                    project_slug=row[2],
                    sequence=row[3],
                    content=row[4] or "",
                    metadata={
                        "id": row[0],
                        "event": row[4],
                        "ts": row[5],
                        "data": json.loads(row[6]) if row[6] else {},
                    },
                    score=1.0,
                    source="event_search",
                )
                for row in results_raw
            ]

        return await asyncio.to_thread(_search)

    # =========================================================================
    # Vector/Embedding Operations
    # =========================================================================

    async def supports_vector_search(self) -> bool:
        """Check if vector search is available."""
        if not self.embedding_provider:
            return False

        # Check if VSS extension is loaded
        def _check() -> bool:
            if self.conn is None:
                return False
            try:
                # Try to use array_cosine_similarity function (from VSS)
                self.conn.execute(
                    "SELECT array_cosine_similarity([1.0]::FLOAT[1], [1.0]::FLOAT[1])"
                ).fetchone()
                return True
            except Exception:
                return False

        return await asyncio.to_thread(_check)

    async def upsert_embeddings(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
        embeddings: list[dict[str, Any]],
    ) -> int:
        """Upsert embeddings for existing transcript messages."""

        def _upsert() -> int:
            if self.conn is None:
                raise StorageIOError("upsert_embeddings", cause=RuntimeError("Not initialized"))

            updated = 0
            for emb in embeddings:
                doc_id = f"{session_id}_msg_{emb['sequence']}"

                self.conn.execute(
                    """
                    UPDATE transcripts
                    SET embedding = ?,
                        embedding_model = ?,
                        synced_at = ?
                    WHERE id = ? AND user_id = ?
                    """,
                    [
                        emb["vector"],
                        emb.get("metadata", {}).get("model", "unknown"),
                        datetime.now(UTC).isoformat(),
                        doc_id,
                        user_id,
                    ],
                )
                updated += 1

            return updated

        return await asyncio.to_thread(_upsert)

    async def vector_search(
        self,
        user_id: str,
        query_vector: list[float],
        filters: SearchFilters | None = None,
        top_k: int = 100,
        vector_columns: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Perform multi-vector similarity search using DuckDB VSS.

        Searches across specified vector columns and returns best matches.

        Args:
            user_id: User identifier
            query_vector: Query embedding vector
            filters: Optional search filters
            top_k: Number of results to return
            vector_columns: Which vector columns to search. Default: all non-null vectors.
                Options: ["user_query_vector", "assistant_response_vector",
                          "assistant_thinking_vector", "tool_output_vector"]
        """
        if vector_columns is None:
            # Search all vector columns by default
            vector_columns = [
                "user_query_vector",
                "assistant_response_vector",
                "assistant_thinking_vector",
                "tool_output_vector",
            ]

        def _search() -> list[SearchResult]:
            if self.conn is None:
                raise StorageIOError("vector_search", cause=RuntimeError("Not initialized"))

            # Build base WHERE clause
            where_parts = ["user_id = ?"]
            params: list[Any] = [user_id]

            if filters:
                if filters.project_slug:
                    where_parts.append("project_slug = ?")
                    params.append(filters.project_slug)

                if filters.start_date:
                    where_parts.append("ts >= ?")
                    params.append(filters.start_date)

                if filters.end_date:
                    where_parts.append("ts <= ?")
                    params.append(filters.end_date)

            # DuckDB vector search - use GREATEST to find best match across all vector columns
            vec_literal = self._format_vector_literal(query_vector)

            # Build similarity expressions for each vector column
            similarity_exprs = []
            for vec_col in vector_columns:
                similarity_exprs.append(
                    f"CASE WHEN {vec_col} IS NOT NULL "
                    f"THEN array_cosine_similarity({vec_col}, {vec_literal}) "
                    f"ELSE 0.0 END"
                )

            # Use GREATEST to find the maximum similarity across all vectors
            max_similarity = f"GREATEST({', '.join(similarity_exprs)})"

            # Ensure at least one vector is not null
            any_vector_not_null = " OR ".join([f"{v} IS NOT NULL" for v in vector_columns])
            where_parts.append(f"({any_vector_not_null})")

            where_clause = " AND ".join(where_parts)

            query = f"""
                SELECT id, session_id, project_slug, sequence, content, role, turn, ts,
                       {max_similarity} AS similarity
                FROM transcripts
                WHERE {where_clause}
                ORDER BY similarity DESC
                LIMIT ?
            """

            params.append(top_k)

            results_raw = self.conn.execute(query, params).fetchall()

            return [
                SearchResult(
                    session_id=row[1],
                    project_slug=row[2],
                    sequence=row[3],
                    content=json.loads(row[4]) if row[4] else "",
                    metadata={
                        "id": row[0],
                        "role": row[5],
                        "turn": row[6],
                        "ts": row[7],
                    },
                    score=float(row[8]) if row[8] is not None else 0.0,
                    source="semantic",
                )
                for row in results_raw
            ]

        return await asyncio.to_thread(_search)

    # =========================================================================
    # Analytics & Aggregations
    # =========================================================================

    async def get_session_statistics(
        self,
        user_id: str,
        filters: SearchFilters | None = None,
    ) -> dict[str, Any]:
        """Get aggregate statistics across sessions."""

        def _get_stats() -> dict[str, Any]:
            if self.conn is None:
                raise StorageIOError("get_statistics", cause=RuntimeError("Not initialized"))

            # Build WHERE clause
            where_parts = ["user_id = ?"]
            params: list[Any] = [user_id]

            if filters:
                if filters.project_slug:
                    where_parts.append("project_slug = ?")
                    params.append(filters.project_slug)

                if filters.start_date:
                    where_parts.append("created >= ?")
                    params.append(filters.start_date)

                if filters.end_date:
                    where_parts.append("created <= ?")
                    params.append(filters.end_date)

            where_clause = " AND ".join(where_parts)

            # Count total sessions
            total_sessions_result = self.conn.execute(
                f"SELECT COUNT(*) FROM sessions WHERE {where_clause}", params
            ).fetchone()
            total_sessions = total_sessions_result[0] if total_sessions_result else 0

            # Aggregate by project
            project_results = self.conn.execute(
                f"""
                SELECT project_slug, COUNT(*) as count
                FROM sessions
                WHERE {where_clause}
                GROUP BY project_slug
                """,
                params,
            ).fetchall()
            projects = {row[0]: row[1] for row in project_results}

            # Aggregate by bundle
            bundle_results = self.conn.execute(
                f"""
                SELECT bundle, COUNT(*) as count
                FROM sessions
                WHERE {where_clause}
                GROUP BY bundle
                """,
                params,
            ).fetchall()
            bundles = {row[0]: row[1] for row in bundle_results}

            return {
                "total_sessions": total_sessions,
                "sessions_by_project": projects,
                "sessions_by_bundle": bundles,
                "filters_applied": filters is not None,
            }

        return await asyncio.to_thread(_get_stats)
