"""
SQLite storage backend with vector search support.

Uses sqlite-vss extension for vector similarity search in SQLite.
Ideal for lightweight deployments, embedded applications, and testing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np

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
class SQLiteConfig:
    """Configuration for SQLite storage."""

    db_path: str | Path = ":memory:"
    vector_dimensions: int = 3072  # text-embedding-3-large

    @classmethod
    def from_env(cls) -> SQLiteConfig:
        """Create config from environment variables."""
        import os

        db_path = os.environ.get("AMPLIFIER_SQLITE_PATH", ":memory:")
        dimensions_str = os.environ.get("AMPLIFIER_SQLITE_VECTOR_DIMENSIONS", "3072")

        return cls(
            db_path=db_path,
            vector_dimensions=int(dimensions_str),
        )


class SQLiteBackend(StorageBackend):
    """
    SQLite storage backend with vector similarity search.

    Features:
    - Single file database
    - sqlite-vss extension for vector search
    - Full SQL capabilities
    - Perfect for embedded applications and testing
    """

    def __init__(
        self,
        config: SQLiteConfig,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        """
        Initialize SQLite backend.

        Args:
            config: SQLite configuration
            embedding_provider: Optional embedding provider
        """
        self.config = config
        self.embedding_provider = embedding_provider
        self.conn: Any = None  # aiosqlite.Connection (using Any due to optional dependency)
        self._initialized = False
        self._vss_available = False

    @classmethod
    async def create(
        cls,
        config: SQLiteConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> SQLiteBackend:
        """Create and initialize SQLite backend."""
        if config is None:
            config = SQLiteConfig.from_env()

        backend = cls(config, embedding_provider)
        await backend.initialize()
        return backend

    async def initialize(self) -> None:
        """Initialize SQLite connection and schema."""
        if self._initialized:
            return

        try:
            self.conn = await aiosqlite.connect(str(self.config.db_path))

            # Enable JSON support
            await self.conn.execute("PRAGMA foreign_keys = ON")

            # Try to load sqlite-vss extension
            try:
                await self.conn.enable_load_extension(True)
                await self.conn.execute("SELECT load_extension('vss0')")
                self._vss_available = True
                logger.info("sqlite-vss extension loaded for vector search")
            except Exception as e:
                logger.warning(f"sqlite-vss not available: {e}")
                self._vss_available = False

            # Create tables
            await self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL PRIMARY KEY,
                    host_id TEXT NOT NULL,
                    project_slug TEXT,
                    bundle TEXT,
                    created TEXT,
                    updated TEXT,
                    turn_count INTEGER,
                    metadata TEXT,
                    synced_at TEXT DEFAULT (datetime('now'))
                )
            """)

            await self.conn.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    id TEXT NOT NULL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    host_id TEXT NOT NULL,
                    project_slug TEXT,
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    role TEXT,
                    content TEXT,
                    turn INTEGER,
                    ts TEXT,
                    embedding_json TEXT,
                    embedding_model TEXT,
                    synced_at TEXT DEFAULT (datetime('now'))
                )
            """)

            await self.conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT NOT NULL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    host_id TEXT NOT NULL,
                    project_slug TEXT,
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    ts TEXT,
                    lvl TEXT,
                    event TEXT,
                    turn INTEGER,
                    data TEXT,
                    data_truncated INTEGER DEFAULT 0,
                    data_size_bytes INTEGER,
                    synced_at TEXT DEFAULT (datetime('now'))
                )
            """)

            # Create indexes
            await self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, created DESC)"
            )
            await self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_transcripts_session "
                "ON transcripts(user_id, project_slug, session_id, sequence)"
            )
            await self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_session "
                "ON events(user_id, project_slug, session_id, sequence)"
            )
            await self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_type ON events(user_id, event, ts)"
            )

            # Create virtual table for vector search if available
            if self._vss_available:
                try:
                    await self.conn.execute(f"""
                        CREATE VIRTUAL TABLE IF NOT EXISTS transcript_embeddings
                        USING vss0(
                            embedding({self.config.vector_dimensions})
                        )
                    """)
                    logger.info("VSS virtual table created for vector search")
                except Exception as e:
                    logger.warning(f"Failed to create VSS table: {e}")
                    self._vss_available = False

            await self.conn.commit()
            self._initialized = True
            logger.info(f"SQLite backend initialized: {self.config.db_path}")

        except Exception as e:
            raise StorageConnectionError(str(self.config.db_path), e) from e

    async def close(self) -> None:
        """Close SQLite connection."""
        if self.conn:
            await self.conn.close()
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
        if self.conn is None:
            raise StorageIOError("upsert_session", cause=RuntimeError("Not initialized"))

        await self.conn.execute(
            """
            INSERT INTO sessions (
                user_id, session_id, host_id, project_slug, bundle,
                created, updated, turn_count, metadata, synced_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (session_id) DO UPDATE SET
                host_id = excluded.host_id,
                project_slug = excluded.project_slug,
                bundle = excluded.bundle,
                updated = excluded.updated,
                turn_count = excluded.turn_count,
                metadata = excluded.metadata,
                synced_at = excluded.synced_at
            """,
            (
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
            ),
        )
        await self.conn.commit()

    async def get_session_metadata(
        self,
        user_id: str,
        session_id: str,
    ) -> dict[str, Any] | None:
        """Get session metadata by ID."""
        if self.conn is None:
            raise StorageIOError("get_session", cause=RuntimeError("Not initialized"))

        async with self.conn.execute(
            """
            SELECT metadata
            FROM sessions
            WHERE user_id = ? AND session_id = ?
            """,
            (user_id, session_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return json.loads(row[0]) if row[0] else {}

    async def search_sessions(
        self,
        user_id: str,
        filters: SearchFilters,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Search sessions with advanced filters."""
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
            SELECT metadata
            FROM sessions
            WHERE {where_clause}
            ORDER BY created DESC
            LIMIT ?
        """
        params.append(limit)

        async with self.conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [json.loads(row[0]) if row[0] else {} for row in rows]

    async def delete_session(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
    ) -> bool:
        """Delete session and all its data."""
        if self.conn is None:
            raise StorageIOError("delete_session", cause=RuntimeError("Not initialized"))

        # Delete in transaction
        await self.conn.execute("BEGIN TRANSACTION")
        try:
            # Delete transcripts
            await self.conn.execute(
                "DELETE FROM transcripts WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )

            # Delete events
            await self.conn.execute(
                "DELETE FROM events WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )

            # Delete session
            cursor = await self.conn.execute(
                "DELETE FROM sessions WHERE user_id = ? AND session_id = ? RETURNING session_id",
                (user_id, session_id),
            )
            row = await cursor.fetchone()

            await self.conn.commit()
            return row is not None

        except Exception:
            await self.conn.rollback()
            raise

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
        embeddings: list[list[float]] | None = None,
    ) -> int:
        """Sync transcript lines with optional embeddings."""
        if not lines:
            return 0

        if self.conn is None:
            raise StorageIOError("sync_transcript", cause=RuntimeError("Not initialized"))

        # Generate embeddings if needed
        if embeddings is None and self.embedding_provider:
            texts = [line.get("content", "") for line in lines]
            embeddings = await self.embedding_provider.embed_batch(texts)
            logger.info(f"Generated {len(embeddings)} embeddings during ingestion")

        synced = 0
        for i, line in enumerate(lines):
            sequence = start_sequence + i
            doc_id = f"{session_id}_msg_{sequence}"

            # Serialize embedding to JSON for SQLite storage
            embedding_json = (
                json.dumps(embeddings[i]) if embeddings and i < len(embeddings) else None
            )

            await self.conn.execute(
                """
                INSERT INTO transcripts (
                    id, user_id, host_id, project_slug, session_id, sequence,
                    role, content, turn, ts, embedding_json, embedding_model, synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    content = excluded.content,
                    embedding_json = excluded.embedding_json,
                    embedding_model = excluded.embedding_model,
                    synced_at = excluded.synced_at
                """,
                (
                    doc_id,
                    user_id,
                    host_id,
                    project_slug,
                    session_id,
                    sequence,
                    line.get("role"),
                    line.get("content"),
                    line.get("turn"),
                    line.get("ts") or line.get("timestamp"),
                    embedding_json,
                    self.embedding_provider.model_name if self.embedding_provider else None,
                    datetime.now(UTC).isoformat(),
                ),
            )

            # If VSS is available and we have an embedding, also insert into virtual table
            if self._vss_available and embedding_json:
                try:
                    await self.conn.execute(
                        """
                        INSERT INTO transcript_embeddings (rowid, embedding)
                        VALUES (?, ?)
                        ON CONFLICT (rowid) DO UPDATE SET embedding = excluded.embedding
                        """,
                        (sequence, embedding_json),
                    )
                except Exception as e:
                    logger.warning(f"Failed to insert into VSS table: {e}")

            synced += 1

        await self.conn.commit()
        return synced

    async def get_transcript_lines(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
        after_sequence: int = -1,
    ) -> list[dict[str, Any]]:
        """Get transcript lines for a session."""
        if self.conn is None:
            raise StorageIOError("get_transcript", cause=RuntimeError("Not initialized"))

        async with self.conn.execute(
            """
            SELECT id, sequence, role, content, turn, ts
            FROM transcripts
            WHERE user_id = ? AND project_slug = ? AND session_id = ?
            AND sequence > ?
            ORDER BY sequence
            """,
            (user_id, project_slug, session_id, after_sequence),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "sequence": row[1],
                    "role": row[2],
                    "content": row[3],
                    "turn": row[4],
                    "ts": row[5],
                }
                for row in rows
            ]

    async def search_transcripts(
        self,
        user_id: str,
        options: TranscriptSearchOptions,
        limit: int = 100,
    ) -> list[SearchResult]:
        """Search transcripts with hybrid support."""
        if options.search_type == "semantic" or options.search_type == "hybrid":
            if not await self.supports_vector_search():
                logger.warning("Vector search not available, falling back to full_text")
                options.search_type = "full_text"

        if options.search_type == "full_text":
            return await self._full_text_search(user_id, options, limit)
        elif options.search_type == "semantic":
            return await self._semantic_search(user_id, options, limit)
        elif options.search_type == "hybrid":
            return await self._hybrid_search(user_id, options, limit)
        else:
            raise ValueError(f"Unknown search_type: {options.search_type}")

    async def _full_text_search(
        self,
        user_id: str,
        options: TranscriptSearchOptions,
        limit: int,
    ) -> list[SearchResult]:
        """Full-text search using SQLite FTS or LIKE."""
        if self.conn is None:
            raise StorageIOError("search", cause=RuntimeError("Not initialized"))

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

        async with self.conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
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
                for row in rows
            ]

    async def _semantic_search(
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

    async def _hybrid_search(
        self,
        user_id: str,
        options: TranscriptSearchOptions,
        limit: int,
    ) -> list[SearchResult]:
        """Hybrid search with MMR re-ranking."""
        if not self.embedding_provider:
            logger.warning("No embedding provider, falling back to full_text")
            return await self._full_text_search(user_id, options, limit)

        # Get more candidates
        candidate_limit = limit * 3

        # Get both result sets
        text_results = await self._full_text_search(user_id, options, candidate_limit)
        semantic_results = await self._semantic_search(user_id, options, candidate_limit)

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

        # Apply MMR
        query_vector = await self.embedding_provider.embed_text(options.query)
        query_np = np.array(query_vector)

        # Get embeddings
        vectors = []
        for result in combined:
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
        """Fetch embedding for a specific transcript."""
        if self.conn is None:
            return None

        async with self.conn.execute(
            """
            SELECT embedding_json
            FROM transcripts
            WHERE user_id = ? AND session_id = ? AND sequence = ?
            """,
            (user_id, session_id, sequence),
        ) as cursor:
            row = await cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return None

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
            is_truncated = 1 if data_size > 400 * 1024 else 0

            await self.conn.execute(
                """
                INSERT INTO events (
                    id, user_id, host_id, project_slug, session_id, sequence,
                    ts, lvl, event, turn, data, data_truncated, data_size_bytes, synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    data = excluded.data,
                    data_truncated = excluded.data_truncated,
                    synced_at = excluded.synced_at
                """,
                (
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
                ),
            )
            synced += 1

        await self.conn.commit()
        return synced

    async def get_event_lines(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
        after_sequence: int = -1,
    ) -> list[dict[str, Any]]:
        """Get event lines for a session."""
        if self.conn is None:
            raise StorageIOError("get_events", cause=RuntimeError("Not initialized"))

        async with self.conn.execute(
            """
            SELECT id, sequence, ts, lvl, event, turn, data_truncated, data_size_bytes
            FROM events
            WHERE user_id = ? AND project_slug = ? AND session_id = ?
            AND sequence > ?
            ORDER BY sequence
            """,
            (user_id, project_slug, session_id, after_sequence),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "sequence": row[1],
                    "ts": row[2],
                    "lvl": row[3],
                    "event": row[4],
                    "turn": row[5],
                    "data_truncated": bool(row[6]),
                    "data_size_bytes": row[7],
                }
                for row in rows
            ]

    async def search_events(
        self,
        user_id: str,
        options: EventSearchOptions,
        limit: int = 100,
    ) -> list[SearchResult]:
        """Search events by type, tool, and filters."""
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

        async with self.conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
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
                for row in rows
            ]

    # =========================================================================
    # Vector/Embedding Operations
    # =========================================================================

    async def supports_vector_search(self) -> bool:
        """Check if vector search is available."""
        return self._vss_available and self.embedding_provider is not None

    async def upsert_embeddings(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
        embeddings: list[dict[str, Any]],
    ) -> int:
        """Upsert embeddings for existing transcripts."""
        if self.conn is None:
            raise StorageIOError("upsert_embeddings", cause=RuntimeError("Not initialized"))

        updated = 0
        for emb in embeddings:
            doc_id = f"{session_id}_msg_{emb['sequence']}"
            embedding_json = json.dumps(emb["vector"])

            await self.conn.execute(
                """
                UPDATE transcripts
                SET embedding_json = ?,
                    embedding_model = ?,
                    synced_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    embedding_json,
                    emb.get("metadata", {}).get("model", "unknown"),
                    datetime.now(UTC).isoformat(),
                    doc_id,
                    user_id,
                ),
            )

            # Update VSS table if available
            if self._vss_available:
                try:
                    await self.conn.execute(
                        """
                        INSERT INTO transcript_embeddings (rowid, embedding)
                        VALUES (?, ?)
                        ON CONFLICT (rowid) DO UPDATE SET embedding = excluded.embedding
                        """,
                        (emb["sequence"], embedding_json),
                    )
                except Exception as e:
                    logger.warning(f"Failed to update VSS table: {e}")

            updated += 1

        await self.conn.commit()
        return updated

    async def vector_search(
        self,
        user_id: str,
        query_vector: list[float],
        filters: SearchFilters | None = None,
        top_k: int = 100,
    ) -> list[SearchResult]:
        """
        Perform vector similarity search.

        Uses sqlite-vss if available, otherwise falls back to brute-force numpy.
        """
        if self.conn is None:
            raise StorageIOError("vector_search", cause=RuntimeError("Not initialized"))

        if self._vss_available:
            return await self._vss_vector_search(user_id, query_vector, filters, top_k)
        else:
            return await self._numpy_vector_search(user_id, query_vector, filters, top_k)

    async def _vss_vector_search(
        self,
        user_id: str,
        query_vector: list[float],
        filters: SearchFilters | None,
        top_k: int,
    ) -> list[SearchResult]:
        """Vector search using sqlite-vss extension."""
        # Note: Implementation depends on sqlite-vss API
        # This is a placeholder for when sqlite-vss is properly integrated
        logger.warning("sqlite-vss integration not yet implemented, using numpy fallback")
        return await self._numpy_vector_search(user_id, query_vector, filters, top_k)

    async def _numpy_vector_search(
        self,
        user_id: str,
        query_vector: list[float],
        filters: SearchFilters | None,
        top_k: int,
    ) -> list[SearchResult]:
        """Brute-force vector search using numpy cosine similarity."""
        if self.conn is None:
            raise StorageIOError("vector_search", cause=RuntimeError("Not initialized"))

        # Build WHERE clause
        where_parts = ["user_id = ?", "embedding_json IS NOT NULL"]
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

        where_clause = " AND ".join(where_parts)

        query = f"""
            SELECT id, session_id, project_slug, sequence, content, role, turn, ts, embedding_json
            FROM transcripts
            WHERE {where_clause}
        """

        # Fetch all candidates
        async with self.conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        # Compute similarities with numpy
        query_np = np.array(query_vector)
        candidates: list[tuple[float, Any]] = []

        for row in rows:
            if row[8]:  # embedding_json
                embedding = json.loads(row[8])
                embedding_np = np.array(embedding)

                # Cosine similarity
                similarity = float(
                    np.dot(query_np, embedding_np)
                    / (np.linalg.norm(query_np) * np.linalg.norm(embedding_np))
                )

                candidates.append((similarity, row))

        # Sort by similarity (descending) and take top_k
        candidates.sort(reverse=True, key=lambda x: x[0])
        top_candidates = candidates[:top_k]

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
                    "embedding": json.loads(row[8]),
                },
                score=similarity,
                source="semantic",
            )
            for similarity, row in top_candidates
        ]

    # =========================================================================
    # Analytics & Aggregations
    # =========================================================================

    async def get_session_statistics(
        self,
        user_id: str,
        filters: SearchFilters | None = None,
    ) -> dict[str, Any]:
        """Get aggregate statistics across sessions."""
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
        async with self.conn.execute(
            f"SELECT COUNT(*) FROM sessions WHERE {where_clause}", params
        ) as cursor:
            row = await cursor.fetchone()
            total_sessions = row[0] if row else 0

        # Aggregate by project
        async with self.conn.execute(
            f"""
            SELECT project_slug, COUNT(*) as count
            FROM sessions
            WHERE {where_clause}
            GROUP BY project_slug
            """,
            params,
        ) as cursor:
            project_rows = await cursor.fetchall()
            projects = {row[0]: row[1] for row in project_rows}

        # Aggregate by bundle
        async with self.conn.execute(
            f"""
            SELECT bundle, COUNT(*) as count
            FROM sessions
            WHERE {where_clause}
            GROUP BY bundle
            """,
            params,
        ) as cursor:
            bundle_rows = await cursor.fetchall()
            bundles = {row[0]: row[1] for row in bundle_rows}

        return {
            "total_sessions": total_sessions,
            "sessions_by_project": projects,
            "sessions_by_bundle": bundles,
            "filters_applied": filters is not None,
        }
