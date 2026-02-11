"""
DuckDB storage backend with vector search support.

Uses DuckDB's VSS (Vector Similarity Search) extension for efficient local vector search.
Ideal for development, testing, and single-machine deployments.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from ..chunking import chunk_text
from ..content_extraction import (
    count_embeddable_content_types,
    count_tokens,
    extract_all_embeddable_content,
)
from ..embeddings import EmbeddingProvider
from ..embeddings.mixin import EmbeddingMixin
from ..exceptions import StorageConnectionError, StorageIOError
from ..search.mmr import compute_mmr
from .base import (
    EmbeddingOperationResult,
    MessageContext,
    SearchFilters,
    SearchResult,
    SessionSyncStats,
    StorageBackend,
    TranscriptMessage,
    TranscriptSearchOptions,
    TurnContext,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Column Definitions - Centralized for consistency and maintainability
# =============================================================================

# Transcript columns for standard read operations
TRANSCRIPT_READ_COLUMNS = (
    "id",
    "user_id",
    "host_id",
    "project_slug",
    "session_id",
    "sequence",
    "role",
    "content",
    "turn",
    "ts",
)

# Event columns for standard read operations
EVENT_READ_COLUMNS = (
    "id",
    "user_id",
    "host_id",
    "project_slug",
    "session_id",
    "sequence",
    "ts",
    "lvl",
    "event",
    "turn",
    "data",
    "data_truncated",
    "data_size_bytes",
)

# Session columns for standard read operations
SESSION_READ_COLUMNS = (
    "user_id",
    "session_id",
    "host_id",
    "project_slug",
    "bundle",
    "created",
    "updated",
    "turn_count",
    "metadata",
)

# SQL for creating views over the externalized vector schema
_CREATE_VIEWS_SQL = """
-- Simple view since transcripts no longer has vectors
CREATE OR REPLACE VIEW transcript_messages AS
SELECT * FROM transcripts;

-- Join view for search result building
CREATE OR REPLACE VIEW vectors_with_context AS
SELECT
    v.id as vector_id, v.parent_id, v.content_type,
    v.chunk_index, v.total_chunks, v.span_start, v.span_end,
    v.source_text, v.token_count, v.vector, v.embedding_model,
    t.user_id, t.session_id, t.project_slug, t.role, t.content,
    t.sequence, t.turn, t.ts
FROM transcript_vectors v
JOIN transcripts t ON t.id = v.parent_id;
"""


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


class DuckDBBackend(EmbeddingMixin, StorageBackend):
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

    async def initialize(self) -> None:
        """Initialize DuckDB connection and schema."""
        if self._initialized:
            return

        def _init() -> None:
            """Run sync initialization in thread."""
            self.conn = duckdb.connect(str(self.config.db_path))

            # Install and load VSS extension for vector search
            # Must be done BEFORE setting hnsw_enable_experimental_persistence
            vss_available = False
            try:
                self.conn.execute("INSTALL vss")
                self.conn.execute("LOAD vss")
                vss_available = True
                logger.info("VSS extension loaded for vector search")
            except Exception as e:
                logger.warning(f"VSS extension not available: {e}")

            # Enable experimental HNSW persistence for disk-based databases
            # Only if VSS extension is loaded (the setting requires VSS)
            if vss_available and str(self.config.db_path) != ":memory:":
                try:
                    self.conn.execute("SET hnsw_enable_experimental_persistence = true")
                    logger.info("Enabled HNSW experimental persistence for disk storage")
                except Exception as e:
                    logger.warning(f"Could not enable HNSW persistence: {e}")

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

            # Create transcripts table (vector-free since schema v2)
            # CREATE TABLE IF NOT EXISTS won't modify existing tables, so old
            # DBs keep their inline vector columns until migration runs.
            self.conn.execute("""
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
                    has_vectors BOOLEAN DEFAULT FALSE,
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

            # Always create schema_meta and transcript_vectors tables
            self._create_schema_meta_table()
            self._create_transcript_vectors_table()

            # Schema versioning and migration
            schema_version = self._get_schema_version()
            if schema_version < 2:
                self._migrate_to_externalized_vectors()
            if schema_version < 3:
                self._migrate_add_has_vectors()

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

            # Create HNSW vector index on transcript_vectors
            try:
                self.conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_vectors_hnsw
                    ON transcript_vectors USING HNSW (vector)
                    WITH (metric = 'cosine')
                """)
                logger.info("HNSW vector index created on transcript_vectors")
            except Exception as e:
                logger.warning(f"Failed to create HNSW index: {e}")

            # Create indexes on transcript_vectors
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vectors_parent ON transcript_vectors (parent_id)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vectors_session "
                "ON transcript_vectors (user_id, session_id)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vectors_user ON transcript_vectors (user_id)"
            )

            # Create views
            self.conn.execute(_CREATE_VIEWS_SQL)
            logger.debug("Created transcript_messages and vectors_with_context views")

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
    # Schema Management
    # =========================================================================

    def _create_schema_meta_table(self) -> None:
        """Create schema_meta table for version tracking."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_meta (
                key VARCHAR PRIMARY KEY,
                value VARCHAR
            )
        """)

    def _create_transcript_vectors_table(self) -> None:
        """Create the externalized transcript_vectors table."""
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS transcript_vectors (
                id VARCHAR NOT NULL PRIMARY KEY,
                parent_id VARCHAR NOT NULL,
                user_id VARCHAR NOT NULL,
                session_id VARCHAR NOT NULL,
                project_slug VARCHAR,
                content_type VARCHAR NOT NULL,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                total_chunks INTEGER NOT NULL DEFAULT 1,
                span_start INTEGER NOT NULL DEFAULT 0,
                span_end INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                source_text TEXT NOT NULL,
                vector FLOAT[{self.config.vector_dimensions}],
                embedding_model VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def _get_schema_version(self) -> int:
        """Get the current schema version (defaults to 1 for pre-versioned DBs)."""
        try:
            result = self.conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'version'"
            ).fetchone()
            return int(result[0]) if result else 1
        except Exception:
            return 1

    def _set_schema_version(self, version: int) -> None:
        """Set the schema version."""
        self.conn.execute(
            """
            INSERT INTO schema_meta (key, value) VALUES ('version', ?)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """,
            [str(version)],
        )

    def _migrate_to_externalized_vectors(self) -> None:
        """Migrate inline vectors from transcripts to the transcript_vectors table."""
        logger.info("Migrating inline vectors to transcript_vectors table...")

        # Ensure new tables exist
        self._create_transcript_vectors_table()

        # Check if transcript_vectors already has data (idempotency)
        count = self.conn.execute("SELECT COUNT(*) FROM transcript_vectors").fetchone()[0]
        if count > 0:
            logger.info(f"transcript_vectors already has {count} rows, skipping migration")
            self._set_schema_version(2)
            return

        # Check if old vector columns exist
        columns = [col[1] for col in self.conn.execute("PRAGMA table_info(transcripts)").fetchall()]
        if "user_query_vector" not in columns:
            logger.info("No inline vector columns found, skipping migration")
            self._set_schema_version(2)
            return

        # Read all transcript rows that have at least one vector
        rows = self.conn.execute("""
            SELECT id, user_id, session_id, project_slug, role, content,
                   user_query_vector, assistant_response_vector,
                   assistant_thinking_vector, tool_output_vector,
                   embedding_model
            FROM transcripts
            WHERE user_query_vector IS NOT NULL
               OR assistant_response_vector IS NOT NULL
               OR assistant_thinking_vector IS NOT NULL
               OR tool_output_vector IS NOT NULL
        """).fetchall()

        migrated = 0
        for row in rows:
            parent_id = row[0]
            user_id = row[1]
            session_id = row[2]
            project_slug = row[3]
            role = row[4]
            content = row[5]
            vectors = {
                "user_query": row[6],
                "assistant_response": row[7],
                "assistant_thinking": row[8],
                "tool_output": row[9],
            }
            embedding_model = row[10]

            # Re-extract content to get correct source_text
            # Parse content from JSON if it's a string
            if isinstance(content, str):
                try:
                    parsed_content = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    parsed_content = content
            else:
                parsed_content = content

            message = {"role": role, "content": parsed_content}
            extracted = extract_all_embeddable_content(message)

            for content_type, vector in vectors.items():
                if vector is None:
                    continue

                source_text = extracted.get(content_type) or ""
                if not source_text:
                    # Fallback: use str(content) if extraction returned None
                    source_text = str(content)[:1000] if content else ""

                token_count_val = count_tokens(source_text) if source_text else 0
                vector_id = f"{parent_id}_{content_type}_0"

                vec_literal = self._format_vector_literal(vector)
                self.conn.execute(
                    f"""
                    INSERT INTO transcript_vectors (
                        id, parent_id, user_id, session_id, project_slug,
                        content_type, chunk_index, total_chunks,
                        span_start, span_end, token_count,
                        source_text, vector, embedding_model
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        ?, 0, 1,
                        0, ?, ?,
                        ?, {vec_literal}, ?
                    )
                    """,
                    [
                        vector_id,
                        parent_id,
                        user_id,
                        session_id,
                        project_slug,
                        content_type,
                        len(source_text),
                        token_count_val,
                        source_text,
                        embedding_model,
                    ],
                )
                migrated += 1

        # Drop old vector columns
        for col in [
            "user_query_vector",
            "assistant_response_vector",
            "assistant_thinking_vector",
            "tool_output_vector",
            "embedding_model",
            "vector_metadata",
        ]:
            try:
                self.conn.execute(f"ALTER TABLE transcripts DROP COLUMN IF EXISTS {col}")
            except Exception as e:
                logger.warning(f"Could not drop column {col}: {e}")

        # Drop old HNSW indexes
        for idx in [
            "idx_user_query_vector",
            "idx_assistant_response_vector",
            "idx_assistant_thinking_vector",
            "idx_tool_output_vector",
        ]:
            try:
                self.conn.execute(f"DROP INDEX IF EXISTS {idx}")
            except Exception:
                pass

        self._set_schema_version(2)
        logger.info(f"Migration complete: {migrated} vectors moved to transcript_vectors")

    def _migrate_add_has_vectors(self) -> None:
        """Add has_vectors column to transcripts table (schema v3)."""
        columns = [col[1] for col in self.conn.execute("PRAGMA table_info(transcripts)").fetchall()]
        if "has_vectors" not in columns:
            logger.info("Adding has_vectors column to transcripts table...")
            self.conn.execute(
                "ALTER TABLE transcripts ADD COLUMN has_vectors BOOLEAN DEFAULT FALSE"
            )
        self._set_schema_version(3)

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
                # Delete transcript vectors
                self.conn.execute(
                    "DELETE FROM transcript_vectors WHERE user_id = ? AND session_id = ?",
                    [user_id, session_id],
                )

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

    async def backfill_embeddings(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
        *,
        batch_size: int = 100,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> EmbeddingOperationResult:
        """Generate embeddings for transcripts where has_vectors is False."""
        if not self.embedding_provider:
            return EmbeddingOperationResult(
                transcripts_found=0,
                vectors_stored=0,
                vectors_failed=0,
                errors=["No embedding provider configured"],
            )

        def _fetch_missing() -> list[dict[str, Any]]:
            if self.conn is None:
                raise StorageIOError("backfill_embeddings", cause=RuntimeError("Not initialized"))
            rows = self.conn.execute(
                """
                SELECT id, user_id, session_id, project_slug, role, content
                FROM transcripts
                WHERE user_id = ? AND session_id = ?
                AND (has_vectors = FALSE OR has_vectors IS NULL)
                ORDER BY sequence
                """,
                [user_id, session_id],
            ).fetchall()
            return [
                {
                    "parent_id": row[0],
                    "user_id": row[1],
                    "session_id": row[2],
                    "project_slug": row[3],
                    "role": row[4],
                    "content": json.loads(row[5]) if row[5] else None,
                }
                for row in rows
            ]

        missing = await asyncio.to_thread(_fetch_missing)
        if not missing:
            return EmbeddingOperationResult(transcripts_found=0, vectors_stored=0, vectors_failed=0)

        total = len(missing)
        total_stored = 0
        total_failed = 0
        errors: list[str] = []
        processed = 0

        for batch_start in range(0, total, batch_size):
            batch = missing[batch_start : batch_start + batch_size]

            try:
                records, failed = await self._prepare_vector_records(
                    batch, context_msg=f"backfill session={session_id}"
                )

                if records:
                    await asyncio.to_thread(self._store_vector_records, records)

                total_stored += sum(1 for r in records if r.get("vector") is not None)
                total_failed += failed

            except Exception as exc:
                msg = f"Batch {batch_start // batch_size + 1} failed: {exc}"
                if len(errors) < 50:
                    errors.append(msg)
                logger.error("backfill_embeddings: %s", msg, exc_info=True)
                total_failed += len(batch)

            processed += len(batch)
            if on_progress:
                on_progress(processed, total)

        return EmbeddingOperationResult(
            transcripts_found=total,
            vectors_stored=total_stored,
            vectors_failed=total_failed,
            errors=errors,
        )

    async def rebuild_vectors(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
        *,
        batch_size: int = 100,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> EmbeddingOperationResult:
        """Delete all vectors for a session, reset has_vectors, regenerate."""
        if not self.embedding_provider:
            return EmbeddingOperationResult(
                transcripts_found=0,
                vectors_stored=0,
                vectors_failed=0,
                errors=["No embedding provider configured"],
            )

        def _delete_and_reset() -> list[dict[str, Any]]:
            if self.conn is None:
                raise StorageIOError("rebuild_vectors", cause=RuntimeError("Not initialized"))
            # Delete all vectors
            self.conn.execute(
                "DELETE FROM transcript_vectors WHERE user_id = ? AND session_id = ?",
                [user_id, session_id],
            )
            # Reset has_vectors
            self.conn.execute(
                "UPDATE transcripts SET has_vectors = FALSE WHERE user_id = ? AND session_id = ?",
                [user_id, session_id],
            )
            # Fetch all transcripts
            rows = self.conn.execute(
                """
                SELECT id, user_id, session_id, project_slug, role, content
                FROM transcripts
                WHERE user_id = ? AND session_id = ?
                ORDER BY sequence
                """,
                [user_id, session_id],
            ).fetchall()
            return [
                {
                    "parent_id": row[0],
                    "user_id": row[1],
                    "session_id": row[2],
                    "project_slug": row[3],
                    "role": row[4],
                    "content": json.loads(row[5]) if row[5] else None,
                }
                for row in rows
            ]

        all_transcripts = await asyncio.to_thread(_delete_and_reset)
        if not all_transcripts:
            return EmbeddingOperationResult(transcripts_found=0, vectors_stored=0, vectors_failed=0)

        total = len(all_transcripts)
        total_stored = 0
        total_failed = 0
        errors: list[str] = []
        processed = 0

        for batch_start in range(0, total, batch_size):
            batch = all_transcripts[batch_start : batch_start + batch_size]

            try:
                records, failed = await self._prepare_vector_records(
                    batch, context_msg=f"rebuild session={session_id}"
                )

                if records:
                    await asyncio.to_thread(self._store_vector_records, records)

                total_stored += sum(1 for r in records if r.get("vector") is not None)
                total_failed += failed

            except Exception as exc:
                msg = f"Batch {batch_start // batch_size + 1} failed: {exc}"
                if len(errors) < 50:
                    errors.append(msg)
                logger.error("rebuild_vectors: %s", msg, exc_info=True)
                total_failed += len(batch)

            processed += len(batch)
            if on_progress:
                on_progress(processed, total)

        return EmbeddingOperationResult(
            transcripts_found=total,
            vectors_stored=total_stored,
            vectors_failed=total_failed,
            errors=errors,
        )

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
        """Sync transcript lines with externalized vector storage.

        Step 1: Store transcript rows (content only, no vectors).
        Step 2: Generate and store vectors in transcript_vectors table.
        """
        if not lines:
            return 0

        # Step 1: Store transcript rows (content only)
        stored = await asyncio.to_thread(
            self._store_transcript_rows,
            user_id,
            host_id,
            project_slug,
            session_id,
            lines,
            start_sequence,
        )

        # Step 2: Generate and store vectors
        # Embedding failures must never prevent transcript storage (step 1 already committed).
        if self.embedding_provider and embeddings is None:
            try:
                await self._generate_and_store_vectors(
                    user_id, project_slug, session_id, lines, start_sequence
                )
            except Exception as exc:
                logger.error(
                    "EMBEDDING_FAILURE: Vector generation failed for "
                    "user=%s project=%s session=%s (%d messages). "
                    "%d transcripts were stored successfully but will lack "
                    "vector embeddings — semantic/hybrid search will not cover "
                    "these messages. error_type=%s error=%s",
                    user_id,
                    project_slug,
                    session_id,
                    len(lines),
                    stored,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
        elif embeddings is not None:
            # Pre-computed embeddings (from sync daemon) - store as single-chunk vectors
            await asyncio.to_thread(
                self._store_precomputed_vectors,
                user_id,
                project_slug,
                session_id,
                lines,
                start_sequence,
                embeddings,
            )

        return stored

    def _store_transcript_rows(
        self,
        user_id: str,
        host_id: str,
        project_slug: str,
        session_id: str,
        lines: list[dict[str, Any]],
        start_sequence: int,
    ) -> int:
        """Store transcript rows (content only, no vectors)."""
        if self.conn is None:
            raise StorageIOError("sync_transcript", cause=RuntimeError("Not initialized"))

        synced = 0
        for i, line in enumerate(lines):
            sequence = start_sequence + i
            doc_id = f"{session_id}_msg_{sequence}"

            # Store content as JSON (handles both string and array formats)
            content = line.get("content")
            content_json = json.dumps(content) if content is not None else None

            now = datetime.now(UTC).isoformat()
            self.conn.execute(
                """
                INSERT INTO transcripts (
                    id, user_id, host_id, project_slug, session_id, sequence,
                    role, content, turn, ts, has_vectors, synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    role = EXCLUDED.role,
                    turn = EXCLUDED.turn,
                    ts = EXCLUDED.ts,
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
                    False,
                    now,
                ],
            )
            synced += 1

        return synced

    async def _generate_and_store_vectors(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
        lines: list[dict[str, Any]],
        start_sequence: int,
    ) -> None:
        """Extract content, chunk, embed, and store vectors."""
        all_vector_records: list[dict[str, Any]] = []
        all_texts_to_embed: list[str] = []

        for i, line in enumerate(lines):
            sequence = start_sequence + i
            parent_id = f"{session_id}_msg_{sequence}"
            content = extract_all_embeddable_content(line)

            for content_type, text in content.items():
                if text is None:
                    continue

                chunks = chunk_text(text, content_type)

                for chunk in chunks:
                    vector_id = f"{parent_id}_{content_type}_{chunk.chunk_index}"
                    record = {
                        "id": vector_id,
                        "parent_id": parent_id,
                        "user_id": user_id,
                        "session_id": session_id,
                        "project_slug": project_slug,
                        "content_type": content_type,
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                        "span_start": chunk.span_start,
                        "span_end": chunk.span_end,
                        "token_count": chunk.token_count,
                        "source_text": chunk.text,
                        "embedding_model": (
                            self.embedding_provider.model_name if self.embedding_provider else None
                        ),
                    }
                    all_vector_records.append(record)
                    all_texts_to_embed.append(chunk.text)

        if not all_texts_to_embed:
            return

        # Batch embed all chunk texts (all entries are non-None strings)
        texts_for_embed: list[str | None] = list(all_texts_to_embed)
        embeddings_list = await self._embed_non_none(texts_for_embed)

        # Attach vectors to records
        for idx, embedding in enumerate(embeddings_list):
            all_vector_records[idx]["vector"] = embedding

        # Bulk store vectors
        await asyncio.to_thread(self._store_vector_records, all_vector_records)

        total_chunks = len(all_vector_records)
        total_messages = len(lines)
        counts = count_embeddable_content_types(lines)
        logger.info(
            f"Stored {total_chunks} vectors for {total_messages} messages "
            f"in session {session_id} "
            f"({counts['user_query']} user, "
            f"{counts['assistant_response']} responses, "
            f"{counts['assistant_thinking']} thinking, "
            f"{counts['tool_output']} tool)"
        )

    def _store_vector_records(self, records: list[dict[str, Any]]) -> None:
        """Store vector records in transcript_vectors table.

        Uses delete-before-insert for idempotent re-sync.
        After storing, marks parent transcripts with has_vectors=TRUE.
        """
        if not records:
            return

        # Group by parent_id for cleanup
        parent_ids = {r["parent_id"] for r in records}

        # Delete existing vectors for these parents
        for parent_id in parent_ids:
            self.conn.execute(
                "DELETE FROM transcript_vectors WHERE parent_id = ?",
                [parent_id],
            )

        # Insert new vectors
        stored_parent_ids: set[str] = set()
        for record in records:
            vector = record.get("vector")
            if vector is not None:
                vec_sql = self._format_vector_literal(vector)
            else:
                vec_sql = "NULL"

            self.conn.execute(
                f"""
                INSERT INTO transcript_vectors (
                    id, parent_id, user_id, session_id, project_slug,
                    content_type, chunk_index, total_chunks,
                    span_start, span_end, token_count,
                    source_text, vector, embedding_model
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, {vec_sql}, ?
                )
                """,
                [
                    record["id"],
                    record["parent_id"],
                    record["user_id"],
                    record["session_id"],
                    record["project_slug"],
                    record["content_type"],
                    record["chunk_index"],
                    record["total_chunks"],
                    record["span_start"],
                    record["span_end"],
                    record["token_count"],
                    record["source_text"],
                    record["embedding_model"],
                ],
            )
            if vector is not None:
                stored_parent_ids.add(record["parent_id"])

        # Mark parent transcripts as having vectors
        for pid in stored_parent_ids:
            self.conn.execute(
                "UPDATE transcripts SET has_vectors = TRUE WHERE id = ?",
                [pid],
            )

    def _store_precomputed_vectors(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
        lines: list[dict[str, Any]],
        start_sequence: int,
        embeddings: dict[str, list[list[float] | None]],
    ) -> None:
        """Store pre-computed embeddings as single-chunk vector records."""
        records: list[dict[str, Any]] = []

        for i, line in enumerate(lines):
            sequence = start_sequence + i
            parent_id = f"{session_id}_msg_{sequence}"
            content = extract_all_embeddable_content(line)

            for content_type in [
                "user_query",
                "assistant_response",
                "assistant_thinking",
                "tool_output",
            ]:
                emb_list = embeddings.get(content_type, [])
                vector = emb_list[i] if i < len(emb_list) else None
                if vector is None:
                    continue

                source_text = content.get(content_type) or ""
                records.append(
                    {
                        "id": f"{parent_id}_{content_type}_0",
                        "parent_id": parent_id,
                        "user_id": user_id,
                        "session_id": session_id,
                        "project_slug": project_slug,
                        "content_type": content_type,
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "span_start": 0,
                        "span_end": len(source_text),
                        "token_count": count_tokens(source_text) if source_text else 0,
                        "source_text": source_text,
                        "vector": vector,
                        "embedding_model": (
                            self.embedding_provider.model_name if self.embedding_provider else None
                        ),
                    }
                )

        self._store_vector_records(records)

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
        """Full-text search across transcripts and vector source texts."""

        def _search_transcripts() -> list[SearchResult]:
            if self.conn is None:
                raise StorageIOError("search_transcripts", cause=RuntimeError("Not initialized"))

            # Build WHERE clause for transcript content search
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

            results = []
            for row in results_raw:
                # Parse JSON content for consistency with semantic search (BUG 2)
                content_raw = row[4] or ""
                try:
                    content = json.loads(content_raw)
                except (json.JSONDecodeError, TypeError):
                    content = content_raw

                results.append(
                    SearchResult(
                        session_id=row[1],
                        project_slug=row[2],
                        sequence=row[3],
                        content=content,
                        metadata={
                            "id": row[0],
                            "role": row[5],
                            "turn": row[6],
                            "ts": row[7],
                        },
                        score=1.0,
                        source="full_text",
                    )
                )
            return results

        transcript_results = await asyncio.to_thread(_search_transcripts)

        # Also search transcript_vectors.source_text (BUG 1+5)
        vector_text_results = await asyncio.to_thread(
            self._full_text_search_vector_sources, user_id, options, limit
        )

        # Merge and deduplicate by (session_id, sequence)
        seen: set[tuple[str, int]] = set()
        combined: list[SearchResult] = []
        for r in transcript_results + vector_text_results:
            key = (r.session_id, r.sequence)
            if key not in seen:
                seen.add(key)
                combined.append(r)

        return combined[:limit]

    def _full_text_search_vector_sources(
        self,
        user_id: str,
        options: TranscriptSearchOptions,
        limit: int,
    ) -> list[SearchResult]:
        """Search source_text in transcript_vectors for span-level full-text matching.

        This catches extracted thinking blocks, tool outputs, and chunked content
        that wouldn't be found by searching transcripts.content alone.
        """
        if self.conn is None:
            return []

        # Build content_type filter from search_in_* flags
        content_types: list[str] = []
        if options.search_in_user:
            content_types.append("user_query")
        if options.search_in_assistant:
            content_types.append("assistant_response")
        if options.search_in_thinking:
            content_types.append("assistant_thinking")
        if options.search_in_tool:
            content_types.append("tool_output")

        if not content_types:
            return []

        where_parts = ["v.user_id = ?", "v.source_text LIKE ?"]
        params: list[Any] = [user_id, f"%{options.query}%"]

        placeholders = ", ".join(["?"] * len(content_types))
        where_parts.append(f"v.content_type IN ({placeholders})")
        params.extend(content_types)

        if options.filters:
            if options.filters.project_slug:
                where_parts.append("t.project_slug = ?")
                params.append(options.filters.project_slug)

            if options.filters.start_date:
                where_parts.append("t.ts >= ?")
                params.append(options.filters.start_date)

            if options.filters.end_date:
                where_parts.append("t.ts <= ?")
                params.append(options.filters.end_date)

        where_clause = " AND ".join(where_parts)

        query = f"""
            SELECT v.parent_id, v.session_id, v.source_text, v.content_type,
                   v.span_start, v.span_end,
                   t.project_slug, t.sequence, t.content, t.role, t.turn, t.ts
            FROM transcript_vectors v
            JOIN transcripts t ON t.id = v.parent_id
            WHERE {where_clause}
            LIMIT ?
        """
        params.append(limit * 3)  # Over-fetch for dedup

        results_raw = self.conn.execute(query, params).fetchall()

        # Deduplicate by parent_id, keeping first match
        seen: dict[str, SearchResult] = {}
        for row in results_raw:
            parent_id = row[0]
            if parent_id not in seen:
                content_raw = row[8] or ""
                try:
                    content = json.loads(content_raw)
                except (json.JSONDecodeError, TypeError):
                    content = content_raw

                seen[parent_id] = SearchResult(
                    session_id=row[1],
                    project_slug=row[6],
                    sequence=row[7],
                    content=content,
                    metadata={
                        "id": parent_id,
                        "role": row[9],
                        "turn": row[10],
                        "ts": row[11],
                        "content_type": row[3],
                        "matched_text": row[2],
                        "span_start": row[4],
                        "span_end": row[5],
                    },
                    score=1.0,
                    source="full_text",
                )

        return list(seen.values())[:limit]

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
        return await self._semantic_search_with_vector(user_id, options, limit, query_vector)

    async def _semantic_search_with_vector(
        self,
        user_id: str,
        options: TranscriptSearchOptions,
        limit: int,
        query_vector: list[float],
    ) -> list[SearchResult]:
        """Semantic search with pre-computed query vector (avoids double embedding)."""
        content_types: list[str] | None = []
        if options.search_in_user:
            content_types.append("user_query")
        if options.search_in_assistant:
            content_types.append("assistant_response")
        if options.search_in_thinking:
            content_types.append("assistant_thinking")
        if options.search_in_tool:
            content_types.append("tool_output")
        if not content_types:
            content_types = None  # None = search all

        return await self.vector_search(
            user_id, query_vector, options.filters, limit, content_types=content_types
        )

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

        # Compute query embedding ONCE
        query_vector = await self.embedding_provider.embed_text(options.query)
        query_np = np.array(query_vector)

        # Get both result sets (semantic reuses pre-computed vector)
        text_results = await self._full_text_search_transcripts(user_id, options, candidate_limit)
        semantic_results = await self._semantic_search_with_vector(
            user_id, options, candidate_limit, query_vector
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

        # Extract embeddings for MMR (fetch from DB if not in metadata)
        vectors = []
        for result in combined:
            if "embedding" in result.metadata and result.metadata["embedding"]:
                vectors.append(np.array(result.metadata["embedding"]))
            else:
                # Fetch embedding from DB, using content_type for precision
                ct = result.metadata.get("content_type")
                embedding = await self._get_embedding(
                    user_id, result.session_id, result.sequence, ct
                )
                if embedding:
                    vectors.append(np.array(embedding))
                else:
                    vectors.append(np.zeros(len(query_vector)))

        # Apply MMR re-ranking
        mmr_results = compute_mmr(
            vectors=vectors,
            query=query_np,
            lambda_param=options.mmr_lambda,
            top_k=limit,
        )

        return [combined[idx] for idx, _ in mmr_results]

    async def _get_embedding(
        self, user_id: str, session_id: str, sequence: int, content_type: str | None = None
    ) -> list[float] | None:
        """
        Fetch embedding for a transcript message, optionally by content_type.

        When content_type is provided, fetches the specific matched vector.
        Otherwise returns the first available vector.
        """

        def _get() -> list[float] | None:
            if self.conn is None:
                return None

            parent_id = f"{session_id}_msg_{sequence}"

            if content_type:
                result = self.conn.execute(
                    """
                    SELECT vector
                    FROM transcript_vectors
                    WHERE parent_id = ? AND user_id = ? AND content_type = ?
                      AND vector IS NOT NULL
                    ORDER BY chunk_index
                    LIMIT 1
                    """,
                    [parent_id, user_id, content_type],
                ).fetchone()
            else:
                result = self.conn.execute(
                    """
                    SELECT vector
                    FROM transcript_vectors
                    WHERE parent_id = ? AND user_id = ? AND vector IS NOT NULL
                    ORDER BY chunk_index
                    LIMIT 1
                    """,
                    [parent_id, user_id],
                ).fetchone()

            if result and result[0] is not None:
                return result[0]
            return None

        return await asyncio.to_thread(_get)

    async def get_turn_context(
        self,
        user_id: str,
        session_id: str,
        turn: int,
        before: int = 2,
        after: int = 2,
        include_tool_outputs: bool = True,
    ) -> TurnContext:
        """Get context window around a specific turn."""

        def _get() -> TurnContext:
            if self.conn is None:
                raise StorageConnectionError("Database not connected")

            # Calculate turn range
            min_turn = max(1, turn - before)
            max_turn = turn + after

            # Build role filter
            role_filter = ""
            if not include_tool_outputs:
                role_filter = "AND role != 'tool'"

            # Query messages in turn range
            rows = self.conn.execute(
                f"""
                SELECT sequence, turn, role, content, ts, project_slug
                FROM transcripts
                WHERE user_id = ? AND session_id = ?
                  AND turn >= ? AND turn <= ?
                  {role_filter}
                ORDER BY turn, sequence
                """,
                [user_id, session_id, min_turn, max_turn],
            ).fetchall()

            # Get session turn range
            range_row = self.conn.execute(
                """
                SELECT MIN(turn), MAX(turn)
                FROM transcripts
                WHERE user_id = ? AND session_id = ?
                """,
                [user_id, session_id],
            ).fetchone()

            first_turn = range_row[0] if range_row and range_row[0] else 1
            last_turn = range_row[1] if range_row and range_row[1] else turn

            # Parse messages
            messages = []
            project_slug = "unknown"
            for row in rows:
                project_slug = row[5] or project_slug
                messages.append(
                    TranscriptMessage(
                        sequence=row[0],
                        turn=row[1],
                        role=row[2] or "unknown",
                        content=row[3] or "",
                        ts=row[4],
                        metadata={"session_id": session_id},
                    )
                )

            # Partition into previous, current, following
            previous = [m for m in messages if m.turn < turn]
            current = [m for m in messages if m.turn == turn]
            following = [m for m in messages if m.turn > turn]

            return TurnContext(
                session_id=session_id,
                project_slug=project_slug,
                target_turn=turn,
                previous=previous,
                current=current,
                following=following,
                has_more_before=min_turn > first_turn,
                has_more_after=max_turn < last_turn,
                first_turn=first_turn,
                last_turn=last_turn,
            )

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
        user_id: str = "",
        session_id: str = "",
        project_slug: str = "",
        event_type: str = "",
        event_category: str = "",
        tool_name: str = "",
        model: str = "",
        provider: str = "",
        level: str = "",
        start_date: str = "",
        end_date: str = "",
        limit: int = 50,
    ) -> list[SearchResult]:
        """Search events with structured filters."""

        def _search() -> list[SearchResult]:
            if self.conn is None:
                raise StorageIOError("search_events", cause=RuntimeError("Not initialized"))

            # Build WHERE clause
            where_parts: list[str] = []
            params: list[Any] = []

            if user_id:
                where_parts.append("user_id = ?")
                params.append(user_id)

            if session_id:
                where_parts.append("session_id = ?")
                params.append(session_id)

            if project_slug:
                where_parts.append("project_slug = ?")
                params.append(project_slug)

            if event_type:
                where_parts.append("event = ?")
                params.append(event_type)

            if tool_name:
                where_parts.append("json_extract(data, '$.tool_name') = ?")
                params.append(tool_name)

            if model:
                where_parts.append("json_extract(data, '$.model') = ?")
                params.append(model)

            if provider:
                where_parts.append("json_extract(data, '$.provider') = ?")
                params.append(provider)

            if level:
                where_parts.append("lvl = ?")
                params.append(level)

            if start_date:
                where_parts.append("ts >= ?")
                params.append(start_date)

            if end_date:
                where_parts.append("ts <= ?")
                params.append(end_date)

            where_clause = " AND ".join(where_parts) if where_parts else "1=1"

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
        """Upsert embeddings for existing transcript messages.

        Stores embeddings as single-chunk vector records in transcript_vectors.
        """

        def _upsert() -> int:
            if self.conn is None:
                raise StorageIOError("upsert_embeddings", cause=RuntimeError("Not initialized"))

            updated = 0
            for emb in embeddings:
                parent_id = f"{session_id}_msg_{emb['sequence']}"
                vector_id = f"{parent_id}_user_query_0"
                vector = emb["vector"]

                # Delete existing vectors for this parent
                self.conn.execute(
                    "DELETE FROM transcript_vectors WHERE parent_id = ?",
                    [parent_id],
                )

                vec_literal = self._format_vector_literal(vector)
                self.conn.execute(
                    f"""
                    INSERT INTO transcript_vectors (
                        id, parent_id, user_id, session_id, project_slug,
                        content_type, chunk_index, total_chunks,
                        span_start, span_end, token_count,
                        source_text, vector, embedding_model
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        'user_query', 0, 1,
                        0, 0, 0,
                        '', {vec_literal}, ?
                    )
                    """,
                    [
                        vector_id,
                        parent_id,
                        user_id,
                        session_id,
                        project_slug,
                        emb.get("metadata", {}).get("model", "unknown"),
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
        content_types: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Perform vector similarity search using the transcript_vectors table.

        Searches across the externalized vector table and joins back to
        transcripts for message context.

        Args:
            user_id: User identifier
            query_vector: Query embedding vector
            filters: Optional search filters
            top_k: Number of results to return
            content_types: Which content types to search. Default: all.
                Options: ["user_query", "assistant_response",
                          "assistant_thinking", "tool_output"]
        """

        def _search() -> list[SearchResult]:
            if self.conn is None:
                raise StorageIOError("vector_search", cause=RuntimeError("Not initialized"))

            vec_literal = self._format_vector_literal(query_vector)

            # Build WHERE clause on the vectors_with_context view
            where_parts = ["user_id = ?"]
            params: list[Any] = [user_id]

            # Filter by content type
            if content_types:
                placeholders = ", ".join(["?"] * len(content_types))
                where_parts.append(f"content_type IN ({placeholders})")
                params.extend(content_types)

            # Require non-null vectors
            where_parts.append("vector IS NOT NULL")

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

            # Query vectors_with_context view, deduplicate by parent message
            # taking the best similarity score per message
            query = f"""
                SELECT parent_id, session_id, project_slug, sequence,
                       content, role, turn, ts, content_type,
                       array_cosine_similarity(vector, {vec_literal}) AS similarity
                FROM vectors_with_context
                WHERE {where_clause}
                ORDER BY similarity DESC
                LIMIT ?
            """

            params.append(top_k * 3)  # Over-fetch to allow dedup

            results_raw = self.conn.execute(query, params).fetchall()

            # Deduplicate by parent message, keeping best score
            seen: dict[str, SearchResult] = {}
            for row in results_raw:
                parent_id = row[0]
                score = float(row[9]) if row[9] is not None else 0.0

                if parent_id not in seen or score > seen[parent_id].score:
                    seen[parent_id] = SearchResult(
                        session_id=row[1],
                        project_slug=row[2],
                        sequence=row[3],
                        content=json.loads(row[4]) if row[4] else "",
                        metadata={
                            "id": parent_id,
                            "role": row[5],
                            "turn": row[6],
                            "ts": row[7],
                            "content_type": row[8],
                        },
                        score=score,
                        source="semantic",
                    )

            # Sort by score descending and limit
            results = sorted(seen.values(), key=lambda r: r.score, reverse=True)
            return results[:top_k]

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

    async def get_session_sync_stats(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
    ) -> SessionSyncStats:
        """Get lightweight sync statistics using aggregate queries."""

        def _query() -> SessionSyncStats:
            if self.conn is None:
                raise StorageIOError(
                    "get_session_sync_stats", cause=RuntimeError("Not initialized")
                )

            results = self.conn.execute(
                """
                SELECT
                    'event' as type,
                    COUNT(*) as count,
                    MIN(ts) as earliest,
                    MAX(ts) as latest
                FROM events
                WHERE user_id = ? AND session_id = ?
                UNION ALL
                SELECT
                    'transcript' as type,
                    COUNT(*) as count,
                    MIN(ts) as earliest,
                    MAX(ts) as latest
                FROM transcripts
                WHERE user_id = ? AND session_id = ?
                """,
                [user_id, session_id, user_id, session_id],
            ).fetchall()

            event_count = transcript_count = 0
            event_earliest = event_latest = None
            transcript_earliest = transcript_latest = None

            for row in results:
                doc_type, count, earliest, latest = row
                earliest_str = str(earliest) if earliest else None
                latest_str = str(latest) if latest else None
                if doc_type == "event":
                    event_count = count
                    event_earliest = earliest_str
                    event_latest = latest_str
                elif doc_type == "transcript":
                    transcript_count = count
                    transcript_earliest = earliest_str
                    transcript_latest = latest_str

            return SessionSyncStats(
                event_count=event_count,
                transcript_count=transcript_count,
                event_ts_range=(event_earliest, event_latest),
                transcript_ts_range=(transcript_earliest, transcript_latest),
            )

        return await asyncio.to_thread(_query)

    async def get_stored_transcript_ids(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
    ) -> list[str]:
        """Return all stored transcript document IDs for a session."""

        def _query() -> list[str]:
            if self.conn is None:
                raise StorageIOError(
                    "get_stored_transcript_ids",
                    cause=RuntimeError("Not initialized"),
                )
            rows = self.conn.execute(
                "SELECT id FROM transcripts WHERE user_id = ? AND session_id = ?",
                [user_id, session_id],
            ).fetchall()
            return [row[0] for row in rows]

        return await asyncio.to_thread(_query)

    async def get_stored_event_ids(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
    ) -> list[str]:
        """Return all stored event document IDs for a session."""

        def _query() -> list[str]:
            if self.conn is None:
                raise StorageIOError(
                    "get_stored_event_ids",
                    cause=RuntimeError("Not initialized"),
                )
            rows = self.conn.execute(
                "SELECT id FROM events WHERE user_id = ? AND session_id = ?",
                [user_id, session_id],
            ).fetchall()
            return [row[0] for row in rows]

        return await asyncio.to_thread(_query)

    # =========================================================================
    # Discovery APIs
    # =========================================================================

    async def list_users(
        self,
        filters: SearchFilters | None = None,
    ) -> list[str]:
        """List all unique user IDs in the storage."""

        def _list_users() -> list[str]:
            where_parts = ["1=1"]
            params: list[Any] = []

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

                if filters.bundle:
                    where_parts.append("bundle = ?")
                    params.append(filters.bundle)

            where_clause = " AND ".join(where_parts)
            query = f"SELECT DISTINCT user_id FROM sessions WHERE {where_clause} ORDER BY user_id"

            results = self.conn.execute(query, params).fetchall()
            return [row[0] for row in results if row[0]]

        return await asyncio.to_thread(_list_users)

    async def list_projects(
        self,
        user_id: str = "",
        filters: SearchFilters | None = None,
    ) -> list[str]:
        """List all unique project slugs."""

        def _list_projects() -> list[str]:
            where_parts = ["1=1"]
            params: list[Any] = []

            if user_id:
                where_parts.append("user_id = ?")
                params.append(user_id)

            if filters:
                if filters.start_date:
                    where_parts.append("created >= ?")
                    params.append(filters.start_date)

                if filters.end_date:
                    where_parts.append("created <= ?")
                    params.append(filters.end_date)

                if filters.bundle:
                    where_parts.append("bundle = ?")
                    params.append(filters.bundle)

            where_clause = " AND ".join(where_parts)
            query = f"SELECT DISTINCT project_slug FROM sessions WHERE {where_clause} ORDER BY project_slug"

            results = self.conn.execute(query, params).fetchall()
            return [row[0] for row in results if row[0]]

        return await asyncio.to_thread(_list_projects)

    async def list_sessions(
        self,
        user_id: str = "",
        project_slug: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List sessions with pagination."""

        def _list_sessions() -> list[dict[str, Any]]:
            where_parts = ["1=1"]
            params: list[Any] = []

            if user_id:
                where_parts.append("user_id = ?")
                params.append(user_id)

            if project_slug:
                where_parts.append("project_slug = ?")
                params.append(project_slug)

            where_clause = " AND ".join(where_parts)
            query = f"""
                SELECT session_id, user_id, project_slug, bundle, created, turn_count
                FROM sessions
                WHERE {where_clause}
                ORDER BY created DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])

            results = self.conn.execute(query, params).fetchall()
            return [
                {
                    "session_id": row[0],
                    "user_id": row[1],
                    "project_slug": row[2],
                    "bundle": row[3],
                    "created": row[4],
                    "turn_count": row[5],
                }
                for row in results
            ]

        return await asyncio.to_thread(_list_sessions)

    async def get_active_sessions(
        self,
        user_id: str = "",
        project_slug: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        min_turn_count: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get sessions with activity in the specified date range.

        Detects sessions by finding transcripts with timestamps in the date range,
        NOT by session creation date.
        """

        def _get_active() -> list[dict[str, Any]]:
            # Step 1: Find session_ids with transcript activity
            activity_where: list[str] = ["type = ?"]
            activity_params: list[Any] = ["transcript"]

            if user_id:
                activity_where.append("user_id = ?")
                activity_params.append(user_id)

            if project_slug:
                activity_where.append("project_slug = ?")
                activity_params.append(project_slug)

            if start_date:
                activity_where.append("ts >= ?")
                activity_params.append(start_date)

            if end_date:
                activity_where.append("ts <= ?")
                activity_params.append(end_date)

            activity_clause = " AND ".join(activity_where)

            # Get distinct session_ids
            activity_query = f"""
                SELECT DISTINCT session_id
                FROM transcripts
                WHERE {activity_clause}
            """

            cursor = self._conn.execute(activity_query, activity_params)
            active_session_ids = [row[0] for row in cursor.fetchall()]

            if not active_session_ids:
                return []

            # Step 2: Fetch session metadata
            placeholders = ",".join("?" * len(active_session_ids))
            session_where: list[str] = [f"session_id IN ({placeholders})"]
            session_params: list[Any] = active_session_ids

            if min_turn_count is not None:
                session_where.append("turn_count >= ?")
                session_params.append(min_turn_count)

            session_clause = " AND ".join(session_where)

            query = f"""
                SELECT session_id, user_id, project_slug, bundle, created,
                       updated, turn_count, metadata
                FROM sessions
                WHERE {session_clause}
                ORDER BY updated DESC
                LIMIT {limit}
            """

            cursor = self._conn.execute(query, session_params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

        return await asyncio.to_thread(_get_active)

    # =========================================================================
    # Sequence-Based Navigation
    # =========================================================================

    async def get_message_context(
        self,
        session_id: str,
        sequence: int,
        user_id: str = "",
        before: int = 5,
        after: int = 5,
        include_tool_outputs: bool = True,
    ) -> MessageContext:
        """Get context window around a specific message by sequence."""

        def _get_context() -> MessageContext:
            # Build user filter
            user_filter = "user_id = ?" if user_id else "1=1"
            params: list[Any] = [session_id]
            if user_id:
                params.append(user_id)

            # Get the target message and surrounding context
            min_seq = max(0, sequence - before)
            max_seq = sequence + after

            role_filter = "" if include_tool_outputs else "AND role != 'tool'"

            query = f"""
                SELECT sequence, turn, role, content, ts, project_slug
                FROM transcripts
                WHERE session_id = ? AND {user_filter}
                  AND sequence >= ? AND sequence <= ?
                  {role_filter}
                ORDER BY sequence
            """
            params.extend([min_seq, max_seq])

            results = self.conn.execute(query, params).fetchall()

            messages: list[TranscriptMessage] = []
            project_slug = "unknown"

            for row in results:
                project_slug = row[5] or project_slug
                messages.append(
                    TranscriptMessage(
                        sequence=row[0],
                        turn=row[1],  # Can be None
                        role=row[2] or "unknown",
                        content=row[3] or "",
                        ts=row[4],
                        metadata={"session_id": session_id},
                    )
                )

            # Separate into previous, current, following
            previous: list[TranscriptMessage] = []
            current: TranscriptMessage | None = None
            following: list[TranscriptMessage] = []

            for msg in messages:
                if msg.sequence < sequence:
                    previous.append(msg)
                elif msg.sequence == sequence:
                    current = msg
                else:
                    following.append(msg)

            # Get session sequence range for navigation metadata
            range_params: list[Any] = [session_id]
            if user_id:
                range_params.append(user_id)

            min_result = self.conn.execute(
                f"SELECT MIN(sequence) FROM transcripts WHERE session_id = ? AND {user_filter}",
                range_params,
            ).fetchone()
            first_sequence = min_result[0] if min_result and min_result[0] is not None else 0

            max_result = self.conn.execute(
                f"SELECT MAX(sequence) FROM transcripts WHERE session_id = ? AND {user_filter}",
                range_params,
            ).fetchone()
            last_sequence = max_result[0] if max_result and max_result[0] is not None else sequence

            return MessageContext(
                session_id=session_id,
                project_slug=project_slug,
                target_sequence=sequence,
                previous=previous,
                current=current,
                following=following,
                has_more_before=first_sequence < min_seq,
                has_more_after=last_sequence > max_seq,
                first_sequence=first_sequence,
                last_sequence=last_sequence,
            )

        return await asyncio.to_thread(_get_context)
