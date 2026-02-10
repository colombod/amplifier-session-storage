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
    "content_json",
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
CREATE VIEW IF NOT EXISTS transcript_messages AS
SELECT id, user_id, host_id, project_slug, session_id, sequence,
    role, content_json, turn, ts, synced_at
FROM transcripts;
"""


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


class SQLiteBackend(EmbeddingMixin, StorageBackend):
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

            # Create transcripts table (vector-free since schema v2)
            # CREATE TABLE IF NOT EXISTS won't modify existing tables, so old
            # DBs keep their inline vector columns until migration runs.
            await self.conn.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    id TEXT NOT NULL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    host_id TEXT NOT NULL,
                    project_slug TEXT,
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    role TEXT,
                    content_json TEXT,
                    turn INTEGER,
                    ts TEXT,
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

            # Always create schema_meta and transcript_vectors tables
            await self._create_schema_meta_table()
            await self._create_transcript_vectors_table()

            # Schema versioning and migration
            schema_version = await self._get_schema_version()
            if schema_version < 2:
                await self._migrate_to_externalized_vectors()

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

            # Create indexes on transcript_vectors
            await self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vectors_parent ON transcript_vectors (parent_id)"
            )
            await self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vectors_session "
                "ON transcript_vectors (user_id, session_id)"
            )

            # Drop old views that reference removed columns, then create new ones
            await self.conn.execute("DROP VIEW IF EXISTS transcript_messages")
            await self.conn.execute("DROP VIEW IF EXISTS transcript_vectors_view")
            await self.conn.executescript(_CREATE_VIEWS_SQL)
            logger.debug("Created transcript_messages view")

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
    # Schema Management
    # =========================================================================

    async def _create_schema_meta_table(self) -> None:
        """Create schema_meta table for version tracking."""
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

    async def _create_transcript_vectors_table(self) -> None:
        """Create the externalized transcript_vectors table."""
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS transcript_vectors (
                id TEXT NOT NULL PRIMARY KEY,
                parent_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                project_slug TEXT,
                content_type TEXT NOT NULL,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                total_chunks INTEGER NOT NULL DEFAULT 1,
                span_start INTEGER NOT NULL DEFAULT 0,
                span_end INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                source_text TEXT NOT NULL,
                vector_json TEXT,
                embedding_model TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

    async def _get_schema_version(self) -> int:
        """Get the current schema version (defaults to 1 for pre-versioned DBs)."""
        try:
            async with self.conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'version'"
            ) as cursor:
                result = await cursor.fetchone()
                return int(result[0]) if result else 1
        except Exception:
            return 1

    async def _set_schema_version(self, version: int) -> None:
        """Set the schema version."""
        await self.conn.execute(
            """
            INSERT INTO schema_meta (key, value) VALUES ('version', ?)
            ON CONFLICT (key) DO UPDATE SET value = excluded.value
            """,
            (str(version),),
        )
        await self.conn.commit()

    async def _migrate_to_externalized_vectors(self) -> None:
        """Migrate inline vectors from transcripts to the transcript_vectors table."""
        logger.info("Migrating inline vectors to transcript_vectors table...")

        # Check if transcript_vectors already has data (idempotency)
        async with self.conn.execute("SELECT COUNT(*) FROM transcript_vectors") as cursor:
            row = await cursor.fetchone()
            count = row[0] if row else 0

        if count > 0:
            logger.info(f"transcript_vectors already has {count} rows, skipping migration")
            await self._set_schema_version(2)
            return

        # Check if old vector columns exist
        async with self.conn.execute("PRAGMA table_info(transcripts)") as cursor:
            columns_info = await cursor.fetchall()
            columns = [col[1] for col in columns_info]

        if "user_query_vector_json" not in columns:
            logger.info("No inline vector columns found, skipping migration")
            await self._set_schema_version(2)
            return

        # Read all transcript rows that have at least one vector
        async with self.conn.execute("""
            SELECT id, user_id, session_id, project_slug, role, content_json,
                   user_query_vector_json, assistant_response_vector_json,
                   assistant_thinking_vector_json, tool_output_vector_json,
                   embedding_model
            FROM transcripts
            WHERE user_query_vector_json IS NOT NULL
               OR assistant_response_vector_json IS NOT NULL
               OR assistant_thinking_vector_json IS NOT NULL
               OR tool_output_vector_json IS NOT NULL
        """) as cursor:
            rows = await cursor.fetchall()

        migrated = 0
        for row in rows:
            parent_id = row[0]
            user_id = row[1]
            session_id = row[2]
            project_slug = row[3]
            role = row[4]
            content_json = row[5]
            vectors = {
                "user_query": row[6],
                "assistant_response": row[7],
                "assistant_thinking": row[8],
                "tool_output": row[9],
            }
            embedding_model = row[10]

            # Re-extract content to get correct source_text
            if isinstance(content_json, str):
                try:
                    parsed_content = json.loads(content_json)
                except (json.JSONDecodeError, TypeError):
                    parsed_content = content_json
            else:
                parsed_content = content_json

            message = {"role": role, "content": parsed_content}
            extracted = extract_all_embeddable_content(message)

            for content_type, vec_json in vectors.items():
                if vec_json is None:
                    continue

                source_text = extracted.get(content_type) or ""
                if not source_text:
                    source_text = str(content_json)[:1000] if content_json else ""

                token_count_val = count_tokens(source_text) if source_text else 0
                vector_id = f"{parent_id}_{content_type}_0"

                await self.conn.execute(
                    """
                    INSERT INTO transcript_vectors (
                        id, parent_id, user_id, session_id, project_slug,
                        content_type, chunk_index, total_chunks,
                        span_start, span_end, token_count,
                        source_text, vector_json, embedding_model
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        ?, 0, 1,
                        0, ?, ?,
                        ?, ?, ?
                    )
                    """,
                    (
                        vector_id,
                        parent_id,
                        user_id,
                        session_id,
                        project_slug,
                        content_type,
                        len(source_text),
                        token_count_val,
                        source_text,
                        vec_json,  # Already JSON text
                        embedding_model,
                    ),
                )
                migrated += 1

        # Leave old columns in place (they'll be NULL for new rows).
        # SQLite < 3.35 doesn't support ALTER TABLE DROP COLUMN reliably,
        # and the columns are just dead weight - no need to risk a table recreate.

        await self._set_schema_version(2)
        logger.info(f"Migration complete: {migrated} vectors moved to transcript_vectors")

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
            # Delete transcript vectors
            await self.conn.execute(
                "DELETE FROM transcript_vectors WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )

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
        embeddings: dict[str, list[list[float] | None]] | None = None,
    ) -> int:
        """Sync transcript lines with externalized vector storage.

        Step 1: Store transcript rows (content only, no vectors).
        Step 2: Generate and store vectors in transcript_vectors table.
        """
        if not lines:
            return 0

        if self.conn is None:
            raise StorageIOError("sync_transcript", cause=RuntimeError("Not initialized"))

        # Step 1: Store transcript rows (content only)
        stored = await self._store_transcript_rows(
            user_id, host_id, project_slug, session_id, lines, start_sequence
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
                    f"Vector generation failed for session {session_id} "
                    f"({len(lines)} messages) — transcripts stored without vectors: {exc}"
                )
        elif embeddings is not None:
            # Pre-computed embeddings (from sync daemon) - store as single-chunk vectors
            await self._store_precomputed_vectors(
                user_id, project_slug, session_id, lines, start_sequence, embeddings
            )

        return stored

    async def _store_transcript_rows(
        self,
        user_id: str,
        host_id: str,
        project_slug: str,
        session_id: str,
        lines: list[dict[str, Any]],
        start_sequence: int,
    ) -> int:
        """Store transcript rows (content only, no vectors)."""
        synced = 0
        for i, line in enumerate(lines):
            sequence = start_sequence + i
            doc_id = f"{session_id}_msg_{sequence}"

            # Store content as JSON (handles both string and array formats)
            content = line.get("content")
            content_json = json.dumps(content) if content is not None else None

            now = datetime.now(UTC).isoformat()
            await self.conn.execute(
                """
                INSERT INTO transcripts (
                    id, user_id, host_id, project_slug, session_id, sequence,
                    role, content_json, turn, ts, synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    content_json = excluded.content_json,
                    role = excluded.role,
                    turn = excluded.turn,
                    ts = excluded.ts,
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
                    content_json,
                    line.get("turn"),
                    line.get("ts") or line.get("timestamp"),
                    now,
                ),
            )
            synced += 1

        await self.conn.commit()
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
        await self._store_vector_records(all_vector_records)

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

    async def _store_vector_records(self, records: list[dict[str, Any]]) -> None:
        """Store vector records in transcript_vectors table.

        Uses delete-before-insert for idempotent re-sync.
        """
        if not records:
            return

        # Group by parent_id for cleanup
        parent_ids = {r["parent_id"] for r in records}

        # Delete existing vectors for these parents
        for parent_id in parent_ids:
            await self.conn.execute(
                "DELETE FROM transcript_vectors WHERE parent_id = ?",
                (parent_id,),
            )

        # Insert new vectors
        for record in records:
            vector = record.get("vector")
            vector_json = json.dumps(vector) if vector is not None else None

            await self.conn.execute(
                """
                INSERT INTO transcript_vectors (
                    id, parent_id, user_id, session_id, project_slug,
                    content_type, chunk_index, total_chunks,
                    span_start, span_end, token_count,
                    source_text, vector_json, embedding_model
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?
                )
                """,
                (
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
                    vector_json,
                    record["embedding_model"],
                ),
            )

        await self.conn.commit()

    async def _store_precomputed_vectors(
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
                        "token_count": (count_tokens(source_text) if source_text else 0),
                        "source_text": source_text,
                        "vector": vector,
                        "embedding_model": (
                            self.embedding_provider.model_name if self.embedding_provider else None
                        ),
                    }
                )

        await self._store_vector_records(records)

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
            SELECT id, sequence, role, content_json, turn, ts
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
                    "content": json.loads(row[3]) if row[3] else "",
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
        """Full-text search across transcripts and vector source texts."""
        if self.conn is None:
            raise StorageIOError("search", cause=RuntimeError("Not initialized"))

        # Search transcript content
        transcript_results = await self._full_text_search_transcript_content(
            user_id, options, limit
        )

        # Also search transcript_vectors.source_text (BUG 1+5)
        vector_text_results = await self._full_text_search_vector_sources(user_id, options, limit)

        # Merge and deduplicate by (session_id, sequence)
        seen: set[tuple[str, int]] = set()
        combined: list[SearchResult] = []
        for r in transcript_results + vector_text_results:
            key = (r.session_id, r.sequence)
            if key not in seen:
                seen.add(key)
                combined.append(r)

        return combined[:limit]

    async def _full_text_search_transcript_content(
        self,
        user_id: str,
        options: TranscriptSearchOptions,
        limit: int,
    ) -> list[SearchResult]:
        """Search transcript content_json column."""
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

        # Content search (search in JSON content)
        where_parts.append("content_json LIKE ?")
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
            SELECT id, session_id, project_slug, sequence, content_json, role, turn, ts
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
                    content=json.loads(row[4]) if row[4] else "",
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

    async def _full_text_search_vector_sources(
        self,
        user_id: str,
        options: TranscriptSearchOptions,
        limit: int,
    ) -> list[SearchResult]:
        """Search source_text in transcript_vectors for span-level full-text matching.

        This catches extracted thinking blocks, tool outputs, and chunked content
        that wouldn't be found by searching transcripts.content_json alone.
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
                   t.project_slug, t.sequence, t.content_json, t.role, t.turn, t.ts
            FROM transcript_vectors v
            JOIN transcripts t ON t.id = v.parent_id
            WHERE {where_clause}
            LIMIT ?
        """
        params.append(limit * 3)  # Over-fetch for dedup

        async with self.conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        # Deduplicate by parent_id, keeping first match
        seen: dict[str, SearchResult] = {}
        for row in rows:
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
            user_id,
            query_vector,
            options.filters,
            limit,
            content_types=content_types,
        )

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

        # Compute query embedding ONCE
        query_vector = await self.embedding_provider.embed_text(options.query)
        query_np = np.array(query_vector)

        # Get both result sets (semantic reuses pre-computed vector)
        text_results = await self._full_text_search(user_id, options, candidate_limit)
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

        # Get embeddings for MMR
        vectors = []
        for result in combined:
            ct = result.metadata.get("content_type")
            embedding = await self._get_embedding(user_id, result.session_id, result.sequence, ct)
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
        self, user_id: str, session_id: str, sequence: int, content_type: str | None = None
    ) -> list[float] | None:
        """Fetch embedding for a transcript message, optionally by content_type.

        When content_type is provided, fetches the specific matched vector.
        Otherwise returns the first available vector.
        """
        if self.conn is None:
            return None

        parent_id = f"{session_id}_msg_{sequence}"

        if content_type:
            async with self.conn.execute(
                """
                SELECT vector_json
                FROM transcript_vectors
                WHERE parent_id = ? AND user_id = ? AND content_type = ?
                  AND vector_json IS NOT NULL
                ORDER BY chunk_index
                LIMIT 1
                """,
                (parent_id, user_id, content_type),
            ) as cursor:
                row = await cursor.fetchone()
        else:
            async with self.conn.execute(
                """
                SELECT vector_json
                FROM transcript_vectors
                WHERE parent_id = ? AND user_id = ? AND vector_json IS NOT NULL
                ORDER BY chunk_index
                LIMIT 1
                """,
                (parent_id, user_id),
            ) as cursor:
                row = await cursor.fetchone()

        if row and row[0]:
            return json.loads(row[0])
        return None

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
        async with self.conn.execute(
            f"""
            SELECT sequence, turn, role, content_json, ts, project_slug
            FROM transcripts
            WHERE user_id = ? AND session_id = ?
              AND turn >= ? AND turn <= ?
              {role_filter}
            ORDER BY turn, sequence
            """,
            (user_id, session_id, min_turn, max_turn),
        ) as cursor:
            rows = await cursor.fetchall()

        # Get session turn range
        async with self.conn.execute(
            """
            SELECT MIN(turn), MAX(turn)
            FROM transcripts
            WHERE user_id = ? AND session_id = ?
            """,
            (user_id, session_id),
        ) as range_cursor:
            range_row = await range_cursor.fetchone()

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
        """Check if vector search is available.

        With externalized vectors, numpy brute-force search is always available
        as long as we have an embedding provider.
        """
        return self.embedding_provider is not None

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
        if self.conn is None:
            raise StorageIOError("upsert_embeddings", cause=RuntimeError("Not initialized"))

        updated = 0
        for emb in embeddings:
            parent_id = f"{session_id}_msg_{emb['sequence']}"
            vector_id = f"{parent_id}_user_query_0"
            vector_json = json.dumps(emb["vector"])

            # Delete existing vectors for this parent
            await self.conn.execute(
                "DELETE FROM transcript_vectors WHERE parent_id = ?",
                (parent_id,),
            )

            await self.conn.execute(
                """
                INSERT INTO transcript_vectors (
                    id, parent_id, user_id, session_id, project_slug,
                    content_type, chunk_index, total_chunks,
                    span_start, span_end, token_count,
                    source_text, vector_json, embedding_model
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    'user_query', 0, 1,
                    0, 0, 0,
                    '', ?, ?
                )
                """,
                (
                    vector_id,
                    parent_id,
                    user_id,
                    session_id,
                    project_slug,
                    vector_json,
                    emb.get("metadata", {}).get("model", "unknown"),
                ),
            )
            updated += 1

        await self.conn.commit()
        return updated

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
        if self.conn is None:
            raise StorageIOError("vector_search", cause=RuntimeError("Not initialized"))

        return await self._numpy_vector_search(user_id, query_vector, filters, top_k, content_types)

    async def _numpy_vector_search(
        self,
        user_id: str,
        query_vector: list[float],
        filters: SearchFilters | None,
        top_k: int,
        content_types: list[str] | None,
    ) -> list[SearchResult]:
        """Brute-force vector search using numpy cosine similarity.

        Reads from transcript_vectors table, computes cosine similarity,
        deduplicates by parent_id (keeping best score per parent),
        then JOINs to transcripts for content.
        """
        if self.conn is None:
            raise StorageIOError("vector_search", cause=RuntimeError("Not initialized"))

        # Build WHERE clause on transcript_vectors
        where_parts = ["v.user_id = ?"]
        params: list[Any] = [user_id]

        # Filter by content type
        if content_types:
            placeholders = ", ".join(["?"] * len(content_types))
            where_parts.append(f"v.content_type IN ({placeholders})")
            params.extend(content_types)

        # Require non-null vectors
        where_parts.append("v.vector_json IS NOT NULL")

        if filters:
            if filters.project_slug:
                where_parts.append("v.project_slug = ?")
                params.append(filters.project_slug)

            if filters.start_date:
                where_parts.append("t.ts >= ?")
                params.append(filters.start_date)

            if filters.end_date:
                where_parts.append("t.ts <= ?")
                params.append(filters.end_date)

        where_clause = " AND ".join(where_parts)

        # Query vectors with transcript content via JOIN
        query = f"""
            SELECT v.parent_id, v.vector_json, v.content_type,
                   t.session_id, t.project_slug, t.sequence,
                   t.content_json, t.role, t.turn, t.ts
            FROM transcript_vectors v
            JOIN transcripts t ON t.id = v.parent_id
            WHERE {where_clause}
        """

        async with self.conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        # Compute similarities with numpy
        query_np = np.array(query_vector)
        query_norm = np.linalg.norm(query_np)

        # Deduplicate by parent_id, keeping best score
        seen: dict[str, tuple[float, Any]] = {}

        for row in rows:
            parent_id = row[0]
            vec_json = row[1]

            vec = json.loads(vec_json)
            vec_np = np.array(vec)

            # Cosine similarity
            vec_norm = np.linalg.norm(vec_np)
            if query_norm == 0 or vec_norm == 0:
                sim = 0.0
            else:
                sim = float(np.dot(query_np, vec_np) / (query_norm * vec_norm))

            if parent_id not in seen or sim > seen[parent_id][0]:
                seen[parent_id] = (sim, row)

        # Sort by score descending and take top_k
        sorted_results = sorted(seen.values(), key=lambda x: x[0], reverse=True)
        top_results = sorted_results[:top_k]

        return [
            SearchResult(
                session_id=row[3],
                project_slug=row[4],
                sequence=row[5],
                content=json.loads(row[6]) if row[6] else "",
                metadata={
                    "id": row[0],
                    "role": row[7],
                    "turn": row[8],
                    "ts": row[9],
                    "content_type": row[2],
                },
                score=similarity,
                source="semantic",
            )
            for similarity, row in top_results
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

    async def get_session_sync_stats(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
    ) -> SessionSyncStats:
        """Get lightweight sync statistics using aggregate queries."""
        if self.conn is None:
            raise StorageIOError("get_session_sync_stats", cause=RuntimeError("Not initialized"))

        async with self.conn.execute(
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
            (user_id, session_id, user_id, session_id),
        ) as cursor:
            rows = await cursor.fetchall()

        event_count = transcript_count = 0
        event_earliest = event_latest = None
        transcript_earliest = transcript_latest = None

        for row in rows:
            doc_type, count, earliest, latest = row
            if doc_type == "event":
                event_count = count
                event_earliest = earliest
                event_latest = latest
            elif doc_type == "transcript":
                transcript_count = count
                transcript_earliest = earliest
                transcript_latest = latest

        return SessionSyncStats(
            event_count=event_count,
            transcript_count=transcript_count,
            event_ts_range=(event_earliest, event_latest),
            transcript_ts_range=(transcript_earliest, transcript_latest),
        )

    # =========================================================================
    # Discovery APIs
    # =========================================================================

    async def list_users(
        self,
        filters: SearchFilters | None = None,
    ) -> list[str]:
        """List all unique user IDs in the storage."""
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

        async with self.conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [row[0] for row in rows if row[0]]

    async def list_projects(
        self,
        user_id: str = "",
        filters: SearchFilters | None = None,
    ) -> list[str]:
        """List all unique project slugs."""
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
        query = (
            f"SELECT DISTINCT project_slug FROM sessions WHERE {where_clause} ORDER BY project_slug"
        )

        async with self.conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [row[0] for row in rows if row[0]]

    async def list_sessions(
        self,
        user_id: str = "",
        project_slug: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List sessions with pagination."""
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

        async with self.conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "session_id": row[0],
                    "user_id": row[1],
                    "project_slug": row[2],
                    "bundle": row[3],
                    "created": row[4],
                    "turn_count": row[5],
                }
                for row in rows
            ]

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

        # Get distinct session_ids with activity
        activity_query = f"""
            SELECT DISTINCT session_id
            FROM transcripts
            WHERE {activity_clause}
        """

        async with self.conn.execute(activity_query, activity_params) as cursor:
            rows = await cursor.fetchall()
            active_session_ids = [row[0] for row in rows]

        if not active_session_ids:
            return []

        # Step 2: Fetch session metadata for those IDs
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

        async with self.conn.execute(query, session_params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

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
            SELECT sequence, turn, role, content_json, ts, project_slug
            FROM transcripts
            WHERE session_id = ? AND {user_filter}
              AND sequence >= ? AND sequence <= ?
              {role_filter}
            ORDER BY sequence
        """
        params.extend([min_seq, max_seq])

        messages: list[TranscriptMessage] = []
        project_slug = "unknown"

        async with self.conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            for row in rows:
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

        async with self.conn.execute(
            f"SELECT MIN(sequence) FROM transcripts WHERE session_id = ? AND {user_filter}",
            range_params,
        ) as cursor:
            row = await cursor.fetchone()
            first_sequence = row[0] if row and row[0] is not None else 0

        async with self.conn.execute(
            f"SELECT MAX(sequence) FROM transcripts WHERE session_id = ? AND {user_filter}",
            range_params,
        ) as cursor:
            row = await cursor.fetchone()
            last_sequence = row[0] if row and row[0] is not None else sequence

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
