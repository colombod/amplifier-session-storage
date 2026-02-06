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

from ..content_extraction import count_embeddable_content_types
from ..embeddings import EmbeddingProvider
from ..embeddings.mixin import EmbeddingMixin
from ..exceptions import StorageConnectionError, StorageIOError
from ..search.mmr import compute_mmr
from .base import (
    MessageContext,
    SearchFilters,
    SearchResult,
    StorageBackend,
    TranscriptMessage,
    TranscriptSearchOptions,
    TurnContext,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Column Definitions - Centralized for consistency and maintainability
# =============================================================================

# Vector column names - excluded from standard queries to avoid bloating LLM context
VECTOR_COLUMNS = frozenset(
    {
        "user_query_vector_json",
        "assistant_response_vector_json",
        "assistant_thinking_vector_json",
        "tool_output_vector_json",
    }
)

# Transcript columns for standard read operations (excludes vectors)
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
    "embedding_model",
    "vector_metadata",
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

# SQL for creating views that exclude vector columns
# These views provide a stable API even if schema changes
_CREATE_VIEWS_SQL = """
-- View for transcript reads without vectors (most common access pattern)
CREATE VIEW IF NOT EXISTS transcript_messages AS
SELECT
    id, user_id, host_id, project_slug, session_id, sequence,
    role, content_json, turn, ts, embedding_model, vector_metadata, synced_at
FROM transcripts;

-- View for vector-only access (used by vector search operations)
CREATE VIEW IF NOT EXISTS transcript_vectors AS
SELECT
    id, user_id, session_id, sequence,
    user_query_vector_json, assistant_response_vector_json,
    assistant_thinking_vector_json, tool_output_vector_json
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
                    user_query_vector_json TEXT,
                    assistant_response_vector_json TEXT,
                    assistant_thinking_vector_json TEXT,
                    tool_output_vector_json TEXT,
                    embedding_model TEXT,
                    vector_metadata TEXT,
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

            # Create views for vector-free access (best practice per SQLITE_BEST_PRACTICES.md)
            # These views provide a stable API even if schema changes
            await self.conn.executescript(_CREATE_VIEWS_SQL)
            logger.debug("Created transcript_messages and transcript_vectors views")

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
        embeddings: dict[str, list[list[float] | None]] | None = None,
    ) -> int:
        """Sync transcript lines with optional multi-vector embeddings."""
        if not lines:
            return 0

        if self.conn is None:
            raise StorageIOError("sync_transcript", cause=RuntimeError("Not initialized"))

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
                assistant_response_vec = embeddings.get("assistant_response", [None] * len(lines))[
                    i
                ]
                assistant_thinking_vec = embeddings.get("assistant_thinking", [None] * len(lines))[
                    i
                ]
                tool_output_vec = embeddings.get("tool_output", [None] * len(lines))[i]

            # Serialize vectors to JSON TEXT for SQLite
            user_query_json = json.dumps(user_query_vec) if user_query_vec else None
            assistant_response_json = (
                json.dumps(assistant_response_vec) if assistant_response_vec else None
            )
            assistant_thinking_json = (
                json.dumps(assistant_thinking_vec) if assistant_thinking_vec else None
            )
            tool_output_json = json.dumps(tool_output_vec) if tool_output_vec else None

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

            await self.conn.execute(
                """
                INSERT INTO transcripts (
                    id, user_id, host_id, project_slug, session_id, sequence,
                    role, content_json, turn, ts,
                    user_query_vector_json, assistant_response_vector_json,
                    assistant_thinking_vector_json, tool_output_vector_json,
                    embedding_model, vector_metadata, synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    content_json = excluded.content_json,
                    user_query_vector_json = excluded.user_query_vector_json,
                    assistant_response_vector_json = excluded.assistant_response_vector_json,
                    assistant_thinking_vector_json = excluded.assistant_thinking_vector_json,
                    tool_output_vector_json = excluded.tool_output_vector_json,
                    embedding_model = excluded.embedding_model,
                    vector_metadata = excluded.vector_metadata,
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
                    user_query_json,
                    assistant_response_json,
                    assistant_thinking_json,
                    tool_output_json,
                    self.embedding_provider.model_name if self.embedding_provider else None,
                    json.dumps(vector_metadata),
                    datetime.now(UTC).isoformat(),
                ),
            )

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

        # Translate search_in_* flags to vector_columns
        vector_columns: list[str] | None = []
        if options.search_in_user:
            vector_columns.append("user_query")
        if options.search_in_assistant:
            vector_columns.append("assistant_response")
        if options.search_in_thinking:
            vector_columns.append("assistant_thinking")
        if options.search_in_tool:
            vector_columns.append("tool_output")
        if not vector_columns:
            vector_columns = None  # None = search all

        return await self.vector_search(
            user_id, query_vector, options.filters, limit, vector_columns=vector_columns
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
        """Fetch embedding for a specific transcript (returns first non-null vector)."""
        if self.conn is None:
            return None

        async with self.conn.execute(
            """
            SELECT user_query_vector_json, assistant_response_vector_json,
                   assistant_thinking_vector_json, tool_output_vector_json
            FROM transcripts
            WHERE user_id = ? AND session_id = ? AND sequence = ?
            """,
            (user_id, session_id, sequence),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                # Return first non-null vector (for backward compatibility with tests)
                for vec_json in row:
                    if vec_json:
                        return json.loads(vec_json)
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
        cursor = self.conn.execute(
            f"""
            SELECT sequence, turn, role, content, ts, project_slug
            FROM transcripts
            WHERE user_id = ? AND session_id = ?
              AND turn >= ? AND turn <= ?
              {role_filter}
            ORDER BY turn, sequence
            """,
            (user_id, session_id, min_turn, max_turn),
        )
        rows = cursor.fetchall()

        # Get session turn range
        range_cursor = self.conn.execute(
            """
            SELECT MIN(turn), MAX(turn)
            FROM transcripts
            WHERE user_id = ? AND session_id = ?
            """,
            (user_id, session_id),
        )
        range_row = range_cursor.fetchone()

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
        """Check if vector search is available."""
        return self._vss_available and self.embedding_provider is not None

    async def upsert_embeddings(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
        embeddings: list[dict[str, Any]],
    ) -> int:
        """Upsert embeddings for existing transcripts (legacy method - deprecated)."""
        if self.conn is None:
            raise StorageIOError("upsert_embeddings", cause=RuntimeError("Not initialized"))

        # This method is deprecated in multi-vector design
        # For backward compatibility, we store single vectors in user_query_vector
        logger.warning(
            "upsert_embeddings is deprecated in multi-vector schema. "
            "Use sync_transcript_lines with multi-vector embeddings instead."
        )

        updated = 0
        for emb in embeddings:
            doc_id = f"{session_id}_msg_{emb['sequence']}"
            embedding_json = json.dumps(emb["vector"])

            # Store in user_query_vector for backward compatibility
            await self.conn.execute(
                """
                UPDATE transcripts
                SET user_query_vector_json = ?,
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

            updated += 1

        await self.conn.commit()
        return updated

    async def vector_search(
        self,
        user_id: str,
        query_vector: list[float],
        filters: SearchFilters | None = None,
        top_k: int = 100,
        vector_columns: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Perform multi-vector similarity search.

        Searches across specified vector columns and returns best matches.

        Args:
            user_id: User identifier
            query_vector: Query embedding vector
            filters: Optional search filters
            top_k: Number of results to return
            vector_columns: Which vector columns to search. Default: all non-null vectors.
                Options: ["user_query", "assistant_response",
                          "assistant_thinking", "tool_output"]
        """
        if self.conn is None:
            raise StorageIOError("vector_search", cause=RuntimeError("Not initialized"))

        if vector_columns is None:
            # Search all vector columns by default
            vector_columns = [
                "user_query",
                "assistant_response",
                "assistant_thinking",
                "tool_output",
            ]

        if self._vss_available:
            return await self._vss_vector_search(
                user_id, query_vector, filters, top_k, vector_columns
            )
        else:
            return await self._numpy_vector_search(
                user_id, query_vector, filters, top_k, vector_columns
            )

    async def _vss_vector_search(
        self,
        user_id: str,
        query_vector: list[float],
        filters: SearchFilters | None,
        top_k: int,
        vector_columns: list[str],
    ) -> list[SearchResult]:
        """Vector search using sqlite-vss extension."""
        # Note: Implementation depends on sqlite-vss API
        # This is a placeholder for when sqlite-vss is properly integrated
        logger.warning("sqlite-vss integration not yet implemented, using numpy fallback")
        return await self._numpy_vector_search(
            user_id, query_vector, filters, top_k, vector_columns
        )

    async def _numpy_vector_search(
        self,
        user_id: str,
        query_vector: list[float],
        filters: SearchFilters | None,
        top_k: int,
        vector_columns: list[str],
    ) -> list[SearchResult]:
        """Brute-force multi-vector search using numpy cosine similarity."""
        if self.conn is None:
            raise StorageIOError("vector_search", cause=RuntimeError("Not initialized"))

        # Map content type names to JSON column names
        column_map = {
            "user_query": "user_query_vector_json",
            "assistant_response": "assistant_response_vector_json",
            "assistant_thinking": "assistant_thinking_vector_json",
            "tool_output": "tool_output_vector_json",
        }

        # Build WHERE clause - require at least one vector column to be non-null
        where_parts = ["user_id = ?"]
        vector_checks = [f"{column_map[col]} IS NOT NULL" for col in vector_columns]
        where_parts.append(f"({' OR '.join(vector_checks)})")
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

        # Select all vector columns
        select_columns = [
            "id",
            "session_id",
            "project_slug",
            "sequence",
            "content_json",
            "role",
            "turn",
            "ts",
            "user_query_vector_json",
            "assistant_response_vector_json",
            "assistant_thinking_vector_json",
            "tool_output_vector_json",
        ]

        query = f"""
            SELECT {", ".join(select_columns)}
            FROM transcripts
            WHERE {where_clause}
        """

        # Fetch all candidates
        async with self.conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        # Compute similarities with numpy (GREATEST strategy across all vectors)
        query_np = np.array(query_vector)
        candidates: list[tuple[float, Any]] = []

        for row in rows:
            # Load all 4 vector types (indices 8-11)
            vectors_to_check = []

            for idx, vec_type in enumerate(
                ["user_query", "assistant_response", "assistant_thinking", "tool_output"]
            ):
                if vec_type in vector_columns:
                    vec_json = row[8 + idx]  # Columns 8-11 are the vector JSONs
                    if vec_json:
                        vec = json.loads(vec_json)
                        vec_np = np.array(vec)

                        # Cosine similarity
                        sim = float(
                            np.dot(query_np, vec_np)
                            / (np.linalg.norm(query_np) * np.linalg.norm(vec_np))
                        )
                        vectors_to_check.append(sim)

            # Use maximum similarity across all vector types (GREATEST strategy)
            if vectors_to_check:
                max_similarity = max(vectors_to_check)
                candidates.append((max_similarity, row))

        # Sort by similarity (descending) and take top_k
        candidates.sort(reverse=True, key=lambda x: x[0])
        top_candidates = candidates[:top_k]

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
            SELECT sequence, turn, role, content, ts, project_slug
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
