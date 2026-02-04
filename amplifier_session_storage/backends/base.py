"""
Abstract base classes for storage backends.

All storage implementations (Cosmos, DuckDB, SQLite) must implement these interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchFilters:
    """Common filters for search operations."""

    project_slug: str | None = None
    session_id: str | None = None
    start_date: str | None = None  # ISO format
    end_date: str | None = None  # ISO format
    bundle: str | None = None
    min_turn_count: int | None = None
    max_turn_count: int | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class TranscriptSearchOptions:
    """Options for transcript search."""

    query: str  # Search query text
    search_in_user: bool = True  # Search in user messages
    search_in_assistant: bool = True  # Search in assistant messages
    search_in_thinking: bool = True  # Search in thinking blocks
    search_type: str = "hybrid"  # "full_text", "semantic", "hybrid"
    mmr_lambda: float = 0.7  # MMR relevance vs diversity (1.0=relevance, 0.0=diversity)
    filters: SearchFilters | None = None


@dataclass
class EventSearchOptions:
    """Options for event search."""

    event_type: str | None = None  # e.g., "llm.request", "tool.call"
    tool_name: str | None = None  # Filter by specific tool
    level: str | None = None  # "debug", "info", "warning", "error"
    filters: SearchFilters | None = None


@dataclass
class SearchResult:
    """Search result with relevance scoring."""

    session_id: str
    project_slug: str
    sequence: int
    content: str
    metadata: dict[str, Any]
    score: float  # Relevance score (higher = more relevant)
    source: str  # "full_text", "semantic", "hybrid"


class StorageBackend(ABC):
    """
    Abstract base for all storage backends.

    Implementations must support:
    - Session metadata storage and retrieval
    - Transcript line storage and search
    - Event line storage and search
    - Vector embeddings for semantic search
    - Graceful degradation when embeddings unavailable
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend (connections, schema, indexes)."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connections and cleanup resources."""
        pass

    async def __aenter__(self) -> StorageBackend:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    # =========================================================================
    # Session Metadata Operations
    # =========================================================================

    @abstractmethod
    async def upsert_session_metadata(
        self,
        user_id: str,
        host_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """
        Upsert session metadata.

        Args:
            user_id: User identifier
            host_id: Host/device identifier
            metadata: Session metadata dict (must include session_id)
        """
        pass

    @abstractmethod
    async def get_session_metadata(
        self,
        user_id: str,
        session_id: str,
    ) -> dict[str, Any] | None:
        """Get session metadata by ID."""
        pass

    @abstractmethod
    async def search_sessions(
        self,
        user_id: str,
        filters: SearchFilters,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Search sessions with advanced filters.

        Args:
            user_id: User identifier
            filters: Search filters (project, date range, bundle, etc.)
            limit: Maximum number of results

        Returns:
            List of session metadata dicts matching criteria
        """
        pass

    @abstractmethod
    async def delete_session(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
    ) -> bool:
        """Delete a session and all its data."""
        pass

    # =========================================================================
    # Transcript Operations
    # =========================================================================

    @abstractmethod
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
        """
        Sync transcript lines with optional embeddings.

        Args:
            user_id: User identifier
            host_id: Host/device identifier
            project_slug: Project slug
            session_id: Session identifier
            lines: Transcript message dicts
            start_sequence: Starting sequence number
            embeddings: Optional pre-computed embeddings (same order as lines)

        Returns:
            Number of lines synced
        """
        pass

    @abstractmethod
    async def get_transcript_lines(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
        after_sequence: int = -1,
    ) -> list[dict[str, Any]]:
        """Get transcript lines for a session."""
        pass

    @abstractmethod
    async def search_transcripts(
        self,
        user_id: str,
        options: TranscriptSearchOptions,
        limit: int = 100,
    ) -> list[SearchResult]:
        """
        Search transcript messages with hybrid search support.

        Args:
            user_id: User identifier
            options: Search configuration
            limit: Maximum results

        Returns:
            List of search results with relevance scores
        """
        pass

    # =========================================================================
    # Event Operations
    # =========================================================================

    @abstractmethod
    async def sync_event_lines(
        self,
        user_id: str,
        host_id: str,
        project_slug: str,
        session_id: str,
        lines: list[dict[str, Any]],
        start_sequence: int = 0,
    ) -> int:
        """
        Sync event lines to storage.

        Args:
            user_id: User identifier
            host_id: Host/device identifier
            project_slug: Project slug
            session_id: Session identifier
            lines: Event dicts
            start_sequence: Starting sequence number

        Returns:
            Number of lines synced
        """
        pass

    @abstractmethod
    async def get_event_lines(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
        after_sequence: int = -1,
    ) -> list[dict[str, Any]]:
        """Get event lines for a session (may return summaries for large events)."""
        pass

    @abstractmethod
    async def search_events(
        self,
        user_id: str,
        options: EventSearchOptions,
        limit: int = 100,
    ) -> list[SearchResult]:
        """
        Search events by type, tool, date range.

        Args:
            user_id: User identifier
            options: Search configuration
            limit: Maximum results

        Returns:
            List of matching events with metadata
        """
        pass

    # =========================================================================
    # Vector/Embedding Operations
    # =========================================================================

    @abstractmethod
    async def supports_vector_search(self) -> bool:
        """Check if this backend supports vector similarity search."""
        pass

    @abstractmethod
    async def upsert_embeddings(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
        embeddings: list[dict[str, Any]],
    ) -> int:
        """
        Upsert embeddings for semantic search.

        Args:
            user_id: User identifier
            project_slug: Project slug
            session_id: Session identifier
            embeddings: List of embedding dicts with:
                - sequence: int
                - text: str (original text)
                - vector: list[float] (embedding vector)
                - metadata: dict (role, turn, etc.)

        Returns:
            Number of embeddings stored
        """
        pass

    @abstractmethod
    async def vector_search(
        self,
        user_id: str,
        query_vector: list[float],
        filters: SearchFilters | None = None,
        top_k: int = 100,
    ) -> list[SearchResult]:
        """
        Perform vector similarity search.

        Args:
            user_id: User identifier
            query_vector: Query embedding vector
            filters: Optional filters (project, date range, etc.)
            top_k: Number of results

        Returns:
            List of search results ordered by similarity
        """
        pass

    # =========================================================================
    # Analytics & Aggregations
    # =========================================================================

    @abstractmethod
    async def get_session_statistics(
        self,
        user_id: str,
        filters: SearchFilters | None = None,
    ) -> dict[str, Any]:
        """
        Get aggregate statistics across sessions.

        Returns dict with:
            - total_sessions: int
            - total_messages: int
            - total_events: int
            - sessions_by_project: dict[str, int]
            - sessions_by_bundle: dict[str, int]
            - events_by_type: dict[str, int]
            - tools_used: dict[str, int]
        """
        pass
