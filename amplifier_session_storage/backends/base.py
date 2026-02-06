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
    search_in_tool: bool = False  # Search in tool output blocks (default off - usually noisy)
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


@dataclass
class TranscriptMessage:
    """A single message in a transcript."""

    sequence: int
    turn: int | None  # Can be None in some transcripts
    role: str  # "user", "assistant", "system", "tool"
    content: str
    ts: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnContext:
    """
    Context window around a specific turn.

    Useful for:
    - Expanding search results with surrounding context
    - Understanding conversation flow around a specific point
    - Building prompts with relevant history
    """

    session_id: str
    project_slug: str
    target_turn: int

    # The turns (each list contains all messages in those turns)
    previous: list[TranscriptMessage]  # K turns before (oldest first)
    current: list[TranscriptMessage]  # All messages in target turn
    following: list[TranscriptMessage]  # K turns after (oldest first)

    # Navigation metadata
    has_more_before: bool  # Are there earlier turns?
    has_more_after: bool  # Are there later turns?
    first_turn: int  # First turn number in session
    last_turn: int  # Last turn number in session

    @property
    def total_messages(self) -> int:
        """Total messages in this context window."""
        return len(self.previous) + len(self.current) + len(self.following)

    @property
    def turns_range(self) -> tuple[int, int]:
        """Range of turns included (min, max)."""
        all_turns = [
            m.turn for m in self.previous + self.current + self.following if m.turn is not None
        ]
        return (min(all_turns), max(all_turns)) if all_turns else (0, 0)


@dataclass
class MessageContext:
    """
    Context window around a specific message by sequence.

    Used when:
    - Turn is null in the transcript
    - You need precise sequence-based navigation
    - You want to expand search results by sequence

    Unlike TurnContext which groups by turn number, this navigates
    by raw sequence numbers which are always present.
    """

    session_id: str
    project_slug: str
    target_sequence: int

    # Messages around the target
    previous: list[TranscriptMessage]  # Messages before (oldest first)
    current: TranscriptMessage | None  # The target message
    following: list[TranscriptMessage]  # Messages after (oldest first)

    # Navigation metadata
    has_more_before: bool
    has_more_after: bool
    first_sequence: int
    last_sequence: int

    @property
    def total_messages(self) -> int:
        """Total messages in this context window."""
        count = len(self.previous) + len(self.following)
        return count + 1 if self.current else count

    @property
    def sequence_range(self) -> tuple[int, int]:
        """Range of sequences included (min, max)."""
        all_seqs = [m.sequence for m in self.previous + self.following]
        if self.current:
            all_seqs.append(self.current.sequence)
        return (min(all_seqs), max(all_seqs)) if all_seqs else (0, 0)


@dataclass
class SessionSyncStats:
    """Lightweight sync statistics for consistency checking.

    Used by the sync daemon to detect partial data loss without
    loading all documents. Counts and timestamp ranges are sufficient
    for consistency verification.
    """

    event_count: int
    transcript_count: int
    event_ts_range: tuple[str | None, str | None]  # (earliest, latest)
    transcript_ts_range: tuple[str | None, str | None]  # (earliest, latest)


class StorageBackend(ABC):
    """
    Abstract base for all storage backends.

    Implementations must support:
    - Session metadata storage and retrieval
    - Transcript line storage and search
    - Event line storage and search
    - Vector embeddings for semantic search
    - Graceful degradation when embeddings unavailable

    Note on user_id parameter:
    - Empty string ("") means search across ALL users (team-wide)
    - Non-empty string filters to that specific user
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
        embeddings: dict[str, list[list[float] | None]] | None = None,
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
            options: Search configuration (includes mmr_lambda for diversity control)
            limit: Maximum results

        Returns:
            List of search results with relevance scores
        """
        pass

    @abstractmethod
    async def get_turn_context(
        self,
        user_id: str,
        session_id: str,
        turn: int,
        before: int = 2,
        after: int = 2,
        include_tool_outputs: bool = True,
    ) -> TurnContext:
        """
        Get context window around a specific turn.

        Useful for:
        - Expanding search results with surrounding context
        - Understanding conversation flow around a specific point
        - Building prompts with relevant history

        Args:
            user_id: User identifier
            session_id: Session to query
            turn: Target turn number (1-based)
            before: Number of turns to include before target (default: 2)
            after: Number of turns to include after target (default: 2)
            include_tool_outputs: Include tool call outputs in results (default: True)

        Returns:
            TurnContext with previous turns, target turn, and following turns

        Example:
            # Vector search returns turn 15 as relevant
            context = await storage.get_turn_context(
                user_id="user1",
                session_id="sess123",
                turn=15,
                before=3,  # Get turns 12, 13, 14
                after=1,   # Get turn 16
            )
            # Now have full context: turns 12-16
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
        """Search events with structured filters.

        Args:
            user_id: Filter by user.
            session_id: Filter by session.
            project_slug: Filter by project.
            event_type: Exact event type match (e.g. ``"tool:pre"``, ``"llm:response"``).
            event_category: Semantic group (e.g. ``"tool"``, ``"llm"``, ``"lifecycle"``).
            tool_name: Filter tool events by ``data.tool_name``.
            model: Filter LLM events by ``data.model``.
            provider: Filter LLM events by ``data.provider``.
            level: Filter by log level (``lvl`` field).
            start_date: Inclusive lower bound on ``ts`` (ISO-8601).
            end_date: Inclusive upper bound on ``ts`` (ISO-8601).
            limit: Maximum results to return.

        Returns:
            List of matching events with metadata, ordered by timestamp descending.
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

    @abstractmethod
    async def get_session_sync_stats(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
    ) -> SessionSyncStats:
        """Get lightweight sync statistics for a session.

        Returns counts and timestamp ranges for events and transcripts
        using aggregate queries. Much cheaper than loading all documents.
        """
        ...

    # =========================================================================
    # Discovery APIs
    # =========================================================================

    @abstractmethod
    async def list_users(
        self,
        filters: SearchFilters | None = None,
    ) -> list[str]:
        """
        List all unique user IDs in the storage.

        Args:
            filters: Optional filters (project_slug, date range, bundle)

        Returns:
            List of unique user IDs, sorted alphabetically
        """
        pass

    @abstractmethod
    async def list_projects(
        self,
        user_id: str = "",
        filters: SearchFilters | None = None,
    ) -> list[str]:
        """
        List all unique project slugs.

        Args:
            user_id: Filter by user (empty = all users)
            filters: Optional filters (date range, bundle)

        Returns:
            List of unique project slugs, sorted alphabetically
        """
        pass

    @abstractmethod
    async def list_sessions(
        self,
        user_id: str = "",
        project_slug: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        List sessions with pagination.

        Simpler than search_sessions - just lists recent sessions
        ordered by creation date (newest first).

        Args:
            user_id: Filter by user (empty = all users)
            project_slug: Filter by project
            limit: Max results (default 100)
            offset: Pagination offset (default 0)

        Returns:
            List of session metadata dicts with:
                - session_id
                - user_id
                - project_slug
                - bundle
                - created
                - turn_count
        """
        pass

    # =========================================================================
    # Sequence-Based Navigation
    # =========================================================================

    @abstractmethod
    async def get_message_context(
        self,
        session_id: str,
        sequence: int,
        user_id: str = "",
        before: int = 5,
        after: int = 5,
        include_tool_outputs: bool = True,
    ) -> MessageContext:
        """
        Get context window around a specific message by sequence.

        Use this when:
        - Turn is null in the transcript
        - You need precise sequence-based navigation
        - You want to expand search results by sequence

        Args:
            session_id: Session identifier
            sequence: Target sequence number
            user_id: User ID (empty = search all users for session)
            before: Messages to include before target (default 5)
            after: Messages to include after target (default 5)
            include_tool_outputs: Include tool outputs (default True)

        Returns:
            MessageContext with surrounding messages

        Example:
            # Search returns sequence 42 as relevant
            context = await storage.get_message_context(
                session_id="sess123",
                sequence=42,
                before=3,  # Get sequences 39, 40, 41
                after=2,   # Get sequences 43, 44
            )
            # Now have full context: sequences 39-44
        """
        pass
