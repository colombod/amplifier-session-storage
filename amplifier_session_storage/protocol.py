"""
Core types and abstract base class for session storage.

This module defines the SessionStorage protocol and all associated data types
that storage implementations must work with.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, TypedDict

# =============================================================================
# Session Visibility
# =============================================================================


class SessionVisibility(Enum):
    """Session visibility levels."""

    PRIVATE = "private"  # Only owner can access (default)
    TEAM = "team"  # Members of owner's teams can view
    ORGANIZATION = "organization"  # All org members can view
    PUBLIC = "public"  # Any authenticated user can view


# =============================================================================
# Event Projection Types
# =============================================================================


class EventProjection(TypedDict, total=False):
    """Fields that can be safely projected from events.

    CRITICAL: NEVER includes 'data' or 'content' - those can be 100k+ tokens.
    This type defines the ONLY fields that query_events() may return.
    """

    event_id: str
    event_type: str
    ts: str
    session_id: str
    turn: int
    model: str
    usage: dict[str, int]
    duration_ms: int
    has_tool_calls: bool
    has_error: bool
    error_type: str
    tool_name: str


# Safe projection fields - used to enforce projection in queries
SAFE_PROJECTION_FIELDS: frozenset[str] = frozenset(EventProjection.__annotations__.keys())


# =============================================================================
# Core Data Types
# =============================================================================


@dataclass
class SessionMetadata:
    """Metadata for a session.

    Contains all non-content information about a session including
    ownership, timestamps, and aggregate statistics.
    """

    session_id: str
    user_id: str
    project_slug: str
    created: datetime
    updated: datetime
    name: str | None = None
    description: str | None = None
    bundle: str | None = None
    model: str | None = None
    turn_count: int = 0
    message_count: int = 0
    event_count: int = 0
    parent_id: str | None = None
    forked_from_turn: int | None = None
    # Sharing fields
    visibility: SessionVisibility = SessionVisibility.PRIVATE
    org_id: str | None = None
    team_ids: list[str] | None = None
    shared_at: datetime | None = None
    tags: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "project_slug": self.project_slug,
            "created": self.created.isoformat(),
            "updated": self.updated.isoformat(),
            "name": self.name,
            "description": self.description,
            "bundle": self.bundle,
            "model": self.model,
            "turn_count": self.turn_count,
            "message_count": self.message_count,
            "event_count": self.event_count,
            "parent_id": self.parent_id,
            "forked_from_turn": self.forked_from_turn,
            "visibility": self.visibility.value,
            "org_id": self.org_id,
            "team_ids": self.team_ids,
            "shared_at": self.shared_at.isoformat() if self.shared_at else None,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMetadata":
        """Create from dictionary."""
        # Parse visibility
        visibility_str = data.get("visibility", "private")
        visibility = (
            SessionVisibility(visibility_str) if isinstance(visibility_str, str) else visibility_str
        )

        # Parse shared_at
        shared_at_raw = data.get("shared_at")
        shared_at = None
        if shared_at_raw:
            shared_at = (
                datetime.fromisoformat(shared_at_raw)
                if isinstance(shared_at_raw, str)
                else shared_at_raw
            )

        return cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            project_slug=data["project_slug"],
            created=(
                datetime.fromisoformat(data["created"])
                if isinstance(data["created"], str)
                else data["created"]
            ),
            updated=(
                datetime.fromisoformat(data["updated"])
                if isinstance(data["updated"], str)
                else data["updated"]
            ),
            name=data.get("name"),
            description=data.get("description"),
            bundle=data.get("bundle"),
            model=data.get("model"),
            turn_count=data.get("turn_count", 0),
            message_count=data.get("message_count", 0),
            event_count=data.get("event_count", 0),
            parent_id=data.get("parent_id"),
            forked_from_turn=data.get("forked_from_turn"),
            visibility=visibility,
            org_id=data.get("org_id"),
            team_ids=data.get("team_ids"),
            shared_at=shared_at,
            tags=data.get("tags"),
        )


@dataclass
class TranscriptMessage:
    """A single message in the conversation transcript."""

    sequence: int
    role: Literal["user", "assistant", "tool", "system"]
    content: Any
    timestamp: datetime
    turn: int
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sequence": self.sequence,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "turn": self.turn,
            "tool_calls": self.tool_calls,
            "tool_call_id": self.tool_call_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptMessage":
        """Create from dictionary."""
        return cls(
            sequence=data["sequence"],
            role=data["role"],
            content=data["content"],
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if isinstance(data["timestamp"], str)
                else data["timestamp"]
            ),
            turn=data["turn"],
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass
class EventSummary:
    """Summary of an event without full data payload.

    This is returned by query_events() to provide event metadata
    without loading potentially massive data payloads.
    """

    event_id: str
    event_type: str
    ts: datetime
    turn: int | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    data_size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "ts": self.ts.isoformat(),
            "turn": self.turn,
            "summary": self.summary,
            "data_size_bytes": self.data_size_bytes,
        }


@dataclass
class AggregateStats:
    """Aggregate statistics for events in a session."""

    event_count: int
    event_types: dict[str, int]
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_duration_ms: int = 0
    error_count: int = 0
    tool_call_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_count": self.event_count,
            "event_types": self.event_types,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_duration_ms": self.total_duration_ms,
            "error_count": self.error_count,
            "tool_call_count": self.tool_call_count,
        }


@dataclass
class RewindResult:
    """Result of a rewind operation."""

    success: bool
    messages_removed: int
    events_removed: int
    new_turn_count: int
    backup_path: str | None = None


# =============================================================================
# Organization and Team Types
# =============================================================================


@dataclass
class Organization:
    """Organization for grouping users and sessions."""

    org_id: str
    name: str
    created: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "org_id": self.org_id,
            "name": self.name,
            "created": self.created.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Organization":
        """Create from dictionary."""
        return cls(
            org_id=data["org_id"],
            name=data["name"],
            created=(
                datetime.fromisoformat(data["created"])
                if isinstance(data["created"], str)
                else data["created"]
            ),
        )


@dataclass
class Team:
    """Team within an organization."""

    team_id: str
    org_id: str
    name: str
    description: str | None
    created: datetime
    parent_team_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "team_id": self.team_id,
            "org_id": self.org_id,
            "name": self.name,
            "description": self.description,
            "created": self.created.isoformat(),
            "parent_team_id": self.parent_team_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Team":
        """Create from dictionary."""
        return cls(
            team_id=data["team_id"],
            org_id=data["org_id"],
            name=data["name"],
            description=data.get("description"),
            created=(
                datetime.fromisoformat(data["created"])
                if isinstance(data["created"], str)
                else data["created"]
            ),
            parent_team_id=data.get("parent_team_id"),
        )


@dataclass
class UserMembership:
    """User's organization and team memberships."""

    user_id: str
    org_id: str
    team_ids: list[str]
    role: str = "member"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "org_id": self.org_id,
            "team_ids": self.team_ids,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserMembership":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            org_id=data["org_id"],
            team_ids=data.get("team_ids", []),
            role=data.get("role", "member"),
        )


@dataclass
class SharedSessionSummary:
    """Summary of a shared session for discovery queries."""

    session_id: str
    owner_user_id: str
    owner_name: str
    name: str | None
    description: str | None
    visibility: SessionVisibility
    team_ids: list[str] | None
    project_slug: str
    bundle: str | None
    model: str | None
    tags: list[str] | None
    turn_count: int
    message_count: int
    created: datetime
    updated: datetime
    shared_at: datetime | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "owner_user_id": self.owner_user_id,
            "owner_name": self.owner_name,
            "name": self.name,
            "description": self.description,
            "visibility": self.visibility.value,
            "team_ids": self.team_ids,
            "project_slug": self.project_slug,
            "bundle": self.bundle,
            "model": self.model,
            "tags": self.tags,
            "turn_count": self.turn_count,
            "message_count": self.message_count,
            "created": self.created.isoformat(),
            "updated": self.updated.isoformat(),
            "shared_at": self.shared_at.isoformat() if self.shared_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SharedSessionSummary":
        """Create from dictionary."""
        visibility_str = data.get("visibility", "private")
        visibility = (
            SessionVisibility(visibility_str) if isinstance(visibility_str, str) else visibility_str
        )

        shared_at_raw = data.get("shared_at")
        shared_at = None
        if shared_at_raw:
            shared_at = (
                datetime.fromisoformat(shared_at_raw)
                if isinstance(shared_at_raw, str)
                else shared_at_raw
            )

        return cls(
            session_id=data["session_id"],
            owner_user_id=data["owner_user_id"],
            owner_name=data.get("owner_name", "Unknown"),
            name=data.get("name"),
            description=data.get("description"),
            visibility=visibility,
            team_ids=data.get("team_ids"),
            project_slug=data["project_slug"],
            bundle=data.get("bundle"),
            model=data.get("model"),
            tags=data.get("tags"),
            turn_count=data.get("turn_count", 0),
            message_count=data.get("message_count", 0),
            created=(
                datetime.fromisoformat(data["created"])
                if isinstance(data["created"], str)
                else data["created"]
            ),
            updated=(
                datetime.fromisoformat(data["updated"])
                if isinstance(data["updated"], str)
                else data["updated"]
            ),
            shared_at=shared_at,
        )


@dataclass
class SharedSessionQuery:
    """Query parameters for finding shared sessions."""

    requester_user_id: str
    scope: Literal["team", "organization", "public"] = "organization"
    team_ids: list[str] | None = None
    project_slug: str | None = None
    tags: list[str] | None = None
    owner_user_id: str | None = None
    name_contains: str | None = None
    created_after: datetime | None = None
    updated_after: datetime | None = None
    limit: int = 50
    offset: int = 0
    order_by: Literal["updated", "created", "shared_at"] = "updated"
    order_desc: bool = True


# =============================================================================
# Sync Types
# =============================================================================


class ConflictResolution(Enum):
    """Strategy for resolving sync conflicts."""

    LOCAL_WINS = "local_wins"
    REMOTE_WINS = "remote_wins"
    MERGE = "merge"


@dataclass
class SyncStatus:
    """Current synchronization status."""

    is_synced: bool
    pending_changes: int
    last_sync: datetime | None
    conflict_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_synced": self.is_synced,
            "pending_changes": self.pending_changes,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "conflict_count": self.conflict_count,
        }


# =============================================================================
# Query Types
# =============================================================================


@dataclass
class SessionQuery:
    """Query parameters for listing sessions."""

    user_id: str
    project_slug: str | None = None
    name_contains: str | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    limit: int = 100
    offset: int = 0
    order_by: Literal["created", "updated", "name"] = "updated"
    order_desc: bool = True


@dataclass
class EventQuery:
    """Query parameters for listing events.

    Note: This query will only return EventSummary objects,
    never full event data.
    """

    session_id: str
    user_id: str
    event_types: list[str] | None = None
    turn: int | None = None
    turn_gte: int | None = None
    turn_lte: int | None = None
    after: datetime | None = None
    before: datetime | None = None
    limit: int = 100
    offset: int = 0


# =============================================================================
# Session Storage ABC
# =============================================================================


class SessionStorage(ABC):
    """Abstract base class for session storage implementations.

    Implementations must handle:
    - Session CRUD operations
    - Transcript message storage
    - Event storage with projection enforcement
    - Rewind operations with backup
    - Search functionality
    - Sync status (may be no-op for local-only)

    CRITICAL SAFETY REQUIREMENTS:
    1. Event projection: query_events() MUST NEVER return full event data
    2. User isolation: All queries MUST filter by user_id
    3. Atomic rewind: Transcript and events must be truncated together
    4. Backup before destructive: Create backups before rewind operations
    """

    # =========================================================================
    # Session CRUD
    # =========================================================================

    @abstractmethod
    async def create_session(self, metadata: SessionMetadata) -> SessionMetadata:
        """Create a new session.

        Args:
            metadata: Session metadata to create

        Returns:
            Created session metadata

        Raises:
            SessionExistsError: If session_id already exists
        """
        ...

    @abstractmethod
    async def get_session(self, user_id: str, session_id: str) -> SessionMetadata | None:
        """Get session metadata by ID.

        Args:
            user_id: User ID for isolation
            session_id: Session to retrieve

        Returns:
            Session metadata or None if not found
        """
        ...

    @abstractmethod
    async def update_session(self, metadata: SessionMetadata) -> SessionMetadata:
        """Update session metadata.

        Args:
            metadata: Updated metadata (must include session_id)

        Returns:
            Updated session metadata

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        ...

    @abstractmethod
    async def delete_session(self, user_id: str, session_id: str) -> bool:
        """Delete a session and all its data.

        Args:
            user_id: User ID for isolation
            session_id: Session to delete

        Returns:
            True if deleted, False if not found
        """
        ...

    @abstractmethod
    async def list_sessions(self, query: SessionQuery) -> list[SessionMetadata]:
        """List sessions matching query.

        Args:
            query: Query parameters

        Returns:
            List of matching sessions
        """
        ...

    # =========================================================================
    # Transcript Operations
    # =========================================================================

    @abstractmethod
    async def append_message(
        self,
        user_id: str,
        session_id: str,
        message: TranscriptMessage,
    ) -> TranscriptMessage:
        """Append a message to the transcript.

        Args:
            user_id: User ID for isolation
            session_id: Session to append to
            message: Message to append

        Returns:
            Appended message with assigned sequence number

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        ...

    @abstractmethod
    async def get_transcript(
        self,
        user_id: str,
        session_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TranscriptMessage]:
        """Get transcript messages.

        Args:
            user_id: User ID for isolation
            session_id: Session to read from
            limit: Maximum messages to return
            offset: Number of messages to skip

        Returns:
            List of transcript messages
        """
        ...

    @abstractmethod
    async def get_transcript_for_turn(
        self,
        user_id: str,
        session_id: str,
        turn: int,
    ) -> list[TranscriptMessage]:
        """Get all messages for a specific turn.

        Args:
            user_id: User ID for isolation
            session_id: Session to read from
            turn: Turn number to retrieve

        Returns:
            List of messages in that turn
        """
        ...

    # =========================================================================
    # Event Operations
    # =========================================================================

    @abstractmethod
    async def append_event(
        self,
        user_id: str,
        session_id: str,
        event_id: str,
        event_type: str,
        data: dict[str, Any],
        turn: int | None = None,
    ) -> EventSummary:
        """Append an event to the session.

        Args:
            user_id: User ID for isolation
            session_id: Session to append to
            event_id: Unique event identifier
            event_type: Type of event
            data: Full event data (may be chunked for large events)
            turn: Optional turn number

        Returns:
            Event summary (NOT full data)

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        ...

    @abstractmethod
    async def query_events(
        self,
        query: EventQuery,
    ) -> list[EventSummary]:
        """Query events, returning summaries only.

        CRITICAL: This method MUST NEVER return full event data.
        Only EventSummary objects with safe projection fields.

        Args:
            query: Event query parameters

        Returns:
            List of event summaries (NOT full data)
        """
        ...

    @abstractmethod
    async def get_event_data(
        self,
        user_id: str,
        session_id: str,
        event_id: str,
    ) -> dict[str, Any] | None:
        """Get full data for a specific event.

        This is the ONLY method that returns full event data.
        Use sparingly as events can be 100k+ tokens.

        Args:
            user_id: User ID for isolation
            session_id: Session containing event
            event_id: Event to retrieve

        Returns:
            Full event data or None if not found
        """
        ...

    @abstractmethod
    async def get_event_aggregates(
        self,
        user_id: str,
        session_id: str,
    ) -> AggregateStats:
        """Get aggregate statistics for all events in a session.

        Args:
            user_id: User ID for isolation
            session_id: Session to aggregate

        Returns:
            Aggregate statistics
        """
        ...

    # =========================================================================
    # Rewind Operations
    # =========================================================================

    @abstractmethod
    async def rewind_to_turn(
        self,
        user_id: str,
        session_id: str,
        turn: int,
        create_backup: bool = True,
    ) -> RewindResult:
        """Rewind session to a specific turn.

        Removes all messages and events after the specified turn.
        MUST be atomic - both transcript and events truncated together.

        Args:
            user_id: User ID for isolation
            session_id: Session to rewind
            turn: Turn to rewind to (inclusive)
            create_backup: Whether to backup before rewind

        Returns:
            Result of rewind operation

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        ...

    @abstractmethod
    async def rewind_to_timestamp(
        self,
        user_id: str,
        session_id: str,
        timestamp: datetime,
        create_backup: bool = True,
    ) -> RewindResult:
        """Rewind session to a specific timestamp.

        Removes all messages and events after the timestamp.
        MUST be atomic - both transcript and events truncated together.

        Args:
            user_id: User ID for isolation
            session_id: Session to rewind
            timestamp: Timestamp to rewind to (inclusive)
            create_backup: Whether to backup before rewind

        Returns:
            Result of rewind operation

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        ...

    # =========================================================================
    # Search Operations
    # =========================================================================

    @abstractmethod
    async def search_sessions(
        self,
        user_id: str,
        query_text: str,
        project_slug: str | None = None,
        limit: int = 20,
    ) -> list[SessionMetadata]:
        """Search sessions by text content.

        Searches session names and descriptions.

        Args:
            user_id: User ID for isolation
            query_text: Text to search for
            project_slug: Optional project filter
            limit: Maximum results

        Returns:
            List of matching sessions
        """
        ...

    # =========================================================================
    # Sync Operations
    # =========================================================================

    @abstractmethod
    async def get_sync_status(
        self,
        user_id: str,
        session_id: str,
    ) -> SyncStatus:
        """Get synchronization status for a session.

        For local-only storage, this returns "always synced".

        Args:
            user_id: User ID for isolation
            session_id: Session to check

        Returns:
            Current sync status
        """
        ...

    @abstractmethod
    async def sync_now(
        self,
        user_id: str,
        session_id: str | None = None,
    ) -> SyncStatus:
        """Trigger immediate sync.

        For local-only storage, this is a no-op.

        Args:
            user_id: User ID for isolation
            session_id: Specific session to sync, or None for all

        Returns:
            Updated sync status
        """
        ...

    # =========================================================================
    # Session Sharing Operations
    # =========================================================================

    @abstractmethod
    async def set_session_visibility(
        self,
        user_id: str,
        session_id: str,
        visibility: SessionVisibility,
        team_ids: list[str] | None = None,
    ) -> SessionMetadata:
        """Change session visibility. Only owner can change.

        Args:
            user_id: User ID (must be session owner)
            session_id: Session to modify
            visibility: New visibility level
            team_ids: Team IDs for TEAM visibility

        Returns:
            Updated session metadata

        Raises:
            SessionNotFoundError: If session doesn't exist
            PermissionDeniedError: If user is not the owner
        """
        ...

    @abstractmethod
    async def query_shared_sessions(
        self,
        query: SharedSessionQuery,
    ) -> list[SharedSessionSummary]:
        """Query shared sessions visible to the requester.

        Args:
            query: Query parameters including requester user ID

        Returns:
            List of shared session summaries
        """
        ...

    @abstractmethod
    async def get_shared_session(
        self,
        requester_user_id: str,
        session_id: str,
    ) -> SessionMetadata | None:
        """Get a shared session with access control.

        Args:
            requester_user_id: User requesting access
            session_id: Session to retrieve

        Returns:
            Session metadata if accessible, None otherwise
        """
        ...

    @abstractmethod
    async def get_shared_transcript(
        self,
        requester_user_id: str,
        session_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TranscriptMessage]:
        """Get transcript from a shared session (read-only).

        Args:
            requester_user_id: User requesting access
            session_id: Session to read from
            limit: Maximum messages to return
            offset: Number of messages to skip

        Returns:
            List of transcript messages

        Raises:
            PermissionDeniedError: If user doesn't have read access
        """
        ...

    @abstractmethod
    async def get_user_membership(
        self,
        user_id: str,
    ) -> UserMembership | None:
        """Get user's organization and team memberships.

        Args:
            user_id: User to look up

        Returns:
            User membership info or None if not found
        """
        ...
