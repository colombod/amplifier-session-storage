"""
Session query and analysis capabilities.

Provides a unified interface for querying session data from
both local storage and Cosmos DB, with support for:
- Searching sessions by topic, date, project
- Reading transcripts and events
- Analyzing session content
- Rewinding sessions to prior states
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..blocks.reader import SessionStateReader
from ..blocks.types import BlockType, SessionBlock
from ..storage.base import BlockStorage


class QueryScope(Enum):
    """Scope for session queries."""

    USER = "user"  # Only user's own sessions
    TEAM = "team"  # User's + team-shared sessions
    ORG = "org"  # User's + org-visible sessions
    ALL = "all"  # All visible sessions (includes public)


@dataclass
class SessionQuery:
    """Query parameters for searching sessions.

    Attributes:
        project_slug: Filter by project
        date_from: Sessions created after this date
        date_to: Sessions created before this date
        search_text: Full-text search in messages/events
        session_ids: Specific session IDs to query
        scope: Visibility scope for the query
        limit: Maximum results
        offset: Pagination offset
        include_transcript: Include message transcript in results
        include_events: Include events in results
    """

    project_slug: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    search_text: str | None = None
    session_ids: list[str] | None = None
    scope: QueryScope = QueryScope.USER
    limit: int = 50
    offset: int = 0
    include_transcript: bool = False
    include_events: bool = False


@dataclass
class TranscriptMessage:
    """A message in the session transcript."""

    role: str
    content: str
    turn: int
    timestamp: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EventSummary:
    """Summary of a session event."""

    event_type: str
    timestamp: str
    tool_name: str | None = None
    provider: str | None = None
    model: str | None = None
    tokens: int | None = None
    duration_ms: int | None = None


@dataclass
class SessionSummary:
    """Summary of a session with optional details."""

    session_id: str
    user_id: str
    created: str
    updated: str | None = None
    name: str | None = None
    project_slug: str | None = None
    visibility: str = "private"
    org_id: str | None = None
    team_ids: list[str] = field(default_factory=list)

    # Counts
    message_count: int = 0
    event_count: int = 0
    turn_count: int = 0

    # Optional details (populated if requested)
    transcript: list[TranscriptMessage] = field(default_factory=list)
    events: list[EventSummary] = field(default_factory=list)

    # Metadata
    first_user_message: str | None = None
    last_user_message: str | None = None
    topics: list[str] = field(default_factory=list)


class SessionAnalyst:
    """Query and analysis interface for session storage.

    Provides capabilities for:
    - Searching sessions by various criteria
    - Reading full session transcripts
    - Analyzing session content and patterns
    - Rewinding sessions to earlier states
    - Generating session summaries

    Works with any BlockStorage implementation (local or Cosmos).

    Example:
        >>> analyst = SessionAnalyst(storage, user_id="user-123")
        >>> sessions = await analyst.search(SessionQuery(project_slug="my-project"))
        >>> for session in sessions:
        ...     print(f"{session.session_id}: {session.message_count} messages")
    """

    def __init__(
        self,
        storage: BlockStorage,
        user_id: str,
        org_id: str | None = None,
        team_ids: list[str] | None = None,
    ) -> None:
        """Initialize the analyst.

        Args:
            storage: Block storage backend
            user_id: Current user ID for access control
            org_id: Optional organization ID for org-scoped queries
            team_ids: Optional team IDs for team-scoped queries
        """
        self.storage = storage
        self.user_id = user_id
        self.org_id = org_id
        self.team_ids = team_ids or []

    async def search(self, query: SessionQuery) -> list[SessionSummary]:
        """Search for sessions matching the query.

        Args:
            query: Search parameters

        Returns:
            List of session summaries matching the query
        """
        # Get base session list from storage
        sessions = await self.storage.list_sessions(
            project_slug=query.project_slug,
            limit=query.limit * 2,  # Get extra for filtering
            offset=query.offset,
        )

        results: list[SessionSummary] = []

        for session_meta in sessions:
            session_id = session_meta["session_id"]

            # Date filtering
            if query.date_from or query.date_to:
                created = datetime.fromisoformat(session_meta["created"].replace("Z", "+00:00"))
                if query.date_from and created < query.date_from:
                    continue
                if query.date_to and created > query.date_to:
                    continue

            # Session ID filtering
            if query.session_ids and session_id not in query.session_ids:
                continue

            # Build summary
            summary = await self._build_session_summary(
                session_id,
                session_meta,
                include_transcript=query.include_transcript,
                include_events=query.include_events,
            )

            # Text search filtering
            if query.search_text:
                if not self._matches_search(summary, query.search_text):
                    continue

            results.append(summary)

            if len(results) >= query.limit:
                break

        return results

    async def get_session(
        self,
        session_id: str,
        include_transcript: bool = True,
        include_events: bool = True,
    ) -> SessionSummary | None:
        """Get detailed information about a specific session.

        Args:
            session_id: The session ID
            include_transcript: Include full message transcript
            include_events: Include event details

        Returns:
            Session summary with requested details, or None if not found
        """
        blocks = await self.storage.read_blocks(session_id)
        if not blocks:
            return None

        # Find the session_created block for metadata
        session_meta: dict[str, Any] = {}
        for block in blocks:
            if block.block_type == BlockType.SESSION_CREATED:
                session_meta = {
                    "session_id": session_id,
                    "user_id": block.user_id,
                    "created": block.timestamp,
                    "name": block.data.get("name"),
                    "project_slug": block.data.get("project_slug"),
                    "visibility": block.data.get("visibility", "private"),
                    "org_id": block.data.get("org_id"),
                }
                break

        if not session_meta:
            # No session_created block, use first block's info
            first_block = blocks[0]
            session_meta = {
                "session_id": session_id,
                "user_id": first_block.user_id,
                "created": first_block.timestamp,
            }

        return await self._build_session_summary(
            session_id,
            session_meta,
            include_transcript=include_transcript,
            include_events=include_events,
            blocks=blocks,
        )

    async def get_transcript(self, session_id: str) -> list[TranscriptMessage]:
        """Get the message transcript for a session.

        Args:
            session_id: The session ID

        Returns:
            List of transcript messages in order
        """
        blocks = await self.storage.read_blocks(session_id)
        return self._extract_transcript(blocks)

    async def get_events(
        self,
        session_id: str,
        event_types: list[str] | None = None,
    ) -> list[EventSummary]:
        """Get events for a session.

        Args:
            session_id: The session ID
            event_types: Optional filter for specific event types

        Returns:
            List of event summaries
        """
        blocks = await self.storage.read_blocks(session_id)
        return self._extract_events(blocks, event_types)

    async def get_state_at_sequence(
        self,
        session_id: str,
        sequence: int,
    ) -> dict[str, Any]:
        """Get the session state at a specific sequence number.

        Useful for analyzing session history or preparing a rewind.

        Args:
            session_id: The session ID
            sequence: The sequence number to read up to

        Returns:
            Dictionary with session state (messages, metadata, etc.)
        """
        # Read blocks up to the sequence
        blocks = await self.storage.read_blocks(session_id, limit=sequence)
        blocks = [b for b in blocks if b.sequence <= sequence]

        # Use SessionStateReader to compute state
        reader = SessionStateReader()
        metadata, messages, _ = reader.compute_current_state(blocks)

        # Check if session was rewound
        is_rewound = False
        rewind_sequence = None
        for block in reversed(blocks):
            if block.block_type == BlockType.REWIND:
                is_rewound = True
                rewind_sequence = block.data.get("target_sequence")
                break

        return {
            "session_id": session_id,
            "sequence": sequence,
            "metadata": {
                "name": metadata.name,
                "project_slug": metadata.project_slug,
                "visibility": metadata.visibility,
            },
            "messages": [{"role": m.role, "content": m.content, "turn": m.turn} for m in messages],
            "is_rewound": is_rewound,
            "rewind_sequence": rewind_sequence,
        }

    async def find_sessions_by_topic(
        self,
        topic: str,
        limit: int = 20,
    ) -> list[SessionSummary]:
        """Find sessions related to a topic.

        Searches session names, first messages, and content.

        Args:
            topic: Topic to search for
            limit: Maximum results

        Returns:
            Sessions related to the topic
        """
        query = SessionQuery(
            search_text=topic,
            limit=limit,
            include_transcript=False,
        )
        return await self.search(query)

    async def get_recent_sessions(
        self,
        limit: int = 10,
        project_slug: str | None = None,
    ) -> list[SessionSummary]:
        """Get most recent sessions.

        Args:
            limit: Maximum results
            project_slug: Optional project filter

        Returns:
            Recent sessions, most recent first
        """
        query = SessionQuery(
            project_slug=project_slug,
            limit=limit,
        )
        return await self.search(query)

    async def get_session_statistics(self, session_id: str) -> dict[str, Any]:
        """Get statistics for a session.

        Args:
            session_id: The session ID

        Returns:
            Dictionary with session statistics
        """
        blocks = await self.storage.read_blocks(session_id)
        if not blocks:
            return {}

        # Compute statistics
        message_count = sum(1 for b in blocks if b.block_type == BlockType.MESSAGE)
        event_count = sum(1 for b in blocks if b.block_type == BlockType.EVENT)

        # Extract timing information
        timestamps = [b.timestamp for b in blocks]
        first_ts = min(timestamps)
        last_ts = max(timestamps)

        # Count by block type
        type_counts: dict[str, int] = {}
        for block in blocks:
            type_name = block.block_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        # Count unique devices
        devices = {b.device_id for b in blocks}

        # Calculate total size
        total_size = sum(b.size_bytes for b in blocks)

        return {
            "session_id": session_id,
            "block_count": len(blocks),
            "message_count": message_count,
            "event_count": event_count,
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
            "type_counts": type_counts,
            "unique_devices": len(devices),
            "device_ids": list(devices),
            "total_size_bytes": total_size,
            "latest_sequence": max(b.sequence for b in blocks),
        }

    # Private methods

    async def _build_session_summary(
        self,
        session_id: str,
        session_meta: dict[str, Any],
        include_transcript: bool = False,
        include_events: bool = False,
        blocks: list[SessionBlock] | None = None,
    ) -> SessionSummary:
        """Build a session summary from metadata and optionally blocks."""
        if blocks is None and (include_transcript or include_events):
            blocks = await self.storage.read_blocks(session_id)

        blocks = blocks or []

        # Count messages and events
        message_count = sum(1 for b in blocks if b.block_type == BlockType.MESSAGE)
        event_count = sum(1 for b in blocks if b.block_type == BlockType.EVENT)

        # Find max turn
        turn_count = 0
        for block in blocks:
            if block.block_type == BlockType.MESSAGE:
                turn = block.data.get("turn", 0)
                turn_count = max(turn_count, turn)

        # Extract first and last user messages
        first_user_message = None
        last_user_message = None
        for block in blocks:
            if block.block_type == BlockType.MESSAGE:
                if block.data.get("role") == "user":
                    content = block.data.get("content", "")
                    if first_user_message is None:
                        first_user_message = content[:200]  # Truncate
                    last_user_message = content[:200]

        # Get updated timestamp
        updated: str | None = None
        if blocks:
            max_ts = max(b.timestamp for b in blocks)
            updated = max_ts.isoformat() if isinstance(max_ts, datetime) else str(max_ts)

        summary = SessionSummary(
            session_id=session_id,
            user_id=session_meta.get("user_id", ""),
            created=session_meta.get("created", ""),
            updated=updated,
            name=session_meta.get("name"),
            project_slug=session_meta.get("project_slug"),
            visibility=session_meta.get("visibility", "private"),
            org_id=session_meta.get("org_id"),
            team_ids=session_meta.get("team_ids", []),
            message_count=message_count,
            event_count=event_count,
            turn_count=turn_count,
            first_user_message=first_user_message,
            last_user_message=last_user_message,
        )

        if include_transcript:
            summary.transcript = self._extract_transcript(blocks)

        if include_events:
            summary.events = self._extract_events(blocks)

        return summary

    def _extract_transcript(self, blocks: list[SessionBlock]) -> list[TranscriptMessage]:
        """Extract transcript messages from blocks."""
        messages: list[TranscriptMessage] = []

        for block in blocks:
            if block.block_type == BlockType.MESSAGE:
                ts = block.timestamp
                ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)
                messages.append(
                    TranscriptMessage(
                        role=block.data.get("role", "unknown"),
                        content=block.data.get("content", ""),
                        turn=block.data.get("turn", 0),
                        timestamp=ts_str,
                        tool_calls=block.data.get("tool_calls", []),
                    )
                )

        return messages

    def _extract_events(
        self,
        blocks: list[SessionBlock],
        event_types: list[str] | None = None,
    ) -> list[EventSummary]:
        """Extract event summaries from blocks."""
        events: list[EventSummary] = []

        for block in blocks:
            if block.block_type == BlockType.EVENT:
                event_type = block.data.get("event_type", "unknown")

                if event_types and event_type not in event_types:
                    continue

                ts = block.timestamp
                ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)
                events.append(
                    EventSummary(
                        event_type=event_type,
                        timestamp=ts_str,
                        tool_name=block.data.get("tool_name"),
                        provider=block.data.get("provider"),
                        model=block.data.get("model"),
                        tokens=block.data.get("tokens"),
                        duration_ms=block.data.get("duration_ms"),
                    )
                )

        return events

    def _matches_search(self, summary: SessionSummary, search_text: str) -> bool:
        """Check if a session matches search text."""
        search_lower = search_text.lower()

        # Check name
        if summary.name and search_lower in summary.name.lower():
            return True

        # Check first/last user messages
        if summary.first_user_message and search_lower in summary.first_user_message.lower():
            return True
        if summary.last_user_message and search_lower in summary.last_user_message.lower():
            return True

        # Check project
        if summary.project_slug and search_lower in summary.project_slug.lower():
            return True

        # Check transcript if available
        for msg in summary.transcript:
            if search_lower in msg.content.lower():
                return True

        return False
