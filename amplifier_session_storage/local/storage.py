"""
Local file-based session storage implementation.

Stores sessions in the local filesystem with backward compatibility:
- Path: ~/.amplifier/projects/{project_slug}/sessions/{session_id}/
- Three files: metadata.json, transcript.jsonl, events.jsonl
- Atomic writes using temp file + rename
- Backup files before destructive operations
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ..exceptions import SessionExistsError, SessionNotFoundError
from ..protocol import (
    AggregateStats,
    EventQuery,
    EventSummary,
    RewindResult,
    SessionMetadata,
    SessionQuery,
    SessionStorage,
    SessionVisibility,
    SharedSessionQuery,
    SharedSessionSummary,
    SyncStatus,
    TranscriptMessage,
    UserMembership,
)
from .file_ops import (
    append_jsonl,
    ensure_directory,
    file_exists,
    list_directories,
    read_json,
    read_jsonl,
    remove_directory,
    write_json_atomic,
    write_jsonl_atomic,
)
from .file_ops import (
    create_backup as create_file_backup,
)

# Default base path for Amplifier sessions
DEFAULT_BASE_PATH = Path.home() / ".amplifier" / "projects"


class LocalFileStorage(SessionStorage):
    """File-based session storage implementation.

    Maintains backward compatibility with existing Amplifier session format.
    All operations are performed locally with atomic writes.

    Directory structure:
        {base_path}/{project_slug}/sessions/{session_id}/
            metadata.json    - Session metadata
            transcript.jsonl - Conversation transcript
            events.jsonl     - Event log

    For sync methods, returns "always synced" status (no-op for local-only).
    """

    def __init__(self, base_path: Path | None = None):
        """Initialize local file storage.

        Args:
            base_path: Base directory for session storage.
                      Defaults to ~/.amplifier/projects
        """
        self.base_path = base_path or DEFAULT_BASE_PATH

    def _session_dir(self, project_slug: str, session_id: str) -> Path:
        """Get the directory path for a session."""
        return self.base_path / project_slug / "sessions" / session_id

    def _metadata_path(self, project_slug: str, session_id: str) -> Path:
        """Get path to metadata.json."""
        return self._session_dir(project_slug, session_id) / "metadata.json"

    def _transcript_path(self, project_slug: str, session_id: str) -> Path:
        """Get path to transcript.jsonl."""
        return self._session_dir(project_slug, session_id) / "transcript.jsonl"

    def _events_path(self, project_slug: str, session_id: str) -> Path:
        """Get path to events.jsonl."""
        return self._session_dir(project_slug, session_id) / "events.jsonl"

    def _backup_dir(self, project_slug: str, session_id: str) -> Path:
        """Get backup directory for a session."""
        return self._session_dir(project_slug, session_id) / ".backups"

    # =========================================================================
    # Session CRUD
    # =========================================================================

    async def create_session(self, metadata: SessionMetadata) -> SessionMetadata:
        """Create a new session."""
        session_dir = self._session_dir(metadata.project_slug, metadata.session_id)

        if await file_exists(session_dir / "metadata.json"):
            raise SessionExistsError(metadata.session_id)

        await ensure_directory(session_dir)
        await write_json_atomic(
            self._metadata_path(metadata.project_slug, metadata.session_id),
            metadata.to_dict(),
        )

        return metadata

    async def get_session(self, user_id: str, session_id: str) -> SessionMetadata | None:
        """Get session metadata by ID.

        Note: For local storage, we need to search across projects since
        we don't know the project_slug from just the session_id.
        """
        # Search all projects for this session
        projects = await list_directories(self.base_path)
        for project_slug in projects:
            metadata_path = self._metadata_path(project_slug, session_id)
            data = await read_json(metadata_path)
            if data is not None:
                metadata = SessionMetadata.from_dict(data)
                # Verify user_id matches for isolation
                if metadata.user_id == user_id:
                    return metadata
        return None

    async def update_session(self, metadata: SessionMetadata) -> SessionMetadata:
        """Update session metadata."""
        metadata_path = self._metadata_path(metadata.project_slug, metadata.session_id)

        if not await file_exists(metadata_path):
            raise SessionNotFoundError(metadata.session_id, metadata.user_id)

        # Update the updated timestamp
        metadata.updated = datetime.utcnow()

        await write_json_atomic(metadata_path, metadata.to_dict())
        return metadata

    async def delete_session(self, user_id: str, session_id: str) -> bool:
        """Delete a session and all its data."""
        # Find the session first to get project_slug
        metadata = await self.get_session(user_id, session_id)
        if metadata is None:
            return False

        session_dir = self._session_dir(metadata.project_slug, session_id)
        return await remove_directory(session_dir)

    async def list_sessions(self, query: SessionQuery) -> list[SessionMetadata]:
        """List sessions matching query."""
        results: list[SessionMetadata] = []

        # Determine which projects to search
        if query.project_slug:
            projects = [query.project_slug]
        else:
            projects = await list_directories(self.base_path)

        for project_slug in projects:
            sessions_dir = self.base_path / project_slug / "sessions"
            session_ids = await list_directories(sessions_dir)

            for session_id in session_ids:
                metadata_path = self._metadata_path(project_slug, session_id)
                data = await read_json(metadata_path)
                if data is None:
                    continue

                metadata = SessionMetadata.from_dict(data)

                # Apply filters
                if metadata.user_id != query.user_id:
                    continue
                if query.name_contains and (
                    not metadata.name or query.name_contains.lower() not in metadata.name.lower()
                ):
                    continue
                if query.created_after and metadata.created < query.created_after:
                    continue
                if query.created_before and metadata.created > query.created_before:
                    continue

                results.append(metadata)

        # Sort results
        if query.order_by == "created":
            results.sort(key=lambda m: m.created, reverse=query.order_desc)
        elif query.order_by == "updated":
            results.sort(key=lambda m: m.updated, reverse=query.order_desc)
        elif query.order_by == "name":
            results.sort(key=lambda m: m.name or "", reverse=query.order_desc)

        # Apply pagination
        return results[query.offset : query.offset + query.limit]

    # =========================================================================
    # Transcript Operations
    # =========================================================================

    async def append_message(
        self,
        user_id: str,
        session_id: str,
        message: TranscriptMessage,
    ) -> TranscriptMessage:
        """Append a message to the transcript."""
        metadata = await self.get_session(user_id, session_id)
        if metadata is None:
            raise SessionNotFoundError(session_id, user_id)

        transcript_path = self._transcript_path(metadata.project_slug, session_id)

        # Get current message count for sequence number
        existing = await read_jsonl(transcript_path)
        message.sequence = len(existing)

        await append_jsonl(transcript_path, message.to_dict())

        # Update metadata counts
        metadata.message_count = len(existing) + 1
        metadata.turn_count = max(metadata.turn_count, message.turn)
        await self.update_session(metadata)

        return message

    async def get_transcript(
        self,
        user_id: str,
        session_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TranscriptMessage]:
        """Get transcript messages."""
        metadata = await self.get_session(user_id, session_id)
        if metadata is None:
            return []

        transcript_path = self._transcript_path(metadata.project_slug, session_id)
        data = await read_jsonl(transcript_path)

        # Apply pagination
        if offset > 0:
            data = data[offset:]
        if limit is not None:
            data = data[:limit]

        return [TranscriptMessage.from_dict(item) for item in data]

    async def get_transcript_for_turn(
        self,
        user_id: str,
        session_id: str,
        turn: int,
    ) -> list[TranscriptMessage]:
        """Get all messages for a specific turn."""
        all_messages = await self.get_transcript(user_id, session_id)
        return [m for m in all_messages if m.turn == turn]

    # =========================================================================
    # Event Operations
    # =========================================================================

    async def append_event(
        self,
        user_id: str,
        session_id: str,
        event_id: str,
        event_type: str,
        data: dict[str, Any],
        turn: int | None = None,
    ) -> EventSummary:
        """Append an event to the session."""
        metadata = await self.get_session(user_id, session_id)
        if metadata is None:
            raise SessionNotFoundError(session_id, user_id)

        events_path = self._events_path(metadata.project_slug, session_id)

        # Create event record
        ts = datetime.utcnow()
        event_record = {
            "event_id": event_id,
            "event_type": event_type,
            "ts": ts.isoformat(),
            "turn": turn,
            "data": data,
        }

        await append_jsonl(events_path, event_record)

        # Update metadata
        metadata.event_count += 1
        await self.update_session(metadata)

        # Return summary (NOT full data)
        return EventSummary(
            event_id=event_id,
            event_type=event_type,
            ts=ts,
            turn=turn,
            summary=_extract_event_summary(data),
            data_size_bytes=len(json.dumps(data)),
        )

    async def query_events(self, query: EventQuery) -> list[EventSummary]:
        """Query events, returning summaries only.

        CRITICAL: This method MUST NEVER return full event data.
        """
        metadata = await self.get_session(query.user_id, query.session_id)
        if metadata is None:
            return []

        events_path = self._events_path(metadata.project_slug, query.session_id)
        all_events = await read_jsonl(events_path)

        results: list[EventSummary] = []
        for event in all_events:
            # Apply filters
            if query.event_types and event.get("event_type") not in query.event_types:
                continue
            if query.turn is not None and event.get("turn") != query.turn:
                continue
            if query.turn_gte is not None and (event.get("turn") or 0) < query.turn_gte:
                continue
            if query.turn_lte is not None and (event.get("turn") or 0) > query.turn_lte:
                continue

            event_ts = datetime.fromisoformat(event["ts"])
            if query.after and event_ts < query.after:
                continue
            if query.before and event_ts > query.before:
                continue

            # Extract summary ONLY - never return full data
            data = event.get("data", {})
            results.append(
                EventSummary(
                    event_id=event["event_id"],
                    event_type=event["event_type"],
                    ts=event_ts,
                    turn=event.get("turn"),
                    summary=_extract_event_summary(data),
                    data_size_bytes=len(json.dumps(data)),
                )
            )

        # Apply pagination
        return results[query.offset : query.offset + query.limit]

    async def get_event_data(
        self,
        user_id: str,
        session_id: str,
        event_id: str,
    ) -> dict[str, Any] | None:
        """Get full data for a specific event.

        This is the ONLY method that returns full event data.
        """
        metadata = await self.get_session(user_id, session_id)
        if metadata is None:
            return None

        events_path = self._events_path(metadata.project_slug, session_id)
        all_events = await read_jsonl(events_path)

        for event in all_events:
            if event.get("event_id") == event_id:
                return event.get("data", {})

        return None

    async def get_event_aggregates(
        self,
        user_id: str,
        session_id: str,
    ) -> AggregateStats:
        """Get aggregate statistics for all events in a session."""
        metadata = await self.get_session(user_id, session_id)
        if metadata is None:
            return AggregateStats(event_count=0, event_types={})

        events_path = self._events_path(metadata.project_slug, session_id)
        all_events = await read_jsonl(events_path)

        event_types: dict[str, int] = {}
        total_input_tokens = 0
        total_output_tokens = 0
        total_duration_ms = 0
        error_count = 0
        tool_call_count = 0

        for event in all_events:
            event_type = event.get("event_type", "unknown")
            event_types[event_type] = event_types.get(event_type, 0) + 1

            data = event.get("data", {})

            # Extract usage stats
            usage = data.get("usage", {})
            total_input_tokens += usage.get("input_tokens", 0)
            total_output_tokens += usage.get("output_tokens", 0)

            # Extract duration
            total_duration_ms += data.get("duration_ms", 0)

            # Count errors
            if data.get("has_error") or data.get("error"):
                error_count += 1

            # Count tool calls
            if data.get("has_tool_calls") or data.get("tool_calls"):
                tool_call_count += 1

        return AggregateStats(
            event_count=len(all_events),
            event_types=event_types,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_duration_ms=total_duration_ms,
            error_count=error_count,
            tool_call_count=tool_call_count,
        )

    # =========================================================================
    # Rewind Operations
    # =========================================================================

    async def rewind_to_turn(
        self,
        user_id: str,
        session_id: str,
        turn: int,
        create_backup: bool = True,
    ) -> RewindResult:
        """Rewind session to a specific turn.

        ATOMIC: Both transcript and events are truncated together.
        """
        metadata = await self.get_session(user_id, session_id)
        if metadata is None:
            raise SessionNotFoundError(session_id, user_id)

        transcript_path = self._transcript_path(metadata.project_slug, session_id)
        events_path = self._events_path(metadata.project_slug, session_id)
        backup_path_str: str | None = None

        # Create backups if requested
        if create_backup:
            backup_dir = self._backup_dir(metadata.project_slug, session_id)
            if await file_exists(transcript_path):
                backup_path = await create_file_backup(transcript_path, backup_dir)
                backup_path_str = str(backup_path)
            if await file_exists(events_path):
                await create_file_backup(events_path, backup_dir)

        # Load current data
        transcript = await read_jsonl(transcript_path)
        events = await read_jsonl(events_path)

        # Filter to keep only up to the specified turn
        new_transcript = [m for m in transcript if m.get("turn", 0) <= turn]
        new_events = [e for e in events if (e.get("turn") or 0) <= turn]

        messages_removed = len(transcript) - len(new_transcript)
        events_removed = len(events) - len(new_events)

        # Write atomically
        await write_jsonl_atomic(transcript_path, new_transcript)
        await write_jsonl_atomic(events_path, new_events)

        # Update metadata
        metadata.message_count = len(new_transcript)
        metadata.event_count = len(new_events)
        metadata.turn_count = turn
        await self.update_session(metadata)

        return RewindResult(
            success=True,
            messages_removed=messages_removed,
            events_removed=events_removed,
            new_turn_count=turn,
            backup_path=backup_path_str,
        )

    async def rewind_to_timestamp(
        self,
        user_id: str,
        session_id: str,
        timestamp: datetime,
        create_backup: bool = True,
    ) -> RewindResult:
        """Rewind session to a specific timestamp.

        ATOMIC: Both transcript and events are truncated together.
        """
        metadata = await self.get_session(user_id, session_id)
        if metadata is None:
            raise SessionNotFoundError(session_id, user_id)

        transcript_path = self._transcript_path(metadata.project_slug, session_id)
        events_path = self._events_path(metadata.project_slug, session_id)
        backup_path_str: str | None = None

        # Create backups if requested
        if create_backup:
            backup_dir = self._backup_dir(metadata.project_slug, session_id)
            if await file_exists(transcript_path):
                backup_path = await create_file_backup(transcript_path, backup_dir)
                backup_path_str = str(backup_path)
            if await file_exists(events_path):
                await create_file_backup(events_path, backup_dir)

        # Load current data
        transcript = await read_jsonl(transcript_path)
        events = await read_jsonl(events_path)

        # Filter by timestamp
        new_transcript = [
            m
            for m in transcript
            if datetime.fromisoformat(m.get("timestamp", "9999-12-31")) <= timestamp
        ]
        new_events = [
            e for e in events if datetime.fromisoformat(e.get("ts", "9999-12-31")) <= timestamp
        ]

        messages_removed = len(transcript) - len(new_transcript)
        events_removed = len(events) - len(new_events)

        # Determine new turn count
        new_turn_count = 0
        if new_transcript:
            new_turn_count = max(m.get("turn", 0) for m in new_transcript)

        # Write atomically
        await write_jsonl_atomic(transcript_path, new_transcript)
        await write_jsonl_atomic(events_path, new_events)

        # Update metadata
        metadata.message_count = len(new_transcript)
        metadata.event_count = len(new_events)
        metadata.turn_count = new_turn_count
        await self.update_session(metadata)

        return RewindResult(
            success=True,
            messages_removed=messages_removed,
            events_removed=events_removed,
            new_turn_count=new_turn_count,
            backup_path=backup_path_str,
        )

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search_sessions(
        self,
        user_id: str,
        query_text: str,
        project_slug: str | None = None,
        limit: int = 20,
    ) -> list[SessionMetadata]:
        """Search sessions by text content."""
        query = SessionQuery(
            user_id=user_id,
            project_slug=project_slug,
            limit=sys.maxsize,  # Get all, then filter
        )
        all_sessions = await self.list_sessions(query)

        # Simple text search in name and description
        query_lower = query_text.lower()
        matches = [
            s
            for s in all_sessions
            if (s.name and query_lower in s.name.lower())
            or (s.description and query_lower in s.description.lower())
        ]

        return matches[:limit]

    # =========================================================================
    # Sync Operations (No-op for local storage)
    # =========================================================================

    async def get_sync_status(
        self,
        user_id: str,
        session_id: str,
    ) -> SyncStatus:
        """Get synchronization status.

        For local-only storage, always returns "synced".
        """
        return SyncStatus(
            is_synced=True,
            pending_changes=0,
            last_sync=datetime.utcnow(),
            conflict_count=0,
        )

    async def sync_now(
        self,
        user_id: str,
        session_id: str | None = None,
    ) -> SyncStatus:
        """Trigger immediate sync.

        For local-only storage, this is a no-op.
        """
        return SyncStatus(
            is_synced=True,
            pending_changes=0,
            last_sync=datetime.utcnow(),
            conflict_count=0,
        )

    # =========================================================================
    # Session Sharing Operations (Not supported for local storage)
    # =========================================================================

    async def set_session_visibility(
        self,
        user_id: str,
        session_id: str,
        visibility: SessionVisibility,
        team_ids: list[str] | None = None,
    ) -> SessionMetadata:
        """Change session visibility.

        Local storage does not support sharing - raises NotImplementedError.
        """
        raise NotImplementedError(
            "Session sharing is not supported for local-only storage. "
            "Use SyncedCosmosStorage or CosmosDBStorage for sharing features."
        )

    async def query_shared_sessions(
        self,
        query: SharedSessionQuery,
    ) -> list[SharedSessionSummary]:
        """Query shared sessions.

        Local storage does not support sharing - returns empty list.
        """
        return []

    async def get_shared_session(
        self,
        requester_user_id: str,
        session_id: str,
    ) -> SessionMetadata | None:
        """Get a shared session.

        Local storage does not support sharing - returns None.
        """
        return None

    async def get_shared_transcript(
        self,
        requester_user_id: str,
        session_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TranscriptMessage]:
        """Get transcript from a shared session.

        Local storage does not support sharing - returns empty list.
        """
        return []

    async def get_user_membership(
        self,
        user_id: str,
    ) -> UserMembership | None:
        """Get user's organization and team memberships.

        Local storage does not support memberships - returns None.
        """
        return None


def _extract_event_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Extract safe summary fields from event data.

    CRITICAL: Never include 'data' or 'content' fields.
    Only extract small, known-safe fields.
    """
    summary: dict[str, Any] = {}

    # Safe fields to extract
    safe_fields = [
        "model",
        "duration_ms",
        "has_tool_calls",
        "has_error",
        "error_type",
        "tool_name",
    ]

    for field in safe_fields:
        if field in data:
            summary[field] = data[field]

    # Extract usage summary
    if "usage" in data:
        usage = data["usage"]
        summary["usage"] = {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }

    return summary
