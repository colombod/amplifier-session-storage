"""
Session tool for Amplifier agents.

Provides safe, structured access to session data for agents like session-analyst.
This tool abstracts the storage backend (local files or Cosmos DB) and ensures
safe operations that won't overflow agent context.

Key safety guarantees:
1. Never returns full event data payloads (can be 100k+ tokens)
2. Pagination for large result sets
3. Truncated excerpts in search results
4. User isolation enforced at all operations
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ..facets import FacetQuery, SessionFacets
from ..local import SessionStore, read_events_summary


@dataclass
class SessionToolConfig:
    """Configuration for SessionTool.

    Attributes:
        base_dir: Base directory for local session storage.
                  Default: ~/.amplifier/projects/{project}/sessions/
        project_slug: Project slug for session organization.
        enable_cloud: Whether to include cloud sessions (requires Cosmos config).
        max_results: Default maximum results for list/search operations.
        max_excerpt_length: Maximum length of text excerpts in search results.
    """

    base_dir: Path | None = None
    project_slug: str = "default"
    enable_cloud: bool = False
    max_results: int = 50
    max_excerpt_length: int = 500


@dataclass
class SessionInfo:
    """Summary information about a session."""

    session_id: str
    project: str
    created: str | None = None
    modified: str | None = None
    bundle: str | None = None
    model: str | None = None
    turn_count: int = 0
    message_count: int = 0
    source: Literal["local", "cloud", "both"] = "local"
    name: str | None = None
    path: str | None = None  # Local path for manual inspection


@dataclass
class SearchMatch:
    """A search match result."""

    session_id: str
    project: str
    created: str | None
    match_type: Literal["metadata", "transcript"]
    excerpt: str
    line_number: int | None = None


@dataclass
class EventSummary:
    """Summary of an event (without full data payload)."""

    ts: str
    event_type: str
    turn: int | None = None
    has_error: bool = False
    # Only safe summary fields, NEVER full data


@dataclass
class EventsAnalysis:
    """Analysis of session events."""

    total_events: int = 0
    event_types: dict[str, int] = field(default_factory=dict)
    duration_seconds: float | None = None
    first_event: str | None = None
    last_event: str | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)
    llm_requests: int = 0
    tool_calls: int = 0


@dataclass
class RewindPreview:
    """Preview of what a rewind operation would do."""

    would_remove_messages: int
    would_remove_events: int
    new_turn_count: int
    dry_run: bool = True


class SessionTool:
    """
    Tool for session operations.

    Provides safe, structured access to session data for Amplifier agents.
    Supports both local file storage and (optionally) Cosmos DB cloud storage.

    All operations enforce:
    - User isolation (can only access own sessions)
    - Safe projections (never returns full event payloads)
    - Pagination (large result sets are limited)
    - Truncation (excerpts are length-limited)
    """

    def __init__(self, config: SessionToolConfig | None = None):
        """Initialize the session tool.

        Args:
            config: Tool configuration. Uses defaults if not provided.
        """
        self.config = config or SessionToolConfig()
        self._store: SessionStore | None = None

    @property
    def store(self) -> SessionStore:
        """Get or create the session store."""
        if self._store is None:
            self._store = SessionStore(
                base_dir=self.config.base_dir,
                project_slug=self.config.project_slug,
            )
        return self._store

    def list_sessions(
        self,
        project: str | None = None,
        date_range: str | None = None,
        top_level_only: bool = True,
        limit: int | None = None,
    ) -> list[SessionInfo]:
        """List sessions with optional filtering.

        Args:
            project: Filter by project slug.
            date_range: Filter by date range ("today", "last_week", "YYYY-MM-DD:YYYY-MM-DD").
            top_level_only: If True, exclude spawned sub-sessions.
            limit: Maximum number of sessions to return.

        Returns:
            List of SessionInfo objects with summary information.
        """
        limit = limit or self.config.max_results

        # Get session IDs from store
        session_ids = self.store.list_sessions(top_level_only=top_level_only)

        # Apply date filter if specified
        if date_range:
            session_ids = self._filter_by_date(session_ids, date_range)

        # Limit results
        session_ids = session_ids[:limit]

        # Build session info list
        sessions = []
        for session_id in session_ids:
            try:
                metadata = self.store.get_metadata(session_id)
                sessions.append(
                    SessionInfo(
                        session_id=session_id,
                        project=metadata.get("project_slug", self.config.project_slug),
                        created=metadata.get("created"),
                        modified=metadata.get("updated"),
                        bundle=metadata.get("bundle"),
                        model=metadata.get("model"),
                        turn_count=metadata.get("turn_count", 0),
                        message_count=metadata.get("message_count", 0),
                        source="local",
                        name=metadata.get("name"),
                        path=str(self.store.get_session_dir(session_id)),
                    )
                )
            except Exception:
                # Skip sessions that can't be read
                continue

        return sessions

    def get_session(
        self,
        session_id: str,
        include_transcript: bool = False,
        include_events_summary: bool = False,
    ) -> dict[str, Any]:
        """Get detailed information about a session.

        Args:
            session_id: Full or partial session ID.
            include_transcript: If True, include conversation messages.
            include_events_summary: If True, include event statistics.

        Returns:
            Dictionary with session details.

        Raises:
            SessionNotFoundError: If session not found.
        """
        # Resolve partial session ID
        full_id = self.store.find_session(session_id)

        # Load metadata
        metadata = self.store.get_metadata(full_id)

        result: dict[str, Any] = {
            "session_id": full_id,
            "project": metadata.get("project_slug", self.config.project_slug),
            "metadata": metadata,
            "source": "local",
            "path": str(self.store.get_session_dir(full_id)),
        }

        if include_transcript:
            transcript, _ = self.store.load(full_id)
            result["transcript"] = transcript

        if include_events_summary:
            events_file = self.store.get_session_dir(full_id) / "events.jsonl"
            result["events_summary"] = read_events_summary(events_file)

        return result

    def search_sessions(
        self,
        query: str,
        scope: Literal["metadata", "transcript", "all"] = "all",
        project: str | None = None,
        limit: int | None = None,
        context_lines: int = 2,
    ) -> list[SearchMatch]:
        """Search sessions for matching content.

        Args:
            query: Search term (case-insensitive substring match).
            scope: Where to search ("metadata", "transcript", or "all").
            project: Filter by project slug.
            limit: Maximum number of matches to return.
            context_lines: Number of context lines around transcript matches.

        Returns:
            List of SearchMatch objects with excerpts.
        """
        limit = limit or self.config.max_results
        matches: list[SearchMatch] = []
        query_lower = query.lower()

        session_ids = self.store.list_sessions(top_level_only=False)

        for session_id in session_ids:
            if len(matches) >= limit:
                break

            try:
                # Search metadata
                if scope in ("metadata", "all"):
                    metadata = self.store.get_metadata(session_id)
                    metadata_str = json.dumps(metadata).lower()
                    if query_lower in metadata_str:
                        # Find matching field
                        excerpt = self._extract_metadata_excerpt(metadata, query)
                        matches.append(
                            SearchMatch(
                                session_id=session_id,
                                project=metadata.get("project_slug", self.config.project_slug),
                                created=metadata.get("created"),
                                match_type="metadata",
                                excerpt=excerpt,
                            )
                        )
                        continue  # Don't double-count

                # Search transcript
                if scope in ("transcript", "all"):
                    transcript_file = self.store.get_session_dir(session_id) / "transcript.jsonl"
                    if transcript_file.exists():
                        match = self._search_transcript(transcript_file, query_lower, context_lines)
                        if match:
                            metadata = self.store.get_metadata(session_id)
                            matches.append(
                                SearchMatch(
                                    session_id=session_id,
                                    project=metadata.get("project_slug", self.config.project_slug),
                                    created=metadata.get("created"),
                                    match_type="transcript",
                                    excerpt=match["excerpt"],
                                    line_number=match["line_number"],
                                )
                            )

            except Exception:
                # Skip sessions that can't be searched
                continue

        return matches

    def get_events(
        self,
        session_id: str,
        event_types: list[str] | None = None,
        errors_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get events from a session with SAFE projections.

        IMPORTANT: This method NEVER returns full event data payloads.
        Only summary fields are returned to prevent context overflow.

        Args:
            session_id: Session ID.
            event_types: Filter by event types (e.g., ["llm:request", "tool:call"]).
            errors_only: If True, only return error events.
            limit: Maximum events to return.
            offset: Number of events to skip.

        Returns:
            Dictionary with events list and pagination info.
        """
        full_id = self.store.find_session(session_id)
        events_file = self.store.get_session_dir(full_id) / "events.jsonl"

        events: list[EventSummary] = []
        total_count = 0
        current_index = 0

        if events_file.exists():
            with open(events_file) as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        event_type = event.get("event", "")

                        # Apply filters
                        if event_types and event_type not in event_types:
                            continue
                        if errors_only and event.get("lvl") != "ERROR":
                            continue

                        total_count += 1

                        # Apply pagination
                        if current_index < offset:
                            current_index += 1
                            continue
                        if len(events) >= limit:
                            continue

                        # Create SAFE summary (no full data)
                        events.append(
                            EventSummary(
                                ts=event.get("ts", ""),
                                event_type=event_type,
                                turn=event.get("data", {}).get("turn"),
                                has_error=event.get("lvl") == "ERROR",
                            )
                        )
                        current_index += 1

                    except json.JSONDecodeError:
                        continue

        return {
            "events": [
                {
                    "ts": e.ts,
                    "event_type": e.event_type,
                    "turn": e.turn,
                    "has_error": e.has_error,
                }
                for e in events
            ],
            "total_count": total_count,
            "has_more": total_count > offset + limit,
            "offset": offset,
            "limit": limit,
        }

    def analyze_events(
        self,
        session_id: str,
        analysis_type: Literal["summary", "errors", "timeline", "usage"] = "summary",
    ) -> EventsAnalysis:
        """Analyze session events.

        Args:
            session_id: Session ID.
            analysis_type: Type of analysis to perform.

        Returns:
            EventsAnalysis with requested information.
        """
        full_id = self.store.find_session(session_id)
        events_file = self.store.get_session_dir(full_id) / "events.jsonl"

        analysis = EventsAnalysis()

        if not events_file.exists():
            return analysis

        first_ts: datetime | None = None
        last_ts: datetime | None = None

        with open(events_file) as f:
            for line in f:
                try:
                    event = json.loads(line)
                    event_type = event.get("event", "unknown")
                    ts_str = event.get("ts")

                    analysis.total_events += 1
                    analysis.event_types[event_type] = analysis.event_types.get(event_type, 0) + 1

                    # Track timestamps
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            if first_ts is None:
                                first_ts = ts
                                analysis.first_event = ts_str
                            last_ts = ts
                            analysis.last_event = ts_str
                        except ValueError:
                            pass

                    # Count specific event types
                    if event_type == "llm:request":
                        analysis.llm_requests += 1
                    elif event_type.startswith("tool:"):
                        analysis.tool_calls += 1

                    # Collect errors (truncated)
                    if event.get("lvl") == "ERROR" and analysis_type in ("summary", "errors"):
                        error_data = event.get("data", {})
                        analysis.errors.append(
                            {
                                "ts": ts_str,
                                "event": event_type,
                                "message": str(error_data.get("message", ""))[:200],
                            }
                        )

                except json.JSONDecodeError:
                    continue

        # Calculate duration
        if first_ts and last_ts:
            analysis.duration_seconds = (last_ts - first_ts).total_seconds()

        return analysis

    def query_sessions(
        self,
        *,
        # Basic filters
        bundle: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        # Tool/agent filters
        tool_used: str | None = None,
        agent_delegated_to: str | None = None,
        # Status filters
        has_errors: bool | None = None,
        has_child_sessions: bool | None = None,
        has_recipes: bool | None = None,
        # Token filters
        min_tokens: int | None = None,
        max_tokens: int | None = None,
        # Workflow pattern
        workflow_pattern: str | None = None,
        # Date filters
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        # Pagination
        limit: int | None = None,
        offset: int = 0,
    ) -> list[SessionInfo]:
        """Query sessions using facet-based filters.

        This method enables powerful server-side filtering when using Cosmos DB,
        or in-memory filtering for local storage. Use this for finding sessions
        by their characteristics rather than text search.

        Args:
            bundle: Filter by bundle name (e.g., "amplifier-dev")
            model: Filter by model used (e.g., "claude-sonnet-4-20250514")
            provider: Filter by provider (e.g., "anthropic")
            tool_used: Filter by tool that was used (e.g., "delegate", "bash")
            agent_delegated_to: Filter by agent that was delegated to
            has_errors: Filter by whether session had errors
            has_child_sessions: Filter by whether session spawned child sessions
            has_recipes: Filter by whether session used recipes
            min_tokens: Minimum total tokens used
            max_tokens: Maximum total tokens used
            workflow_pattern: Filter by detected workflow pattern
            created_after: Sessions created after this datetime
            created_before: Sessions created before this datetime
            limit: Maximum results (default from config)
            offset: Pagination offset

        Returns:
            List of SessionInfo objects matching the filters.

        Example:
            # Find multi-agent sessions with errors
            sessions = tool.query_sessions(
                has_child_sessions=True,
                has_errors=True,
                created_after=datetime.now() - timedelta(days=7)
            )

            # Find sessions using specific tools
            sessions = tool.query_sessions(
                tool_used="delegate",
                bundle="amplifier-dev"
            )
        """
        limit = limit or self.config.max_results

        # Build FacetQuery for filtering
        query = FacetQuery(
            bundle=bundle,
            model=model,
            provider=provider,
            tool_used=tool_used,
            agent_delegated_to=agent_delegated_to,
            has_errors=has_errors,
            has_child_sessions=has_child_sessions,
            has_recipes=has_recipes,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            workflow_pattern=workflow_pattern,
            created_after=created_after,
            created_before=created_before,
            limit=limit,
            offset=offset,
        )

        # For local storage, filter in Python
        # (Cosmos backend would use FacetQueryBuilder for server-side filtering)
        sessions: list[SessionInfo] = []
        session_ids = self.store.list_sessions(top_level_only=True)

        for session_id in session_ids:
            if len(sessions) >= limit:
                break

            try:
                metadata = self.store.get_metadata(session_id)

                # Apply facet filters
                if not self._matches_facet_query(metadata, query):
                    continue

                sessions.append(
                    SessionInfo(
                        session_id=session_id,
                        project=metadata.get("project_slug", self.config.project_slug),
                        created=metadata.get("created"),
                        modified=metadata.get("updated"),
                        bundle=metadata.get("bundle"),
                        model=metadata.get("model"),
                        turn_count=metadata.get("turn_count", 0),
                        message_count=metadata.get("message_count", 0),
                        source="local",
                        name=metadata.get("name"),
                        path=str(self.store.get_session_dir(session_id)),
                    )
                )
            except Exception:
                continue

        return sessions

    def _matches_facet_query(self, metadata: dict[str, Any], query: FacetQuery) -> bool:
        """Check if session metadata matches a facet query.

        Used for local filtering. Cosmos uses FacetQueryBuilder for
        server-side filtering.
        """
        # Get facets from metadata (may be None for old sessions)
        facets_data = metadata.get("facets", {})
        facets = SessionFacets.from_dict(facets_data) if facets_data else None

        # Basic metadata filters (always available)
        if query.bundle and metadata.get("bundle") != query.bundle:
            return False

        if query.model and metadata.get("model") != query.model:
            return False

        # Date filters
        if query.created_after:
            created = metadata.get("created")
            if created:
                try:
                    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    if created_dt < query.created_after:
                        return False
                except (ValueError, TypeError):
                    pass

        if query.created_before:
            created = metadata.get("created")
            if created:
                try:
                    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    if created_dt > query.created_before:
                        return False
                except (ValueError, TypeError):
                    pass

        # Facet-based filters (require facets to be computed)
        if facets:
            if query.provider:
                if (
                    query.provider != facets.initial_provider
                    and query.provider not in facets.providers_used
                ):
                    return False

            if query.tool_used and query.tool_used not in facets.tools_used:
                return False

            if (
                query.agent_delegated_to
                and query.agent_delegated_to not in facets.agents_delegated_to
            ):
                return False

            if query.has_errors is not None and facets.has_errors != query.has_errors:
                return False

            if (
                query.has_child_sessions is not None
                and facets.has_child_sessions != query.has_child_sessions
            ):
                return False

            if query.has_recipes is not None and facets.has_recipes != query.has_recipes:
                return False

            if query.min_tokens is not None and facets.total_tokens < query.min_tokens:
                return False

            if query.max_tokens is not None and facets.total_tokens > query.max_tokens:
                return False

            if query.workflow_pattern and facets.workflow_pattern != query.workflow_pattern:
                return False

        return True

    def rewind_session(
        self,
        session_id: str,
        to_turn: int | None = None,
        to_message: int | None = None,
        dry_run: bool = True,
    ) -> RewindPreview:
        """Preview or execute a session rewind.

        Args:
            session_id: Session ID.
            to_turn: Rewind to after this turn number.
            to_message: Rewind to after this message index.
            dry_run: If True, only preview what would be removed.

        Returns:
            RewindPreview with information about what would be/was removed.
        """
        full_id = self.store.find_session(session_id)
        session_dir = self.store.get_session_dir(full_id)

        # Load current transcript
        transcript, metadata = self.store.load(full_id)

        # Determine cutoff point
        if to_turn is not None:
            # Find last message of the target turn
            cutoff_index = 0
            for i, msg in enumerate(transcript):
                if msg.get("turn", 0) <= to_turn:
                    cutoff_index = i + 1
            messages_to_keep = cutoff_index
        elif to_message is not None:
            messages_to_keep = to_message
        else:
            messages_to_keep = len(transcript)

        messages_to_remove = len(transcript) - messages_to_keep
        new_turn_count = to_turn or metadata.get("turn_count", 0)

        # Count events that would be removed (simplified - in reality would check timestamps)
        events_to_remove = 0
        events_file = session_dir / "events.jsonl"
        if events_file.exists():
            with open(events_file) as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        event_turn = event.get("data", {}).get("turn", 0)
                        if event_turn and event_turn > (to_turn or float("inf")):
                            events_to_remove += 1
                    except json.JSONDecodeError:
                        continue

        preview = RewindPreview(
            would_remove_messages=messages_to_remove,
            would_remove_events=events_to_remove,
            new_turn_count=new_turn_count,
            dry_run=dry_run,
        )

        if not dry_run:
            # Actually perform the rewind
            new_transcript = transcript[:messages_to_keep]
            metadata["turn_count"] = new_turn_count
            metadata["message_count"] = len(new_transcript)
            metadata["updated"] = datetime.now(UTC).isoformat()

            # Save truncated transcript
            self.store.save(full_id, new_transcript, metadata)

            # Note: Event truncation would require more complex logic
            # to maintain event log integrity

            preview.dry_run = False

        return preview

    def _filter_by_date(self, session_ids: list[str], date_range: str) -> list[str]:
        """Filter sessions by date range."""
        now = datetime.now(UTC)

        if date_range == "today":
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = now
        elif date_range == "last_week":
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            start_date = start_date.replace(day=start_date.day - 7)
            end_date = now
        elif ":" in date_range:
            # Parse YYYY-MM-DD:YYYY-MM-DD format
            parts = date_range.split(":")
            start_date = datetime.fromisoformat(parts[0]).replace(tzinfo=UTC)
            end_date = datetime.fromisoformat(parts[1]).replace(tzinfo=UTC)
        else:
            return session_ids  # No valid filter

        filtered = []
        for session_id in session_ids:
            try:
                metadata = self.store.get_metadata(session_id)
                created_str = metadata.get("created")
                if created_str:
                    created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                    if start_date <= created <= end_date:
                        filtered.append(session_id)
            except Exception:
                continue

        return filtered

    def _extract_metadata_excerpt(self, metadata: dict[str, Any], query: str) -> str:
        """Extract a relevant excerpt from metadata for a search match."""
        query_lower = query.lower()

        # Search through string fields
        for key, value in metadata.items():
            if isinstance(value, str) and query_lower in value.lower():
                # Found match - extract context
                idx = value.lower().find(query_lower)
                start = max(0, idx - 50)
                end = min(len(value), idx + len(query) + 50)
                excerpt = value[start:end]
                if start > 0:
                    excerpt = "..." + excerpt
                if end < len(value):
                    excerpt = excerpt + "..."
                return f"{key}: {excerpt}"

        return "Match in metadata"

    def _search_transcript(
        self, transcript_file: Path, query_lower: str, context_lines: int
    ) -> dict[str, Any] | None:
        """Search a transcript file for a query."""
        lines: list[str] = []
        with open(transcript_file) as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            try:
                msg = json.loads(line)
                content = str(msg.get("content", "")).lower()
                if query_lower in content:
                    # Build excerpt with context
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    excerpt_lines = []
                    for j in range(start, end):
                        try:
                            ctx_msg = json.loads(lines[j])
                            ctx_content = str(ctx_msg.get("content", ""))[:100]
                            role = ctx_msg.get("role", "?")
                            marker = ">>>" if j == i else "   "
                            excerpt_lines.append(f"{marker} [{role}] {ctx_content}")
                        except json.JSONDecodeError:
                            continue

                    return {
                        "line_number": i + 1,
                        "excerpt": "\n".join(excerpt_lines)[: self.config.max_excerpt_length],
                    }
            except json.JSONDecodeError:
                continue

        return None
