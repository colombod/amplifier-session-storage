"""
Events log writer for session audit logging.

This module provides an EventsLog class for writing events.jsonl files
that are compatible with the session-analyst agent. The events.jsonl
file is an audit log - it is NOT used for session resume, but provides
complete traceability for debugging and analysis.

CRITICAL: events.jsonl lines can be 100k+ tokens. The session-analyst
agent uses surgical extraction patterns (jq, grep -n | cut) to avoid
loading full lines into context.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

logger = logging.getLogger(__name__)


class EventsLog:
    """
    Writer for events.jsonl audit log.

    This class provides a simple interface for appending events to the
    audit log. Events are written as JSONL (one JSON object per line).

    The events.jsonl file format is compatible with session-analyst:
    - ts: ISO timestamp
    - lvl: Log level (INFO, DEBUG, ERROR, etc.)
    - event: Event type (session:start, llm:request, tool:pre, etc.)
    - session_id: Session identifier
    - data: Event-specific payload (can be large)

    Example usage:
        with EventsLog(session_dir) as log:
            log.append({
                "event": "session:start",
                "data": {"bundle": "foundation"}
            })

    Or manual management:
        log = EventsLog(session_dir)
        log.open()
        log.append({"event": "llm:request", "data": {...}})
        log.close()
    """

    def __init__(
        self,
        session_dir: Path,
        session_id: str | None = None,
        *,
        buffer_size: int = 1,  # Line buffering by default
    ):
        """Initialize events log for a session.

        Args:
            session_dir: Directory containing session files
            session_id: Session ID to include in events. If None, uses directory name.
            buffer_size: Write buffer size. 1 = line buffered (default), 0 = unbuffered.
        """
        self.session_dir = Path(session_dir)
        self.session_id = session_id or self.session_dir.name
        self.buffer_size = buffer_size
        self._file: TextIO | None = None
        self._event_count = 0

    def open(self) -> EventsLog:
        """Open the events log for writing.

        Returns:
            Self for method chaining
        """
        if self._file is not None:
            return self

        self.session_dir.mkdir(parents=True, exist_ok=True)
        events_path = self.session_dir / "events.jsonl"

        # Open in append mode with specified buffering
        self._file = open(events_path, "a", encoding="utf-8", buffering=self.buffer_size)

        return self

    def close(self) -> None:
        """Close the events log file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> EventsLog:
        """Context manager entry."""
        return self.open()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    def append(
        self,
        event: dict[str, Any],
        *,
        level: str = "INFO",
        timestamp: str | None = None,
    ) -> None:
        """Append an event to the log.

        Args:
            event: Event dict. Must contain 'event' key with event type.
                   'data' key contains event-specific payload.
            level: Log level (INFO, DEBUG, ERROR, WARNING)
            timestamp: ISO timestamp. If None, uses current time.

        Raises:
            ValueError: If event dict is missing 'event' key
            RuntimeError: If log is not open
        """
        if self._file is None:
            # Auto-open if not already open
            self.open()

        if "event" not in event:
            raise ValueError("Event dict must contain 'event' key")

        # Build the log entry
        entry = {
            "ts": timestamp or datetime.now(UTC).isoformat(timespec="milliseconds"),
            "lvl": level,
            "schema": {"name": "amplifier.log", "ver": "1.0.0"},
            "event": event["event"],
            "session_id": self.session_id,
        }

        # Add data if present
        if "data" in event:
            entry["data"] = event["data"]

        # Add any additional fields from the event
        for key, value in event.items():
            if key not in ("event", "data"):
                entry[key] = value

        # Write the line
        line = json.dumps(entry, ensure_ascii=False)
        self._file.write(line + "\n")
        self._event_count += 1

    def append_session_start(
        self,
        bundle: str | None = None,
        model: str | None = None,
        parent_id: str | None = None,
        **extra: Any,
    ) -> None:
        """Append a session:start event.

        Args:
            bundle: Bundle name
            model: Model name
            parent_id: Parent session ID (for sub-sessions)
            **extra: Additional data fields
        """
        data: dict[str, Any] = {}
        if bundle:
            data["bundle"] = bundle
        if model:
            data["model"] = model
        if parent_id:
            data["parent_id"] = parent_id
        data.update(extra)

        self.append({"event": "session:start", "data": data})

    def append_session_end(
        self,
        turn_count: int | None = None,
        duration_ms: int | None = None,
        **extra: Any,
    ) -> None:
        """Append a session:end event.

        Args:
            turn_count: Number of turns in session
            duration_ms: Session duration in milliseconds
            **extra: Additional data fields
        """
        data: dict[str, Any] = {}
        if turn_count is not None:
            data["turn_count"] = turn_count
        if duration_ms is not None:
            data["duration_ms"] = duration_ms
        data.update(extra)

        self.append({"event": "session:end", "data": data})

    def append_llm_request(
        self,
        model: str,
        message_count: int,
        **extra: Any,
    ) -> None:
        """Append an llm:request event (summary only - not full payload).

        Note: For full payload logging, use append() directly with data.
        This method logs summary info only to keep event sizes reasonable.

        Args:
            model: Model name
            message_count: Number of messages in request
            **extra: Additional data fields
        """
        data = {
            "model": model,
            "message_count": message_count,
            **extra,
        }
        self.append({"event": "llm:request", "data": data})

    def append_llm_response(
        self,
        model: str,
        usage: dict[str, int] | None = None,
        duration_ms: int | None = None,
        **extra: Any,
    ) -> None:
        """Append an llm:response event.

        Args:
            model: Model name
            usage: Token usage dict (input_tokens, output_tokens, etc.)
            duration_ms: Response duration in milliseconds
            **extra: Additional data fields
        """
        data: dict[str, Any] = {"model": model}
        if usage:
            data["usage"] = usage
        if duration_ms is not None:
            data["duration_ms"] = duration_ms
        data.update(extra)

        self.append({"event": "llm:response", "data": data})

    def append_tool_pre(
        self,
        tool_name: str,
        tool_call_id: str | None = None,
        **extra: Any,
    ) -> None:
        """Append a tool:pre event (tool execution starting).

        Args:
            tool_name: Name of the tool being called
            tool_call_id: Tool call ID from LLM
            **extra: Additional data fields (be careful with size)
        """
        data: dict[str, Any] = {"tool_name": tool_name}
        if tool_call_id:
            data["tool_call_id"] = tool_call_id
        data.update(extra)

        self.append({"event": "tool:pre", "data": data})

    def append_tool_post(
        self,
        tool_name: str,
        tool_call_id: str | None = None,
        duration_ms: int | None = None,
        success: bool = True,
        **extra: Any,
    ) -> None:
        """Append a tool:post event (tool execution completed).

        Args:
            tool_name: Name of the tool that was called
            tool_call_id: Tool call ID from LLM
            duration_ms: Execution duration in milliseconds
            success: Whether the tool execution succeeded
            **extra: Additional data fields (be careful with size)
        """
        data: dict[str, Any] = {"tool_name": tool_name, "success": success}
        if tool_call_id:
            data["tool_call_id"] = tool_call_id
        if duration_ms is not None:
            data["duration_ms"] = duration_ms
        data.update(extra)

        self.append({"event": "tool:post", "data": data})

    def append_error(
        self,
        error_type: str,
        message: str,
        **extra: Any,
    ) -> None:
        """Append an error event.

        Args:
            error_type: Type of error (e.g., "api_error", "tool_error")
            message: Error message
            **extra: Additional data fields
        """
        data = {
            "error_type": error_type,
            "message": message,
            **extra,
        }
        self.append({"event": "error", "data": data}, level="ERROR")

    @property
    def event_count(self) -> int:
        """Number of events written to this log."""
        return self._event_count

    @property
    def is_open(self) -> bool:
        """Whether the log file is currently open."""
        return self._file is not None


def read_events_summary(events_path: Path) -> dict[str, Any]:
    """Read a summary of events from an events.jsonl file.

    This function safely reads events without loading full lines into memory.
    It extracts only small fields for summary purposes.

    Args:
        events_path: Path to events.jsonl file

    Returns:
        Dictionary with:
        - total_events: Total count
        - event_types: Dict of event type -> count
        - first_timestamp: First event timestamp
        - last_timestamp: Last event timestamp
        - errors: List of error event summaries
    """
    if not events_path.exists():
        return {
            "total_events": 0,
            "event_types": {},
            "first_timestamp": None,
            "last_timestamp": None,
            "errors": [],
        }

    event_types: dict[str, int] = {}
    first_ts: str | None = None
    last_ts: str | None = None
    errors: list[dict[str, str]] = []
    total = 0

    with open(events_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                # Parse just enough to get summary info
                event = json.loads(line)
                total += 1

                # Count event types
                event_type = event.get("event", "unknown")
                event_types[event_type] = event_types.get(event_type, 0) + 1

                # Track timestamps
                ts = event.get("ts") or event.get("timestamp")
                if ts:
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts

                # Collect error summaries (truncated)
                if event.get("lvl") == "ERROR" or event_type == "error":
                    error_summary = {
                        "ts": ts or "unknown",
                        "event": event_type,
                    }
                    if "data" in event and isinstance(event["data"], dict):
                        msg = event["data"].get("message", "")
                        if msg:
                            error_summary["message"] = msg[:200]  # Truncate
                    errors.append(error_summary)

            except json.JSONDecodeError:
                continue

    return {
        "total_events": total,
        "event_types": event_types,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "errors": errors[:10],  # Limit to 10 errors
    }
