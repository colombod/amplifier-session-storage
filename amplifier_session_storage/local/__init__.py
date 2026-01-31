"""
Local file-based session storage.

Provides file-based storage for offline operation and local caching.
Uses JSONL files for efficient append operations and atomic writes
for data integrity.

Key classes:
- SessionStore: Drop-in replacement for amplifier-app-cli SessionStore
- EventsLog: Writer for events.jsonl audit log
- LocalFileStorage: Block-based storage for sync operations
"""

from .events_log import EventsLog, read_events_summary
from .file_ops import (
    append_jsonl,
    create_backup,
    read_json,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from .session_store import (
    SessionStore,
    extract_session_mode,
    is_top_level_session,
)
from .storage import LocalFileStorage

__all__ = [
    # Primary session management (compatible with amplifier-app-cli)
    "SessionStore",
    "EventsLog",
    "read_events_summary",
    "is_top_level_session",
    "extract_session_mode",
    # Block-based storage for sync
    "LocalFileStorage",
    # Low-level file operations
    "read_json",
    "write_json_atomic",
    "read_jsonl",
    "write_jsonl_atomic",
    "append_jsonl",
    "create_backup",
]
