"""
Local file-based session storage.

Provides file-based storage for offline operation and local caching.
Uses JSONL files for efficient append operations and atomic writes
for data integrity.
"""

from .file_ops import (
    append_jsonl,
    create_backup,
    read_json,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from .storage import LocalFileStorage

__all__ = [
    "LocalFileStorage",
    "read_json",
    "write_json_atomic",
    "read_jsonl",
    "write_jsonl_atomic",
    "append_jsonl",
    "create_backup",
]
