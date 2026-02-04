"""
Storage backend abstraction layer.

Provides abstract interfaces for different storage backends (Cosmos DB, DuckDB, SQLite).
Each backend implements the same interface, allowing seamless switching.
"""

from .base import (
    EventSearchOptions,
    SearchFilters,
    SearchResult,
    StorageBackend,
    TranscriptSearchOptions,
)

__all__ = [
    "StorageBackend",
    "SearchFilters",
    "SearchResult",
    "TranscriptSearchOptions",
    "EventSearchOptions",
]
