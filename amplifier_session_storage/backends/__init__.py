"""
Storage backend abstraction layer.

Provides abstract interfaces for different storage backends (Cosmos DB, DuckDB, SQLite).
Each backend implements the same interface, allowing seamless switching.
"""

from .base import (
    EventSearchOptions,
    MessageContext,
    SearchFilters,
    SearchResult,
    SessionSyncStats,
    StorageBackend,
    TranscriptMessage,
    TranscriptSearchOptions,
    TurnContext,
)

__all__ = [
    # Core classes
    "StorageBackend",
    # Search options
    "SearchFilters",
    "TranscriptSearchOptions",
    "EventSearchOptions",
    # Result types
    "SearchResult",
    "TranscriptMessage",
    "SessionSyncStats",
    # Context types
    "TurnContext",
    "MessageContext",
]
