"""
Cosmos DB session storage.

Provides cloud-based storage using Azure Cosmos DB with:
- User isolation via partition keys
- Event projection enforcement
- Large event chunking (>400KB events)
- Transactional operations
"""

from .chunking import (
    CHUNK_SIZE,
    MAX_INLINE_SIZE,
    EventChunk,
    chunk_event,
    reassemble_event,
    should_chunk,
)
from .client import CosmosClientWrapper, CosmosConfig
from .storage import CosmosDBStorage

__all__ = [
    "CosmosDBStorage",
    "CosmosClientWrapper",
    "CosmosConfig",
    "EventChunk",
    "chunk_event",
    "reassemble_event",
    "should_chunk",
    "CHUNK_SIZE",
    "MAX_INLINE_SIZE",
]
