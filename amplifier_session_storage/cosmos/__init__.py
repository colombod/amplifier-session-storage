"""
Cosmos DB session storage.

Provides cloud-based storage using Azure Cosmos DB with:
- User isolation via partition keys
- Event projection enforcement
- Large event chunking (>400KB events)
- Transactional operations
- CLI-compatible file format storage (CosmosFileStorage)
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
from .file_storage import CosmosFileConfig, CosmosFileStorage
from .storage import CosmosDBStorage

__all__ = [
    "CosmosDBStorage",
    "CosmosClientWrapper",
    "CosmosConfig",
    "CosmosFileStorage",
    "CosmosFileConfig",
    "EventChunk",
    "chunk_event",
    "reassemble_event",
    "should_chunk",
    "CHUNK_SIZE",
    "MAX_INLINE_SIZE",
]
