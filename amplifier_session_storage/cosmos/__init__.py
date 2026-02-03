"""
Cosmos DB session storage.

Provides cloud-based storage using Azure Cosmos DB with:
- User isolation via partition keys
- Separate containers for sessions, transcripts, and events
- CLI-compatible file format storage (CosmosFileStorage)
"""

from .file_storage import CosmosFileConfig, CosmosFileStorage

__all__ = [
    "CosmosFileStorage",
    "CosmosFileConfig",
]
