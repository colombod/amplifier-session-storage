"""
Hybrid file storage module.

Provides offline-first session storage that:
- Uses local file storage (CLI format) for all operations
- Background syncs to Cosmos DB for cloud backup
- Supports directory exclusion patterns for local-only storage
"""

from .file_storage import HybridFileStorage, HybridFileStorageConfig, SyncResult

__all__ = [
    "HybridFileStorage",
    "HybridFileStorageConfig",
    "SyncResult",
]
