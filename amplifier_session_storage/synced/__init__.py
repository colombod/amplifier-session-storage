"""
Synced Cosmos storage combining local and cloud.

Provides offline-first storage that:
- Writes to local first, queues for sync
- Reads from local with cloud fallback
- Handles offline operation transparently
- Triggers sync on connectivity changes

This is the recommended storage implementation for production use.
"""

from .storage import SyncedCosmosStorage

__all__ = [
    "SyncedCosmosStorage",
]
