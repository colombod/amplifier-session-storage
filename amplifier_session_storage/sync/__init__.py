"""
Real-time sync module.

Provides WebSocket and SSE-based real-time sync for session blocks,
enabling live collaboration and multi-device synchronization.
"""

from .client import SyncClient, SyncEvent, SyncEventType
from .server import SyncHandler, SyncServer

__all__ = [
    "SyncClient",
    "SyncEvent",
    "SyncEventType",
    "SyncServer",
    "SyncHandler",
]
