"""
Amplifier Session Storage

A session storage library for syncing Amplifier CLI sessions to Azure Cosmos DB.

Provides:
- CosmosFileStorage: Cloud storage that mirrors CLI file format
  - Sessions container (metadata.json equivalent)
  - Transcripts container (transcript.jsonl lines)
  - Events container (events.jsonl lines)
- Identity management for user/device tracking

Usage:

    >>> from amplifier_session_storage import CosmosFileStorage, CosmosFileConfig
    >>> config = CosmosFileConfig.from_env()
    >>> async with CosmosFileStorage(config) as storage:
    ...     await storage.upsert_session_metadata(user_id, host_id, metadata)
    ...     await storage.sync_transcript_lines(user_id, host_id, project, session, lines)
    ...     await storage.sync_event_lines(user_id, host_id, project, session, lines)

Identity Management:

    >>> from amplifier_session_storage import IdentityContext
    >>> IdentityContext.initialize()
    >>> user_id = IdentityContext.get_user_id()
"""

# Cosmos file storage - CLI-compatible cloud storage
from .cosmos import CosmosFileConfig, CosmosFileStorage

# Exceptions
from .exceptions import (
    AuthenticationError,
    ChunkingError,
    ConflictError,
    EventNotFoundError,
    EventTooLargeError,
    PermissionDeniedError,
    RewindError,
    SessionExistsError,
    SessionNotFoundError,
    SessionStorageError,
    SessionValidationError,
    StorageConnectionError,
    StorageIOError,
    SyncError,
    ValidationError,
)

# Identity module - user/device/org management
from .identity import (
    ConfigFileIdentityProvider,
    IdentityContext,
    IdentityProvider,
    UserIdentity,
)

__all__ = [
    # Cosmos file storage (CLI-compatible)
    "CosmosFileStorage",
    "CosmosFileConfig",
    # Identity
    "IdentityProvider",
    "UserIdentity",
    "ConfigFileIdentityProvider",
    "IdentityContext",
    # Exceptions
    "SessionStorageError",
    "SessionNotFoundError",
    "SessionValidationError",
    "SessionExistsError",
    "EventNotFoundError",
    "StorageIOError",
    "ChunkingError",
    "SyncError",
    "ConflictError",
    "StorageConnectionError",
    "AuthenticationError",
    "ValidationError",
    "RewindError",
    "EventTooLargeError",
    "PermissionDeniedError",
]

__version__ = "0.1.0"
