"""
Amplifier Session Storage

A session storage system for Amplifier with:
- Drop-in compatible SessionStore for amplifier-app-cli
- Local file-based storage (same format as session-analyst expects)
- Cosmos DB cloud storage for multi-tenant persistence
- Block-based sync for efficient multi-device operation
- Identity management for user/org/team access control

PRIMARY API (Compatible with amplifier-app-cli):

    >>> from amplifier_session_storage import SessionStore
    >>> store = SessionStore()  # Uses ~/.amplifier/projects/default/sessions/
    >>> store.save(session_id, transcript, metadata)
    >>> transcript, metadata = store.load(session_id)

    This SessionStore is a drop-in replacement for amplifier-app-cli's
    SessionStore and maintains full compatibility with session-analyst.

Events Logging:

    >>> from amplifier_session_storage import EventsLog
    >>> with EventsLog(session_dir, session_id) as log:
    ...     log.append_session_start(bundle="foundation")
    ...     log.append_llm_request(model="claude-sonnet", message_count=5)

BLOCK-BASED API (For sync operations):

    >>> from amplifier_session_storage import LocalBlockStorage, StorageConfig
    >>> config = StorageConfig(user_id="user-123")
    >>> storage = LocalBlockStorage(config)
    >>> # Write blocks via BlockWriter, read via storage.read_blocks()

Cloud Storage:

    >>> from amplifier_session_storage import CosmosBlockStorage, StorageConfig
    >>> config = StorageConfig(
    ...     user_id="user-123",
    ...     cosmos_endpoint="https://example.cosmos.azure.com",
    ... )
    >>> storage = CosmosBlockStorage(config)

Identity Management:

    >>> from amplifier_session_storage import IdentityContext
    >>> IdentityContext.initialize()
    >>> user_id = IdentityContext.get_user_id()
"""

# Local storage - compatible with amplifier-app-cli and session-analyst
# Blocks module - session block types and utilities
# Analyst module - session query and analysis
from .analyst import SessionAnalyst, SessionQuery, SessionSummary
from .blocks import (
    BlockType,
    BlockWriter,
    EventData,
    EventDataChunk,
    MessageData,
    RewindData,
    SequenceAllocator,
    SessionBlock,
    SessionCreatedData,
    SessionStateReader,
    SessionUpdatedData,
)

# Identity module - user/device/org management
from .identity import (
    ConfigFileIdentityProvider,
    IdentityContext,
    IdentityProvider,
    UserIdentity,
)
from .local import (
    EventsLog,
    SessionStore,
    extract_session_mode,
    is_top_level_session,
    read_events_summary,
)

# Migration module - legacy session migration
from .migration import (
    MigrationResult,
    MigrationStatus,
    SessionMigrator,
    SessionSource,
)

# Storage module - block storage backends
from .storage import (
    AccessDeniedError,
    AuthenticationError,
    BlockStorage,
    ConflictResolution,
    CosmosAuthMethod,
    CosmosBlockStorage,
    HybridBlockStorage,
    LocalBlockStorage,
    SessionNotFoundError,
    StorageConfig,
    StorageError,
    SyncConflict,
    SyncState,
)

# Sync module - real-time synchronization
from .sync import SyncClient, SyncEvent, SyncEventType, SyncHandler, SyncServer

__all__ = [
    # Primary API - compatible with amplifier-app-cli
    "SessionStore",
    "EventsLog",
    "read_events_summary",
    "is_top_level_session",
    "extract_session_mode",
    # Identity
    "IdentityProvider",
    "UserIdentity",
    "ConfigFileIdentityProvider",
    "IdentityContext",
    # Blocks
    "BlockType",
    "SessionBlock",
    "SessionCreatedData",
    "SessionUpdatedData",
    "MessageData",
    "EventData",
    "EventDataChunk",
    "RewindData",
    "BlockWriter",
    "SequenceAllocator",
    "SessionStateReader",
    # Storage
    "StorageConfig",
    "CosmosAuthMethod",
    "BlockStorage",
    "LocalBlockStorage",
    "CosmosBlockStorage",
    "HybridBlockStorage",
    "StorageError",
    "SessionNotFoundError",
    "AccessDeniedError",
    "AuthenticationError",
    # Sync types
    "SyncState",
    "SyncConflict",
    "ConflictResolution",
    # Analyst
    "SessionAnalyst",
    "SessionQuery",
    "SessionSummary",
    # Sync
    "SyncClient",
    "SyncEvent",
    "SyncEventType",
    "SyncHandler",
    "SyncServer",
    # Migration
    "MigrationStatus",
    "MigrationResult",
    "SessionSource",
    "SessionMigrator",
]

__version__ = "0.1.0"
