"""
Amplifier Session Storage

A block-based session storage system for Amplifier with:
- Event-sourced session blocks for efficient sync
- Local file-based storage for offline operation
- Cosmos DB cloud storage for multi-tenant persistence
- Identity management for user/org/team access control
- Migration tools for existing sessions

Architecture Overview:
    Sessions are stored as streams of immutable blocks:
    - SESSION_CREATED: Initial session metadata
    - SESSION_UPDATED: Metadata changes
    - MESSAGE: Conversation messages
    - EVENT: Tool calls, LLM responses, etc.
    - EVENT_CHUNK: Large event continuations
    - REWIND: History truncation markers

    This design enables:
    - Efficient sync (only new blocks transferred)
    - Offline-first operation (local + cloud)
    - Multi-device support (sequence-based merging)
    - Team visibility (access control at block level)

Basic Usage:
    >>> from amplifier_session_storage import LocalBlockStorage, StorageConfig
    >>> config = StorageConfig(user_id="user-123")
    >>> storage = LocalBlockStorage(config)
    >>> # Write blocks via BlockWriter, read via storage.read_blocks()

Cloud Storage:
    >>> from amplifier_session_storage import CosmosBlockStorage, StorageConfig
    >>> config = StorageConfig(
    ...     user_id="user-123",
    ...     cosmos_endpoint="https://example.cosmos.azure.com",
    ...     cosmos_key="secret-key",
    ... )
    >>> storage = CosmosBlockStorage(config)

Identity Management:
    >>> from amplifier_session_storage import IdentityContext
    >>> IdentityContext.initialize()
    >>> user_id = IdentityContext.get_user_id()

Migration from Legacy Format:
    >>> from amplifier_session_storage import SessionMigrator, LocalBlockStorage
    >>> storage = LocalBlockStorage(config)
    >>> migrator = SessionMigrator(storage, user_id="user-123")
    >>> sources = await migrator.discover_sessions(Path("~/.amplifier/sessions"))
    >>> batch = await migrator.migrate_batch(sources)
"""

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
