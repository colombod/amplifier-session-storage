"""
Block storage backends.

Provides both local file storage and Cosmos DB storage
for session blocks, with consistent interfaces.

Authentication Methods:
    For Cosmos DB, multiple authentication methods are supported:
    - KEY: Connection string or key (not recommended for production)
    - DEFAULT_CREDENTIAL: Azure DefaultAzureCredential (recommended)
    - MANAGED_IDENTITY: Azure Managed Identity
    - SERVICE_PRINCIPAL: Service Principal with client secret

Example:
    >>> from amplifier_session_storage.storage import (
    ...     StorageConfig, CosmosAuthMethod, CosmosBlockStorage
    ... )
    >>> config = StorageConfig(
    ...     user_id="user-123",
    ...     cosmos_endpoint="https://example.cosmos.azure.com:443/",
    ...     cosmos_auth_method=CosmosAuthMethod.DEFAULT_CREDENTIAL,
    ... )
    >>> storage = CosmosBlockStorage(config)
"""

from .base import (
    AccessDeniedError,
    AuthenticationError,
    BlockStorage,
    CosmosAuthMethod,
    SessionNotFoundError,
    StorageConfig,
    StorageError,
)
from .cosmos import CosmosBlockStorage
from .hybrid import (
    ConflictResolution,
    HybridBlockStorage,
    SyncConflict,
    SyncFilter,
    SyncPolicy,
    SyncState,
)
from .local import LocalBlockStorage

__all__ = [
    # Configuration
    "StorageConfig",
    "CosmosAuthMethod",
    # Storage implementations
    "BlockStorage",
    "LocalBlockStorage",
    "CosmosBlockStorage",
    "HybridBlockStorage",
    # Sync types
    "SyncState",
    "SyncConflict",
    "ConflictResolution",
    "SyncPolicy",
    "SyncFilter",
    # Exceptions
    "StorageError",
    "SessionNotFoundError",
    "AccessDeniedError",
    "AuthenticationError",
]
