"""
Abstract block storage interface.

Defines the contract that all storage backends must implement.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..blocks.types import SessionBlock


class CosmosAuthMethod(Enum):
    """Authentication method for Cosmos DB.

    KEY: Use connection string or key (not recommended for production)
    DEFAULT_CREDENTIAL: Use Azure DefaultAzureCredential (recommended)
        - Works with Azure CLI, Managed Identity, Environment variables, etc.
    MANAGED_IDENTITY: Use Azure Managed Identity explicitly
    SERVICE_PRINCIPAL: Use Service Principal with client_id/client_secret
    """

    KEY = "key"
    DEFAULT_CREDENTIAL = "default_credential"
    MANAGED_IDENTITY = "managed_identity"
    SERVICE_PRINCIPAL = "service_principal"


@dataclass
class StorageConfig:
    """Configuration for block storage.

    Supports multiple authentication methods for Cosmos DB:
    - Key-based (for development/testing, if org policy allows)
    - Azure AD via DefaultAzureCredential (recommended)
    - Azure Managed Identity (for Azure-hosted services)
    - Service Principal (for CI/CD and automation)

    Configuration can be provided directly or via environment variables:

    Environment Variables:
        AMPLIFIER_COSMOS_ENDPOINT: Cosmos DB endpoint URL
        AMPLIFIER_COSMOS_KEY: Cosmos DB key (if using key auth)
        AMPLIFIER_COSMOS_DATABASE: Database name (default: amplifier-db)
        AMPLIFIER_COSMOS_CONTAINER: Container name (default: items)
        AMPLIFIER_COSMOS_AUTH_METHOD: Auth method (default: default_credential)
        AZURE_TENANT_ID: Azure tenant ID (for service principal)
        AZURE_CLIENT_ID: Azure client ID (for service principal/managed identity)
        AZURE_CLIENT_SECRET: Azure client secret (for service principal)

    Attributes:
        user_id: The authenticated user ID
        org_id: Optional organization ID for shared access
        enable_sync: Whether to enable cloud sync

        cosmos_endpoint: Cosmos DB endpoint URL
        cosmos_auth_method: Authentication method (default: DEFAULT_CREDENTIAL)
        cosmos_key: Cosmos DB key (only for KEY auth method)
        cosmos_database: Cosmos DB database name
        cosmos_container: Cosmos DB container name
        cosmos_partition_key_path: Partition key path in container

        azure_tenant_id: Azure tenant ID (for SERVICE_PRINCIPAL)
        azure_client_id: Azure client/app ID (for SERVICE_PRINCIPAL/MANAGED_IDENTITY)
        azure_client_secret: Azure client secret (for SERVICE_PRINCIPAL)

        local_path: Path for local storage
    """

    user_id: str
    org_id: str | None = None
    enable_sync: bool = False

    # Cosmos DB connection settings
    cosmos_endpoint: str | None = None
    cosmos_auth_method: CosmosAuthMethod = CosmosAuthMethod.DEFAULT_CREDENTIAL
    cosmos_key: str | None = None  # Only used if auth_method is KEY
    cosmos_database: str = "amplifier-db"
    cosmos_container: str = "items"
    cosmos_partition_key_path: str = "/partitionKey"

    # Azure AD authentication settings (for SERVICE_PRINCIPAL)
    azure_tenant_id: str | None = None
    azure_client_id: str | None = None
    azure_client_secret: str | None = None

    # Local storage settings
    local_path: str | None = None

    # Additional options
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_environment(cls, user_id: str, org_id: str | None = None) -> StorageConfig:
        """Create configuration from environment variables.

        Args:
            user_id: The authenticated user ID
            org_id: Optional organization ID

        Returns:
            StorageConfig populated from environment variables
        """
        auth_method_str = os.environ.get("AMPLIFIER_COSMOS_AUTH_METHOD", "default_credential")
        try:
            auth_method = CosmosAuthMethod(auth_method_str.lower())
        except ValueError:
            auth_method = CosmosAuthMethod.DEFAULT_CREDENTIAL

        return cls(
            user_id=user_id,
            org_id=org_id,
            cosmos_endpoint=os.environ.get("AMPLIFIER_COSMOS_ENDPOINT"),
            cosmos_auth_method=auth_method,
            cosmos_key=os.environ.get("AMPLIFIER_COSMOS_KEY"),
            cosmos_database=os.environ.get("AMPLIFIER_COSMOS_DATABASE", "amplifier-db"),
            cosmos_container=os.environ.get("AMPLIFIER_COSMOS_CONTAINER", "items"),
            cosmos_partition_key_path=os.environ.get(
                "AMPLIFIER_COSMOS_PARTITION_KEY", "/partitionKey"
            ),
            azure_tenant_id=os.environ.get("AZURE_TENANT_ID"),
            azure_client_id=os.environ.get("AZURE_CLIENT_ID"),
            azure_client_secret=os.environ.get("AZURE_CLIENT_SECRET"),
            local_path=os.environ.get("AMPLIFIER_LOCAL_STORAGE_PATH"),
            enable_sync=os.environ.get("AMPLIFIER_ENABLE_SYNC", "").lower() == "true",
        )


class BlockStorage(ABC):
    """Abstract interface for block storage.

    All storage implementations (local, cosmos, hybrid) must
    implement this interface.
    """

    @abstractmethod
    async def write_block(self, block: SessionBlock) -> None:
        """Write a single block to storage.

        Args:
            block: The block to write

        Raises:
            StorageError: If write fails
        """
        ...

    @abstractmethod
    async def write_blocks(self, blocks: list[SessionBlock]) -> None:
        """Write multiple blocks atomically.

        Args:
            blocks: List of blocks to write

        Raises:
            StorageError: If write fails
        """
        ...

    @abstractmethod
    async def read_blocks(
        self,
        session_id: str,
        since_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[SessionBlock]:
        """Read blocks for a session.

        Args:
            session_id: The session ID
            since_sequence: Optional - only return blocks with seq > this
            limit: Optional - maximum number of blocks to return

        Returns:
            List of blocks sorted by sequence

        Raises:
            StorageError: If read fails
        """
        ...

    @abstractmethod
    async def get_latest_sequence(self, session_id: str) -> int:
        """Get the latest sequence number for a session.

        Args:
            session_id: The session ID

        Returns:
            Latest sequence number, or 0 if no blocks exist
        """
        ...

    @abstractmethod
    async def list_sessions(
        self,
        project_slug: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List sessions for the current user.

        Args:
            project_slug: Optional - filter by project
            limit: Maximum number of sessions
            offset: Pagination offset

        Returns:
            List of session metadata dictionaries
        """
        ...

    @abstractmethod
    async def delete_session(self, session_id: str) -> None:
        """Delete all blocks for a session.

        Args:
            session_id: The session ID

        Raises:
            StorageError: If delete fails
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the storage connection and cleanup resources."""
        ...


class StorageError(Exception):
    """Base exception for storage errors."""

    pass


class SessionNotFoundError(StorageError):
    """Raised when a session is not found."""

    pass


class AccessDeniedError(StorageError):
    """Raised when access to a resource is denied."""

    pass


class AuthenticationError(StorageError):
    """Raised when authentication fails."""

    pass
