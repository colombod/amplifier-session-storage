"""
Cosmos DB block storage.

Stores blocks in Azure Cosmos DB with proper partitioning
for multi-tenant access and efficient queries.

Supports multiple authentication methods:
- Key-based authentication (if org policy allows)
- Azure AD via DefaultAzureCredential (recommended)
- Azure Managed Identity
- Service Principal
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from azure.cosmos import PartitionKey
from azure.cosmos.aio import ContainerProxy, CosmosClient, DatabaseProxy
from azure.cosmos.exceptions import CosmosResourceNotFoundError

from ..blocks.types import BlockType, SessionBlock
from .base import (
    AccessDeniedError,
    AuthenticationError,
    BlockStorage,
    CosmosAuthMethod,
    SessionNotFoundError,
    StorageConfig,
    StorageError,
)

logger = logging.getLogger(__name__)


def _get_credential(config: StorageConfig) -> Any:
    """Get the appropriate credential based on auth method.

    Args:
        config: Storage configuration with auth settings

    Returns:
        Credential object for Cosmos DB authentication

    Raises:
        AuthenticationError: If credential cannot be created
    """
    auth_method = config.cosmos_auth_method

    if auth_method == CosmosAuthMethod.KEY:
        if not config.cosmos_key:
            raise AuthenticationError("cosmos_key required for KEY authentication")
        return config.cosmos_key

    if auth_method == CosmosAuthMethod.DEFAULT_CREDENTIAL:
        try:
            from azure.identity.aio import DefaultAzureCredential

            return DefaultAzureCredential()
        except ImportError as e:
            raise AuthenticationError(
                "azure-identity package required for Azure AD authentication. "
                "Install with: pip install azure-identity"
            ) from e

    if auth_method == CosmosAuthMethod.MANAGED_IDENTITY:
        try:
            from azure.identity.aio import ManagedIdentityCredential

            # If client_id is provided, use user-assigned managed identity
            if config.azure_client_id:
                return ManagedIdentityCredential(client_id=config.azure_client_id)
            # Otherwise use system-assigned managed identity
            return ManagedIdentityCredential()
        except ImportError as e:
            raise AuthenticationError(
                "azure-identity package required for Managed Identity authentication. "
                "Install with: pip install azure-identity"
            ) from e

    if auth_method == CosmosAuthMethod.SERVICE_PRINCIPAL:
        if not all([config.azure_tenant_id, config.azure_client_id, config.azure_client_secret]):
            raise AuthenticationError(
                "azure_tenant_id, azure_client_id, and azure_client_secret "
                "required for SERVICE_PRINCIPAL authentication"
            )
        try:
            from azure.identity.aio import ClientSecretCredential

            return ClientSecretCredential(
                tenant_id=config.azure_tenant_id,
                client_id=config.azure_client_id,
                client_secret=config.azure_client_secret,
            )
        except ImportError as e:
            raise AuthenticationError(
                "azure-identity package required for Service Principal authentication. "
                "Install with: pip install azure-identity"
            ) from e

    raise AuthenticationError(f"Unsupported auth method: {auth_method}")


class CosmosBlockStorage(BlockStorage):
    """Cosmos DB block storage.

    Uses a single container with composite partition key for
    efficient multi-tenant access.

    Default partition key path: /partitionKey
    Partition key value: {user_id}_{session_id}

    This allows:
    - All blocks for a session in one partition (efficient reads)
    - User-scoped queries without cross-partition scans
    - Team/org access via secondary indexing

    Container schema:
    {
        "id": "{block_id}",
        "partitionKey": "{user_id}_{session_id}",
        "block_id": "{block_id}",
        "session_id": "{session_id}",
        "user_id": "{user_id}",
        "sequence": {int},
        "timestamp": "{iso_timestamp}",
        "device_id": "{device_id}",
        "block_type": "{type}",
        "data": {...},
        "checksum": "{sha256}",
        "size_bytes": {int},
        // Visibility fields for shared access
        "org_id": "{org_id}",  // nullable
        "visibility": "private|team|org|public",
        "team_ids": ["team1", "team2"],  // nullable
    }

    Indexing policy:
    - Included paths: /user_id, /session_id, /sequence, /block_type,
                      /org_id, /visibility, /team_ids/[]
    - Composite indexes for common queries
    """

    def __init__(self, config: StorageConfig) -> None:
        """Initialize Cosmos DB storage.

        Args:
            config: Storage configuration with Cosmos connection info
        """
        if not config.cosmos_endpoint:
            raise StorageError("Cosmos endpoint is required")

        self.config = config
        self.user_id = config.user_id
        self.org_id = config.org_id
        self._partition_key_path = config.cosmos_partition_key_path

        self._credential: Any = None
        self._client: CosmosClient | None = None
        self._database: DatabaseProxy | None = None
        self._container: ContainerProxy | None = None
        self._initialized = False

    def _get_partition_key(self, user_id: str, session_id: str) -> str:
        """Generate partition key value.

        Args:
            user_id: User ID
            session_id: Session ID

        Returns:
            Partition key value
        """
        return f"{user_id}_{session_id}"

    async def _ensure_initialized(self) -> None:
        """Ensure client and container are initialized."""
        if self._initialized:
            return

        # Get credential based on auth method
        self._credential = _get_credential(self.config)

        try:
            client = CosmosClient(
                self.config.cosmos_endpoint,  # type: ignore[arg-type]
                credential=self._credential,
            )
            self._client = client

            # Get or create database
            database = await client.create_database_if_not_exists(id=self.config.cosmos_database)
            self._database = database

            # Get or create container with partition key
            container = await database.create_container_if_not_exists(
                id=self.config.cosmos_container,
                partition_key=PartitionKey(path=self._partition_key_path),
                indexing_policy=self._get_indexing_policy(),
            )
            self._container = container

            self._initialized = True
            logger.info(
                f"Connected to Cosmos DB: {self.config.cosmos_endpoint} "
                f"(database={self.config.cosmos_database}, "
                f"container={self.config.cosmos_container}, "
                f"auth={self.config.cosmos_auth_method.value})"
            )

        except Exception as e:
            error_msg = str(e)
            if "unauthorized" in error_msg.lower() or "403" in error_msg:
                raise AuthenticationError(
                    f"Authentication failed for Cosmos DB. "
                    f"Ensure your identity has 'Cosmos DB Data Contributor' role. "
                    f"Error: {error_msg}"
                ) from e
            raise StorageError(f"Failed to connect to Cosmos DB: {error_msg}") from e

    def _get_indexing_policy(self) -> dict[str, Any]:
        """Get the indexing policy for the container."""
        return {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [
                {"path": "/data/*"},  # Don't index data payload
                {"path": '/"_etag"/?'},
            ],
            "compositeIndexes": [
                # User's sessions ordered by time
                [
                    {"path": "/user_id", "order": "ascending"},
                    {"path": "/timestamp", "order": "descending"},
                ],
                # Session blocks ordered by sequence
                [
                    {"path": "/session_id", "order": "ascending"},
                    {"path": "/sequence", "order": "ascending"},
                ],
                # Org-visible sessions
                [
                    {"path": "/org_id", "order": "ascending"},
                    {"path": "/visibility", "order": "ascending"},
                    {"path": "/timestamp", "order": "descending"},
                ],
            ],
        }

    async def write_block(self, block: SessionBlock) -> None:
        """Write a single block to Cosmos DB."""
        await self._ensure_initialized()

        # Verify ownership
        if block.user_id != self.user_id:
            raise AccessDeniedError("Cannot write blocks for another user")

        doc = self._block_to_document(block)
        await self._container.upsert_item(doc)  # type: ignore[union-attr]

    async def write_blocks(self, blocks: list[SessionBlock]) -> None:
        """Write multiple blocks to Cosmos DB.

        Uses batch operations for efficiency.
        """
        if not blocks:
            return

        await self._ensure_initialized()

        # Verify all blocks belong to the same session and user
        session_id = blocks[0].session_id
        if not all(b.session_id == session_id for b in blocks):
            raise StorageError("All blocks must belong to the same session")

        if not all(b.user_id == self.user_id for b in blocks):
            raise AccessDeniedError("Cannot write blocks for another user")

        # Write blocks in parallel (Cosmos handles concurrency)
        tasks = [
            self._container.upsert_item(self._block_to_document(block))  # type: ignore[union-attr]
            for block in blocks
        ]
        await asyncio.gather(*tasks)

    async def read_blocks(
        self,
        session_id: str,
        since_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[SessionBlock]:
        """Read blocks for a session from Cosmos DB."""
        await self._ensure_initialized()

        # Build query with access control
        query = """
            SELECT * FROM c
            WHERE c.session_id = @session_id
            AND (
                c.user_id = @user_id
                OR (c.visibility = 'public')
                OR (c.visibility = 'org' AND c.org_id = @org_id)
                OR (c.visibility = 'team' AND ARRAY_CONTAINS(c.team_ids, @user_id))
            )
        """
        params: list[dict[str, Any]] = [
            {"name": "@session_id", "value": session_id},
            {"name": "@user_id", "value": self.user_id},
            {"name": "@org_id", "value": self.org_id},
        ]

        if since_sequence is not None:
            query += " AND c.sequence > @since_sequence"
            params.append({"name": "@since_sequence", "value": since_sequence})

        query += " ORDER BY c.sequence ASC"

        # Execute query
        blocks: list[SessionBlock] = []
        async for doc in self._container.query_items(  # type: ignore[union-attr]
            query=query,
            parameters=params,
            max_item_count=limit or 1000,
        ):
            blocks.append(self._document_to_block(doc))
            if limit and len(blocks) >= limit:
                break

        return blocks

    async def get_latest_sequence(self, session_id: str) -> int:
        """Get the latest sequence number for a session."""
        await self._ensure_initialized()

        query = """
            SELECT VALUE MAX(c.sequence)
            FROM c
            WHERE c.session_id = @session_id
            AND c.user_id = @user_id
        """
        params = [
            {"name": "@session_id", "value": session_id},
            {"name": "@user_id", "value": self.user_id},
        ]

        async for result in self._container.query_items(  # type: ignore[union-attr]
            query=query, parameters=params
        ):
            return result or 0

        return 0

    async def list_sessions(
        self,
        project_slug: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List sessions visible to the current user.

        Returns sessions that are:
        - Owned by the user
        - Public
        - In the user's org (if org visibility)
        - Shared with user's teams
        """
        await self._ensure_initialized()

        # Query for SESSION_CREATED blocks (session headers)
        query = """
            SELECT c.session_id, c.user_id, c.timestamp, c.data,
                   c.org_id, c.visibility
            FROM c
            WHERE c.block_type = 'session_created'
            AND (
                c.user_id = @user_id
                OR c.visibility = 'public'
                OR (c.visibility = 'org' AND c.org_id = @org_id)
                OR (c.visibility = 'team' AND ARRAY_CONTAINS(c.team_ids, @user_id))
            )
        """
        params: list[dict[str, Any]] = [
            {"name": "@user_id", "value": self.user_id},
            {"name": "@org_id", "value": self.org_id},
        ]

        if project_slug:
            query += " AND c.data.project_slug = @project_slug"
            params.append({"name": "@project_slug", "value": project_slug})

        query += " ORDER BY c.timestamp DESC"
        query += f" OFFSET {offset} LIMIT {limit}"

        sessions: list[dict[str, Any]] = []
        async for doc in self._container.query_items(  # type: ignore[union-attr]
            query=query, parameters=params
        ):
            sessions.append(
                {
                    "session_id": doc["session_id"],
                    "user_id": doc["user_id"],
                    "created": doc["timestamp"],
                    "name": doc["data"].get("name"),
                    "project_slug": doc["data"].get("project_slug"),
                    "visibility": doc.get("visibility", "private"),
                    "org_id": doc.get("org_id"),
                }
            )

        return sessions

    async def delete_session(self, session_id: str) -> None:
        """Delete all blocks for a session.

        Only the owner can delete a session.
        """
        await self._ensure_initialized()

        # Verify ownership first
        query = """
            SELECT c.id, c.user_id, c.partitionKey
            FROM c
            WHERE c.session_id = @session_id
        """
        params = [{"name": "@session_id", "value": session_id}]

        items_to_delete: list[tuple[str, str]] = []  # (id, partition_key)
        found_any = False

        async for doc in self._container.query_items(  # type: ignore[union-attr]
            query=query, parameters=params
        ):
            found_any = True
            if doc["user_id"] != self.user_id:
                raise AccessDeniedError("Cannot delete another user's session")
            items_to_delete.append((doc["id"], doc["partitionKey"]))

        if not found_any:
            raise SessionNotFoundError(f"Session not found: {session_id}")

        # Delete all items
        for item_id, partition_key in items_to_delete:
            try:
                await self._container.delete_item(  # type: ignore[union-attr]
                    item=item_id, partition_key=partition_key
                )
            except CosmosResourceNotFoundError:
                pass  # Already deleted

    async def close(self) -> None:
        """Close the Cosmos client."""
        if self._client:
            await self._client.close()
            self._client = None
            self._database = None
            self._container = None
            self._initialized = False

        # Close credential if it has a close method (AAD credentials do)
        if self._credential and hasattr(self._credential, "close"):
            await self._credential.close()
            self._credential = None

    def _block_to_document(self, block: SessionBlock) -> dict[str, Any]:
        """Convert a block to a Cosmos document."""
        doc = block.to_dict()

        # Add partition key
        doc["partitionKey"] = self._get_partition_key(block.user_id, block.session_id)

        # Add visibility fields
        if block.block_type == BlockType.SESSION_CREATED:
            doc["visibility"] = block.data.get("visibility", "private")
            doc["org_id"] = block.data.get("org_id") or self.org_id
            doc["team_ids"] = block.data.get("team_ids", [])
        else:
            # Inherit visibility from session
            doc["visibility"] = "private"
            doc["org_id"] = self.org_id
            doc["team_ids"] = []

        return doc

    def _document_to_block(self, doc: dict[str, Any]) -> SessionBlock:
        """Convert a Cosmos document to a block."""
        return SessionBlock.from_dict(doc)
