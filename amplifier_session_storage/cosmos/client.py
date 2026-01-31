"""
Cosmos DB client wrapper.

Provides a clean interface to Azure Cosmos DB with:
- Connection management
- Container access
- Query execution with user isolation
- Retry logic for transient failures
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from azure.cosmos import PartitionKey
from azure.cosmos.aio import ContainerProxy, CosmosClient, DatabaseProxy
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError

from ..exceptions import AuthenticationError, ConnectionError, StorageIOError

# Container names
SESSIONS_CONTAINER = "sessions"
TRANSCRIPT_CONTAINER = "transcript_messages"
EVENTS_CONTAINER = "events"
SYNC_STATE_CONTAINER = "sync_state"
CHUNKS_CONTAINER = "event_chunks"

# Sharing-related container names
SHARED_SESSIONS_CONTAINER = "shared_sessions"
ORGANIZATIONS_CONTAINER = "organizations"
TEAMS_CONTAINER = "teams"
USER_MEMBERSHIPS_CONTAINER = "user_memberships"

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds


@dataclass
class CosmosConfig:
    """Configuration for Cosmos DB connection.

    Attributes:
        endpoint: Cosmos DB account endpoint URL
        key: Cosmos DB account key or connection string
        database_name: Name of the database to use
        max_retries: Maximum retry attempts for transient failures
        retry_delay: Base delay between retries (seconds)
    """

    endpoint: str
    key: str
    database_name: str
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY

    @classmethod
    def from_env(cls, database_name: str = "amplifier_sessions") -> "CosmosConfig":
        """Create config from environment variables.

        Expected environment variables:
        - COSMOS_ENDPOINT: Cosmos DB account endpoint
        - COSMOS_KEY: Cosmos DB account key

        Args:
            database_name: Name of the database to use

        Returns:
            CosmosConfig instance

        Raises:
            AuthenticationError: If required environment variables are missing
        """
        endpoint = os.environ.get("COSMOS_ENDPOINT")
        key = os.environ.get("COSMOS_KEY")

        if not endpoint:
            raise AuthenticationError("cosmos", "COSMOS_ENDPOINT environment variable not set")
        if not key:
            raise AuthenticationError("cosmos", "COSMOS_KEY environment variable not set")

        return cls(endpoint=endpoint, key=key, database_name=database_name)


class CosmosClientWrapper:
    """Wrapper for Azure Cosmos DB async client.

    Manages connection lifecycle, provides container access,
    and handles retry logic for transient failures.

    Container partition keys:
    - sessions: /user_id
    - transcript_messages: /user_id_session_id (composite)
    - events: /user_id_session_id (composite)
    - sync_state: /user_id
    - event_chunks: /user_id_session_id (composite)
    """

    def __init__(self, config: CosmosConfig):
        """Initialize the Cosmos DB client wrapper.

        Args:
            config: Cosmos DB configuration
        """
        self.config = config
        self._client: CosmosClient | None = None
        self._database: DatabaseProxy | None = None
        self._containers: dict[str, ContainerProxy] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the Cosmos DB connection and ensure containers exist.

        Creates the database and containers if they don't exist.
        """
        if self._initialized:
            return

        try:
            client = CosmosClient(self.config.endpoint, credential=self.config.key)
            self._client = client

            # Create database if not exists
            self._database = await client.create_database_if_not_exists(
                id=self.config.database_name
            )

            # Create containers with appropriate partition keys
            await self._ensure_container(SESSIONS_CONTAINER, "/user_id")
            await self._ensure_container(TRANSCRIPT_CONTAINER, "/user_id_session_id")
            await self._ensure_container(EVENTS_CONTAINER, "/user_id_session_id")
            await self._ensure_container(SYNC_STATE_CONTAINER, "/user_id")
            await self._ensure_container(CHUNKS_CONTAINER, "/user_id_session_id")

            self._initialized = True

        except CosmosHttpResponseError as e:
            if e.status_code == 401:
                raise AuthenticationError(self.config.endpoint, str(e)) from e
            raise ConnectionError(self.config.endpoint, e) from e
        except Exception as e:
            raise ConnectionError(self.config.endpoint, e) from e

    async def _ensure_container(self, name: str, partition_key_path: str) -> None:
        """Ensure a container exists, creating if necessary."""
        if self._database is None:
            raise StorageIOError("ensure_container", cause=RuntimeError("Database not initialized"))

        container = await self._database.create_container_if_not_exists(
            id=name,
            partition_key=PartitionKey(path=partition_key_path),
        )
        self._containers[name] = container

    async def close(self) -> None:
        """Close the Cosmos DB connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._database = None
            self._containers = {}
            self._initialized = False

    async def __aenter__(self) -> "CosmosClientWrapper":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def _get_container(self, name: str) -> ContainerProxy:
        """Get a container proxy by name."""
        if not self._initialized:
            raise StorageIOError("get_container", cause=RuntimeError("Client not initialized"))
        if name not in self._containers:
            raise StorageIOError("get_container", cause=KeyError(f"Unknown container: {name}"))
        return self._containers[name]

    # =========================================================================
    # Container Access
    # =========================================================================

    @property
    def sessions(self) -> ContainerProxy:
        """Get the sessions container."""
        return self._get_container(SESSIONS_CONTAINER)

    @property
    def transcript_messages(self) -> ContainerProxy:
        """Get the transcript_messages container."""
        return self._get_container(TRANSCRIPT_CONTAINER)

    @property
    def events(self) -> ContainerProxy:
        """Get the events container."""
        return self._get_container(EVENTS_CONTAINER)

    @property
    def sync_state(self) -> ContainerProxy:
        """Get the sync_state container."""
        return self._get_container(SYNC_STATE_CONTAINER)

    @property
    def chunks(self) -> ContainerProxy:
        """Get the event_chunks container."""
        return self._get_container(CHUNKS_CONTAINER)

    # =========================================================================
    # CRUD Operations with Retry
    # =========================================================================

    async def create_item(
        self,
        container_name: str,
        item: dict[str, Any],
        partition_key: str,
    ) -> dict[str, Any]:
        """Create an item in a container with retry logic.

        Args:
            container_name: Name of the container
            item: Item to create
            partition_key: Partition key value

        Returns:
            Created item
        """
        container = self._get_container(container_name)
        return await self._with_retry(
            lambda: container.create_item(body=item, partition_key=partition_key)
        )

    async def upsert_item(
        self,
        container_name: str,
        item: dict[str, Any],
        partition_key: str,
    ) -> dict[str, Any]:
        """Upsert an item in a container with retry logic.

        Args:
            container_name: Name of the container
            item: Item to upsert
            partition_key: Partition key value

        Returns:
            Upserted item
        """
        container = self._get_container(container_name)
        return await self._with_retry(
            lambda: container.upsert_item(body=item, partition_key=partition_key)
        )

    async def read_item(
        self,
        container_name: str,
        item_id: str,
        partition_key: str,
    ) -> dict[str, Any] | None:
        """Read an item from a container.

        Args:
            container_name: Name of the container
            item_id: ID of the item to read
            partition_key: Partition key value

        Returns:
            Item or None if not found
        """
        container = self._get_container(container_name)
        try:
            return await self._with_retry(
                lambda: container.read_item(item=item_id, partition_key=partition_key)
            )
        except CosmosResourceNotFoundError:
            return None

    async def delete_item(
        self,
        container_name: str,
        item_id: str,
        partition_key: str,
    ) -> bool:
        """Delete an item from a container.

        Args:
            container_name: Name of the container
            item_id: ID of the item to delete
            partition_key: Partition key value

        Returns:
            True if deleted, False if not found
        """
        container = self._get_container(container_name)
        try:
            await self._with_retry(
                lambda: container.delete_item(item=item_id, partition_key=partition_key)
            )
            return True
        except CosmosResourceNotFoundError:
            return False

    async def query_items(
        self,
        container_name: str,
        query: str,
        parameters: list[dict[str, Any]] | None = None,
        partition_key: str | None = None,
        max_items: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query items from a container.

        CRITICAL: All queries MUST include user isolation in the WHERE clause.
        The partition key should always include user_id.

        Args:
            container_name: Name of the container
            query: SQL query string
            parameters: Query parameters
            partition_key: Optional partition key for single-partition queries
            max_items: Maximum items to return

        Returns:
            List of matching items
        """
        container = self._get_container(container_name)

        query_options: dict[str, Any] = {}
        if partition_key:
            query_options["partition_key"] = partition_key
        if max_items:
            query_options["max_item_count"] = max_items

        results: list[dict[str, Any]] = []
        async for item in container.query_items(
            query=query,
            parameters=parameters or [],
            **query_options,
        ):
            results.append(item)
            if max_items and len(results) >= max_items:
                break

        return results

    async def _with_retry(self, operation: Any) -> Any:
        """Execute an operation with retry logic for transient failures.

        Args:
            operation: Async callable to execute

        Returns:
            Result of the operation

        Raises:
            The original exception after max retries
        """
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                return await operation()
            except CosmosHttpResponseError as e:
                # Don't retry client errors (4xx)
                if 400 <= e.status_code < 500:
                    raise

                # Retry server errors (5xx) and rate limiting (429)
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)
            except Exception as e:
                # Don't retry unknown errors
                raise StorageIOError("cosmos_operation", cause=e) from e

        if last_error:
            raise StorageIOError("cosmos_operation", cause=last_error) from last_error
        raise StorageIOError("cosmos_operation", cause=RuntimeError("Unexpected retry failure"))


def make_partition_key(user_id: str, session_id: str | None = None) -> str:
    """Create a partition key value.

    For containers with composite partition keys (user_id_session_id),
    this creates the combined key value.

    Args:
        user_id: User ID
        session_id: Optional session ID for composite keys

    Returns:
        Partition key value
    """
    if session_id:
        return f"{user_id}_{session_id}"
    return user_id
