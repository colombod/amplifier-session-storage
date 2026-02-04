"""
Cosmos DB file-format storage.

Mirrors CLI file format to Cosmos DB containers:
- sessions: Session metadata (mirrors metadata.json)
- transcripts: Message documents (mirrors transcript.jsonl lines)
- events: Event documents (mirrors events.jsonl lines)

This storage is designed for syncing CLI sessions to cloud,
NOT as a standalone storage implementation.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from azure.cosmos import PartitionKey
from azure.cosmos.aio import ContainerProxy, CosmosClient, DatabaseProxy
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError
from azure.identity.aio import DefaultAzureCredential

from ..exceptions import AuthenticationError, StorageConnectionError, StorageIOError

logger = logging.getLogger(__name__)

# Container names for file-format storage
SESSIONS_CONTAINER = "sessions"
TRANSCRIPTS_CONTAINER = "transcripts"
EVENTS_CONTAINER = "events"

# Auth methods
AUTH_KEY = "key"
AUTH_DEFAULT_CREDENTIAL = "default_credential"


@dataclass
class CosmosFileConfig:
    """Configuration for Cosmos file storage.

    Attributes:
        endpoint: Cosmos DB account endpoint URL
        database_name: Name of the database
        auth_method: Authentication method ('key' or 'default_credential')
        key: Cosmos DB account key (only needed if auth_method='key')
    """

    endpoint: str
    database_name: str
    auth_method: str = AUTH_DEFAULT_CREDENTIAL
    key: str | None = None

    @classmethod
    def from_env(cls) -> CosmosFileConfig:
        """Create config from environment variables.

        Expected environment variables:
        - AMPLIFIER_COSMOS_ENDPOINT: Cosmos DB account endpoint
        - AMPLIFIER_COSMOS_DATABASE: Database name
        - AMPLIFIER_COSMOS_AUTH_METHOD: 'key' or 'default_credential' (default)
        - AMPLIFIER_COSMOS_KEY: Account key (only if auth_method='key')
        """
        endpoint = os.environ.get("AMPLIFIER_COSMOS_ENDPOINT")
        database = os.environ.get("AMPLIFIER_COSMOS_DATABASE", "amplifier-db")
        auth_method = os.environ.get("AMPLIFIER_COSMOS_AUTH_METHOD", AUTH_DEFAULT_CREDENTIAL)
        key = os.environ.get("AMPLIFIER_COSMOS_KEY")

        if not endpoint:
            raise AuthenticationError(
                "cosmos", "AMPLIFIER_COSMOS_ENDPOINT environment variable not set"
            )

        if auth_method == AUTH_KEY and not key:
            raise AuthenticationError(
                "cosmos", "AMPLIFIER_COSMOS_KEY required when auth_method='key'"
            )

        return cls(
            endpoint=endpoint,
            database_name=database,
            auth_method=auth_method,
            key=key,
        )


class CosmosFileStorage:
    """Cosmos DB storage that mirrors CLI file format.

    Unlike the full SessionStorage ABC implementations, this is a simpler
    storage designed specifically for syncing CLI sessions to cloud.

    Key differences from CosmosDBStorage:
    - Documents mirror JSONL line format exactly
    - Partition key strategy optimized for project-level queries
    - Optimized for bulk upload of file contents
    - No complex abstractions - just file line sync

    Container partition keys:
    - sessions: /user_id
    - transcripts: /partition_key (= "{user_id}|{project_slug}|{session_id}")
    - events: /partition_key (same format)
    """

    def __init__(self, config: CosmosFileConfig):
        """Initialize Cosmos file storage.

        Args:
            config: Cosmos configuration
        """
        self.config = config
        self._client: CosmosClient | None = None
        self._credential: DefaultAzureCredential | None = None
        self._database: DatabaseProxy | None = None
        self._containers: dict[str, ContainerProxy] = {}
        self._initialized = False

    @classmethod
    async def create(cls, config: CosmosFileConfig | None = None) -> CosmosFileStorage:
        """Create and initialize a CosmosFileStorage instance.

        Args:
            config: Cosmos configuration (defaults to env vars)

        Returns:
            Initialized CosmosFileStorage instance
        """
        if config is None:
            config = CosmosFileConfig.from_env()

        storage = cls(config)
        await storage.initialize()
        return storage

    async def initialize(self) -> None:
        """Initialize connection and ensure containers exist."""
        if self._initialized:
            return

        try:
            # Create client based on auth method
            if self.config.auth_method == AUTH_KEY:
                self._client = CosmosClient(
                    self.config.endpoint,
                    credential=self.config.key,
                )
            else:
                # Use DefaultAzureCredential for managed identity, etc.
                self._credential = DefaultAzureCredential()
                self._client = CosmosClient(
                    self.config.endpoint,
                    credential=self._credential,
                )

            # Create database if not exists
            if self._client is None:
                raise StorageIOError("initialize", cause=RuntimeError("Client creation failed"))
            self._database = await self._client.create_database_if_not_exists(
                id=self.config.database_name
            )

            # Create containers with appropriate partition keys
            await self._ensure_container(SESSIONS_CONTAINER, "/user_id")
            await self._ensure_container(TRANSCRIPTS_CONTAINER, "/partition_key")
            await self._ensure_container(EVENTS_CONTAINER, "/partition_key")

            self._initialized = True
            logger.info(f"Cosmos file storage initialized: {self.config.endpoint}")

        except CosmosHttpResponseError as e:
            if e.status_code == 401 or e.status_code == 403:
                raise AuthenticationError(self.config.endpoint, str(e)) from e
            raise StorageConnectionError(self.config.endpoint, e) from e
        except Exception as e:
            raise StorageConnectionError(self.config.endpoint, e) from e

    async def _ensure_container(self, name: str, partition_key_path: str) -> None:
        """Ensure a container exists."""
        if self._database is None:
            raise StorageIOError("ensure_container", cause=RuntimeError("Database not initialized"))

        container = await self._database.create_container_if_not_exists(
            id=name,
            partition_key=PartitionKey(path=partition_key_path),
        )
        self._containers[name] = container

    async def close(self) -> None:
        """Close the connection."""
        if self._client:
            await self._client.close()
            self._client = None

        if self._credential:
            await self._credential.close()
            self._credential = None

        self._database = None
        self._containers = {}
        self._initialized = False

    async def __aenter__(self) -> CosmosFileStorage:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def _get_container(self, name: str) -> ContainerProxy:
        """Get a container by name."""
        if not self._initialized:
            raise StorageIOError("get_container", cause=RuntimeError("Storage not initialized"))
        if name not in self._containers:
            raise StorageIOError("get_container", cause=KeyError(f"Unknown container: {name}"))
        return self._containers[name]

    @staticmethod
    def make_partition_key(user_id: str, project_slug: str, session_id: str) -> str:
        """Create composite partition key for transcripts/events."""
        return f"{user_id}|{project_slug}|{session_id}"

    # =========================================================================
    # Session Metadata Operations
    # =========================================================================

    async def upsert_session_metadata(
        self,
        user_id: str,
        host_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """Upsert session metadata to Cosmos.

        Args:
            user_id: User ID (partition key)
            host_id: Host/machine ID where session originated
            metadata: Session metadata dict (mirrors metadata.json)
        """
        container = self._get_container(SESSIONS_CONTAINER)

        doc = {
            **metadata,
            "id": metadata.get("session_id"),
            "user_id": user_id,
            "host_id": host_id,
            "_type": "session",
            "synced_at": datetime.now(UTC).isoformat(),
        }

        # Don't pass partition_key - SDK extracts it from doc["user_id"]
        await container.upsert_item(body=doc)

    async def get_session_metadata(
        self,
        user_id: str,
        session_id: str,
    ) -> dict[str, Any] | None:
        """Get session metadata from Cosmos."""
        container = self._get_container(SESSIONS_CONTAINER)

        try:
            # Query by user_id and session_id since we don't have partition_key access
            query = "SELECT * FROM c WHERE c.user_id = @user_id AND c.id = @session_id"
            params = [
                {"name": "@user_id", "value": user_id},
                {"name": "@session_id", "value": session_id},
            ]
            async for doc in container.query_items(query=query, parameters=params):
                return doc
            return None
        except CosmosResourceNotFoundError:
            return None

    async def list_sessions(
        self,
        user_id: str,
        project_slug: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List sessions for a user, optionally filtered by project."""
        container = self._get_container(SESSIONS_CONTAINER)

        if project_slug:
            query = (
                "SELECT * FROM c WHERE c.user_id = @user_id "
                "AND c.project_slug = @project_slug "
                "ORDER BY c.updated DESC"
            )
            params = [
                {"name": "@user_id", "value": user_id},
                {"name": "@project_slug", "value": project_slug},
            ]
        else:
            query = "SELECT * FROM c WHERE c.user_id = @user_id ORDER BY c.updated DESC"
            params = [{"name": "@user_id", "value": user_id}]

        results: list[dict[str, Any]] = []
        async for item in container.query_items(
            query=query,
            parameters=params,
            max_item_count=limit,
        ):
            results.append(item)
            if len(results) >= limit:
                break

        return results

    async def delete_session(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
    ) -> bool:
        """Delete session and all its data from Cosmos."""
        partition_key = self.make_partition_key(user_id, project_slug, session_id)

        # Delete transcript messages
        transcripts = self._get_container(TRANSCRIPTS_CONTAINER)
        async for doc in transcripts.query_items(
            query="SELECT c.id, c.partition_key FROM c WHERE c.partition_key = @pk",
            parameters=[{"name": "@pk", "value": partition_key}],
        ):
            try:
                # Use query to find and delete - SDK extracts partition key from doc
                await transcripts.delete_item(item=doc["id"], partition_key=doc["partition_key"])
            except CosmosResourceNotFoundError:
                pass

        # Delete events
        events = self._get_container(EVENTS_CONTAINER)
        async for doc in events.query_items(
            query="SELECT c.id, c.partition_key FROM c WHERE c.partition_key = @pk",
            parameters=[{"name": "@pk", "value": partition_key}],
        ):
            try:
                await events.delete_item(item=doc["id"], partition_key=doc["partition_key"])
            except CosmosResourceNotFoundError:
                pass

        # Delete session metadata - query first to get user_id
        sessions = self._get_container(SESSIONS_CONTAINER)
        try:
            # Query to find then delete
            query = "SELECT c.id, c.user_id FROM c WHERE c.id = @sid AND c.user_id = @uid"
            params = [
                {"name": "@sid", "value": session_id},
                {"name": "@uid", "value": user_id},
            ]
            async for doc in sessions.query_items(query=query, parameters=params):
                await sessions.delete_item(item=doc["id"], partition_key=doc["user_id"])
                return True
            return False
        except CosmosResourceNotFoundError:
            return False

    # =========================================================================
    # Transcript Operations
    # =========================================================================

    async def sync_transcript_lines(
        self,
        user_id: str,
        host_id: str,
        project_slug: str,
        session_id: str,
        lines: list[dict[str, Any]],
        start_sequence: int = 0,
    ) -> int:
        """Sync transcript lines to Cosmos.

        Documents are idempotent: same session_id + sequence = same document ID.
        Re-pushing the same transcript message will upsert (update) not duplicate.

        Args:
            user_id: User ID
            host_id: Host/machine ID where session originated
            project_slug: Project slug
            session_id: Session ID
            lines: List of message dicts (each = one JSONL line)
            start_sequence: Starting sequence number

        Returns:
            Number of lines synced
        """
        if not lines:
            return 0

        container = self._get_container(TRANSCRIPTS_CONTAINER)
        partition_key = self.make_partition_key(user_id, project_slug, session_id)

        synced = 0
        for i, line in enumerate(lines):
            sequence = start_sequence + i

            # Document ID is deterministic: same session + sequence = same doc
            # This ensures idempotent upserts - pushing same message twice won't duplicate
            doc_id = f"{session_id}_msg_{sequence}"

            # Get timestamp from the line data for tracking
            ts = line.get("ts") or line.get("timestamp")

            doc = {
                **line,
                "id": doc_id,
                "partition_key": partition_key,
                "user_id": user_id,
                "host_id": host_id,
                "project_slug": project_slug,
                "session_id": session_id,
                "sequence": sequence,
                "ts": ts,  # Preserve/add ts for resume queries
                "_type": "transcript_message",
                "synced_at": datetime.now(UTC).isoformat(),
            }

            # Upsert ensures idempotency - same ID = update, not create
            await container.upsert_item(body=doc)
            synced += 1

        return synced

    async def get_transcript_lines(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
        after_sequence: int = -1,
    ) -> list[dict[str, Any]]:
        """Get transcript lines from Cosmos.

        Args:
            after_sequence: Only return lines after this sequence number
        """
        container = self._get_container(TRANSCRIPTS_CONTAINER)
        partition_key = self.make_partition_key(user_id, project_slug, session_id)

        query = (
            "SELECT * FROM c WHERE c.partition_key = @pk "
            "AND c.sequence > @after_seq ORDER BY c.sequence"
        )
        params = [
            {"name": "@pk", "value": partition_key},
            {"name": "@after_seq", "value": after_sequence},
        ]

        results: list[dict[str, Any]] = []
        async for item in container.query_items(
            query=query,
            parameters=params,
        ):
            results.append(item)

        return results

    async def get_transcript_count(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
    ) -> int:
        """Get count of transcript messages in Cosmos."""
        container = self._get_container(TRANSCRIPTS_CONTAINER)
        partition_key = self.make_partition_key(user_id, project_slug, session_id)

        query = "SELECT VALUE COUNT(1) FROM c WHERE c.partition_key = @pk"
        params = [{"name": "@pk", "value": partition_key}]

        results: list[Any] = []
        async for item in container.query_items(
            query=query,
            parameters=params,
        ):
            results.append(item)

        return int(results[0]) if results else 0

    async def get_last_transcript_ts(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
    ) -> str | None:
        """Get the timestamp of the last transcript message.

        Returns:
            ISO timestamp string of last message, or None if no messages exist.
        """
        container = self._get_container(TRANSCRIPTS_CONTAINER)
        partition_key = self.make_partition_key(user_id, project_slug, session_id)

        # Get the message with the highest sequence number
        query = (
            "SELECT TOP 1 c.ts, c.timestamp, c.synced_at FROM c "
            "WHERE c.partition_key = @pk "
            "ORDER BY c.sequence DESC"
        )
        params = [{"name": "@pk", "value": partition_key}]

        async for item in container.query_items(query=query, parameters=params):
            # Return ts (from original data) or timestamp or synced_at
            return item.get("ts") or item.get("timestamp") or item.get("synced_at")

        return None

    async def get_last_transcript_sequence(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
    ) -> int:
        """Get the sequence number of the last transcript message.

        Returns:
            Last sequence number, or -1 if no messages exist.
        """
        container = self._get_container(TRANSCRIPTS_CONTAINER)
        partition_key = self.make_partition_key(user_id, project_slug, session_id)

        query = (
            "SELECT TOP 1 c.sequence FROM c WHERE c.partition_key = @pk ORDER BY c.sequence DESC"
        )
        params = [{"name": "@pk", "value": partition_key}]

        async for item in container.query_items(query=query, parameters=params):
            return item.get("sequence", -1)

        return -1

    # =========================================================================
    # Event Operations
    # =========================================================================

    async def sync_event_lines(
        self,
        user_id: str,
        host_id: str,
        project_slug: str,
        session_id: str,
        lines: list[dict[str, Any]],
        start_sequence: int = 0,
    ) -> int:
        """Sync event lines to Cosmos.

        Documents are idempotent: same session_id + sequence = same document ID.
        Re-pushing the same event will upsert (update) not duplicate.

        For large events (>400KB), stores summary only with data_truncated flag.

        Args:
            user_id: User ID
            host_id: Host/machine ID where session originated
            project_slug: Project slug
            session_id: Session ID
            lines: List of event dicts (each = one JSONL line)
            start_sequence: Starting sequence number

        Returns:
            Number of lines synced
        """
        if not lines:
            return 0

        container = self._get_container(EVENTS_CONTAINER)
        partition_key = self.make_partition_key(user_id, project_slug, session_id)

        synced = 0
        for i, line in enumerate(lines):
            sequence = start_sequence + i

            # Document ID is deterministic: same session + sequence = same doc
            # This ensures idempotent upserts - pushing same event twice won't duplicate
            doc_id = f"{session_id}_evt_{sequence}"

            # Check event size - Cosmos has 2MB limit, we use 400KB threshold
            line_json = json.dumps(line)
            data_size = len(line_json.encode("utf-8"))

            if data_size > 400 * 1024:  # 400KB
                # Store summary only for large events
                doc = {
                    "id": doc_id,
                    "partition_key": partition_key,
                    "user_id": user_id,
                    "host_id": host_id,
                    "project_slug": project_slug,
                    "session_id": session_id,
                    "sequence": sequence,
                    "ts": line.get("ts"),
                    "lvl": line.get("lvl"),
                    "event": line.get("event"),
                    "turn": line.get("turn"),
                    "data_truncated": True,
                    "data_size_bytes": data_size,
                    "_type": "event",
                    "synced_at": datetime.now(UTC).isoformat(),
                }
            else:
                # Store full event
                doc = {
                    **line,
                    "id": doc_id,
                    "partition_key": partition_key,
                    "user_id": user_id,
                    "host_id": host_id,
                    "project_slug": project_slug,
                    "session_id": session_id,
                    "sequence": sequence,
                    "data_truncated": False,
                    "data_size_bytes": data_size,
                    "_type": "event",
                    "synced_at": datetime.now(UTC).isoformat(),
                }

            # Upsert ensures idempotency - same ID = update, not create
            await container.upsert_item(body=doc)
            synced += 1

        return synced

    async def get_event_lines(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
        after_sequence: int = -1,
    ) -> list[dict[str, Any]]:
        """Get event lines from Cosmos (summaries, not full data)."""
        container = self._get_container(EVENTS_CONTAINER)
        partition_key = self.make_partition_key(user_id, project_slug, session_id)

        # Only select summary fields, not full data
        query = (
            "SELECT c.id, c.sequence, c.ts, c.lvl, c.event, c.turn, "
            "c.data_truncated, c.data_size_bytes "
            "FROM c WHERE c.partition_key = @pk "
            "AND c.sequence > @after_seq ORDER BY c.sequence"
        )
        params = [
            {"name": "@pk", "value": partition_key},
            {"name": "@after_seq", "value": after_sequence},
        ]

        results: list[dict[str, Any]] = []
        async for item in container.query_items(
            query=query,
            parameters=params,
        ):
            results.append(item)

        return results

    async def get_event_count(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
    ) -> int:
        """Get count of events in Cosmos."""
        container = self._get_container(EVENTS_CONTAINER)
        partition_key = self.make_partition_key(user_id, project_slug, session_id)

        query = "SELECT VALUE COUNT(1) FROM c WHERE c.partition_key = @pk"
        params = [{"name": "@pk", "value": partition_key}]

        results: list[Any] = []
        async for item in container.query_items(
            query=query,
            parameters=params,
        ):
            results.append(item)

        return int(results[0]) if results else 0

    async def get_last_event_ts(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
    ) -> str | None:
        """Get the timestamp of the last event.

        Returns:
            ISO timestamp string of last event, or None if no events exist.
        """
        container = self._get_container(EVENTS_CONTAINER)
        partition_key = self.make_partition_key(user_id, project_slug, session_id)

        # Get the event with the highest sequence number
        query = (
            "SELECT TOP 1 c.ts, c.synced_at FROM c "
            "WHERE c.partition_key = @pk "
            "ORDER BY c.sequence DESC"
        )
        params = [{"name": "@pk", "value": partition_key}]

        async for item in container.query_items(query=query, parameters=params):
            # Return ts (original event timestamp) or synced_at as fallback
            return item.get("ts") or item.get("synced_at")

        return None

    async def get_last_event_sequence(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
    ) -> int:
        """Get the sequence number of the last event.

        Returns:
            Last sequence number, or -1 if no events exist.
        """
        container = self._get_container(EVENTS_CONTAINER)
        partition_key = self.make_partition_key(user_id, project_slug, session_id)

        query = (
            "SELECT TOP 1 c.sequence FROM c WHERE c.partition_key = @pk ORDER BY c.sequence DESC"
        )
        params = [{"name": "@pk", "value": partition_key}]

        async for item in container.query_items(query=query, parameters=params):
            return item.get("sequence", -1)

        return -1

    # =========================================================================
    # Sync Status (for daemon resume)
    # =========================================================================

    async def get_sync_status(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
    ) -> dict[str, Any]:
        """Get sync status for a session.

        Returns all information needed for daemon to resume sync:
        - Whether session exists
        - Last event sequence and timestamp
        - Last transcript sequence and timestamp

        Returns:
            Dict with sync status information
        """
        # Check if session exists
        session = await self.get_session_metadata(user_id, session_id)
        session_exists = session is not None

        # Get last event info
        last_event_sequence = await self.get_last_event_sequence(user_id, project_slug, session_id)
        last_event_ts = await self.get_last_event_ts(user_id, project_slug, session_id)

        # Get last transcript info
        last_transcript_sequence = await self.get_last_transcript_sequence(
            user_id, project_slug, session_id
        )
        last_transcript_ts = await self.get_last_transcript_ts(user_id, project_slug, session_id)

        return {
            "session_exists": session_exists,
            "last_event_sequence": last_event_sequence,
            "last_event_ts": last_event_ts,
            "event_count": last_event_sequence + 1 if last_event_sequence >= 0 else 0,
            "last_transcript_sequence": last_transcript_sequence,
            "last_transcript_ts": last_transcript_ts,
            "message_count": (last_transcript_sequence + 1 if last_transcript_sequence >= 0 else 0),
        }

    # =========================================================================
    # Connection Verification
    # =========================================================================

    async def verify_connection(self) -> bool:
        """Verify connection to Cosmos DB.

        Returns:
            True if connection is working

        Raises:
            AuthenticationError: If auth fails
            StorageConnectionError: If connection fails
        """
        if not self._initialized:
            await self.initialize()

        # Try to read container to verify connection
        try:
            # Read container properties - simplest verification
            if self._database is None:
                raise StorageIOError(
                    "verify_connection", cause=RuntimeError("Database not initialized")
                )
            # Just check that we can read database properties
            await self._database.read()
            return True
        except CosmosHttpResponseError as e:
            if e.status_code in (401, 403):
                raise AuthenticationError(self.config.endpoint, str(e)) from e
            raise StorageConnectionError(self.config.endpoint, e) from e
