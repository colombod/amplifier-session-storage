"""
Cosmos DB storage backend with vector search support.

Implements the StorageBackend interface with Azure Cosmos DB as the storage layer.
Supports both traditional queries and vector similarity search.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from azure.cosmos import PartitionKey
from azure.cosmos.aio import ContainerProxy, CosmosClient, DatabaseProxy
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError
from azure.identity.aio import DefaultAzureCredential

from ..embeddings import EmbeddingProvider
from ..exceptions import AuthenticationError, StorageConnectionError, StorageIOError
from ..search.mmr import compute_mmr
from .base import (
    EventSearchOptions,
    SearchFilters,
    SearchResult,
    StorageBackend,
    TranscriptSearchOptions,
)

logger = logging.getLogger(__name__)

# Container names
SESSIONS_CONTAINER = "sessions"
TRANSCRIPTS_CONTAINER = "transcripts"
EVENTS_CONTAINER = "events"

# Auth methods
AUTH_KEY = "key"
AUTH_DEFAULT_CREDENTIAL = "default_credential"


@dataclass
class CosmosConfig:
    """Configuration for Cosmos DB storage."""

    endpoint: str
    database_name: str
    auth_method: str = AUTH_DEFAULT_CREDENTIAL
    key: str | None = None
    enable_vector_search: bool = True  # Enable vector indexes on containers

    @classmethod
    def from_env(cls) -> CosmosConfig:
        """Create config from environment variables."""
        import os

        endpoint = os.environ.get("AMPLIFIER_COSMOS_ENDPOINT")
        database = os.environ.get("AMPLIFIER_COSMOS_DATABASE", "amplifier-db")
        auth_method = os.environ.get("AMPLIFIER_COSMOS_AUTH_METHOD", AUTH_DEFAULT_CREDENTIAL)
        key = os.environ.get("AMPLIFIER_COSMOS_KEY")
        enable_vector = os.environ.get("AMPLIFIER_COSMOS_ENABLE_VECTOR", "true").lower() == "true"

        if not endpoint:
            raise AuthenticationError("cosmos", "AMPLIFIER_COSMOS_ENDPOINT not set")

        if auth_method == AUTH_KEY and not key:
            raise AuthenticationError("cosmos", "AMPLIFIER_COSMOS_KEY required for key auth")

        return cls(
            endpoint=endpoint,
            database_name=database,
            auth_method=auth_method,
            key=key,
            enable_vector_search=enable_vector,
        )


class CosmosBackend(StorageBackend):
    """
    Cosmos DB storage backend with hybrid search support.

    Features:
    - Traditional SQL queries for metadata and structured search
    - Vector similarity search for semantic queries
    - Hybrid search combining both approaches
    - MMR re-ranking for diversity
    """

    def __init__(
        self,
        config: CosmosConfig,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        """
        Initialize Cosmos backend.

        Args:
            config: Cosmos configuration
            embedding_provider: Optional embedding provider for semantic search
        """
        self.config = config
        self.embedding_provider = embedding_provider
        self._client: CosmosClient | None = None
        self._credential: DefaultAzureCredential | None = None
        self._database: DatabaseProxy | None = None
        self._containers: dict[str, ContainerProxy] = {}
        self._initialized = False

    @classmethod
    async def create(
        cls,
        config: CosmosConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> CosmosBackend:
        """
        Create and initialize a Cosmos backend.

        Args:
            config: Cosmos configuration (from env if None)
            embedding_provider: Optional embedding provider

        Returns:
            Initialized CosmosBackend instance
        """
        if config is None:
            config = CosmosConfig.from_env()

        backend = cls(config, embedding_provider)
        await backend.initialize()
        return backend

    async def initialize(self) -> None:
        """Initialize Cosmos connection and containers."""
        if self._initialized:
            return

        try:
            # Create client based on auth method
            if self.config.auth_method == AUTH_KEY:
                if not self.config.key:
                    raise AuthenticationError("cosmos", "Key required for key auth")
                self._client = CosmosClient(
                    self.config.endpoint,
                    credential=self.config.key,
                )
            else:
                self._credential = DefaultAzureCredential()
                self._client = CosmosClient(
                    self.config.endpoint,
                    credential=self._credential,
                )

            if self._client is None:
                raise StorageIOError("initialize", cause=RuntimeError("Client creation failed"))

            # Create database
            self._database = await self._client.create_database_if_not_exists(
                id=self.config.database_name
            )

            # Create containers with vector support if enabled
            await self._ensure_container(SESSIONS_CONTAINER, "/user_id", vector_enabled=False)
            await self._ensure_container(
                TRANSCRIPTS_CONTAINER,
                "/partition_key",
                vector_enabled=self.config.enable_vector_search,
            )
            await self._ensure_container(EVENTS_CONTAINER, "/partition_key", vector_enabled=False)

            self._initialized = True
            logger.info(
                f"Cosmos backend initialized: {self.config.endpoint}, "
                f"vector_search={self.config.enable_vector_search}"
            )

        except CosmosHttpResponseError as e:
            if e.status_code in (401, 403):
                raise AuthenticationError(self.config.endpoint, str(e)) from e
            raise StorageConnectionError(self.config.endpoint, e) from e
        except Exception as e:
            raise StorageConnectionError(self.config.endpoint, e) from e

    async def _ensure_container(
        self, name: str, partition_key_path: str, vector_enabled: bool = False
    ) -> None:
        """
        Ensure container exists with optional vector indexing.

        Args:
            name: Container name
            partition_key_path: Partition key path
            vector_enabled: Enable vector search indexes
        """
        if self._database is None:
            raise StorageIOError("ensure_container", cause=RuntimeError("Database not initialized"))

        # Base indexing policy
        indexing_policy: dict[str, Any] = {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [{"path": "/*"}],
        }

        # Add vector index for transcripts if enabled
        if vector_enabled and name == TRANSCRIPTS_CONTAINER:
            # Vector index configuration for Cosmos DB
            indexing_policy["vectorIndexes"] = [
                {
                    "path": "/embedding",
                    "type": "quantizedFlat",  # Good for <100k docs, diskANN for larger
                    "quantizationByteSize": 128,
                    "vectorDataType": "float32",
                    "dimensions": 3072,  # text-embedding-3-large
                }
            ]
            logger.info(f"Creating {name} with vector search support (3072 dimensions)")

        container = await self._database.create_container_if_not_exists(
            id=name,
            partition_key=PartitionKey(path=partition_key_path),
            indexing_policy=indexing_policy,
        )
        self._containers[name] = container

    async def close(self) -> None:
        """Close Cosmos connections."""
        if self._client:
            await self._client.close()
            self._client = None

        if self._credential:
            await self._credential.close()
            self._credential = None

        if self.embedding_provider:
            await self.embedding_provider.close()

        self._database = None
        self._containers = {}
        self._initialized = False

    def _get_container(self, name: str) -> ContainerProxy:
        """Get container by name."""
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
        """Upsert session metadata."""
        container = self._get_container(SESSIONS_CONTAINER)

        doc = {
            **metadata,
            "id": metadata.get("session_id"),
            "user_id": user_id,
            "host_id": host_id,
            "_type": "session",
            "synced_at": datetime.now(UTC).isoformat(),
        }

        await container.upsert_item(body=doc)

    async def get_session_metadata(
        self,
        user_id: str,
        session_id: str,
    ) -> dict[str, Any] | None:
        """Get session metadata by ID."""
        container = self._get_container(SESSIONS_CONTAINER)

        try:
            query = "SELECT * FROM c WHERE c.user_id = @user_id AND c.id = @session_id"
            params: list[dict[str, object]] = [
                {"name": "@user_id", "value": user_id},
                {"name": "@session_id", "value": session_id},
            ]
            async for doc in container.query_items(query=query, parameters=params):
                return doc
            return None
        except CosmosResourceNotFoundError:
            return None

    async def search_sessions(
        self,
        user_id: str,
        filters: SearchFilters,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Search sessions with advanced filters."""
        container = self._get_container(SESSIONS_CONTAINER)

        # Build query
        query_parts = ["SELECT * FROM c WHERE c.user_id = @user_id"]
        params = [{"name": "@user_id", "value": user_id}]

        if filters.project_slug:
            query_parts.append("AND c.project_slug = @project")
            params.append({"name": "@project", "value": filters.project_slug})

        if filters.bundle:
            query_parts.append("AND c.bundle = @bundle")
            params.append({"name": "@bundle", "value": filters.bundle})

        if filters.start_date:
            query_parts.append("AND c.created >= @start_date")
            params.append({"name": "@start_date", "value": filters.start_date})

        if filters.end_date:
            query_parts.append("AND c.created <= @end_date")
            params.append({"name": "@end_date", "value": filters.end_date})

        if filters.min_turn_count is not None:
            query_parts.append("AND c.turn_count >= @min_turns")
            params.append({"name": "@min_turns", "value": int(filters.min_turn_count)})

        if filters.max_turn_count is not None:
            query_parts.append("AND c.turn_count <= @max_turns")
            params.append({"name": "@max_turns", "value": int(filters.max_turn_count)})

        query_parts.append("ORDER BY c.created DESC")

        query = " ".join(query_parts)

        results: list[dict[str, Any]] = []
        async for item in container.query_items(
            query=query,
            parameters=params,  # type: ignore
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
        """Delete session and all its data."""
        partition_key = self.make_partition_key(user_id, project_slug, session_id)

        # Delete transcripts
        transcripts = self._get_container(TRANSCRIPTS_CONTAINER)
        async for doc in transcripts.query_items(
            query="SELECT c.id, c.partition_key FROM c WHERE c.partition_key = @pk",
            parameters=[{"name": "@pk", "value": partition_key}],  # type: ignore
        ):
            try:
                await transcripts.delete_item(item=doc["id"], partition_key=doc["partition_key"])
            except CosmosResourceNotFoundError:
                pass

        # Delete events
        events = self._get_container(EVENTS_CONTAINER)
        async for doc in events.query_items(
            query="SELECT c.id, c.partition_key FROM c WHERE c.partition_key = @pk",
            parameters=[{"name": "@pk", "value": partition_key}],  # type: ignore
        ):
            try:
                await events.delete_item(item=doc["id"], partition_key=doc["partition_key"])
            except CosmosResourceNotFoundError:
                pass

        # Delete session metadata
        sessions = self._get_container(SESSIONS_CONTAINER)
        try:
            query = "SELECT c.id, c.user_id FROM c WHERE c.id = @sid AND c.user_id = @uid"
            params: list[dict[str, object]] = [
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
        embeddings: list[list[float]] | None = None,
    ) -> int:
        """
        Sync transcript lines with optional embeddings.

        Args:
            embeddings: Pre-computed embeddings (same order as lines)
        """
        if not lines:
            return 0

        # If embeddings not provided but we have a provider, generate them
        if embeddings is None and self.embedding_provider:
            texts = [line.get("content", "") for line in lines]
            embeddings = await self.embedding_provider.embed_batch(texts)
            logger.info(f"Generated {len(embeddings)} embeddings during ingestion")

        container = self._get_container(TRANSCRIPTS_CONTAINER)
        partition_key = self.make_partition_key(user_id, project_slug, session_id)

        synced = 0
        for i, line in enumerate(lines):
            sequence = start_sequence + i

            doc_id = f"{session_id}_msg_{sequence}"
            ts = line.get("ts") or line.get("timestamp")

            doc: dict[str, Any] = {
                **line,
                "id": doc_id,
                "partition_key": partition_key,
                "user_id": user_id,
                "host_id": host_id,
                "project_slug": project_slug,
                "session_id": session_id,
                "sequence": sequence,
                "ts": ts,
                "_type": "transcript_message",
                "synced_at": datetime.now(UTC).isoformat(),
            }

            # Add embedding if available
            if embeddings and i < len(embeddings):
                doc["embedding"] = embeddings[i]
                doc["embedding_model"] = (
                    self.embedding_provider.model_name if self.embedding_provider else "unknown"
                )

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
        """Get transcript lines for a session."""
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
        async for item in container.query_items(query=query, parameters=params):
            results.append(item)

        return results

    async def search_transcripts(
        self,
        user_id: str,
        options: TranscriptSearchOptions,
        limit: int = 100,
    ) -> list[SearchResult]:
        """
        Search transcript messages with hybrid search.

        Supports:
        - full_text: CONTAINS queries
        - semantic: Vector similarity search
        - hybrid: Combines both with MMR re-ranking
        """
        if options.search_type == "semantic" or options.search_type == "hybrid":
            if not await self.supports_vector_search():
                logger.warning(
                    "Vector search requested but not available, falling back to full_text"
                )
                options.search_type = "full_text"

        if options.search_type == "full_text":
            return await self._full_text_search_transcripts(user_id, options, limit)
        elif options.search_type == "semantic":
            return await self._semantic_search_transcripts(user_id, options, limit)
        elif options.search_type == "hybrid":
            return await self._hybrid_search_transcripts(user_id, options, limit)
        else:
            raise ValueError(f"Unknown search_type: {options.search_type}")

    async def _full_text_search_transcripts(
        self,
        user_id: str,
        options: TranscriptSearchOptions,
        limit: int,
    ) -> list[SearchResult]:
        """Full-text search using CONTAINS."""
        container = self._get_container(TRANSCRIPTS_CONTAINER)

        # Build query with role filters
        query_parts = ["SELECT * FROM c WHERE c.user_id = @user_id"]
        params: list[dict[str, object]] = [{"name": "@user_id", "value": user_id}]

        # Role filters
        role_conditions = []
        if options.search_in_user:
            role_conditions.append("c.role = 'user'")
        if options.search_in_assistant:
            role_conditions.append("c.role = 'assistant'")

        if role_conditions:
            query_parts.append(f"AND ({' OR '.join(role_conditions)})")

        # Content search
        query_parts.append("AND CONTAINS(c.content, @query)")
        params.append({"name": "@query", "value": options.query})

        # Apply filters
        if options.filters:
            if options.filters.project_slug:
                query_parts.append("AND c.project_slug = @project")
                params.append({"name": "@project", "value": options.filters.project_slug})

            if options.filters.start_date:
                query_parts.append("AND c.ts >= @start_date")
                params.append({"name": "@start_date", "value": options.filters.start_date})

            if options.filters.end_date:
                query_parts.append("AND c.ts <= @end_date")
                params.append({"name": "@end_date", "value": options.filters.end_date})

        query_parts.append("ORDER BY c.ts DESC")
        query = " ".join(query_parts)

        results: list[SearchResult] = []
        async for item in container.query_items(
            query=query, parameters=params, max_item_count=limit
        ):
            results.append(
                SearchResult(
                    session_id=item["session_id"],
                    project_slug=item["project_slug"],
                    sequence=item["sequence"],
                    content=item.get("content", ""),
                    metadata=item,
                    score=1.0,  # Full-text doesn't have scores
                    source="full_text",
                )
            )
            if len(results) >= limit:
                break

        return results

    async def _semantic_search_transcripts(
        self,
        user_id: str,
        options: TranscriptSearchOptions,
        limit: int,
    ) -> list[SearchResult]:
        """Semantic search using vector similarity."""
        if not self.embedding_provider:
            raise ValueError("Embedding provider required for semantic search")

        # Generate query embedding
        query_vector = await self.embedding_provider.embed_text(options.query)

        # Perform vector search
        return await self.vector_search(
            user_id=user_id,
            query_vector=query_vector,
            filters=options.filters,
            top_k=limit,
        )

    async def _hybrid_search_transcripts(
        self,
        user_id: str,
        options: TranscriptSearchOptions,
        limit: int,
    ) -> list[SearchResult]:
        """Hybrid search combining full-text and semantic with MMR."""
        if not self.embedding_provider:
            logger.warning("No embedding provider, falling back to full_text")
            return await self._full_text_search_transcripts(user_id, options, limit)

        # Get more candidates for MMR re-ranking
        candidate_limit = limit * 3

        # Get full-text results
        text_results = await self._full_text_search_transcripts(user_id, options, candidate_limit)

        # Get semantic results
        semantic_results = await self._semantic_search_transcripts(
            user_id, options, candidate_limit
        )

        # Merge and deduplicate
        seen = set()
        combined: list[SearchResult] = []
        for result in text_results + semantic_results:
            key = (result.session_id, result.sequence)
            if key not in seen:
                seen.add(key)
                combined.append(result)

        if len(combined) <= limit:
            return combined[:limit]

        # Apply MMR re-ranking for diversity
        query_vector = await self.embedding_provider.embed_text(options.query)
        query_np = np.array(query_vector)

        # Extract embeddings from results
        vectors = []
        for result in combined:
            if "embedding" in result.metadata and result.metadata["embedding"]:
                vectors.append(np.array(result.metadata["embedding"]))
            else:
                # No embedding available, use zero vector (will rank low)
                vectors.append(np.zeros(len(query_vector)))

        # Apply MMR
        mmr_results = compute_mmr(
            vectors=vectors,
            query=query_np,
            lambda_param=options.mmr_lambda,
            top_k=limit,
        )

        # Return re-ranked results
        return [combined[idx] for idx, _ in mmr_results]

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
        """Sync event lines with large event handling."""
        if not lines:
            return 0

        container = self._get_container(EVENTS_CONTAINER)
        partition_key = self.make_partition_key(user_id, project_slug, session_id)

        synced = 0
        for i, line in enumerate(lines):
            sequence = start_sequence + i
            doc_id = f"{session_id}_evt_{sequence}"

            # Check size (Cosmos has 2MB limit, use 400KB threshold)
            line_json = json.dumps(line)
            data_size = len(line_json.encode("utf-8"))

            if data_size > 400 * 1024:
                # Store summary only
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
        """Get event lines (summaries for large events)."""
        container = self._get_container(EVENTS_CONTAINER)
        partition_key = self.make_partition_key(user_id, project_slug, session_id)

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
        async for item in container.query_items(query=query, parameters=params):
            results.append(item)

        return results

    async def search_events(
        self,
        user_id: str,
        options: EventSearchOptions,
        limit: int = 100,
    ) -> list[SearchResult]:
        """Search events by type, tool, and filters."""
        container = self._get_container(EVENTS_CONTAINER)

        # Build query
        query_parts = ["SELECT * FROM c WHERE c.user_id = @user_id"]
        params: list[dict[str, object]] = [{"name": "@user_id", "value": user_id}]

        if options.event_type:
            query_parts.append("AND c.event = @event_type")
            params.append({"name": "@event_type", "value": options.event_type})

        if options.tool_name:
            # Search in data field for tool name (this requires full event data)
            query_parts.append("AND CONTAINS(c.data, @tool_name)")
            params.append({"name": "@tool_name", "value": options.tool_name})

        if options.level:
            query_parts.append("AND c.lvl = @level")
            params.append({"name": "@level", "value": options.level})

        # Apply filters
        if options.filters:
            if options.filters.project_slug:
                query_parts.append("AND c.project_slug = @project")
                params.append({"name": "@project", "value": options.filters.project_slug})

            if options.filters.start_date:
                query_parts.append("AND c.ts >= @start_date")
                params.append({"name": "@start_date", "value": options.filters.start_date})

            if options.filters.end_date:
                query_parts.append("AND c.ts <= @end_date")
                params.append({"name": "@end_date", "value": options.filters.end_date})

        query_parts.append("ORDER BY c.ts DESC")
        query = " ".join(query_parts)

        results: list[SearchResult] = []
        async for item in container.query_items(
            query=query, parameters=params, max_item_count=limit
        ):
            results.append(
                SearchResult(
                    session_id=item["session_id"],
                    project_slug=item["project_slug"],
                    sequence=item["sequence"],
                    content=str(item.get("event", "")),
                    metadata=item,
                    score=1.0,
                    source="event_search",
                )
            )
            if len(results) >= limit:
                break

        return results

    # =========================================================================
    # Vector/Embedding Operations
    # =========================================================================

    async def supports_vector_search(self) -> bool:
        """Check if vector search is available."""
        return self.config.enable_vector_search and self.embedding_provider is not None

    async def upsert_embeddings(
        self,
        user_id: str,
        project_slug: str,
        session_id: str,
        embeddings: list[dict[str, Any]],
    ) -> int:
        """
        Upsert embeddings for existing transcript messages.

        Used for backfilling embeddings on existing data.
        """
        container = self._get_container(TRANSCRIPTS_CONTAINER)
        partition_key = self.make_partition_key(user_id, project_slug, session_id)

        updated = 0
        for emb in embeddings:
            doc_id = f"{session_id}_msg_{emb['sequence']}"

            # Read existing document
            try:
                existing = await container.read_item(item=doc_id, partition_key=partition_key)

                # Update with embedding
                existing["embedding"] = emb["vector"]
                existing["embedding_model"] = emb.get("metadata", {}).get("model", "unknown")
                existing["synced_at"] = datetime.now(UTC).isoformat()

                await container.upsert_item(body=existing)
                updated += 1

            except CosmosResourceNotFoundError:
                logger.warning(f"Transcript message not found: {doc_id}")
                continue

        return updated

    async def vector_search(
        self,
        user_id: str,
        query_vector: list[float],
        filters: SearchFilters | None = None,
        top_k: int = 100,
    ) -> list[SearchResult]:
        """
        Perform vector similarity search in Cosmos DB.

        Uses VectorDistance function for similarity scoring.
        """
        container = self._get_container(TRANSCRIPTS_CONTAINER)

        # Build query with VectorDistance
        query_parts = [
            "SELECT c.id, c.session_id, c.project_slug, c.sequence, c.content, c.role, "
            "c.turn, c.ts, c.embedding, "
            "VectorDistance(c.embedding, @query_vector) AS similarity_score "
            "FROM c "
            "WHERE c.user_id = @user_id AND c.embedding != null"
        ]

        params: list[dict[str, object]] = [
            {"name": "@user_id", "value": user_id},
            {"name": "@query_vector", "value": query_vector},
        ]

        # Apply filters
        if filters:
            if filters.project_slug:
                query_parts.append("AND c.project_slug = @project")
                params.append({"name": "@project", "value": filters.project_slug})

            if filters.start_date:
                query_parts.append("AND c.ts >= @start_date")
                params.append({"name": "@start_date", "value": filters.start_date})

            if filters.end_date:
                query_parts.append("AND c.ts <= @end_date")
                params.append({"name": "@end_date", "value": filters.end_date})

        query_parts.append("ORDER BY VectorDistance(c.embedding, @query_vector)")

        query = " ".join(query_parts)

        results: list[SearchResult] = []
        async for item in container.query_items(
            query=query, parameters=params, max_item_count=top_k
        ):
            results.append(
                SearchResult(
                    session_id=item["session_id"],
                    project_slug=item["project_slug"],
                    sequence=item["sequence"],
                    content=item.get("content", ""),
                    metadata=item,
                    score=1.0 - item.get("similarity_score", 1.0),  # Convert distance to similarity
                    source="semantic",
                )
            )
            if len(results) >= top_k:
                break

        return results

    # =========================================================================
    # Analytics & Aggregations
    # =========================================================================

    async def get_session_statistics(
        self,
        user_id: str,
        filters: SearchFilters | None = None,
    ) -> dict[str, Any]:
        """Get aggregate statistics across sessions."""
        sessions = self._get_container(SESSIONS_CONTAINER)

        # Build base query
        where_parts = ["c.user_id = @user_id"]
        params: list[dict[str, object]] = [{"name": "@user_id", "value": user_id}]

        if filters:
            if filters.project_slug:
                where_parts.append("c.project_slug = @project")
                params.append({"name": "@project", "value": filters.project_slug})

            if filters.start_date:
                where_parts.append("c.created >= @start_date")
                params.append({"name": "@start_date", "value": filters.start_date})

            if filters.end_date:
                where_parts.append("c.created <= @end_date")
                params.append({"name": "@end_date", "value": filters.end_date})

        where_clause = " AND ".join(where_parts)

        # Count total sessions
        count_query = f"SELECT VALUE COUNT(1) FROM c WHERE {where_clause}"
        total_sessions = 0
        async for count_val in sessions.query_items(query=count_query, parameters=params):  # type: ignore
            total_sessions = int(count_val) if isinstance(count_val, (int, str)) else 0
            break

        # Aggregate by project
        project_query = f"""
            SELECT c.project_slug, COUNT(1) as count
            FROM c
            WHERE {where_clause}
            GROUP BY c.project_slug
        """
        projects: dict[str, int] = {}
        async for item in sessions.query_items(query=project_query, parameters=params):  # type: ignore
            projects[item["project_slug"]] = item["count"]

        # Aggregate by bundle
        bundle_query = f"""
            SELECT c.bundle, COUNT(1) as count
            FROM c
            WHERE {where_clause}
            GROUP BY c.bundle
        """
        bundles: dict[str, int] = {}
        async for item in sessions.query_items(query=bundle_query, parameters=params):  # type: ignore
            bundles[item["bundle"]] = item["count"]

        return {
            "total_sessions": total_sessions,
            "sessions_by_project": projects,
            "sessions_by_bundle": bundles,
            "filters_applied": filters is not None,
        }
