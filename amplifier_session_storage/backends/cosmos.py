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

from ..content_extraction import count_embeddable_content_types, extract_all_embeddable_content
from ..embeddings import EmbeddingProvider
from ..exceptions import AuthenticationError, StorageConnectionError, StorageIOError
from ..search.mmr import compute_mmr
from .base import (
    MessageContext,
    SearchFilters,
    SearchResult,
    StorageBackend,
    TranscriptMessage,
    TranscriptSearchOptions,
    TurnContext,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Single Container Architecture (per COSMOS_DB_BEST_PRACTICES.md)
# All session data (sessions, transcripts, events) stored in ONE container
# with type discriminator for efficient single-partition queries
# =============================================================================

# Single container for all session data
CONTAINER_NAME = "session_data"

# Document types (type discriminator)
DOC_TYPE_SESSION = "session"
DOC_TYPE_TRANSCRIPT = "transcript"
DOC_TYPE_EVENT = "event"

# Auth methods
AUTH_KEY = "key"
AUTH_DEFAULT_CREDENTIAL = "default_credential"

# Vector field names to exclude from metadata (to avoid bloating LLM context)
VECTOR_FIELDS = {
    "user_query_vector",
    "assistant_response_vector",
    "assistant_thinking_vector",
    "tool_output_vector",
}

# =============================================================================
# Column Projections for RU Optimization (per COSMOS_DB_BEST_PRACTICES.md)
# Using explicit projections reduces RU cost by not reading large vector fields
# All projections include 'type' field for discriminator awareness
# =============================================================================

# Session fields (no vectors)
SESSION_PROJECTION = """
    c.id, c.type, c.partition_key, c.user_id, c.session_id, c.host_id, c.project_slug,
    c.bundle, c.created, c.updated, c.turn_count, c.metadata, c.synced_at
"""

# Transcript fields WITHOUT vectors (most common read pattern)
# This dramatically reduces RU cost vs SELECT * when vectors are ~48KB each
TRANSCRIPT_PROJECTION = """
    c.id, c.type, c.partition_key, c.user_id, c.host_id, c.project_slug, c.session_id, c.sequence,
    c.role, c.content, c.turn, c.ts, c.embedding_model, c.vector_metadata, c.synced_at
"""

# Event fields (no vectors)
EVENT_PROJECTION = """
    c.id, c.type, c.partition_key, c.user_id, c.host_id, c.project_slug, c.session_id, c.sequence,
    c.ts, c.lvl, c.event, c.turn, c.data, c.data_truncated, c.data_size_bytes, c.synced_at
"""


def _strip_vectors(item: dict[str, Any]) -> dict[str, Any]:
    """Remove vector fields from item to avoid bloating LLM context.

    Note: This is a safety fallback. Prefer using explicit projections in queries
    (TRANSCRIPT_PROJECTION) to avoid reading vectors in the first place, which
    reduces RU cost significantly.
    """
    return {k: v for k, v in item.items() if k not in VECTOR_FIELDS}


def _extract_display_content(raw_content: Any) -> str:
    """Extract readable text from raw transcript content for display.

    Handles:
    - Simple strings (user messages) -> returned as-is
    - List of content blocks (assistant messages) -> extracts text/thinking
    - None -> empty string
    """
    if raw_content is None:
        return ""
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts = []
        for block in raw_content:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "text":
                    text = block.get("text", "")
                    if text:
                        parts.append(text)
                elif block_type == "thinking":
                    thinking = block.get("thinking", "")
                    if thinking:  # Skip empty thinking with only signature
                        parts.append(f"[thinking] {thinking}")
                elif block_type == "tool_use":
                    name = block.get("name", "unknown")
                    parts.append(f"[tool_use: {name}]")
                elif block_type == "tool_result":
                    content = block.get("content", "")
                    if isinstance(content, str) and content:
                        parts.append(f"[tool_result] {content[:200]}")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else str(raw_content)[:500]
    return str(raw_content)[:500]


def _extract_event_content(item: dict[str, Any]) -> str:
    """Build useful display content from an event document."""
    event_type = item.get("event", "")
    data = item.get("data", {})

    if not isinstance(data, dict):
        return event_type

    parts: list[str] = [event_type]

    # Tool events
    if event_type in ("tool:pre", "tool:post"):
        tool = data.get("tool_name", "")
        if tool:
            parts.append(f"tool={tool}")

    # LLM events
    elif event_type in ("llm:request", "llm:response"):
        model = data.get("model", "")
        provider = data.get("provider", "")
        if model:
            parts.append(f"model={model}")
        if provider:
            parts.append(f"provider={provider}")
        usage = data.get("usage")
        if isinstance(usage, dict):
            inp = usage.get("input", 0)
            out = usage.get("output", 0)
            parts.append(f"tokens={inp}in/{out}out")

    # Content blocks
    elif event_type.startswith("content_block:"):
        block_type = data.get("block_type", "")
        if block_type:
            parts.append(f"type={block_type}")
        idx = data.get("block_index")
        total = data.get("total_blocks")
        if idx is not None and total:
            parts.append(f"block={idx}/{total}")

    # Orchestrator
    elif event_type == "orchestrator:complete":
        orch = data.get("orchestrator", "")
        turns = data.get("turn_count", "")
        if orch:
            parts.append(f"orchestrator={orch}")
        if turns:
            parts.append(f"turns={turns}")

    # Delegate
    elif event_type == "delegate:agent_completed":
        agent = data.get("agent", data.get("agent_name", ""))
        if agent:
            parts.append(f"agent={agent}")

    return " | ".join(parts)


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

            # Create single container with vector support if enabled
            # All document types (session, transcript, event) share the same container
            # partitioned by /partition_key (user_id|project_slug|session_id)
            await self._ensure_container(
                CONTAINER_NAME,
                "/partition_key",
                vector_enabled=self.config.enable_vector_search,
            )

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

        If vector_enabled is True and the container already exists without vector support,
        this will attempt to migrate by replacing the container (after logging a warning).

        Args:
            name: Container name
            partition_key_path: Partition key path
            vector_enabled: Enable vector search indexes
        """
        if self._database is None:
            raise StorageIOError("ensure_container", cause=RuntimeError("Database not initialized"))

        logger.info(
            "Ensuring container exists",
            extra={
                "container": name,
                "partition_key": partition_key_path,
                "vector_enabled": vector_enabled,
            },
        )

        # Optimized indexing policy per COSMOS_DB_BEST_PRACTICES.md
        # - Include only queried scalar properties (reduces write RU cost)
        # - Exclude large text content and vectors from standard indexes
        # - Default exclude everything else to prevent indexing new unknown fields
        indexing_policy: dict[str, Any] = {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [
                {"path": "/user_id/?"},
                {"path": "/session_id/?"},
                {"path": "/type/?"},  # Type discriminator for single-container
                {"path": "/partition_key/?"},
                {"path": "/project_slug/?"},
                {"path": "/ts/?"},
                {"path": "/sequence/?"},
                {"path": "/role/?"},
                {"path": "/turn/?"},
                {"path": "/created/?"},
                {"path": "/updated/?"},
                {"path": "/bundle/?"},
                {"path": "/event/?"},
                {"path": "/lvl/?"},
            ],
            "excludedPaths": [
                {"path": "/content/*"},  # Don't index full content text
                {"path": "/user_query_vector/*"},  # Handled by vector index
                {"path": "/assistant_response_vector/*"},
                {"path": "/assistant_thinking_vector/*"},
                {"path": "/tool_output_vector/*"},
                {"path": "/*"},  # Exclude everything else by default
            ],
        }

        # Vector embedding policy for Cosmos DB (required for vector search)
        vector_embedding_policy: dict[str, Any] | None = None

        # Add vector indexes for transcripts if enabled
        if vector_enabled and name == CONTAINER_NAME:
            # Vector index configuration for Cosmos DB - one index per content type
            indexing_policy["vectorIndexes"] = [
                {
                    "path": "/user_query_vector",
                    "type": "quantizedFlat",
                },
                {
                    "path": "/assistant_response_vector",
                    "type": "quantizedFlat",
                },
                {
                    "path": "/assistant_thinking_vector",
                    "type": "quantizedFlat",
                },
                {
                    "path": "/tool_output_vector",
                    "type": "quantizedFlat",
                },
            ]

            # Vector embedding policy defines the embedding paths and their properties
            # This is REQUIRED for vector search to work in Cosmos DB
            vector_embedding_policy = {
                "vectorEmbeddings": [
                    {
                        "path": "/user_query_vector",
                        "dataType": "float32",
                        "dimensions": 3072,
                        "distanceFunction": "cosine",
                    },
                    {
                        "path": "/assistant_response_vector",
                        "dataType": "float32",
                        "dimensions": 3072,
                        "distanceFunction": "cosine",
                    },
                    {
                        "path": "/assistant_thinking_vector",
                        "dataType": "float32",
                        "dimensions": 3072,
                        "distanceFunction": "cosine",
                    },
                    {
                        "path": "/tool_output_vector",
                        "dataType": "float32",
                        "dimensions": 3072,
                        "distanceFunction": "cosine",
                    },
                ]
            }
            logger.info(
                "Configuring vector search support",
                extra={
                    "container": name,
                    "dimensions": 3072,
                    "distance_function": "cosine",
                    "vector_paths": [
                        "/user_query_vector",
                        "/assistant_response_vector",
                        "/assistant_thinking_vector",
                        "/tool_output_vector",
                    ],
                },
            )

        # Check if container exists and needs migration
        needs_migration = False
        try:
            existing_container = self._database.get_container_client(name)
            # Try to read container properties to verify it exists
            props = await existing_container.read()

            # Check if vector upgrade is needed
            if vector_enabled and name == CONTAINER_NAME:
                existing_vector_policy = props.get("vectorEmbeddingPolicy")
                if not existing_vector_policy or not existing_vector_policy.get("vectorEmbeddings"):
                    logger.warning(
                        "Container exists without vector support - migration required",
                        extra={
                            "container": name,
                            "action": "will_recreate",
                            "warning": "Existing data will be preserved but vector indexes need rebuild",
                        },
                    )
                    needs_migration = True
                else:
                    logger.info(
                        "Container already has vector support",
                        extra={"container": name, "status": "no_migration_needed"},
                    )
        except CosmosResourceNotFoundError:
            # Container doesn't exist, will be created
            logger.info(
                "Container does not exist, will create",
                extra={"container": name, "vector_enabled": vector_enabled},
            )

        # Create or replace container
        if needs_migration:
            # For migration, we need to replace the container
            # Note: This preserves data but requires reindexing
            logger.warning(
                "Replacing container to add vector support",
                extra={
                    "container": name,
                    "action": "replace_container",
                    "note": "Data preserved, indexes will rebuild",
                },
            )
            try:
                container = await self._database.replace_container(
                    container=name,
                    partition_key=PartitionKey(path=partition_key_path),
                    indexing_policy=indexing_policy,
                    vector_embedding_policy=vector_embedding_policy,
                )
                logger.info(
                    "Container replaced with vector support",
                    extra={"container": name, "status": "migration_complete"},
                )
            except CosmosHttpResponseError as e:
                logger.error(
                    "Failed to replace container for vector migration",
                    extra={"container": name, "error": str(e), "status_code": e.status_code},
                )
                # Fall back to using existing container without vector
                container = self._database.get_container_client(name)
        elif vector_embedding_policy:
            container = await self._database.create_container_if_not_exists(
                id=name,
                partition_key=PartitionKey(path=partition_key_path),
                indexing_policy=indexing_policy,
                vector_embedding_policy=vector_embedding_policy,
            )
            logger.info(
                "Container created/verified with vector support",
                extra={"container": name, "status": "ready"},
            )
        else:
            container = await self._database.create_container_if_not_exists(
                id=name,
                partition_key=PartitionKey(path=partition_key_path),
                indexing_policy=indexing_policy,
            )
            logger.info("Container created/verified", extra={"container": name, "status": "ready"})
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
        container = self._get_container(CONTAINER_NAME)

        session_id = metadata.get("session_id", "")
        project_slug = metadata.get("project_slug", "default")
        partition_key = self.make_partition_key(user_id, project_slug, str(session_id))

        doc = {
            **metadata,
            "id": f"session_{session_id}",  # Prefix to avoid ID collision with transcripts
            "type": DOC_TYPE_SESSION,  # Type discriminator
            "partition_key": partition_key,
            "user_id": user_id,
            "host_id": host_id,
            "synced_at": datetime.now(UTC).isoformat(),
        }

        await container.upsert_item(body=doc)

    async def get_session_metadata(
        self,
        user_id: str,
        session_id: str,
    ) -> dict[str, Any] | None:
        """Get session metadata by ID."""
        container = self._get_container(CONTAINER_NAME)

        try:
            # Use explicit projection for RU optimization (per COSMOS_DB_BEST_PRACTICES.md)
            # Filter by type discriminator for single-container architecture
            query = f"SELECT {SESSION_PROJECTION} FROM c WHERE c.type = @type AND c.user_id = @user_id AND c.session_id = @session_id"
            params: list[dict[str, object]] = [
                {"name": "@type", "value": DOC_TYPE_SESSION},
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
        container = self._get_container(CONTAINER_NAME)

        # Build query with explicit projection for RU optimization
        # Filter by type discriminator for single-container architecture
        query_parts = [f"SELECT {SESSION_PROJECTION} FROM c WHERE c.type = @type"]
        params: list[dict[str, object]] = [
            {"name": "@type", "value": DOC_TYPE_SESSION},
        ]

        if user_id:
            query_parts.append("AND c.user_id = @user_id")
            params.append({"name": "@user_id", "value": user_id})

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
        """Delete session and all its data (sessions, transcripts, events).

        In single-container architecture, all documents for a session share
        the same partition_key, so we can delete them all in one query.
        """
        partition_key = self.make_partition_key(user_id, project_slug, session_id)
        container = self._get_container(CONTAINER_NAME)

        deleted_count = 0
        try:
            # Delete all documents with this partition key (sessions, transcripts, events)
            async for doc in container.query_items(
                query="SELECT c.id FROM c WHERE c.partition_key = @pk",
                parameters=[{"name": "@pk", "value": partition_key}],  # type: ignore
            ):
                try:
                    await container.delete_item(item=doc["id"], partition_key=partition_key)
                    deleted_count += 1
                except CosmosResourceNotFoundError:
                    pass

            return deleted_count > 0
        except CosmosResourceNotFoundError:
            return False

    # =========================================================================
    # Embedding Helper Methods
    # =========================================================================

    async def _embed_non_none(self, texts: list[str | None]) -> list[list[float] | None]:
        """Generate embeddings only for non-None texts."""
        if not self.embedding_provider:
            return [None] * len(texts)

        # Collect non-None texts with their indices
        texts_to_embed: list[tuple[int, str]] = []
        for i, text in enumerate(texts):
            if text is not None:
                texts_to_embed.append((i, text))

        if not texts_to_embed:
            return [None] * len(texts)

        # Generate embeddings for non-None texts
        non_none_texts = [text for _, text in texts_to_embed]
        embeddings_batch = await self.embedding_provider.embed_batch(non_none_texts)

        # Build result list with None for skipped texts
        results: list[list[float] | None] = [None] * len(texts)
        for (idx, _), embedding in zip(texts_to_embed, embeddings_batch, strict=True):
            results[idx] = embedding

        return results

    async def _generate_multi_vector_embeddings(
        self, lines: list[dict[str, Any]]
    ) -> dict[str, list[list[float] | None]]:
        """Generate embeddings for all content types in a batch."""
        if not self.embedding_provider:
            return {}

        # Extract all embeddable content types
        user_queries: list[str | None] = []
        assistant_responses: list[str | None] = []
        assistant_thinking: list[str | None] = []
        tool_outputs: list[str | None] = []

        for line in lines:
            extracted = extract_all_embeddable_content(line)
            user_queries.append(extracted["user_query"])
            assistant_responses.append(extracted["assistant_response"])
            assistant_thinking.append(extracted["assistant_thinking"])
            tool_outputs.append(extracted["tool_output"])

        # Generate embeddings for each content type (only for non-None values)
        return {
            "user_query": await self._embed_non_none(user_queries),
            "assistant_response": await self._embed_non_none(assistant_responses),
            "assistant_thinking": await self._embed_non_none(assistant_thinking),
            "tool_output": await self._embed_non_none(tool_outputs),
        }

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
        embeddings: dict[str, list[list[float] | None]] | None = None,
    ) -> int:
        """
        Sync transcript lines with optional multi-vector embeddings.

        Args:
            embeddings: Pre-computed embeddings dict with keys: user_query, assistant_response,
                        assistant_thinking, tool_output
        """
        if not lines:
            return 0

        # Generate embeddings if not provided and we have a provider
        if embeddings is None and self.embedding_provider:
            embeddings = await self._generate_multi_vector_embeddings(lines)
            counts = count_embeddable_content_types(lines)
            logger.info(
                f"Generated embeddings: {counts['user_query']} user, "
                f"{counts['assistant_response']} responses, "
                f"{counts['assistant_thinking']} thinking, "
                f"{counts['tool_output']} tool"
            )

        container = self._get_container(CONTAINER_NAME)
        partition_key = self.make_partition_key(user_id, project_slug, session_id)

        synced = 0
        for i, line in enumerate(lines):
            sequence = start_sequence + i

            doc_id = f"{session_id}_msg_{sequence}"
            ts = line.get("ts") or line.get("timestamp")

            doc: dict[str, Any] = {
                **line,
                "id": doc_id,
                "type": DOC_TYPE_TRANSCRIPT,  # Type discriminator for single-container
                "partition_key": partition_key,
                "user_id": user_id,
                "host_id": host_id,
                "project_slug": project_slug,
                "session_id": session_id,
                "sequence": sequence,
                "ts": ts,
                "synced_at": datetime.now(UTC).isoformat(),
            }

            # Add multi-vector embeddings if available
            if embeddings:
                user_query_vec = embeddings.get("user_query", [None] * len(lines))[i]
                assistant_response_vec = embeddings.get("assistant_response", [None] * len(lines))[
                    i
                ]
                assistant_thinking_vec = embeddings.get("assistant_thinking", [None] * len(lines))[
                    i
                ]
                tool_output_vec = embeddings.get("tool_output", [None] * len(lines))[i]

                if user_query_vec:
                    doc["user_query_vector"] = user_query_vec
                if assistant_response_vec:
                    doc["assistant_response_vector"] = assistant_response_vec
                if assistant_thinking_vec:
                    doc["assistant_thinking_vector"] = assistant_thinking_vec
                if tool_output_vec:
                    doc["tool_output_vector"] = tool_output_vec

                # Store metadata about which vectors exist
                doc["vector_metadata"] = {
                    "has_user_query": user_query_vec is not None,
                    "has_assistant_response": assistant_response_vec is not None,
                    "has_assistant_thinking": assistant_thinking_vec is not None,
                    "has_tool_output": tool_output_vec is not None,
                }

                if self.embedding_provider:
                    doc["embedding_model"] = self.embedding_provider.model_name

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
        container = self._get_container(CONTAINER_NAME)
        partition_key = self.make_partition_key(user_id, project_slug, session_id)

        # Use explicit projection to exclude vectors at query time (reduces RU cost)
        # This is more efficient than SELECT * + _strip_vectors() because Cosmos
        # doesn't have to read the large vector fields at all
        # Filter by type discriminator for single-container architecture
        query = (
            f"SELECT {TRANSCRIPT_PROJECTION} FROM c WHERE c.type = @type "
            "AND c.partition_key = @pk AND c.sequence > @after_seq ORDER BY c.sequence"
        )
        params = [
            {"name": "@type", "value": DOC_TYPE_TRANSCRIPT},
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
        container = self._get_container(CONTAINER_NAME)

        # Build query with explicit projection (excludes vectors, reduces RU cost)
        # Filter by type discriminator for single-container architecture
        # Allow empty user_id for team-wide search
        if user_id:
            query_parts = [
                f"SELECT {TRANSCRIPT_PROJECTION} FROM c WHERE c.type = @type AND c.user_id = @user_id"
            ]
            params: list[dict[str, object]] = [
                {"name": "@type", "value": DOC_TYPE_TRANSCRIPT},
                {"name": "@user_id", "value": user_id},
            ]
        else:
            # Team-wide search (all users)
            query_parts = [f"SELECT {TRANSCRIPT_PROJECTION} FROM c WHERE c.type = @type"]
            params: list[dict[str, object]] = [{"name": "@type", "value": DOC_TYPE_TRANSCRIPT}]

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
                    content=_extract_display_content(item.get("content")),
                    metadata=_strip_vectors(item),
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

        # Translate search_in_* flags to vector_columns
        vector_columns: list[str] | None = []
        if options.search_in_user:
            vector_columns.append("user_query")
        if options.search_in_assistant:
            vector_columns.append("assistant_response")
        if options.search_in_thinking:
            vector_columns.append("assistant_thinking")
        if options.search_in_tool:
            vector_columns.append("tool_output")
        if not vector_columns:
            vector_columns = None  # None = search all

        # Perform vector search
        return await self.vector_search(
            user_id=user_id,
            query_vector=query_vector,
            filters=options.filters,
            top_k=limit,
            vector_columns=vector_columns,
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

    async def get_turn_context(
        self,
        user_id: str,
        session_id: str,
        turn: int,
        before: int = 2,
        after: int = 2,
        include_tool_outputs: bool = True,
    ) -> TurnContext:
        """
        Get context window around a specific turn.

        Queries Cosmos DB for messages in the turn range [turn-before, turn+after].
        """
        container = self._get_container(CONTAINER_NAME)

        # Calculate turn range
        min_turn = max(1, turn - before)
        max_turn = turn + after

        # Query for messages in the turn range (no ORDER BY - Cosmos needs composite index)
        # We'll sort in Python instead
        # Filter by type discriminator for single-container architecture
        query = """
            SELECT c.sequence, c.turn, c.role, c.content, c.ts, c.project_slug
            FROM c
            WHERE c.type = @type
              AND c.user_id = @user_id
              AND c.session_id = @session_id
              AND c.turn >= @min_turn
              AND c.turn <= @max_turn
        """
        params = [
            {"name": "@type", "value": DOC_TYPE_TRANSCRIPT},
            {"name": "@user_id", "value": user_id},
            {"name": "@session_id", "value": session_id},
            {"name": "@min_turn", "value": min_turn},
            {"name": "@max_turn", "value": max_turn},
        ]

        # Fetch messages
        messages: list[TranscriptMessage] = []
        project_slug = "unknown"

        async for item in container.query_items(query=query, parameters=params):
            # Filter out tool outputs if requested
            if not include_tool_outputs and item.get("role") == "tool":
                continue

            project_slug = item.get("project_slug", project_slug)
            messages.append(
                TranscriptMessage(
                    sequence=item.get("sequence", 0),
                    turn=item.get("turn", 0),
                    role=item.get("role", "unknown"),
                    content=item.get("content", ""),
                    ts=item.get("ts"),
                    metadata={"session_id": session_id},
                )
            )

        # Sort by turn, then sequence (Cosmos doesn't support multi-column ORDER BY without composite index)
        messages.sort(key=lambda m: (m.turn, m.sequence))

        # Get session turn range for navigation metadata
        # Cosmos DB doesn't support composite aggregates, so run separate queries
        # Filter by type discriminator for single-container architecture
        min_query = "SELECT VALUE MIN(c.turn) FROM c WHERE c.type = @type AND c.user_id = @user_id AND c.session_id = @session_id"
        max_query = "SELECT VALUE MAX(c.turn) FROM c WHERE c.type = @type AND c.user_id = @user_id AND c.session_id = @session_id"
        range_params = [
            {"name": "@type", "value": DOC_TYPE_TRANSCRIPT},
            {"name": "@user_id", "value": user_id},
            {"name": "@session_id", "value": session_id},
        ]

        first_turn = 1
        last_turn = turn

        async for item in container.query_items(query=min_query, parameters=range_params):
            if item is not None:
                first_turn = item
            break

        async for item in container.query_items(query=max_query, parameters=range_params):
            if item is not None:
                last_turn = item
            break

        # Partition messages into previous, current, following
        # Handle case where turn might be None
        previous = [m for m in messages if m.turn is not None and m.turn < turn]
        current = [m for m in messages if m.turn == turn]
        following = [m for m in messages if m.turn is not None and m.turn > turn]

        return TurnContext(
            session_id=session_id,
            project_slug=project_slug,
            target_turn=turn,
            previous=previous,
            current=current,
            following=following,
            has_more_before=min_turn > first_turn,
            has_more_after=max_turn < last_turn,
            first_turn=first_turn,
            last_turn=last_turn,
        )

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

        container = self._get_container(CONTAINER_NAME)
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
                    "type": DOC_TYPE_EVENT,  # Type discriminator for single-container
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
                    "synced_at": datetime.now(UTC).isoformat(),
                }
            else:
                # Store full event
                doc = {
                    **line,
                    "id": doc_id,
                    "type": DOC_TYPE_EVENT,  # Type discriminator for single-container
                    "partition_key": partition_key,
                    "user_id": user_id,
                    "host_id": host_id,
                    "project_slug": project_slug,
                    "session_id": session_id,
                    "sequence": sequence,
                    "data_truncated": False,
                    "data_size_bytes": data_size,
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
        container = self._get_container(CONTAINER_NAME)
        partition_key = self.make_partition_key(user_id, project_slug, session_id)

        # Filter by type discriminator for single-container architecture
        query = (
            "SELECT c.id, c.sequence, c.ts, c.lvl, c.event, c.turn, "
            "c.data, c.data_truncated, c.data_size_bytes "
            "FROM c WHERE c.type = @type AND c.partition_key = @pk "
            "AND c.sequence > @after_seq ORDER BY c.sequence"
        )
        params = [
            {"name": "@type", "value": DOC_TYPE_EVENT},
            {"name": "@pk", "value": partition_key},
            {"name": "@after_seq", "value": after_sequence},
        ]

        results: list[dict[str, Any]] = []
        async for item in container.query_items(query=query, parameters=params):
            results.append(item)

        return results

    async def search_events(
        self,
        user_id: str = "",
        session_id: str = "",
        project_slug: str = "",
        event_type: str = "",
        event_category: str = "",
        tool_name: str = "",
        model: str = "",
        provider: str = "",
        level: str = "",
        start_date: str = "",
        end_date: str = "",
        limit: int = 50,
    ) -> list[SearchResult]:
        """Search events with structured filters.

        Args:
            user_id: Filter by user.
            session_id: Filter by session.
            project_slug: Filter by project.
            event_type: Exact event type match (e.g. "tool:pre", "llm:response").
            event_category: Prefix-based group filter (e.g. "tool", "llm", "session").
                Uses STARTSWITH on the event type prefix for fully dynamic discovery.
            tool_name: Filter tool events by ``data.tool_name``.
            model: Filter LLM events by ``data.model``.
            provider: Filter LLM events by ``data.provider``.
            level: Filter by log level (``lvl`` field).
            start_date: Inclusive lower bound on ``ts`` (ISO-8601).
            end_date: Inclusive upper bound on ``ts`` (ISO-8601).
            limit: Maximum results to return.

        Returns:
            List of :class:`SearchResult` ordered by timestamp descending.
        """
        container = self._get_container(CONTAINER_NAME)

        # Always filter on the type discriminator
        query_parts = [f"SELECT {EVENT_PROJECTION} FROM c WHERE c.type = @doc_type"]
        params: list[dict[str, object]] = [
            {"name": "@doc_type", "value": DOC_TYPE_EVENT},
        ]

        if user_id:
            query_parts.append("AND c.user_id = @user_id")
            params.append({"name": "@user_id", "value": user_id})

        if session_id:
            query_parts.append("AND c.session_id = @session_id")
            params.append({"name": "@session_id", "value": session_id})

        if project_slug:
            query_parts.append("AND c.project_slug = @project")
            params.append({"name": "@project", "value": project_slug})

        # event_type takes precedence over event_category
        if event_type:
            query_parts.append("AND c.event = @event_type")
            params.append({"name": "@event_type", "value": event_type})
        elif event_category:
            # Category is the prefix before first colon (e.g. "tool" matches "tool:pre", "tool:post", "tool:error")
            # Use STARTSWITH for fully dynamic discovery - no hardcoded mapping needed
            cat_prefix = event_category.rstrip(":") + ":"
            # Allow shorthand: "content" matches "content_block:*"
            if cat_prefix == "content:" and not event_category.startswith("content_block"):
                cat_prefix = "content_block:"
            query_parts.append("AND STARTSWITH(c.event, @cat_prefix)")
            params.append({"name": "@cat_prefix", "value": cat_prefix})

        # Data-level filters (correct path into the nested dict)
        if tool_name:
            query_parts.append("AND c.data.tool_name = @tool_name")
            params.append({"name": "@tool_name", "value": tool_name})

        if model:
            query_parts.append("AND c.data.model = @model")
            params.append({"name": "@model", "value": model})

        if provider:
            query_parts.append("AND c.data.provider = @provider")
            params.append({"name": "@provider", "value": provider})

        if level:
            query_parts.append("AND c.lvl = @level")
            params.append({"name": "@level", "value": level})

        if start_date:
            query_parts.append("AND c.ts >= @start_date")
            params.append({"name": "@start_date", "value": start_date})

        if end_date:
            query_parts.append("AND c.ts <= @end_date")
            params.append({"name": "@end_date", "value": end_date})

        query_parts.append("ORDER BY c.ts DESC")
        query = " ".join(query_parts)

        results: list[SearchResult] = []
        async for item in container.query_items(
            query=query, parameters=params, max_item_count=limit
        ):
            results.append(
                SearchResult(
                    session_id=item.get("session_id", ""),
                    project_slug=item.get("project_slug", ""),
                    sequence=item.get("sequence", 0),
                    content=_extract_event_content(item),
                    metadata=_strip_vectors(item),
                    score=1.0,
                    source="event_search",
                )
            )
            if len(results) >= limit:
                break

        return results

    async def list_event_types(self, user_id: str = "", session_id: str = "") -> dict:
        """Return distinct event types with counts and runtime-discovered categories.

        Categories are derived from event type prefixes (e.g. tool:pre -> tool).
        No hardcoded mapping needed - new event types are auto-categorized.
        """
        container = self._get_container(CONTAINER_NAME)
        query = "SELECT c.event FROM c WHERE c.type = @t"
        params: list[dict[str, str]] = [{"name": "@t", "value": DOC_TYPE_EVENT}]
        if user_id:
            query += " AND c.user_id = @uid"
            params.append({"name": "@uid", "value": user_id})
        if session_id:
            query += " AND c.session_id = @sid"
            params.append({"name": "@sid", "value": session_id})

        from collections import Counter

        type_counter: Counter[str] = Counter()
        async for item in container.query_items(query=query, parameters=params):
            type_counter[item.get("event", "?")] += 1

        # Derive categories from event type prefixes
        category_map: dict[str, list[str]] = {}
        for event_type in sorted(type_counter.keys()):
            # Category = everything before first colon
            prefix = event_type.split(":")[0] if ":" in event_type else event_type
            # Normalize: content_block -> content_block (keep underscore as part of prefix)
            category_map.setdefault(prefix, []).append(event_type)

        return {
            "event_types": [{"event_type": k, "count": v} for k, v in type_counter.most_common()],
            "categories": category_map,
            "total_events": sum(type_counter.values()),
        }

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
        container = self._get_container(CONTAINER_NAME)
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
        vector_columns: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Perform multi-vector similarity search in Cosmos DB.

        Since Cosmos DB doesn't support LEAST/GREATEST functions, we run separate
        queries for each vector column and merge results, keeping the best score
        for each document.

        Args:
            user_id: User identifier
            query_vector: Query embedding vector
            filters: Optional search filters
            top_k: Number of results to return
            vector_columns: Which vector columns to search. Default: all non-null vectors.
                Options: ["user_query", "assistant_response",
                          "assistant_thinking", "tool_output"]
        """
        container = self._get_container(CONTAINER_NAME)

        if vector_columns is None:
            # Search all vector columns by default
            vector_columns = [
                "user_query",
                "assistant_response",
                "assistant_thinking",
                "tool_output",
            ]

        vector_field_names = {
            "user_query": "user_query_vector",
            "assistant_response": "assistant_response_vector",
            "assistant_thinking": "assistant_thinking_vector",
            "tool_output": "tool_output_vector",
        }

        # Build base filter conditions - allow empty user_id for team-wide search
        # Always filter by type discriminator for single-container architecture
        if user_id:
            filter_parts = ["c.type = @doc_type", "c.user_id = @user_id"]
            base_params: list[dict[str, object]] = [
                {"name": "@doc_type", "value": DOC_TYPE_TRANSCRIPT},
                {"name": "@user_id", "value": user_id},
                {"name": "@query_vector", "value": query_vector},
            ]
        else:
            # Team-wide search (all users)
            filter_parts: list[str] = ["c.type = @doc_type"]
            base_params: list[dict[str, object]] = [
                {"name": "@doc_type", "value": DOC_TYPE_TRANSCRIPT},
                {"name": "@query_vector", "value": query_vector},
            ]

        if filters:
            if filters.project_slug:
                filter_parts.append("c.project_slug = @project")
                base_params.append({"name": "@project", "value": filters.project_slug})

            if filters.start_date:
                filter_parts.append("c.ts >= @start_date")
                base_params.append({"name": "@start_date", "value": filters.start_date})

            if filters.end_date:
                filter_parts.append("c.ts <= @end_date")
                base_params.append({"name": "@end_date", "value": filters.end_date})

        # Build where clause - handle empty filter_parts for team-wide search
        if filter_parts:
            where_clause = " AND ".join(filter_parts)
            where_prefix = f"WHERE {where_clause} AND"
        else:
            where_prefix = "WHERE"

        # Run separate query for each vector column and merge results
        # This is necessary because Cosmos DB doesn't support LEAST() function
        all_results: dict[str, SearchResult] = {}  # keyed by doc id

        for vec_type in vector_columns:
            field = vector_field_names[vec_type]

            # Query for this vector column only - TOP is REQUIRED for vector search in Cosmos DB
            query = f"""
                SELECT TOP {top_k} c.id, c.session_id, c.project_slug, c.user_id, c.host_id,
                       c.sequence, c.content, c.role, c.turn, c.ts,
                       VectorDistance(c.{field}, @query_vector) AS distance
                FROM c
                {where_prefix} IS_DEFINED(c.{field})
                ORDER BY VectorDistance(c.{field}, @query_vector)
            """

            try:
                async for item in container.query_items(
                    query=query, parameters=base_params, max_item_count=top_k
                ):
                    doc_id = item["id"]
                    distance = item.get("distance", 1.0)
                    similarity = 1.0 - min(distance, 1.0)

                    # Keep the best score for each document
                    if doc_id not in all_results or similarity > all_results[doc_id].score:
                        all_results[doc_id] = SearchResult(
                            session_id=item["session_id"],
                            project_slug=item["project_slug"],
                            sequence=item["sequence"],
                            content=_extract_display_content(item.get("content")),
                            metadata=_strip_vectors(item),
                            score=similarity,
                            source=f"semantic_{vec_type}",
                        )
            except CosmosHttpResponseError as e:
                # Log but continue with other vector columns
                logger.warning(f"Vector search failed for {vec_type}: {e}")
                continue

        # Sort by score and return top_k
        results = sorted(all_results.values(), key=lambda r: r.score, reverse=True)[:top_k]
        return results

    async def _vector_search_single_column(
        self,
        user_id: str,
        query_vector: list[float],
        vector_column: str,
        filters: SearchFilters | None = None,
        top_k: int = 100,
    ) -> list[SearchResult]:
        """Helper for single-column vector search."""
        return await self.vector_search(
            user_id=user_id,
            query_vector=query_vector,
            filters=filters,
            top_k=top_k,
            vector_columns=[vector_column],
        )

    async def _old_vector_search(
        self,
        user_id: str,
        query_vector: list[float],
        filters: SearchFilters | None = None,
        top_k: int = 100,
        vector_columns: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        DEPRECATED: Old implementation that used LEAST() - not supported by Cosmos DB.
        Kept for reference only.
        """
        container = self._get_container(CONTAINER_NAME)

        if vector_columns is None:
            vector_columns = [
                "user_query",
                "assistant_response",
                "assistant_thinking",
                "tool_output",
            ]

        vector_field_names = {
            "user_query": "user_query_vector",
            "assistant_response": "assistant_response_vector",
            "assistant_thinking": "assistant_thinking_vector",
            "tool_output": "tool_output_vector",
        }

        vector_distance_exprs = []
        for vec_type in vector_columns:
            field = vector_field_names[vec_type]
            vector_distance_exprs.append(
                f"(IS_DEFINED(c.{field}) ? VectorDistance(c.{field}, @query_vector) : 1.0)"
            )

        # NOTE: LEAST() is NOT supported in Cosmos DB SQL
        min_distance_expr = (
            f"LEAST({', '.join(vector_distance_exprs)})"
            if len(vector_distance_exprs) > 1
            else vector_distance_exprs[0]
        )

        query_parts = [
            f"SELECT c.id, c.session_id, c.project_slug, c.sequence, c.content, c.role, "
            f"c.turn, c.ts, {min_distance_expr} AS best_distance "
            f"FROM c "
            f"WHERE c.user_id = @user_id"
        ]

        params: list[dict[str, object]] = [
            {"name": "@user_id", "value": user_id},
            {"name": "@query_vector", "value": query_vector},
        ]

        # Require at least one vector to be defined
        vector_exists_conditions = [
            f"IS_DEFINED(c.{vector_field_names[v]})" for v in vector_columns
        ]
        query_parts.append(f"AND ({' OR '.join(vector_exists_conditions)})")

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

        # Order by best distance (lowest = most similar)
        query_parts.append(f"ORDER BY {min_distance_expr}")

        query = " ".join(query_parts)

        results: list[SearchResult] = []
        async for item in container.query_items(
            query=query, parameters=params, max_item_count=top_k
        ):
            # Convert distance to similarity score (1 - distance)
            distance = item.get("best_distance", 1.0)
            similarity = 1.0 - min(distance, 1.0)

            results.append(
                SearchResult(
                    session_id=item["session_id"],
                    project_slug=item["project_slug"],
                    sequence=item["sequence"],
                    content=item.get("content", ""),
                    metadata=_strip_vectors(item),
                    score=similarity,
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
        sessions = self._get_container(CONTAINER_NAME)

        # Build base query - filter by type discriminator for single-container
        where_parts = ["c.type = @doc_type", "c.user_id = @user_id"]
        params: list[dict[str, object]] = [
            {"name": "@doc_type", "value": DOC_TYPE_SESSION},
            {"name": "@user_id", "value": user_id},
        ]

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

        # Cosmos DB cross-partition queries don't support GROUP BY with aggregates
        # So we fetch minimal data and aggregate in Python
        fetch_query = f"SELECT c.project_slug, c.bundle FROM c WHERE {where_clause}"

        projects: dict[str, int] = {}
        bundles: dict[str, int] = {}
        total_sessions = 0

        async for item in sessions.query_items(query=fetch_query, parameters=params):  # type: ignore
            total_sessions += 1

            # Count by project
            proj = item.get("project_slug", "unknown")
            projects[proj] = projects.get(proj, 0) + 1

            # Count by bundle
            bundle = item.get("bundle", "unknown")
            bundles[bundle] = bundles.get(bundle, 0) + 1

        return {
            "total_sessions": total_sessions,
            "sessions_by_project": projects,
            "sessions_by_bundle": bundles,
            "filters_applied": filters is not None,
        }

    # =========================================================================
    # Discovery APIs
    # =========================================================================

    async def list_users(
        self,
        filters: SearchFilters | None = None,
    ) -> list[str]:
        """List all unique user IDs in the storage."""
        container = self._get_container(CONTAINER_NAME)

        # Build query with optional filters - filter by type for single-container
        where_parts: list[str] = ["c.type = @doc_type"]
        params: list[dict[str, object]] = [{"name": "@doc_type", "value": DOC_TYPE_SESSION}]

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

            if filters.bundle:
                where_parts.append("c.bundle = @bundle")
                params.append({"name": "@bundle", "value": filters.bundle})

        where_clause = " AND ".join(where_parts)
        query = f"SELECT DISTINCT VALUE c.user_id FROM c WHERE {where_clause}"

        users: list[str] = []
        async for user_id in container.query_items(query=query, parameters=params):
            if user_id:
                users.append(user_id)

        return sorted(users)

    async def list_projects(
        self,
        user_id: str = "",
        filters: SearchFilters | None = None,
    ) -> list[str]:
        """List all unique project slugs."""
        container = self._get_container(CONTAINER_NAME)

        # Build query with optional filters - filter by type for single-container
        where_parts: list[str] = ["c.type = @doc_type"]
        params: list[dict[str, object]] = [{"name": "@doc_type", "value": DOC_TYPE_SESSION}]

        if user_id:
            where_parts.append("c.user_id = @user_id")
            params.append({"name": "@user_id", "value": user_id})

        if filters:
            if filters.start_date:
                where_parts.append("c.created >= @start_date")
                params.append({"name": "@start_date", "value": filters.start_date})

            if filters.end_date:
                where_parts.append("c.created <= @end_date")
                params.append({"name": "@end_date", "value": filters.end_date})

            if filters.bundle:
                where_parts.append("c.bundle = @bundle")
                params.append({"name": "@bundle", "value": filters.bundle})

        where_clause = " AND ".join(where_parts)
        query = f"SELECT DISTINCT VALUE c.project_slug FROM c WHERE {where_clause}"

        projects: list[str] = []
        async for project in container.query_items(query=query, parameters=params):
            if project:
                projects.append(project)

        return sorted(projects)

    async def list_sessions(
        self,
        user_id: str = "",
        project_slug: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List sessions with pagination."""
        container = self._get_container(CONTAINER_NAME)

        # Build query with optional filters - filter by type for single-container
        where_parts: list[str] = ["c.type = @doc_type"]
        params: list[dict[str, object]] = [{"name": "@doc_type", "value": DOC_TYPE_SESSION}]

        if user_id:
            where_parts.append("c.user_id = @user_id")
            params.append({"name": "@user_id", "value": user_id})

        if project_slug:
            where_parts.append("c.project_slug = @project")
            params.append({"name": "@project", "value": project_slug})

        where_clause = " AND ".join(where_parts)

        # Cosmos DB doesn't support OFFSET directly, use continuation token pattern
        # For simplicity, we'll fetch all and slice in Python (fine for moderate datasets)
        query = f"""
            SELECT c.session_id, c.user_id, c.project_slug, c.bundle, c.created, c.turn_count
            FROM c WHERE {where_clause}
            ORDER BY c.created DESC
        """

        sessions: list[dict[str, Any]] = []
        async for item in container.query_items(query=query, parameters=params):
            sessions.append(item)

        # Apply offset and limit
        return sessions[offset : offset + limit]

    # =========================================================================
    # Sequence-Based Navigation
    # =========================================================================

    async def get_message_context(
        self,
        session_id: str,
        sequence: int,
        user_id: str = "",
        before: int = 5,
        after: int = 5,
        include_tool_outputs: bool = True,
    ) -> MessageContext:
        """Get context window around a specific message by sequence."""
        from .base import MessageContext, TranscriptMessage

        container = self._get_container(CONTAINER_NAME)

        # Build user filter - filter by type for single-container architecture
        user_filter = "c.user_id = @user_id" if user_id else "1=1"
        params: list[dict[str, object]] = [
            {"name": "@doc_type", "value": DOC_TYPE_TRANSCRIPT},
            {"name": "@session_id", "value": session_id},
        ]
        if user_id:
            params.append({"name": "@user_id", "value": user_id})

        # Get the target message and surrounding context
        min_seq = max(0, sequence - before)
        max_seq = sequence + after

        query = f"""
            SELECT c.sequence, c.turn, c.role, c.content, c.ts, c.project_slug
            FROM c
            WHERE c.type = @doc_type AND c.session_id = @session_id AND {user_filter}
              AND c.sequence >= @min_seq AND c.sequence <= @max_seq
        """
        params.extend(
            [
                {"name": "@min_seq", "value": min_seq},
                {"name": "@max_seq", "value": max_seq},
            ]
        )

        messages: list[TranscriptMessage] = []
        project_slug = "unknown"

        async for item in container.query_items(query=query, parameters=params):
            # Filter out tool outputs if requested
            if not include_tool_outputs and item.get("role") == "tool":
                continue

            project_slug = item.get("project_slug", project_slug)
            messages.append(
                TranscriptMessage(
                    sequence=item.get("sequence", 0),
                    turn=item.get("turn"),  # Can be None
                    role=item.get("role", "unknown"),
                    content=item.get("content", ""),
                    ts=item.get("ts"),
                    metadata={"session_id": session_id},
                )
            )

        # Sort by sequence
        messages.sort(key=lambda m: m.sequence)

        # Separate into previous, current, following
        previous: list[TranscriptMessage] = []
        current: TranscriptMessage | None = None
        following: list[TranscriptMessage] = []

        for msg in messages:
            if msg.sequence < sequence:
                previous.append(msg)
            elif msg.sequence == sequence:
                current = msg
            else:
                following.append(msg)

        # Get session sequence range for navigation metadata
        # Filter by type for single-container architecture
        range_query = f"""
            SELECT VALUE MIN(c.sequence) FROM c
            WHERE c.type = @doc_type AND c.session_id = @session_id AND {user_filter}
        """
        first_sequence = 0
        # params[:3] includes @doc_type, @session_id, and optionally @user_id
        range_params = params[:3] if user_id else params[:2]
        async for item in container.query_items(query=range_query, parameters=range_params):
            if item is not None:
                first_sequence = item
            break

        range_query = f"""
            SELECT VALUE MAX(c.sequence) FROM c
            WHERE c.type = @doc_type AND c.session_id = @session_id AND {user_filter}
        """
        last_sequence = sequence
        async for item in container.query_items(query=range_query, parameters=range_params):
            if item is not None:
                last_sequence = item
            break

        return MessageContext(
            session_id=session_id,
            project_slug=project_slug,
            target_sequence=sequence,
            previous=previous,
            current=current,
            following=following,
            has_more_before=first_sequence < min_seq,
            has_more_after=last_sequence > max_seq,
            first_sequence=first_sequence,
            last_sequence=last_sequence,
        )
