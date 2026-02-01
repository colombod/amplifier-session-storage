"""
Cosmos DB session storage implementation.

Implements the SessionStorage ABC using Azure Cosmos DB with:
- User isolation via partition keys
- Event projection enforcement
- Large event chunking
- Transactional batch operations where possible
"""

from datetime import UTC, datetime
from typing import Any

from ..access.controller import AccessController
from ..access.permissions import Permission
from ..exceptions import PermissionDeniedError, SessionExistsError, SessionNotFoundError
from ..membership.store import MembershipStore
from ..protocol import (
    AggregateStats,
    EventQuery,
    EventSummary,
    RewindResult,
    SessionMetadata,
    SessionQuery,
    SessionStorage,
    SessionVisibility,
    SharedSessionQuery,
    SharedSessionSummary,
    SyncStatus,
    TranscriptMessage,
    UserMembership,
)
from ..utils import extract_event_summary as _extract_event_summary
from .chunking import (
    EventChunk,
    chunk_event,
    get_data_size,
    reassemble_event,
    should_chunk,
)
from .client import (
    CHUNKS_CONTAINER,
    EVENTS_CONTAINER,
    SESSIONS_CONTAINER,
    SHARED_SESSIONS_CONTAINER,
    TRANSCRIPT_CONTAINER,
    CosmosClientWrapper,
    CosmosConfig,
    make_partition_key,
)


class CosmosDBStorage(SessionStorage):
    """Cosmos DB-based session storage implementation.

    Uses Azure Cosmos DB for cloud storage with:
    - User isolation via partition keys
    - Event projection enforcement (never returns full event data from queries)
    - Large event chunking (>400KB events are split)
    - Optimized queries for common access patterns
    - Session sharing with access control

    Container structure:
    - sessions: Session metadata (partition: user_id)
    - transcript_messages: Conversation messages (partition: user_id_session_id)
    - events: Event metadata and small events (partition: user_id_session_id)
    - event_chunks: Large event chunks (partition: user_id_session_id)
    - sync_state: Sync tracking (partition: user_id)
    - shared_sessions: Index of shared sessions (partition: visibility)
    - organizations: Organization data (partition: org_id)
    - teams: Team data (partition: org_id)
    - user_memberships: User membership data (partition: user_id)
    """

    def __init__(self, client: CosmosClientWrapper):
        """Initialize Cosmos DB storage.

        Args:
            client: Initialized CosmosClientWrapper instance
        """
        self.client = client
        self._membership_store = MembershipStore(client)
        self._access_controller = AccessController()

    @classmethod
    async def create(cls, config: CosmosConfig | None = None) -> "CosmosDBStorage":
        """Create and initialize a CosmosDBStorage instance.

        Args:
            config: Cosmos DB configuration (defaults to env vars)

        Returns:
            Initialized CosmosDBStorage instance
        """
        if config is None:
            config = CosmosConfig.from_env()

        client = CosmosClientWrapper(config)
        await client.initialize()
        return cls(client)

    async def close(self) -> None:
        """Close the Cosmos DB connection."""
        await self.client.close()

    # =========================================================================
    # Session CRUD
    # =========================================================================

    async def create_session(self, metadata: SessionMetadata) -> SessionMetadata:
        """Create a new session."""
        # Check if session already exists
        existing = await self.client.read_item(
            SESSIONS_CONTAINER,
            metadata.session_id,
            metadata.user_id,
        )
        if existing is not None:
            raise SessionExistsError(metadata.session_id)

        # Create session document
        doc = metadata.to_dict()
        doc["id"] = metadata.session_id
        doc["_type"] = "session"

        await self.client.create_item(SESSIONS_CONTAINER, doc, metadata.user_id)
        return metadata

    async def get_session(self, user_id: str, session_id: str) -> SessionMetadata | None:
        """Get session metadata by ID."""
        doc = await self.client.read_item(SESSIONS_CONTAINER, session_id, user_id)
        if doc is None:
            return None
        return SessionMetadata.from_dict(doc)

    async def update_session(self, metadata: SessionMetadata) -> SessionMetadata:
        """Update session metadata."""
        existing = await self.get_session(metadata.user_id, metadata.session_id)
        if existing is None:
            raise SessionNotFoundError(metadata.session_id, metadata.user_id)

        metadata.updated = datetime.now(UTC)

        doc = metadata.to_dict()
        doc["id"] = metadata.session_id
        doc["_type"] = "session"

        await self.client.upsert_item(SESSIONS_CONTAINER, doc, metadata.user_id)
        return metadata

    async def delete_session(self, user_id: str, session_id: str) -> bool:
        """Delete a session and all its data."""
        # Check if session exists
        existing = await self.get_session(user_id, session_id)
        if existing is None:
            return False

        partition_key = make_partition_key(user_id, session_id)

        # Delete all transcript messages
        messages = await self.client.query_items(
            TRANSCRIPT_CONTAINER,
            "SELECT c.id FROM c WHERE c.user_id_session_id = @pk",
            [{"name": "@pk", "value": partition_key}],
            partition_key,
        )
        for msg in messages:
            await self.client.delete_item(TRANSCRIPT_CONTAINER, msg["id"], partition_key)

        # Delete all events
        events = await self.client.query_items(
            EVENTS_CONTAINER,
            "SELECT c.id FROM c WHERE c.user_id_session_id = @pk",
            [{"name": "@pk", "value": partition_key}],
            partition_key,
        )
        for event in events:
            await self.client.delete_item(EVENTS_CONTAINER, event["id"], partition_key)

        # Delete all chunks
        chunks = await self.client.query_items(
            CHUNKS_CONTAINER,
            "SELECT c.id FROM c WHERE c.user_id_session_id = @pk",
            [{"name": "@pk", "value": partition_key}],
            partition_key,
        )
        for chunk in chunks:
            await self.client.delete_item(CHUNKS_CONTAINER, chunk["id"], partition_key)

        # Delete session
        return await self.client.delete_item(SESSIONS_CONTAINER, session_id, user_id)

    async def list_sessions(self, query: SessionQuery) -> list[SessionMetadata]:
        """List sessions matching query."""
        # Build query with user isolation
        conditions = ["c.user_id = @user_id"]
        params: list[dict[str, Any]] = [{"name": "@user_id", "value": query.user_id}]

        if query.project_slug:
            conditions.append("c.project_slug = @project_slug")
            params.append({"name": "@project_slug", "value": query.project_slug})

        if query.name_contains:
            conditions.append("CONTAINS(LOWER(c.name), @name_contains)")
            params.append({"name": "@name_contains", "value": query.name_contains.lower()})

        if query.created_after:
            conditions.append("c.created >= @created_after")
            params.append({"name": "@created_after", "value": query.created_after.isoformat()})

        if query.created_before:
            conditions.append("c.created <= @created_before")
            params.append({"name": "@created_before", "value": query.created_before.isoformat()})

        # Build ORDER BY clause
        order_field = {"created": "c.created", "updated": "c.updated", "name": "c.name"}[
            query.order_by
        ]
        order_dir = "DESC" if query.order_desc else "ASC"

        sql = f"""
            SELECT * FROM c
            WHERE {" AND ".join(conditions)}
            ORDER BY {order_field} {order_dir}
            OFFSET {query.offset} LIMIT {query.limit}
        """

        results = await self.client.query_items(
            SESSIONS_CONTAINER,
            sql,
            params,
            query.user_id,
        )

        return [SessionMetadata.from_dict(doc) for doc in results]

    # =========================================================================
    # Transcript Operations
    # =========================================================================

    async def append_message(
        self,
        user_id: str,
        session_id: str,
        message: TranscriptMessage,
    ) -> TranscriptMessage:
        """Append a message to the transcript."""
        # Verify session exists
        metadata = await self.get_session(user_id, session_id)
        if metadata is None:
            raise SessionNotFoundError(session_id, user_id)

        partition_key = make_partition_key(user_id, session_id)

        # Get current message count for sequence number
        count_result = await self.client.query_items(
            TRANSCRIPT_CONTAINER,
            "SELECT VALUE COUNT(1) FROM c WHERE c.user_id_session_id = @pk",
            [{"name": "@pk", "value": partition_key}],
            partition_key,
        )
        # SELECT VALUE returns raw values, not dicts - cast appropriately
        raw_count = count_result[0] if count_result else 0
        message.sequence = int(raw_count) if isinstance(raw_count, (int, float, str)) else 0

        # Create message document
        doc = message.to_dict()
        doc["id"] = f"{session_id}_msg_{message.sequence}"
        doc["user_id_session_id"] = partition_key
        doc["user_id"] = user_id
        doc["session_id"] = session_id
        doc["_type"] = "transcript_message"

        await self.client.create_item(TRANSCRIPT_CONTAINER, doc, partition_key)

        # Update session metadata
        metadata.message_count = message.sequence + 1
        metadata.turn_count = max(metadata.turn_count, message.turn)
        await self.update_session(metadata)

        return message

    async def get_transcript(
        self,
        user_id: str,
        session_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TranscriptMessage]:
        """Get transcript messages."""
        partition_key = make_partition_key(user_id, session_id)

        sql = f"""
            SELECT * FROM c
            WHERE c.user_id_session_id = @pk
            ORDER BY c.sequence ASC
            OFFSET {offset} LIMIT {limit or 1000}
        """

        results = await self.client.query_items(
            TRANSCRIPT_CONTAINER,
            sql,
            [{"name": "@pk", "value": partition_key}],
            partition_key,
        )

        return [TranscriptMessage.from_dict(doc) for doc in results]

    async def get_transcript_for_turn(
        self,
        user_id: str,
        session_id: str,
        turn: int,
    ) -> list[TranscriptMessage]:
        """Get all messages for a specific turn."""
        partition_key = make_partition_key(user_id, session_id)

        results = await self.client.query_items(
            TRANSCRIPT_CONTAINER,
            "SELECT * FROM c WHERE c.user_id_session_id = @pk AND c.turn = @turn ORDER BY c.sequence",
            [
                {"name": "@pk", "value": partition_key},
                {"name": "@turn", "value": turn},
            ],
            partition_key,
        )

        return [TranscriptMessage.from_dict(doc) for doc in results]

    # =========================================================================
    # Event Operations
    # =========================================================================

    async def append_event(
        self,
        user_id: str,
        session_id: str,
        event_id: str,
        event_type: str,
        data: dict[str, Any],
        turn: int | None = None,
    ) -> EventSummary:
        """Append an event to the session."""
        # Verify session exists
        metadata = await self.get_session(user_id, session_id)
        if metadata is None:
            raise SessionNotFoundError(session_id, user_id)

        partition_key = make_partition_key(user_id, session_id)
        ts = datetime.now(UTC)
        data_size = get_data_size(data)

        # Check if we need to chunk the event
        if should_chunk(data):
            # Store chunked event
            chunks = chunk_event(event_id, data)
            for chunk in chunks:
                chunk_doc = chunk.to_dict()
                chunk_doc["user_id_session_id"] = partition_key
                chunk_doc["user_id"] = user_id
                chunk_doc["session_id"] = session_id
                await self.client.create_item(CHUNKS_CONTAINER, chunk_doc, partition_key)

            # Store event metadata only (no data)
            event_doc = {
                "id": event_id,
                "event_id": event_id,
                "event_type": event_type,
                "ts": ts.isoformat(),
                "turn": turn,
                "user_id_session_id": partition_key,
                "user_id": user_id,
                "session_id": session_id,
                "is_chunked": True,
                "chunk_count": len(chunks),
                "data_size_bytes": data_size,
                "summary": _extract_event_summary(data),
                "_type": "event",
            }
        else:
            # Store event with inline data
            event_doc = {
                "id": event_id,
                "event_id": event_id,
                "event_type": event_type,
                "ts": ts.isoformat(),
                "turn": turn,
                "user_id_session_id": partition_key,
                "user_id": user_id,
                "session_id": session_id,
                "is_chunked": False,
                "data": data,
                "data_size_bytes": data_size,
                "summary": _extract_event_summary(data),
                "_type": "event",
            }

        await self.client.create_item(EVENTS_CONTAINER, event_doc, partition_key)

        # Update session metadata
        metadata.event_count += 1
        await self.update_session(metadata)

        return EventSummary(
            event_id=event_id,
            event_type=event_type,
            ts=ts,
            turn=turn,
            summary=_extract_event_summary(data),
            data_size_bytes=data_size,
        )

    async def query_events(self, query: EventQuery) -> list[EventSummary]:
        """Query events, returning summaries only.

        CRITICAL: This method MUST NEVER return full event data.
        Only returns EventSummary objects with safe projection fields.
        """
        partition_key = make_partition_key(query.user_id, query.session_id)

        # Build query - NEVER select 'data' field
        conditions = ["c.user_id_session_id = @pk"]
        params: list[dict[str, Any]] = [{"name": "@pk", "value": partition_key}]

        if query.event_types:
            conditions.append("ARRAY_CONTAINS(@event_types, c.event_type)")
            params.append({"name": "@event_types", "value": query.event_types})

        if query.turn is not None:
            conditions.append("c.turn = @turn")
            params.append({"name": "@turn", "value": query.turn})

        if query.turn_gte is not None:
            conditions.append("c.turn >= @turn_gte")
            params.append({"name": "@turn_gte", "value": query.turn_gte})

        if query.turn_lte is not None:
            conditions.append("c.turn <= @turn_lte")
            params.append({"name": "@turn_lte", "value": query.turn_lte})

        if query.after:
            conditions.append("c.ts > @after")
            params.append({"name": "@after", "value": query.after.isoformat()})

        if query.before:
            conditions.append("c.ts < @before")
            params.append({"name": "@before", "value": query.before.isoformat()})

        # CRITICAL: Only select safe projection fields, NEVER 'data'
        sql = f"""
            SELECT c.event_id, c.event_type, c.ts, c.turn,
                   c.summary, c.data_size_bytes
            FROM c
            WHERE {" AND ".join(conditions)}
            ORDER BY c.ts ASC
            OFFSET {query.offset} LIMIT {query.limit}
        """

        results = await self.client.query_items(
            EVENTS_CONTAINER,
            sql,
            params,
            partition_key,
        )

        return [
            EventSummary(
                event_id=doc["event_id"],
                event_type=doc["event_type"],
                ts=datetime.fromisoformat(doc["ts"]),
                turn=doc.get("turn"),
                summary=doc.get("summary", {}),
                data_size_bytes=doc.get("data_size_bytes", 0),
            )
            for doc in results
        ]

    async def get_event_data(
        self,
        user_id: str,
        session_id: str,
        event_id: str,
    ) -> dict[str, Any] | None:
        """Get full data for a specific event.

        This is the ONLY method that returns full event data.
        """
        partition_key = make_partition_key(user_id, session_id)

        event_doc = await self.client.read_item(EVENTS_CONTAINER, event_id, partition_key)
        if event_doc is None:
            return None

        # If not chunked, return inline data
        if not event_doc.get("is_chunked"):
            return event_doc.get("data", {})

        # Reassemble from chunks
        chunks_data = await self.client.query_items(
            CHUNKS_CONTAINER,
            "SELECT * FROM c WHERE c.event_id = @event_id ORDER BY c.chunk_index",
            [{"name": "@event_id", "value": event_id}],
            partition_key,
        )

        if not chunks_data:
            return None

        chunks = [EventChunk.from_dict(c) for c in chunks_data]
        return reassemble_event(chunks)

    async def get_event_aggregates(
        self,
        user_id: str,
        session_id: str,
    ) -> AggregateStats:
        """Get aggregate statistics for all events in a session."""
        partition_key = make_partition_key(user_id, session_id)

        # Get event type counts
        type_counts = await self.client.query_items(
            EVENTS_CONTAINER,
            """
            SELECT c.event_type, COUNT(1) as count
            FROM c
            WHERE c.user_id_session_id = @pk
            GROUP BY c.event_type
            """,
            [{"name": "@pk", "value": partition_key}],
            partition_key,
        )

        event_types = {doc["event_type"]: doc["count"] for doc in type_counts}
        event_count = sum(event_types.values())

        # Get aggregated stats from summaries
        stats = await self.client.query_items(
            EVENTS_CONTAINER,
            """
            SELECT
                SUM(c.summary.usage.input_tokens) as total_input_tokens,
                SUM(c.summary.usage.output_tokens) as total_output_tokens,
                SUM(c.summary.duration_ms) as total_duration_ms,
                SUM(CASE WHEN c.summary.has_error = true THEN 1 ELSE 0 END) as error_count,
                SUM(CASE WHEN c.summary.has_tool_calls = true THEN 1 ELSE 0 END) as tool_call_count
            FROM c
            WHERE c.user_id_session_id = @pk
            """,
            [{"name": "@pk", "value": partition_key}],
            partition_key,
        )

        agg = stats[0] if stats else {}

        return AggregateStats(
            event_count=event_count,
            event_types=event_types,
            total_input_tokens=agg.get("total_input_tokens") or 0,
            total_output_tokens=agg.get("total_output_tokens") or 0,
            total_duration_ms=agg.get("total_duration_ms") or 0,
            error_count=agg.get("error_count") or 0,
            tool_call_count=agg.get("tool_call_count") or 0,
        )

    # =========================================================================
    # Rewind Operations
    # =========================================================================

    async def rewind_to_turn(
        self,
        user_id: str,
        session_id: str,
        turn: int,
        create_backup: bool = True,
    ) -> RewindResult:
        """Rewind session to a specific turn.

        Note: create_backup is ignored for Cosmos DB as backups are
        handled differently (point-in-time restore at account level).
        """
        metadata = await self.get_session(user_id, session_id)
        if metadata is None:
            raise SessionNotFoundError(session_id, user_id)

        partition_key = make_partition_key(user_id, session_id)

        # Find and delete messages after turn
        messages_to_delete = await self.client.query_items(
            TRANSCRIPT_CONTAINER,
            "SELECT c.id FROM c WHERE c.user_id_session_id = @pk AND c.turn > @turn",
            [
                {"name": "@pk", "value": partition_key},
                {"name": "@turn", "value": turn},
            ],
            partition_key,
        )

        for msg in messages_to_delete:
            await self.client.delete_item(TRANSCRIPT_CONTAINER, msg["id"], partition_key)

        # Find and delete events after turn
        events_to_delete = await self.client.query_items(
            EVENTS_CONTAINER,
            "SELECT c.id, c.is_chunked FROM c WHERE c.user_id_session_id = @pk AND c.turn > @turn",
            [
                {"name": "@pk", "value": partition_key},
                {"name": "@turn", "value": turn},
            ],
            partition_key,
        )

        for event in events_to_delete:
            # Delete chunks if event was chunked
            if event.get("is_chunked"):
                chunks = await self.client.query_items(
                    CHUNKS_CONTAINER,
                    "SELECT c.id FROM c WHERE c.event_id = @event_id",
                    [{"name": "@event_id", "value": event["id"]}],
                    partition_key,
                )
                for chunk in chunks:
                    await self.client.delete_item(CHUNKS_CONTAINER, chunk["id"], partition_key)

            await self.client.delete_item(EVENTS_CONTAINER, event["id"], partition_key)

        # Update metadata
        metadata.turn_count = turn
        metadata.message_count -= len(messages_to_delete)
        metadata.event_count -= len(events_to_delete)
        await self.update_session(metadata)

        return RewindResult(
            success=True,
            messages_removed=len(messages_to_delete),
            events_removed=len(events_to_delete),
            new_turn_count=turn,
            backup_path=None,  # No local backup for Cosmos DB
        )

    async def rewind_to_timestamp(
        self,
        user_id: str,
        session_id: str,
        timestamp: datetime,
        create_backup: bool = True,
    ) -> RewindResult:
        """Rewind session to a specific timestamp."""
        metadata = await self.get_session(user_id, session_id)
        if metadata is None:
            raise SessionNotFoundError(session_id, user_id)

        partition_key = make_partition_key(user_id, session_id)
        ts_str = timestamp.isoformat()

        # Find and delete messages after timestamp
        messages_to_delete = await self.client.query_items(
            TRANSCRIPT_CONTAINER,
            "SELECT c.id FROM c WHERE c.user_id_session_id = @pk AND c.timestamp > @ts",
            [
                {"name": "@pk", "value": partition_key},
                {"name": "@ts", "value": ts_str},
            ],
            partition_key,
        )

        for msg in messages_to_delete:
            await self.client.delete_item(TRANSCRIPT_CONTAINER, msg["id"], partition_key)

        # Find and delete events after timestamp
        events_to_delete = await self.client.query_items(
            EVENTS_CONTAINER,
            "SELECT c.id, c.is_chunked FROM c WHERE c.user_id_session_id = @pk AND c.ts > @ts",
            [
                {"name": "@pk", "value": partition_key},
                {"name": "@ts", "value": ts_str},
            ],
            partition_key,
        )

        for event in events_to_delete:
            if event.get("is_chunked"):
                chunks = await self.client.query_items(
                    CHUNKS_CONTAINER,
                    "SELECT c.id FROM c WHERE c.event_id = @event_id",
                    [{"name": "@event_id", "value": event["id"]}],
                    partition_key,
                )
                for chunk in chunks:
                    await self.client.delete_item(CHUNKS_CONTAINER, chunk["id"], partition_key)

            await self.client.delete_item(EVENTS_CONTAINER, event["id"], partition_key)

        # Get new turn count
        remaining_messages = await self.client.query_items(
            TRANSCRIPT_CONTAINER,
            "SELECT VALUE MAX(c.turn) FROM c WHERE c.user_id_session_id = @pk",
            [{"name": "@pk", "value": partition_key}],
            partition_key,
        )
        # SELECT VALUE returns raw values, not dicts - cast appropriately
        raw_turn = remaining_messages[0] if remaining_messages and remaining_messages[0] else 0
        new_turn_count = int(raw_turn) if isinstance(raw_turn, (int, float, str)) else 0

        # Update metadata
        metadata.turn_count = new_turn_count
        metadata.message_count -= len(messages_to_delete)
        metadata.event_count -= len(events_to_delete)
        await self.update_session(metadata)

        return RewindResult(
            success=True,
            messages_removed=len(messages_to_delete),
            events_removed=len(events_to_delete),
            new_turn_count=new_turn_count,
            backup_path=None,
        )

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search_sessions(
        self,
        user_id: str,
        query_text: str,
        project_slug: str | None = None,
        limit: int = 20,
    ) -> list[SessionMetadata]:
        """Search sessions by text content."""
        conditions = [
            "c.user_id = @user_id",
            "(CONTAINS(LOWER(c.name), @query) OR CONTAINS(LOWER(c.description), @query))",
        ]
        params: list[dict[str, Any]] = [
            {"name": "@user_id", "value": user_id},
            {"name": "@query", "value": query_text.lower()},
        ]

        if project_slug:
            conditions.append("c.project_slug = @project_slug")
            params.append({"name": "@project_slug", "value": project_slug})

        sql = f"""
            SELECT * FROM c
            WHERE {" AND ".join(conditions)}
            ORDER BY c.updated DESC
            OFFSET 0 LIMIT {limit}
        """

        results = await self.client.query_items(
            SESSIONS_CONTAINER,
            sql,
            params,
            user_id,
        )

        return [SessionMetadata.from_dict(doc) for doc in results]

    # =========================================================================
    # Sync Operations
    # =========================================================================

    async def get_sync_status(
        self,
        user_id: str,
        session_id: str,
    ) -> SyncStatus:
        """Get synchronization status.

        For pure Cosmos DB storage, always returns synced
        since all writes go directly to the cloud.
        """
        return SyncStatus(
            is_synced=True,
            pending_changes=0,
            last_sync=datetime.now(UTC),
            conflict_count=0,
        )

    async def sync_now(
        self,
        user_id: str,
        session_id: str | None = None,
    ) -> SyncStatus:
        """Trigger immediate sync.

        For pure Cosmos DB storage, this is a no-op.
        """
        return SyncStatus(
            is_synced=True,
            pending_changes=0,
            last_sync=datetime.now(UTC),
            conflict_count=0,
        )

    # =========================================================================
    # Session Sharing Operations
    # =========================================================================

    async def set_session_visibility(
        self,
        user_id: str,
        session_id: str,
        visibility: SessionVisibility,
        team_ids: list[str] | None = None,
    ) -> SessionMetadata:
        """Change session visibility. Only owner can change."""
        # Get session and verify ownership
        session = await self.get_session(user_id, session_id)
        if session is None:
            raise SessionNotFoundError(session_id, user_id)

        if session.user_id != user_id:
            raise PermissionDeniedError(user_id, session_id, "only_owner_can_change")

        # Get user's membership for org_id
        membership = await self._membership_store.get_membership(user_id)

        # Update session metadata
        old_visibility = session.visibility
        session.visibility = visibility
        session.team_ids = team_ids if visibility == SessionVisibility.TEAM else None
        session.org_id = membership.org_id if membership else None

        # Set shared_at timestamp when first sharing
        if old_visibility == SessionVisibility.PRIVATE and visibility != SessionVisibility.PRIVATE:
            session.shared_at = datetime.now(UTC)

        # Update session in sessions container
        await self.update_session(session)

        # Manage shared_sessions index
        if visibility == SessionVisibility.PRIVATE:
            # Remove from shared_sessions index
            await self._remove_from_shared_index(session)
        else:
            # Add/update in shared_sessions index
            await self._upsert_to_shared_index(session, user_id)

        return session

    async def _upsert_to_shared_index(self, session: SessionMetadata, owner_name: str) -> None:
        """Add or update session in shared_sessions index."""
        doc = {
            "id": session.session_id,
            "session_id": session.session_id,
            "owner_user_id": session.user_id,
            "owner_name": owner_name,
            "name": session.name,
            "description": session.description,
            "visibility": session.visibility.value,
            "org_id": session.org_id,
            "team_ids": session.team_ids,
            "project_slug": session.project_slug,
            "bundle": session.bundle,
            "model": session.model,
            "tags": session.tags,
            "turn_count": session.turn_count,
            "message_count": session.message_count,
            "created": session.created.isoformat(),
            "updated": session.updated.isoformat(),
            "shared_at": session.shared_at.isoformat() if session.shared_at else None,
            "_type": "shared_session",
        }
        # Partition by visibility for efficient querying
        await self.client.upsert_item(SHARED_SESSIONS_CONTAINER, doc, session.visibility.value)

    async def _remove_from_shared_index(self, session: SessionMetadata) -> None:
        """Remove session from shared_sessions index."""
        # Try to remove from all visibility partitions
        for vis in [
            SessionVisibility.TEAM,
            SessionVisibility.ORGANIZATION,
            SessionVisibility.PUBLIC,
        ]:
            try:
                await self.client.delete_item(
                    SHARED_SESSIONS_CONTAINER, session.session_id, vis.value
                )
            except Exception:
                # Ignore if not found in this partition
                pass

    async def query_shared_sessions(
        self,
        query: SharedSessionQuery,
    ) -> list[SharedSessionSummary]:
        """Query shared sessions visible to the requester."""
        # Get requester's membership
        membership = await self._membership_store.get_membership(query.requester_user_id)

        results: list[SharedSessionSummary] = []

        # Build visibility conditions based on scope and membership
        if query.scope == "public":
            # Only query public sessions
            results.extend(
                await self._query_shared_by_visibility(query, SessionVisibility.PUBLIC, membership)
            )
        elif query.scope == "organization" and membership:
            # Query org + public sessions
            results.extend(
                await self._query_shared_by_visibility(
                    query, SessionVisibility.ORGANIZATION, membership
                )
            )
            results.extend(
                await self._query_shared_by_visibility(query, SessionVisibility.PUBLIC, membership)
            )
        elif query.scope == "team" and membership and membership.team_ids:
            # Query team + org + public sessions
            results.extend(
                await self._query_shared_by_visibility(query, SessionVisibility.TEAM, membership)
            )
            results.extend(
                await self._query_shared_by_visibility(
                    query, SessionVisibility.ORGANIZATION, membership
                )
            )
            results.extend(
                await self._query_shared_by_visibility(query, SessionVisibility.PUBLIC, membership)
            )

        # Sort results
        if query.order_by == "updated":
            results.sort(key=lambda s: s.updated, reverse=query.order_desc)
        elif query.order_by == "created":
            results.sort(key=lambda s: s.created, reverse=query.order_desc)
        elif query.order_by == "shared_at":
            results.sort(key=lambda s: s.shared_at or s.created, reverse=query.order_desc)

        # Apply pagination
        return results[query.offset : query.offset + query.limit]

    async def _query_shared_by_visibility(
        self,
        query: SharedSessionQuery,
        visibility: SessionVisibility,
        membership: UserMembership | None,
    ) -> list[SharedSessionSummary]:
        """Query shared sessions for a specific visibility level."""
        conditions = ["c.visibility = @visibility"]
        params: list[dict[str, Any]] = [{"name": "@visibility", "value": visibility.value}]

        # For team visibility, filter by user's teams
        if visibility == SessionVisibility.TEAM:
            if not membership or not membership.team_ids:
                return []
            if query.team_ids:
                # Filter to specific requested teams that user is member of
                allowed_teams = set(membership.team_ids) & set(query.team_ids)
                if not allowed_teams:
                    return []
                conditions.append("ARRAY_LENGTH(SetIntersect(c.team_ids, @team_ids)) > 0")
                params.append({"name": "@team_ids", "value": list(allowed_teams)})
            else:
                conditions.append("ARRAY_LENGTH(SetIntersect(c.team_ids, @team_ids)) > 0")
                params.append({"name": "@team_ids", "value": membership.team_ids})

        # For org visibility, filter by user's org
        if visibility == SessionVisibility.ORGANIZATION:
            if not membership:
                return []
            conditions.append("c.org_id = @org_id")
            params.append({"name": "@org_id", "value": membership.org_id})

        # Apply additional filters
        if query.project_slug:
            conditions.append("c.project_slug = @project_slug")
            params.append({"name": "@project_slug", "value": query.project_slug})

        if query.owner_user_id:
            conditions.append("c.owner_user_id = @owner_user_id")
            params.append({"name": "@owner_user_id", "value": query.owner_user_id})

        if query.name_contains:
            conditions.append("CONTAINS(LOWER(c.name), @name_contains)")
            params.append({"name": "@name_contains", "value": query.name_contains.lower()})

        if query.tags:
            conditions.append("ARRAY_LENGTH(SetIntersect(c.tags, @tags)) > 0")
            params.append({"name": "@tags", "value": query.tags})

        if query.created_after:
            conditions.append("c.created >= @created_after")
            params.append({"name": "@created_after", "value": query.created_after.isoformat()})

        if query.updated_after:
            conditions.append("c.updated >= @updated_after")
            params.append({"name": "@updated_after", "value": query.updated_after.isoformat()})

        sql = f"""
            SELECT * FROM c
            WHERE {" AND ".join(conditions)}
        """

        docs = await self.client.query_items(
            SHARED_SESSIONS_CONTAINER,
            sql,
            params,
            visibility.value,
        )

        return [SharedSessionSummary.from_dict(doc) for doc in docs]

    async def get_shared_session(
        self,
        requester_user_id: str,
        session_id: str,
    ) -> SessionMetadata | None:
        """Get a shared session with access control."""
        # First, try to find the session in the shared index to get owner info
        # We need to check all visibility partitions since we don't know visibility
        session_doc = None
        for vis in [
            SessionVisibility.PUBLIC,
            SessionVisibility.ORGANIZATION,
            SessionVisibility.TEAM,
        ]:
            doc = await self.client.read_item(SHARED_SESSIONS_CONTAINER, session_id, vis.value)
            if doc:
                session_doc = doc
                break

        if not session_doc:
            return None

        # Get the actual session
        owner_user_id = session_doc["owner_user_id"]
        session = await self.get_session(owner_user_id, session_id)
        if session is None:
            return None

        # Check access
        membership = await self._membership_store.get_membership(requester_user_id)
        decision = await self._access_controller.check_access(
            requester_user_id, session, membership, Permission.READ
        )

        if not decision.allowed:
            return None

        return session

    async def get_shared_transcript(
        self,
        requester_user_id: str,
        session_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TranscriptMessage]:
        """Get transcript from a shared session (read-only)."""
        # Get the shared session with access check
        session = await self.get_shared_session(requester_user_id, session_id)
        if session is None:
            raise PermissionDeniedError(requester_user_id, session_id, "no_read_access")

        # Use the owner's user_id to get the transcript
        return await self.get_transcript(session.user_id, session_id, limit, offset)

    async def get_user_membership(
        self,
        user_id: str,
    ) -> UserMembership | None:
        """Get user's organization and team memberships."""
        return await self._membership_store.get_membership(user_id)
