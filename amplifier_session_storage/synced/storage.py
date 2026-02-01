"""
Synced Cosmos storage implementation.

Combines LocalFileStorage + CosmosDBStorage for offline-first operation:
- Writes go to local first, then queued for sync
- Reads prefer local cache, fallback to cloud
- Handles offline operation transparently
- Triggers sync on connectivity changes
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..cosmos.client import CosmosConfig
from ..cosmos.storage import CosmosDBStorage
from ..local.storage import LocalFileStorage
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
from ..sync.conflict import ConflictResolver
from ..sync.engine import SyncEngine
from ..sync.tracker import ChangeTracker, EntityType


class SyncedCosmosStorage(SessionStorage):
    """Synced storage combining local and cloud.

    Implements the SessionStorage ABC with offline-first behavior:
    - All writes go to local storage first
    - Changes are tracked and queued for sync
    - Reads prefer local, with cloud fallback
    - Background sync keeps local and cloud in sync

    This is the recommended storage implementation for production use.
    """

    def __init__(
        self,
        local: LocalFileStorage,
        cloud: CosmosDBStorage,
        tracker: ChangeTracker,
        resolver: ConflictResolver,
        sync_engine: SyncEngine | None = None,
    ):
        """Initialize synced storage.

        Args:
            local: Local file storage instance
            cloud: Cloud Cosmos DB storage instance
            tracker: Change tracker for sync queue
            resolver: Conflict resolver
            sync_engine: Optional sync engine (created if not provided)
        """
        self.local = local
        self.cloud = cloud
        self.tracker = tracker
        self.resolver = resolver
        self._sync_engine = sync_engine or SyncEngine(
            local_storage=local,
            cloud_storage=cloud,
            tracker=tracker,
            resolver=resolver,
        )
        self._is_online = True

    @classmethod
    async def create(
        cls,
        local_base_path: Path | None = None,
        cosmos_config: CosmosConfig | None = None,
        device_id: str | None = None,
    ) -> "SyncedCosmosStorage":
        """Create and initialize a SyncedCosmosStorage instance.

        Args:
            local_base_path: Base path for local storage
            cosmos_config: Cosmos DB configuration
            device_id: Unique device ID

        Returns:
            Initialized SyncedCosmosStorage instance
        """
        import uuid

        if device_id is None:
            device_id = str(uuid.uuid4())

        # Initialize local storage
        local = LocalFileStorage(local_base_path)

        # Initialize cloud storage
        cloud = await CosmosDBStorage.create(cosmos_config)

        # Initialize tracker and resolver
        base_path = local_base_path or Path.home() / ".amplifier"
        queue_path = base_path / ".sync" / "queue.jsonl"
        tracker = ChangeTracker(queue_path, device_id)
        resolver = ConflictResolver(device_id)

        return cls(local, cloud, tracker, resolver)

    async def close(self) -> None:
        """Close storage connections."""
        await self._sync_engine.stop_auto_sync()
        await self.cloud.close()

    # =========================================================================
    # Session CRUD - Write to local, queue for sync
    # =========================================================================

    async def create_session(self, metadata: SessionMetadata) -> SessionMetadata:
        """Create a new session locally, queue for sync."""
        # Create locally first
        result = await self.local.create_session(metadata)

        # Track change for sync
        await self.tracker.track_create(
            entity_type=EntityType.SESSION,
            entity_id=metadata.session_id,
            session_id=metadata.session_id,
            user_id=metadata.user_id,
            data=metadata.to_dict(),
        )

        # Try immediate sync if online
        if self._is_online:
            try:
                await self.cloud.create_session(metadata)
                await self.tracker.mark_synced(
                    (await self.tracker.get_pending_changes(metadata.session_id))[0].change_id
                )
            except Exception:
                # Will be synced later
                pass

        return result

    async def get_session(self, user_id: str, session_id: str) -> SessionMetadata | None:
        """Get session, preferring local cache."""
        # Try local first
        local_session = await self.local.get_session(user_id, session_id)
        if local_session is not None:
            return local_session

        # Fallback to cloud
        if self._is_online:
            try:
                cloud_session = await self.cloud.get_session(user_id, session_id)
                if cloud_session is not None:
                    # Cache locally
                    await self.local.create_session(cloud_session)
                return cloud_session
            except Exception:
                pass

        return None

    async def update_session(self, metadata: SessionMetadata) -> SessionMetadata:
        """Update session locally, queue for sync."""
        # Update locally first
        result = await self.local.update_session(metadata)

        # Track change for sync
        await self.tracker.track_update(
            entity_type=EntityType.SESSION,
            entity_id=metadata.session_id,
            session_id=metadata.session_id,
            user_id=metadata.user_id,
            data=metadata.to_dict(),
        )

        # Try immediate sync if online
        if self._is_online:
            try:
                await self.cloud.update_session(metadata)
                changes = await self.tracker.get_pending_changes(metadata.session_id)
                if changes:
                    await self.tracker.mark_synced(changes[-1].change_id)
            except Exception:
                pass

        return result

    async def delete_session(self, user_id: str, session_id: str) -> bool:
        """Delete session locally, queue for sync."""
        # Delete locally first
        result = await self.local.delete_session(user_id, session_id)

        if result:
            # Track change for sync
            await self.tracker.track_delete(
                entity_type=EntityType.SESSION,
                entity_id=session_id,
                session_id=session_id,
                user_id=user_id,
            )

            # Try immediate sync if online
            if self._is_online:
                try:
                    await self.cloud.delete_session(user_id, session_id)
                    changes = await self.tracker.get_pending_changes(session_id)
                    if changes:
                        await self.tracker.mark_synced(changes[-1].change_id)
                except Exception:
                    pass

        return result

    async def list_sessions(self, query: SessionQuery) -> list[SessionMetadata]:
        """List sessions from local, with cloud sync."""
        # Get local sessions
        local_sessions = await self.local.list_sessions(query)

        # If online, sync with cloud
        if self._is_online:
            try:
                cloud_sessions = await self.cloud.list_sessions(query)

                # Merge: cloud sessions not in local get added
                local_ids = {s.session_id for s in local_sessions}
                for cloud_session in cloud_sessions:
                    if cloud_session.session_id not in local_ids:
                        # Cache locally
                        try:
                            await self.local.create_session(cloud_session)
                            local_sessions.append(cloud_session)
                        except Exception:
                            pass
            except Exception:
                pass

        return local_sessions

    # =========================================================================
    # Transcript Operations - Write to local, queue for sync
    # =========================================================================

    async def append_message(
        self,
        user_id: str,
        session_id: str,
        message: TranscriptMessage,
    ) -> TranscriptMessage:
        """Append message locally, queue for sync."""
        # Append locally first
        result = await self.local.append_message(user_id, session_id, message)

        # Track change for sync
        await self.tracker.track_append(
            entity_type=EntityType.MESSAGE,
            entity_id=f"{session_id}_msg_{result.sequence}",
            session_id=session_id,
            user_id=user_id,
            data=result.to_dict(),
        )

        # Try immediate sync if online
        if self._is_online:
            try:
                await self.cloud.append_message(user_id, session_id, result)
                changes = await self.tracker.get_pending_changes(session_id)
                if changes:
                    await self.tracker.mark_synced(changes[-1].change_id)
            except Exception:
                pass

        return result

    async def get_transcript(
        self,
        user_id: str,
        session_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TranscriptMessage]:
        """Get transcript from local."""
        return await self.local.get_transcript(user_id, session_id, limit, offset)

    async def get_transcript_for_turn(
        self,
        user_id: str,
        session_id: str,
        turn: int,
    ) -> list[TranscriptMessage]:
        """Get messages for a specific turn from local."""
        return await self.local.get_transcript_for_turn(user_id, session_id, turn)

    # =========================================================================
    # Event Operations - Write to local, queue for sync
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
        """Append event locally, queue for sync."""
        # Append locally first
        result = await self.local.append_event(
            user_id, session_id, event_id, event_type, data, turn
        )

        # Track change for sync
        await self.tracker.track_append(
            entity_type=EntityType.EVENT,
            entity_id=event_id,
            session_id=session_id,
            user_id=user_id,
            data={
                "event_id": event_id,
                "event_type": event_type,
                "data": data,
                "turn": turn,
            },
        )

        # Try immediate sync if online
        if self._is_online:
            try:
                await self.cloud.append_event(user_id, session_id, event_id, event_type, data, turn)
                changes = await self.tracker.get_pending_changes(session_id)
                if changes:
                    await self.tracker.mark_synced(changes[-1].change_id)
            except Exception:
                pass

        return result

    async def query_events(self, query: EventQuery) -> list[EventSummary]:
        """Query events from local.

        CRITICAL: This method MUST NEVER return full event data.
        """
        return await self.local.query_events(query)

    async def get_event_data(
        self,
        user_id: str,
        session_id: str,
        event_id: str,
    ) -> dict[str, Any] | None:
        """Get full event data from local."""
        return await self.local.get_event_data(user_id, session_id, event_id)

    async def get_event_aggregates(
        self,
        user_id: str,
        session_id: str,
    ) -> AggregateStats:
        """Get event aggregates from local."""
        return await self.local.get_event_aggregates(user_id, session_id)

    # =========================================================================
    # Rewind Operations - Apply locally, queue for sync
    # =========================================================================

    async def rewind_to_turn(
        self,
        user_id: str,
        session_id: str,
        turn: int,
        create_backup: bool = True,
    ) -> RewindResult:
        """Rewind session locally, queue for sync."""
        # Rewind locally first (with backup)
        result = await self.local.rewind_to_turn(user_id, session_id, turn, create_backup)

        if result.success:
            # Track session update for sync
            session = await self.local.get_session(user_id, session_id)
            if session:
                await self.tracker.track_update(
                    entity_type=EntityType.SESSION,
                    entity_id=session_id,
                    session_id=session_id,
                    user_id=user_id,
                    data=session.to_dict(),
                )

            # Try immediate sync if online
            if self._is_online:
                try:
                    await self.cloud.rewind_to_turn(user_id, session_id, turn, False)
                    changes = await self.tracker.get_pending_changes(session_id)
                    if changes:
                        await self.tracker.mark_synced(changes[-1].change_id)
                except Exception:
                    pass

        return result

    async def rewind_to_timestamp(
        self,
        user_id: str,
        session_id: str,
        timestamp: datetime,
        create_backup: bool = True,
    ) -> RewindResult:
        """Rewind session to timestamp locally, queue for sync."""
        # Rewind locally first (with backup)
        result = await self.local.rewind_to_timestamp(user_id, session_id, timestamp, create_backup)

        if result.success:
            # Track session update for sync
            session = await self.local.get_session(user_id, session_id)
            if session:
                await self.tracker.track_update(
                    entity_type=EntityType.SESSION,
                    entity_id=session_id,
                    session_id=session_id,
                    user_id=user_id,
                    data=session.to_dict(),
                )

            # Try immediate sync if online
            if self._is_online:
                try:
                    await self.cloud.rewind_to_timestamp(user_id, session_id, timestamp, False)
                    changes = await self.tracker.get_pending_changes(session_id)
                    if changes:
                        await self.tracker.mark_synced(changes[-1].change_id)
                except Exception:
                    pass

        return result

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
        """Search sessions locally."""
        return await self.local.search_sessions(user_id, query_text, project_slug, limit)

    # =========================================================================
    # Sync Operations
    # =========================================================================

    async def get_sync_status(
        self,
        user_id: str,
        session_id: str,
    ) -> SyncStatus:
        """Get synchronization status."""
        return await self._sync_engine.get_status(user_id, session_id)

    async def sync_now(
        self,
        user_id: str,
        session_id: str | None = None,
    ) -> SyncStatus:
        """Trigger immediate sync."""
        result = await self._sync_engine.sync_now(user_id, session_id)

        return SyncStatus(
            is_synced=result.success and result.conflicts == 0,
            pending_changes=result.pushed + result.pulled,
            last_sync=datetime.now(UTC) if result.success else None,
            conflict_count=result.conflicts,
        )

    async def start_auto_sync(self, user_id: str) -> None:
        """Start automatic background sync."""
        await self._sync_engine.start_auto_sync(user_id)

    async def stop_auto_sync(self) -> None:
        """Stop automatic background sync."""
        await self._sync_engine.stop_auto_sync()

    def set_online(self, is_online: bool) -> None:
        """Set online/offline status.

        Call this when network status changes to enable/disable
        immediate sync attempts.
        """
        self._is_online = is_online
        if is_online:
            self._sync_engine.resume()
        else:
            self._sync_engine.pause()

    # =========================================================================
    # Session Sharing Operations - Delegate to cloud
    # =========================================================================

    async def set_session_visibility(
        self,
        user_id: str,
        session_id: str,
        visibility: SessionVisibility,
        team_ids: list[str] | None = None,
    ) -> SessionMetadata:
        """Change session visibility.

        Updates local cache and syncs to cloud.
        """
        # Update in cloud first (sharing is cloud-based)
        result = await self.cloud.set_session_visibility(user_id, session_id, visibility, team_ids)

        # Update local cache
        try:
            await self.local.update_session(result)
        except Exception:
            # Local update failure shouldn't block sharing
            pass

        # Track change for sync
        await self.tracker.track_update(
            entity_type=EntityType.SESSION,
            entity_id=session_id,
            session_id=session_id,
            user_id=user_id,
            data=result.to_dict(),
        )

        return result

    async def query_shared_sessions(
        self,
        query: SharedSessionQuery,
    ) -> list[SharedSessionSummary]:
        """Query shared sessions.

        Shared sessions are cloud-only, delegates to cloud storage.
        """
        if not self._is_online:
            return []
        return await self.cloud.query_shared_sessions(query)

    async def get_shared_session(
        self,
        requester_user_id: str,
        session_id: str,
    ) -> SessionMetadata | None:
        """Get a shared session with access control.

        Delegates to cloud storage for access control.
        """
        if not self._is_online:
            return None
        return await self.cloud.get_shared_session(requester_user_id, session_id)

    async def get_shared_transcript(
        self,
        requester_user_id: str,
        session_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TranscriptMessage]:
        """Get transcript from a shared session (read-only).

        Delegates to cloud storage for access control.
        """
        if not self._is_online:
            return []
        return await self.cloud.get_shared_transcript(requester_user_id, session_id, limit, offset)

    async def get_user_membership(
        self,
        user_id: str,
    ) -> UserMembership | None:
        """Get user's organization and team memberships.

        Delegates to cloud storage.
        """
        if not self._is_online:
            return None
        return await self.cloud.get_user_membership(user_id)
