"""
Synchronization engine for bidirectional sync.

Orchestrates synchronization between local and cloud storage:
- Push: Local changes → Cloud
- Pull: Cloud changes → Local
- Conflict detection and resolution
- Exponential backoff for retries
- Network availability detection
"""

import asyncio
import socket
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ..cosmos.storage import CosmosDBStorage
from ..local.storage import LocalFileStorage
from ..protocol import SessionMetadata, SyncStatus, TranscriptMessage
from .conflict import Conflict, ConflictDecision, ConflictResolver
from .tracker import ChangeRecord, ChangeTracker, ChangeType, EntityType
from .version import VersionVector


class SyncState(Enum):
    """Current state of the sync engine."""

    IDLE = "idle"
    SYNCING = "syncing"
    PAUSED = "paused"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class SyncResult:
    """Result of a sync operation."""

    success: bool
    pushed: int = 0
    pulled: int = 0
    conflicts: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: int = 0


@dataclass
class SyncConfig:
    """Configuration for the sync engine."""

    # Retry settings
    max_retries: int = 5
    initial_backoff_ms: int = 1000
    max_backoff_ms: int = 60000
    backoff_multiplier: float = 2.0

    # Sync behavior
    auto_sync_interval_ms: int = 30000  # 30 seconds
    batch_size: int = 50
    sync_on_change: bool = True

    # Network detection
    connectivity_check_url: str = "https://login.microsoftonline.com"
    connectivity_timeout_ms: int = 5000


class SyncEngine:
    """Bidirectional sync engine for local and cloud storage.

    Handles:
    - Tracking local changes
    - Pushing changes to cloud with retry logic
    - Pulling remote changes to local
    - Conflict detection and resolution
    - Network availability detection
    """

    def __init__(
        self,
        local_storage: LocalFileStorage,
        cloud_storage: CosmosDBStorage,
        tracker: ChangeTracker,
        resolver: ConflictResolver,
        config: SyncConfig | None = None,
    ):
        """Initialize the sync engine.

        Args:
            local_storage: Local file storage instance
            cloud_storage: Cloud Cosmos DB storage instance
            tracker: Change tracker instance
            resolver: Conflict resolver instance
            config: Sync configuration
        """
        self.local = local_storage
        self.cloud = cloud_storage
        self.tracker = tracker
        self.resolver = resolver
        self.config = config or SyncConfig()

        self._state = SyncState.IDLE
        self._last_sync: datetime | None = None
        self._pending_conflicts: list[Conflict] = []
        self._sync_task: asyncio.Task[None] | None = None
        self._is_online = True

    @property
    def state(self) -> SyncState:
        """Get current sync state."""
        return self._state

    @property
    def is_online(self) -> bool:
        """Check if we're currently online."""
        return self._is_online

    async def check_connectivity(self) -> bool:
        """Check if we have network connectivity.

        Returns:
            True if online, False otherwise
        """
        try:
            # Simple DNS resolution check
            socket.setdefaulttimeout(self.config.connectivity_timeout_ms / 1000)
            socket.gethostbyname("login.microsoftonline.com")
            self._is_online = True
            return True
        except (OSError, TimeoutError):
            self._is_online = False
            return False

    async def get_status(self, user_id: str, session_id: str | None = None) -> SyncStatus:
        """Get current sync status.

        Args:
            user_id: User ID
            session_id: Optional session ID to check

        Returns:
            Current sync status
        """
        pending = await self.tracker.get_pending_count(session_id, user_id)

        return SyncStatus(
            is_synced=pending == 0 and self._state != SyncState.ERROR,
            pending_changes=pending,
            last_sync=self._last_sync,
            conflict_count=len(self._pending_conflicts),
        )

    async def sync_now(
        self,
        user_id: str,
        session_id: str | None = None,
    ) -> SyncResult:
        """Trigger immediate sync.

        Args:
            user_id: User ID
            session_id: Optional session ID to sync

        Returns:
            Result of sync operation
        """
        if self._state == SyncState.SYNCING:
            return SyncResult(success=False, errors=["Sync already in progress"])

        if not await self.check_connectivity():
            self._state = SyncState.OFFLINE
            return SyncResult(success=False, errors=["No network connectivity"])

        self._state = SyncState.SYNCING
        start_time = datetime.now(UTC)

        try:
            # Push local changes
            push_result = await self._push_changes(user_id, session_id)

            # Pull remote changes
            pull_result = await self._pull_changes(user_id, session_id)

            self._last_sync = datetime.now(UTC)
            self._state = SyncState.IDLE

            duration = int((datetime.now(UTC) - start_time).total_seconds() * 1000)

            return SyncResult(
                success=len(push_result["errors"]) == 0 and len(pull_result["errors"]) == 0,
                pushed=push_result["count"],
                pulled=pull_result["count"],
                conflicts=push_result["conflicts"] + pull_result["conflicts"],
                errors=push_result["errors"] + pull_result["errors"],
                duration_ms=duration,
            )

        except Exception as e:
            self._state = SyncState.ERROR
            return SyncResult(success=False, errors=[str(e)])

    async def _push_changes(
        self,
        user_id: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Push local changes to cloud.

        Returns:
            Dict with count, conflicts, errors
        """
        result: dict[str, Any] = {"count": 0, "conflicts": 0, "errors": []}

        changes = await self.tracker.get_pending_changes(session_id, user_id)

        for change in changes[: self.config.batch_size]:
            try:
                success = await self._push_single_change(change)
                if success:
                    await self.tracker.mark_synced(change.change_id)
                    result["count"] += 1
                else:
                    result["conflicts"] += 1
            except Exception as e:
                await self.tracker.mark_failed(change.change_id, str(e))
                result["errors"].append(f"Failed to push {change.entity_id}: {e}")

        return result

    async def _push_single_change(self, change: ChangeRecord) -> bool:
        """Push a single change to cloud.

        Returns:
            True if pushed successfully, False if conflict detected
        """
        # Get remote version for conflict detection
        remote_data, remote_version = await self._get_remote_version(change)

        if remote_data is not None:
            # Check for conflict
            conflict = self.resolver.detect_conflict(change, remote_data, remote_version)

            if conflict is not None:
                # Try automatic resolution
                decision = self.resolver.resolve_automatically(conflict)

                if decision is None:
                    # Needs manual resolution
                    self._pending_conflicts.append(conflict)
                    return False

                # Apply resolution
                resolved_data = self.resolver.apply_resolution(conflict, decision)
                await self._apply_change_to_cloud(change, resolved_data)
                return True

        # No conflict - apply change directly
        await self._apply_change_to_cloud(change, change.data)
        return True

    async def _get_remote_version(
        self,
        change: ChangeRecord,
    ) -> tuple[dict[str, Any] | None, VersionVector]:
        """Get remote data and version for an entity."""
        if change.entity_type == EntityType.SESSION:
            session = await self.cloud.get_session(change.user_id, change.entity_id)
            if session:
                data = session.to_dict()
                # Extract version from sync state if available
                version = VersionVector.from_dict(data.get("_version", {}))
                return data, version
        elif change.entity_type == EntityType.MESSAGE:
            # Messages are append-only, check for sequence collision
            transcript = await self.cloud.get_transcript(
                change.user_id, change.session_id, limit=1000
            )
            for msg in transcript:
                if msg.sequence == (change.data or {}).get("sequence"):
                    return msg.to_dict(), VersionVector()
        elif change.entity_type == EntityType.EVENT:
            event_data = await self.cloud.get_event_data(
                change.user_id, change.session_id, change.entity_id
            )
            if event_data:
                version = VersionVector.from_dict(event_data.get("_version", {}))
                return event_data, version

        return None, VersionVector()

    async def _apply_change_to_cloud(
        self,
        change: ChangeRecord,
        data: dict[str, Any] | None,
    ) -> None:
        """Apply a change to cloud storage."""
        if change.change_type == ChangeType.DELETE:
            if change.entity_type == EntityType.SESSION:
                await self.cloud.delete_session(change.user_id, change.entity_id)
            return

        if data is None:
            return

        if change.entity_type == EntityType.SESSION:
            if change.change_type == ChangeType.CREATE:
                metadata = SessionMetadata.from_dict(data)
                await self.cloud.create_session(metadata)
            else:
                metadata = SessionMetadata.from_dict(data)
                await self.cloud.update_session(metadata)

        elif change.entity_type == EntityType.MESSAGE:
            message = TranscriptMessage.from_dict(data)
            await self.cloud.append_message(change.user_id, change.session_id, message)

        elif change.entity_type == EntityType.EVENT:
            await self.cloud.append_event(
                change.user_id,
                change.session_id,
                change.entity_id,
                data.get("event_type", "unknown"),
                data.get("data", {}),
                data.get("turn"),
            )

    async def _pull_changes(
        self,
        user_id: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Pull remote changes to local.

        Returns:
            Dict with count, conflicts, errors
        """
        result: dict[str, Any] = {"count": 0, "conflicts": 0, "errors": []}

        try:
            # Get sessions from cloud
            from ..protocol import SessionQuery

            query = SessionQuery(user_id=user_id, project_slug=None, limit=100)
            if session_id:
                # Single session pull
                remote_session = await self.cloud.get_session(user_id, session_id)
                if remote_session:
                    await self._pull_session(remote_session)
                    result["count"] += 1
            else:
                # Pull all sessions
                remote_sessions = await self.cloud.list_sessions(query)
                for session in remote_sessions:
                    try:
                        await self._pull_session(session)
                        result["count"] += 1
                    except Exception as e:
                        result["errors"].append(f"Failed to pull {session.session_id}: {e}")

        except Exception as e:
            result["errors"].append(f"Pull failed: {e}")

        return result

    async def _pull_session(self, remote_session: SessionMetadata) -> None:
        """Pull a single session from cloud to local."""
        local_session = await self.local.get_session(
            remote_session.user_id, remote_session.session_id
        )

        if local_session is None:
            # Create locally
            await self.local.create_session(remote_session)
        else:
            # Check if remote is newer
            if remote_session.updated > local_session.updated:
                await self.local.update_session(remote_session)

        # Pull transcript messages
        remote_transcript = await self.cloud.get_transcript(
            remote_session.user_id, remote_session.session_id
        )
        local_transcript = await self.local.get_transcript(
            remote_session.user_id, remote_session.session_id
        )

        local_sequences = {m.sequence for m in local_transcript}

        for message in remote_transcript:
            if message.sequence not in local_sequences:
                await self.local.append_message(
                    remote_session.user_id, remote_session.session_id, message
                )

    async def resolve_conflict(
        self,
        conflict_id: str,
        decision: ConflictDecision,
    ) -> bool:
        """Resolve a pending conflict manually.

        Args:
            conflict_id: ID of the conflict to resolve
            decision: Resolution decision

        Returns:
            True if conflict was resolved
        """
        conflict = None
        for c in self._pending_conflicts:
            if c.conflict_id == conflict_id:
                conflict = c
                break

        if conflict is None:
            return False

        # Apply resolution
        resolved_data = self.resolver.apply_resolution(conflict, decision)
        await self._apply_change_to_cloud(conflict.local_change, resolved_data)

        # Mark change as synced
        await self.tracker.mark_synced(conflict.local_change.change_id)

        # Remove from pending
        self._pending_conflicts = [
            c for c in self._pending_conflicts if c.conflict_id != conflict_id
        ]

        return True

    def get_pending_conflicts(self) -> list[Conflict]:
        """Get list of pending conflicts requiring manual resolution."""
        return self._pending_conflicts.copy()

    async def start_auto_sync(self, user_id: str) -> None:
        """Start automatic background sync.

        Args:
            user_id: User ID to sync for
        """
        if self._sync_task is not None:
            return

        async def sync_loop() -> None:
            while True:
                try:
                    await asyncio.sleep(self.config.auto_sync_interval_ms / 1000)
                    if self._state != SyncState.PAUSED:
                        await self.sync_now(user_id)
                except asyncio.CancelledError:
                    break
                except Exception:
                    # Log error but continue
                    pass

        self._sync_task = asyncio.create_task(sync_loop())

    async def stop_auto_sync(self) -> None:
        """Stop automatic background sync."""
        if self._sync_task is not None:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

    def pause(self) -> None:
        """Pause sync operations."""
        self._state = SyncState.PAUSED

    def resume(self) -> None:
        """Resume sync operations."""
        if self._state == SyncState.PAUSED:
            self._state = SyncState.IDLE


async def create_sync_engine(
    local_base_path: Path | None = None,
    device_id: str | None = None,
    cosmos_config: Any | None = None,
) -> SyncEngine:
    """Create and initialize a sync engine.

    Args:
        local_base_path: Base path for local storage
        device_id: Unique device ID (generated if not provided)
        cosmos_config: Cosmos DB configuration

    Returns:
        Initialized SyncEngine
    """
    import uuid

    if device_id is None:
        device_id = str(uuid.uuid4())

    # Initialize storages
    local_storage = LocalFileStorage(local_base_path)
    cloud_storage = await CosmosDBStorage.create(cosmos_config)

    # Initialize tracker and resolver
    queue_path = (local_base_path or Path.home() / ".amplifier") / ".sync" / "queue.jsonl"
    tracker = ChangeTracker(queue_path, device_id)
    resolver = ConflictResolver(device_id)

    return SyncEngine(
        local_storage=local_storage,
        cloud_storage=cloud_storage,
        tracker=tracker,
        resolver=resolver,
    )
