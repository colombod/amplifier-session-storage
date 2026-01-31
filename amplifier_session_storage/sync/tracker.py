"""
Change tracking for synchronization.

Tracks local changes that need to be synced to the cloud,
and maintains a persistent queue of pending changes.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os

from .version import VersionVector


class ChangeType(Enum):
    """Type of change being tracked."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    APPEND = "append"  # For append-only operations (messages, events)


class EntityType(Enum):
    """Type of entity being changed."""

    SESSION = "session"
    MESSAGE = "message"
    EVENT = "event"


@dataclass
class ChangeRecord:
    """Record of a single change that needs to be synced.

    Attributes:
        change_id: Unique identifier for this change
        entity_type: Type of entity changed
        entity_id: ID of the entity
        session_id: Session this change belongs to
        user_id: User who made the change
        change_type: Type of change
        data: Change payload (for create/update/append)
        version: Version vector at time of change
        timestamp: When the change occurred
        retries: Number of sync attempts
        last_error: Last error message if sync failed
    """

    change_id: str
    entity_type: EntityType
    entity_id: str
    session_id: str
    user_id: str
    change_type: ChangeType
    data: dict[str, Any] | None
    version: VersionVector
    timestamp: datetime = field(default_factory=datetime.utcnow)
    retries: int = 0
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "change_id": self.change_id,
            "entity_type": self.entity_type.value,
            "entity_id": self.entity_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "change_type": self.change_type.value,
            "data": self.data,
            "version": self.version.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "retries": self.retries,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChangeRecord":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()

        return cls(
            change_id=data["change_id"],
            entity_type=EntityType(data["entity_type"]),
            entity_id=data["entity_id"],
            session_id=data["session_id"],
            user_id=data["user_id"],
            change_type=ChangeType(data["change_type"]),
            data=data.get("data"),
            version=VersionVector.from_dict(data.get("version", {})),
            timestamp=timestamp,
            retries=data.get("retries", 0),
            last_error=data.get("last_error"),
        )


class ChangeTracker:
    """Tracks and persists changes for synchronization.

    Maintains a persistent queue of changes that need to be synced
    to the cloud. Changes are written to a local JSONL file for
    durability across restarts.
    """

    def __init__(self, queue_path: Path, device_id: str):
        """Initialize the change tracker.

        Args:
            queue_path: Path to the change queue file
            device_id: Unique ID for this device
        """
        self.queue_path = queue_path
        self.device_id = device_id
        self._changes: list[ChangeRecord] = []
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        """Load changes from disk if not already loaded."""
        if self._loaded:
            return

        if await aiofiles.os.path.exists(self.queue_path):
            try:
                async with aiofiles.open(self.queue_path, encoding="utf-8") as f:
                    content = await f.read()
                    for line in content.strip().split("\n"):
                        if line:
                            data = json.loads(line)
                            self._changes.append(ChangeRecord.from_dict(data))
            except (json.JSONDecodeError, OSError):
                # If queue is corrupted, start fresh
                self._changes = []

        self._loaded = True

    async def _persist(self) -> None:
        """Persist changes to disk."""
        # Ensure directory exists
        await aiofiles.os.makedirs(self.queue_path.parent, exist_ok=True)

        async with aiofiles.open(self.queue_path, "w", encoding="utf-8") as f:
            for change in self._changes:
                await f.write(json.dumps(change.to_dict()) + "\n")

    def _create_version(self) -> VersionVector:
        """Create a new version vector for a change."""
        return VersionVector.initial(self.device_id)

    async def track_create(
        self,
        entity_type: EntityType,
        entity_id: str,
        session_id: str,
        user_id: str,
        data: dict[str, Any],
    ) -> ChangeRecord:
        """Track a create operation.

        Args:
            entity_type: Type of entity created
            entity_id: ID of the entity
            session_id: Session this belongs to
            user_id: User who made the change
            data: Entity data

        Returns:
            Created ChangeRecord
        """
        await self._ensure_loaded()

        change = ChangeRecord(
            change_id=str(uuid.uuid4()),
            entity_type=entity_type,
            entity_id=entity_id,
            session_id=session_id,
            user_id=user_id,
            change_type=ChangeType.CREATE,
            data=data,
            version=self._create_version(),
        )

        self._changes.append(change)
        await self._persist()
        return change

    async def track_update(
        self,
        entity_type: EntityType,
        entity_id: str,
        session_id: str,
        user_id: str,
        data: dict[str, Any],
        previous_version: VersionVector | None = None,
    ) -> ChangeRecord:
        """Track an update operation.

        Args:
            entity_type: Type of entity updated
            entity_id: ID of the entity
            session_id: Session this belongs to
            user_id: User who made the change
            data: Updated entity data
            previous_version: Previous version (for incrementing)

        Returns:
            Created ChangeRecord
        """
        await self._ensure_loaded()

        version = (
            previous_version.increment(self.device_id)
            if previous_version
            else self._create_version()
        )

        change = ChangeRecord(
            change_id=str(uuid.uuid4()),
            entity_type=entity_type,
            entity_id=entity_id,
            session_id=session_id,
            user_id=user_id,
            change_type=ChangeType.UPDATE,
            data=data,
            version=version,
        )

        self._changes.append(change)
        await self._persist()
        return change

    async def track_delete(
        self,
        entity_type: EntityType,
        entity_id: str,
        session_id: str,
        user_id: str,
    ) -> ChangeRecord:
        """Track a delete operation.

        Args:
            entity_type: Type of entity deleted
            entity_id: ID of the entity
            session_id: Session this belongs to
            user_id: User who made the change

        Returns:
            Created ChangeRecord
        """
        await self._ensure_loaded()

        change = ChangeRecord(
            change_id=str(uuid.uuid4()),
            entity_type=entity_type,
            entity_id=entity_id,
            session_id=session_id,
            user_id=user_id,
            change_type=ChangeType.DELETE,
            data=None,
            version=self._create_version(),
        )

        self._changes.append(change)
        await self._persist()
        return change

    async def track_append(
        self,
        entity_type: EntityType,
        entity_id: str,
        session_id: str,
        user_id: str,
        data: dict[str, Any],
    ) -> ChangeRecord:
        """Track an append operation (for messages, events).

        Args:
            entity_type: Type of entity appended
            entity_id: ID of the entity
            session_id: Session this belongs to
            user_id: User who made the change
            data: Appended data

        Returns:
            Created ChangeRecord
        """
        await self._ensure_loaded()

        change = ChangeRecord(
            change_id=str(uuid.uuid4()),
            entity_type=entity_type,
            entity_id=entity_id,
            session_id=session_id,
            user_id=user_id,
            change_type=ChangeType.APPEND,
            data=data,
            version=self._create_version(),
        )

        self._changes.append(change)
        await self._persist()
        return change

    async def get_pending_changes(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        max_retries: int = 5,
    ) -> list[ChangeRecord]:
        """Get pending changes that need to be synced.

        Args:
            session_id: Filter by session (optional)
            user_id: Filter by user (optional)
            max_retries: Exclude changes with more retries than this

        Returns:
            List of pending changes
        """
        await self._ensure_loaded()

        changes = self._changes
        if session_id:
            changes = [c for c in changes if c.session_id == session_id]
        if user_id:
            changes = [c for c in changes if c.user_id == user_id]
        changes = [c for c in changes if c.retries < max_retries]

        return changes

    async def get_pending_count(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> int:
        """Get count of pending changes.

        Args:
            session_id: Filter by session (optional)
            user_id: Filter by user (optional)

        Returns:
            Number of pending changes
        """
        changes = await self.get_pending_changes(session_id, user_id)
        return len(changes)

    async def mark_synced(self, change_id: str) -> bool:
        """Mark a change as synced (remove from queue).

        Args:
            change_id: ID of the change to mark

        Returns:
            True if change was found and removed
        """
        await self._ensure_loaded()

        original_count = len(self._changes)
        self._changes = [c for c in self._changes if c.change_id != change_id]

        if len(self._changes) < original_count:
            await self._persist()
            return True
        return False

    async def mark_failed(self, change_id: str, error: str) -> bool:
        """Mark a change as failed (increment retry count).

        Args:
            change_id: ID of the change
            error: Error message

        Returns:
            True if change was found and updated
        """
        await self._ensure_loaded()

        for change in self._changes:
            if change.change_id == change_id:
                change.retries += 1
                change.last_error = error
                await self._persist()
                return True
        return False

    async def clear_session(self, session_id: str) -> int:
        """Clear all changes for a session.

        Args:
            session_id: Session to clear

        Returns:
            Number of changes removed
        """
        await self._ensure_loaded()

        original_count = len(self._changes)
        self._changes = [c for c in self._changes if c.session_id != session_id]

        if len(self._changes) < original_count:
            await self._persist()
        return original_count - len(self._changes)

    async def clear_all(self) -> int:
        """Clear all pending changes.

        Returns:
            Number of changes removed
        """
        await self._ensure_loaded()

        count = len(self._changes)
        self._changes = []
        await self._persist()
        return count

    async def get_failed_changes(self, min_retries: int = 1) -> list[ChangeRecord]:
        """Get changes that have failed at least once.

        Args:
            min_retries: Minimum retry count to include

        Returns:
            List of failed changes
        """
        await self._ensure_loaded()
        return [c for c in self._changes if c.retries >= min_retries]
