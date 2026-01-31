"""
Conflict resolution strategies for synchronization.

Handles conflicts that arise when the same data is modified on
multiple devices before synchronization occurs.

Conflict types:
- Metadata conflicts: Same session metadata modified differently
- Append conflicts: Same sequence number for messages/events
- Rewind conflicts: Session rewound on one device while continued on another
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..protocol import ConflictResolution
from .tracker import ChangeRecord, ChangeType, EntityType
from .version import VersionVector, compare_versions


class ConflictType(Enum):
    """Type of conflict detected."""

    METADATA_DIVERGED = "metadata_diverged"  # Session metadata changed differently
    SEQUENCE_COLLISION = "sequence_collision"  # Same sequence number used
    REWIND_CONFLICT = "rewind_conflict"  # One side rewound, other continued
    DELETE_UPDATE = "delete_update"  # One side deleted, other updated
    CONCURRENT_CREATE = "concurrent_create"  # Same entity created on both sides


@dataclass
class Conflict:
    """Represents a detected conflict.

    Attributes:
        conflict_id: Unique identifier for this conflict
        conflict_type: Type of conflict
        session_id: Session where conflict occurred
        user_id: User who owns the session
        entity_type: Type of entity in conflict
        entity_id: ID of the entity in conflict
        local_change: Local change record
        remote_change: Remote change data
        local_version: Local version vector
        remote_version: Remote version vector
        detected_at: When the conflict was detected
        resolved: Whether conflict has been resolved
        resolution: How the conflict was resolved (if resolved)
    """

    conflict_id: str
    conflict_type: ConflictType
    session_id: str
    user_id: str
    entity_type: EntityType
    entity_id: str
    local_change: ChangeRecord
    remote_change: dict[str, Any]
    local_version: VersionVector
    remote_version: VersionVector
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolution: ConflictResolution | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type.value,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "entity_type": self.entity_type.value,
            "entity_id": self.entity_id,
            "local_change": self.local_change.to_dict(),
            "remote_change": self.remote_change,
            "local_version": self.local_version.to_dict(),
            "remote_version": self.remote_version.to_dict(),
            "detected_at": self.detected_at.isoformat(),
            "resolved": self.resolved,
            "resolution": self.resolution.value if self.resolution else None,
        }


@dataclass
class ConflictDecision:
    """Decision about how to resolve a conflict.

    For automatic resolution, the resolver will make the decision.
    For manual resolution, this captures the user's choice.
    """

    conflict_id: str
    resolution: ConflictResolution
    merged_data: dict[str, Any] | None = None  # For MERGE resolution
    decided_by: str = "auto"  # "auto" or user_id
    decided_at: datetime = field(default_factory=datetime.utcnow)


class ConflictResolver:
    """Resolves conflicts between local and remote changes.

    Implements different strategies for different conflict types:
    - Metadata: Last-writer-wins based on version vector timestamps
    - Append-only: Merge by timestamp + device_id for ordering
    - Rewind: Requires user decision (returns unresolved conflict)
    """

    def __init__(self, device_id: str):
        """Initialize the conflict resolver.

        Args:
            device_id: ID of this device (for tie-breaking)
        """
        self.device_id = device_id

    def detect_conflict(
        self,
        local_change: ChangeRecord,
        remote_data: dict[str, Any],
        remote_version: VersionVector,
    ) -> Conflict | None:
        """Detect if there's a conflict between local and remote.

        Args:
            local_change: Local change record
            remote_data: Remote entity data
            remote_version: Remote version vector

        Returns:
            Conflict if detected, None otherwise
        """
        comparison = compare_versions(local_change.version, remote_version)

        # No conflict if versions are ordered
        if comparison in ("equal", "v1_before_v2", "v2_before_v1"):
            return None

        # Concurrent versions - conflict detected
        conflict_type = self._determine_conflict_type(local_change, remote_data)

        return Conflict(
            conflict_id=f"{local_change.session_id}_{local_change.entity_id}_{local_change.change_id}",
            conflict_type=conflict_type,
            session_id=local_change.session_id,
            user_id=local_change.user_id,
            entity_type=local_change.entity_type,
            entity_id=local_change.entity_id,
            local_change=local_change,
            remote_change=remote_data,
            local_version=local_change.version,
            remote_version=remote_version,
        )

    def _determine_conflict_type(
        self,
        local_change: ChangeRecord,
        remote_data: dict[str, Any],
    ) -> ConflictType:
        """Determine the type of conflict."""
        # Delete vs update
        if local_change.change_type == ChangeType.DELETE:
            return ConflictType.DELETE_UPDATE

        # Check for rewind indicators
        local_turn = (local_change.data or {}).get("turn_count", 0)
        remote_turn = remote_data.get("turn_count", 0)
        if local_turn < remote_turn or remote_turn < local_turn:
            # Check if this looks like a rewind
            if local_change.entity_type == EntityType.SESSION:
                local_event_count = (local_change.data or {}).get("event_count", 0)
                remote_event_count = remote_data.get("event_count", 0)
                if local_event_count < remote_event_count or remote_event_count < local_event_count:
                    return ConflictType.REWIND_CONFLICT

        # Sequence collision for append operations
        if local_change.change_type == ChangeType.APPEND:
            local_seq = (local_change.data or {}).get("sequence")
            remote_seq = remote_data.get("sequence")
            if local_seq is not None and local_seq == remote_seq:
                return ConflictType.SEQUENCE_COLLISION

        # Default to metadata diverged
        return ConflictType.METADATA_DIVERGED

    def resolve_automatically(self, conflict: Conflict) -> ConflictDecision | None:
        """Attempt to resolve a conflict automatically.

        Args:
            conflict: Conflict to resolve

        Returns:
            ConflictDecision if auto-resolved, None if requires manual resolution
        """
        # Rewind conflicts always require user decision
        if conflict.conflict_type == ConflictType.REWIND_CONFLICT:
            return None

        # Delete/update conflicts use remote wins (preserve remote data)
        if conflict.conflict_type == ConflictType.DELETE_UPDATE:
            return ConflictDecision(
                conflict_id=conflict.conflict_id,
                resolution=ConflictResolution.REMOTE_WINS,
            )

        # Metadata conflicts use last-writer-wins
        if conflict.conflict_type == ConflictType.METADATA_DIVERGED:
            return self._resolve_last_writer_wins(conflict)

        # Sequence collisions use merge (reorder by timestamp + device)
        if conflict.conflict_type == ConflictType.SEQUENCE_COLLISION:
            return self._resolve_sequence_collision(conflict)

        # Concurrent creates - merge if possible
        if conflict.conflict_type == ConflictType.CONCURRENT_CREATE:
            return self._resolve_concurrent_create(conflict)

        return None

    def _resolve_last_writer_wins(self, conflict: Conflict) -> ConflictDecision:
        """Resolve using last-writer-wins strategy.

        Compares timestamps, with device_id as tie-breaker.
        """
        local_ts = conflict.local_version.timestamp
        remote_ts = conflict.remote_version.timestamp

        if local_ts > remote_ts:
            return ConflictDecision(
                conflict_id=conflict.conflict_id,
                resolution=ConflictResolution.LOCAL_WINS,
            )
        elif remote_ts > local_ts:
            return ConflictDecision(
                conflict_id=conflict.conflict_id,
                resolution=ConflictResolution.REMOTE_WINS,
            )
        else:
            # Timestamps equal - use device_id as tie-breaker
            # Lexicographically smaller device_id wins for determinism
            local_device = max(conflict.local_version.entries.keys(), default="")
            remote_device = max(conflict.remote_version.entries.keys(), default="")

            if local_device <= remote_device:
                return ConflictDecision(
                    conflict_id=conflict.conflict_id,
                    resolution=ConflictResolution.LOCAL_WINS,
                )
            else:
                return ConflictDecision(
                    conflict_id=conflict.conflict_id,
                    resolution=ConflictResolution.REMOTE_WINS,
                )

    def _resolve_sequence_collision(self, conflict: Conflict) -> ConflictDecision:
        """Resolve sequence collision by merging with new ordering.

        Both items are kept, but reordered by timestamp + device_id.
        """
        local_data = conflict.local_change.data or {}
        remote_data = conflict.remote_change

        # Determine ordering
        local_ts = conflict.local_version.timestamp
        remote_ts = conflict.remote_version.timestamp

        if local_ts <= remote_ts:
            # Local comes first, remote gets bumped sequence
            merged = {
                "local": local_data,
                "remote": remote_data,
                "local_sequence": local_data.get("sequence", 0),
                "remote_sequence": local_data.get("sequence", 0) + 1,
            }
        else:
            # Remote comes first, local gets bumped sequence
            merged = {
                "local": local_data,
                "remote": remote_data,
                "local_sequence": remote_data.get("sequence", 0) + 1,
                "remote_sequence": remote_data.get("sequence", 0),
            }

        return ConflictDecision(
            conflict_id=conflict.conflict_id,
            resolution=ConflictResolution.MERGE,
            merged_data=merged,
        )

    def _resolve_concurrent_create(self, conflict: Conflict) -> ConflictDecision:
        """Resolve concurrent create by merging data.

        For session creation, merge non-conflicting fields.
        """
        local_data = conflict.local_change.data or {}
        remote_data = conflict.remote_change

        # Merge: take newer values for each field
        merged: dict[str, Any] = {}
        all_keys = set(local_data.keys()) | set(remote_data.keys())

        local_ts = conflict.local_version.timestamp
        remote_ts = conflict.remote_version.timestamp

        for key in all_keys:
            local_val = local_data.get(key)
            remote_val = remote_data.get(key)

            if local_val is None:
                merged[key] = remote_val
            elif remote_val is None:
                merged[key] = local_val
            elif local_val == remote_val:
                merged[key] = local_val
            else:
                # Different values - use timestamp to decide
                merged[key] = local_val if local_ts >= remote_ts else remote_val

        return ConflictDecision(
            conflict_id=conflict.conflict_id,
            resolution=ConflictResolution.MERGE,
            merged_data=merged,
        )

    def apply_resolution(
        self,
        conflict: Conflict,
        decision: ConflictDecision,
    ) -> dict[str, Any]:
        """Apply a resolution decision and return the resulting data.

        Args:
            conflict: The conflict being resolved
            decision: The resolution decision

        Returns:
            The data to use after resolution
        """
        if decision.resolution == ConflictResolution.LOCAL_WINS:
            return conflict.local_change.data or {}
        elif decision.resolution == ConflictResolution.REMOTE_WINS:
            return conflict.remote_change
        elif decision.resolution == ConflictResolution.MERGE:
            return decision.merged_data or {}
        else:
            # Shouldn't happen, but default to remote
            return conflict.remote_change


def merge_metadata(
    local: dict[str, Any],
    remote: dict[str, Any],
    local_ts: datetime,
    remote_ts: datetime,
) -> dict[str, Any]:
    """Merge two metadata dictionaries.

    Takes the newer value for each field, with special handling for:
    - Counts (message_count, event_count): take max
    - Timestamps: take newer

    Args:
        local: Local metadata
        remote: Remote metadata
        local_ts: Local modification timestamp
        remote_ts: Remote modification timestamp

    Returns:
        Merged metadata
    """
    merged: dict[str, Any] = {}

    # Fields that should take max value
    max_fields = {"turn_count", "message_count", "event_count"}

    # Fields that should take newer timestamp
    timestamp_fields = {"created", "updated"}

    all_keys = set(local.keys()) | set(remote.keys())

    for key in all_keys:
        local_val = local.get(key)
        remote_val = remote.get(key)

        if local_val is None:
            merged[key] = remote_val
        elif remote_val is None:
            merged[key] = local_val
        elif key in max_fields:
            merged[key] = max(local_val, remote_val)
        elif key in timestamp_fields:
            # Parse if string, then take newer
            if isinstance(local_val, str):
                local_val = datetime.fromisoformat(local_val)
            if isinstance(remote_val, str):
                remote_val = datetime.fromisoformat(remote_val)
            merged[key] = max(local_val, remote_val).isoformat()
        elif local_val == remote_val:
            merged[key] = local_val
        else:
            # Different values - use modification timestamp
            merged[key] = local_val if local_ts >= remote_ts else remote_val

    return merged
