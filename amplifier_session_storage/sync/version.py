"""
Version vectors for distributed synchronization.

Version vectors track the causal ordering of changes across multiple
devices/replicas, enabling conflict detection and resolution.

Key concepts:
- Each device has a unique ID and maintains a sequence counter
- Version vectors capture the "happens-before" relationship
- Concurrent modifications (neither happens-before the other) indicate conflicts
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class VersionVector:
    """Version vector for tracking distributed state.

    A version vector maps device IDs to sequence numbers, representing
    the last known state from each device. This enables detection of:
    - Updates that happened-before others (can be safely ordered)
    - Concurrent updates (require conflict resolution)

    Attributes:
        entries: Mapping of device_id to sequence number
        timestamp: When this version was created/updated
    """

    entries: dict[str, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def increment(self, device_id: str) -> "VersionVector":
        """Increment the sequence for a device.

        Call this when making a local change on the device.

        Args:
            device_id: ID of the device making the change

        Returns:
            New VersionVector with incremented sequence
        """
        new_entries = self.entries.copy()
        new_entries[device_id] = new_entries.get(device_id, 0) + 1
        return VersionVector(entries=new_entries, timestamp=datetime.utcnow())

    def get_sequence(self, device_id: str) -> int:
        """Get the sequence number for a device.

        Args:
            device_id: Device ID to look up

        Returns:
            Sequence number (0 if device not in vector)
        """
        return self.entries.get(device_id, 0)

    def merge_with(self, other: "VersionVector") -> "VersionVector":
        """Merge two version vectors, taking the maximum of each entry.

        Used after receiving updates from another replica to incorporate
        their state into our view.

        Args:
            other: Version vector to merge with

        Returns:
            New VersionVector with merged entries
        """
        all_devices = set(self.entries.keys()) | set(other.entries.keys())
        merged = {
            device: max(self.entries.get(device, 0), other.entries.get(device, 0))
            for device in all_devices
        }
        return VersionVector(
            entries=merged,
            timestamp=max(self.timestamp, other.timestamp),
        )

    def happens_before(self, other: "VersionVector") -> bool:
        """Check if this version happens-before another.

        A happens-before B if all entries in A are <= the corresponding
        entries in B, and at least one entry is strictly less.

        Args:
            other: Version vector to compare with

        Returns:
            True if this version causally precedes the other
        """
        all_devices = set(self.entries.keys()) | set(other.entries.keys())

        all_lte = True
        any_lt = False

        for device in all_devices:
            self_seq = self.entries.get(device, 0)
            other_seq = other.entries.get(device, 0)

            if self_seq > other_seq:
                all_lte = False
                break
            if self_seq < other_seq:
                any_lt = True

        return all_lte and any_lt

    def concurrent_with(self, other: "VersionVector") -> bool:
        """Check if this version is concurrent with another.

        Two versions are concurrent if neither happens-before the other.
        This indicates a conflict that requires resolution.

        Args:
            other: Version vector to compare with

        Returns:
            True if versions are concurrent (conflict exists)
        """
        return not self.happens_before(other) and not other.happens_before(self) and self != other

    def dominates(self, other: "VersionVector") -> bool:
        """Check if this version dominates (happens-after) another.

        Args:
            other: Version vector to compare with

        Returns:
            True if this version causally follows the other
        """
        return other.happens_before(self)

    def equals(self, other: "VersionVector") -> bool:
        """Check if two version vectors are identical.

        Args:
            other: Version vector to compare with

        Returns:
            True if all entries are equal
        """
        all_devices = set(self.entries.keys()) | set(other.entries.keys())
        return all(
            self.entries.get(device, 0) == other.entries.get(device, 0) for device in all_devices
        )

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, VersionVector):
            return False
        return self.equals(other)

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash(tuple(sorted(self.entries.items())))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entries": self.entries.copy(),
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VersionVector":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()

        return cls(
            entries=data.get("entries", {}),
            timestamp=timestamp,
        )

    @classmethod
    def initial(cls, device_id: str) -> "VersionVector":
        """Create an initial version vector for a device.

        Args:
            device_id: ID of the device

        Returns:
            VersionVector with sequence 1 for the device
        """
        return cls(entries={device_id: 1}, timestamp=datetime.utcnow())


@dataclass
class VersionedData:
    """Data with an associated version vector.

    Used to track the version of any piece of data for sync purposes.
    """

    version: VersionVector
    data: dict[str, Any]
    device_id: str
    modified_at: datetime = field(default_factory=datetime.utcnow)

    def update(self, new_data: dict[str, Any]) -> "VersionedData":
        """Create a new version with updated data.

        Args:
            new_data: New data to store

        Returns:
            New VersionedData with incremented version
        """
        return VersionedData(
            version=self.version.increment(self.device_id),
            data=new_data,
            device_id=self.device_id,
            modified_at=datetime.utcnow(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version.to_dict(),
            "data": self.data,
            "device_id": self.device_id,
            "modified_at": self.modified_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VersionedData":
        """Create from dictionary."""
        modified_at = data.get("modified_at")
        if isinstance(modified_at, str):
            modified_at = datetime.fromisoformat(modified_at)
        elif modified_at is None:
            modified_at = datetime.utcnow()

        return cls(
            version=VersionVector.from_dict(data.get("version", {})),
            data=data.get("data", {}),
            device_id=data.get("device_id", "unknown"),
            modified_at=modified_at,
        )


def compare_versions(v1: VersionVector, v2: VersionVector) -> str:
    """Compare two version vectors and return their relationship.

    Args:
        v1: First version vector
        v2: Second version vector

    Returns:
        One of: "equal", "v1_before_v2", "v2_before_v1", "concurrent"
    """
    if v1.equals(v2):
        return "equal"
    if v1.happens_before(v2):
        return "v1_before_v2"
    if v2.happens_before(v1):
        return "v2_before_v1"
    return "concurrent"
