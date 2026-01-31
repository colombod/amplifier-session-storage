"""
Sequence number allocation for blocks.

Handles sequence numbering strategy for multi-device
scenarios where blocks may be created offline and
need to be merged later.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass
class SequenceAllocator:
    """Allocates sequence numbers for blocks.

    Strategy:
    - Local writes: use local sequence with device suffix
    - Cloud writes: use atomic increment
    - Sync: resolve conflicts by interleaving

    The sequence number format allows ~1000 local writes
    between syncs without collision:
        sequence = base * 1000 + local_offset

    Example:
        base=5, local_offset=1 → sequence=5001
        base=5, local_offset=2 → sequence=5002
        After sync with cloud sequence=6:
        base=6, local_offset=1 → sequence=6001
    """

    device_id: str
    _base_sequence: int = 0
    _local_counter: int = 0
    _lock: Lock | None = None

    def __post_init__(self) -> None:
        """Initialize the lock."""
        self._lock = Lock()

    def next_sequence(self) -> int:
        """Generate next sequence number.

        Thread-safe sequence allocation.
        """
        if self._lock is None:
            self._lock = Lock()

        with self._lock:
            self._local_counter += 1
            return self._base_sequence * 1000 + self._local_counter

    def set_base_sequence(self, sequence: int) -> None:
        """Set the base sequence after sync.

        Call this after syncing with cloud to update
        the base sequence to the cloud's latest.

        Args:
            sequence: The latest sequence from cloud
        """
        if self._lock is None:
            self._lock = Lock()

        with self._lock:
            self._base_sequence = sequence
            self._local_counter = 0

    def get_current_base(self) -> int:
        """Get the current base sequence."""
        return self._base_sequence

    def get_local_counter(self) -> int:
        """Get the current local counter."""
        return self._local_counter


class SimpleSequenceAllocator:
    """Simple sequential allocator for migration and testing.

    Just increments from a starting point, no device suffix.
    """

    def __init__(self, start: int = 1) -> None:
        """Initialize with starting sequence.

        Args:
            start: Starting sequence number
        """
        self._current = start - 1
        self._lock = Lock()

    def next_sequence(self) -> int:
        """Get next sequence number."""
        with self._lock:
            self._current += 1
            return self._current

    def get_current(self) -> int:
        """Get current sequence (last allocated)."""
        return self._current
