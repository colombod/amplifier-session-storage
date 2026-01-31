"""
Migration types and data structures.

Defines the types used during session migration from
legacy disk format to block-based storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class MigrationStatus(Enum):
    """Status of a migration operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SessionSource:
    """Information about a session to migrate.

    Represents an existing session in the legacy disk format
    (events.jsonl + transcript.jsonl + state.json).
    """

    session_id: str
    path: Path
    project_slug: str

    # Discovered metadata
    created: datetime | None = None
    updated: datetime | None = None
    message_count: int = 0
    event_count: int = 0

    # File info
    events_file: Path | None = None
    transcript_file: Path | None = None
    state_file: Path | None = None

    # Size info for progress tracking
    total_size_bytes: int = 0

    def __post_init__(self) -> None:
        """Discover files in the session directory."""
        if self.path.exists():
            events = self.path / "events.jsonl"
            if events.exists():
                self.events_file = events
                self.total_size_bytes += events.stat().st_size

            transcript = self.path / "transcript.jsonl"
            if transcript.exists():
                self.transcript_file = transcript
                self.total_size_bytes += transcript.stat().st_size

            state = self.path / "state.json"
            if state.exists():
                self.state_file = state


@dataclass
class MigrationResult:
    """Result of migrating a single session.

    Contains details about the migration outcome including
    any errors encountered.
    """

    session_id: str
    status: MigrationStatus
    blocks_created: int = 0
    messages_migrated: int = 0
    events_migrated: int = 0

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Error info (if failed)
    error_message: str | None = None
    error_details: dict[str, Any] = field(default_factory=dict)

    # Source info
    source_path: str | None = None
    source_size_bytes: int = 0

    @property
    def duration_seconds(self) -> float | None:
        """Calculate migration duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "blocks_created": self.blocks_created,
            "messages_migrated": self.messages_migrated,
            "events_migrated": self.events_migrated,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "source_path": self.source_path,
            "source_size_bytes": self.source_size_bytes,
        }


@dataclass
class MigrationBatch:
    """Batch migration result.

    Contains results for multiple sessions migrated together.
    """

    total_sessions: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[MigrationResult] = field(default_factory=list)

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_sessions == 0:
            return 0.0
        return (self.completed / self.total_sessions) * 100

    def add_result(self, result: MigrationResult) -> None:
        """Add a migration result to the batch."""
        self.results.append(result)
        if result.status == MigrationStatus.COMPLETED:
            self.completed += 1
        elif result.status == MigrationStatus.FAILED:
            self.failed += 1
        elif result.status == MigrationStatus.SKIPPED:
            self.skipped += 1
