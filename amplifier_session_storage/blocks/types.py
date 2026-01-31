"""
Block types and schemas for event-sourced session storage.

Defines the core block types that make up a session stream.
Each block is immutable once written and has a sequence number
for ordering within the session.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


class BlockType(Enum):
    """Types of blocks in a session stream."""

    # Session lifecycle
    SESSION_CREATED = "session_created"
    SESSION_UPDATED = "session_updated"
    SESSION_DELETED = "session_deleted"

    # Transcript
    MESSAGE = "message"

    # Events
    EVENT = "event"
    EVENT_DATA = "event_data"  # Continuation for large events

    # History modification
    REWIND = "rewind"
    FORK = "fork"


@dataclass
class SessionBlock:
    """A single block in the session event stream.

    Blocks are immutable once written. The sequence number
    provides total ordering within a session.

    Attributes:
        block_id: Unique identifier for this block
        session_id: Parent session ID
        user_id: Owner user ID
        sequence: Monotonic sequence number within session
        timestamp: When the block was created
        device_id: Which device created this block
        block_type: Type of block (message, event, etc.)
        data: Type-specific payload
        parent_block_id: For continuation blocks, the parent block
        chunk_index: For chunked data, the chunk number (0-indexed)
        is_final_chunk: Whether this is the last chunk
        checksum: SHA-256 checksum of data for integrity
        size_bytes: Size of the data payload in bytes
    """

    # Identity
    block_id: str
    session_id: str
    user_id: str

    # Ordering
    sequence: int
    timestamp: datetime

    # Source
    device_id: str

    # Content
    block_type: BlockType
    data: dict[str, Any]

    # Chunking (for large data)
    parent_block_id: str | None = None
    chunk_index: int | None = None
    is_final_chunk: bool = True

    # Metadata
    checksum: str | None = None
    size_bytes: int = 0

    def __post_init__(self) -> None:
        """Calculate checksum and size if not provided."""
        if self.data and self.checksum is None:
            serialized = json.dumps(self.data, sort_keys=True)
            self.checksum = hashlib.sha256(serialized.encode()).hexdigest()
            self.size_bytes = len(serialized.encode())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "id": self.block_id,
            "block_id": self.block_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "user_id_session_id": f"{self.user_id}_{self.session_id}",
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
            "device_id": self.device_id,
            "block_type": self.block_type.value,
            "data": self.data,
            "parent_block_id": self.parent_block_id,
            "chunk_index": self.chunk_index,
            "is_final_chunk": self.is_final_chunk,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionBlock:
        """Deserialize from dictionary."""
        return cls(
            block_id=data.get("block_id", data.get("id", "")),
            session_id=data["session_id"],
            user_id=data["user_id"],
            sequence=data["sequence"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            device_id=data["device_id"],
            block_type=BlockType(data["block_type"]),
            data=data["data"],
            parent_block_id=data.get("parent_block_id"),
            chunk_index=data.get("chunk_index"),
            is_final_chunk=data.get("is_final_chunk", True),
            checksum=data.get("checksum"),
            size_bytes=data.get("size_bytes", 0),
        )


# Type-specific data schemas


@dataclass
class SessionCreatedData:
    """Data for SESSION_CREATED block.

    Contains the initial metadata when a session is created.
    """

    project_slug: str
    name: str | None = None
    description: str | None = None
    bundle: str | None = None
    model: str | None = None
    visibility: str = "private"
    org_id: str | None = None
    team_ids: list[str] | None = None
    tags: list[str] | None = None
    parent_session_id: str | None = None
    forked_from_sequence: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "project_slug": self.project_slug,
            "name": self.name,
            "description": self.description,
            "bundle": self.bundle,
            "model": self.model,
            "visibility": self.visibility,
            "org_id": self.org_id,
            "team_ids": self.team_ids,
            "tags": self.tags,
            "parent_session_id": self.parent_session_id,
            "forked_from_sequence": self.forked_from_sequence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionCreatedData:
        """Deserialize from dictionary."""
        return cls(
            project_slug=data["project_slug"],
            name=data.get("name"),
            description=data.get("description"),
            bundle=data.get("bundle"),
            model=data.get("model"),
            visibility=data.get("visibility", "private"),
            org_id=data.get("org_id"),
            team_ids=data.get("team_ids"),
            tags=data.get("tags"),
            parent_session_id=data.get("parent_session_id"),
            forked_from_sequence=data.get("forked_from_sequence"),
        )


@dataclass
class SessionUpdatedData:
    """Data for SESSION_UPDATED block.

    Only includes changed fields. None means no change.
    """

    name: str | None = None
    description: str | None = None
    visibility: str | None = None
    team_ids: list[str] | None = None
    tags: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary, excluding None values."""
        result: dict[str, Any] = {}
        if self.name is not None:
            result["name"] = self.name
        if self.description is not None:
            result["description"] = self.description
        if self.visibility is not None:
            result["visibility"] = self.visibility
        if self.team_ids is not None:
            result["team_ids"] = self.team_ids
        if self.tags is not None:
            result["tags"] = self.tags
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionUpdatedData:
        """Deserialize from dictionary."""
        return cls(
            name=data.get("name"),
            description=data.get("description"),
            visibility=data.get("visibility"),
            team_ids=data.get("team_ids"),
            tags=data.get("tags"),
        )


@dataclass
class MessageData:
    """Data for MESSAGE block.

    Represents a single message in the conversation transcript.
    """

    role: Literal["user", "assistant", "tool", "system"]
    content: Any  # String or structured content
    turn: int
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    timestamp: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "turn": self.turn,
        }
        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessageData:
        """Deserialize from dictionary."""
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])
        return cls(
            role=data["role"],
            content=data["content"],
            turn=data["turn"],
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
            timestamp=timestamp,
        )


@dataclass
class EventData:
    """Data for EVENT block.

    Contains summary/metadata only. Large payloads go into
    EVENT_DATA continuation blocks.
    """

    event_id: str
    event_type: str
    turn: int | None = None

    # Safe summary fields (never full content)
    summary: dict[str, Any] = field(default_factory=dict)

    # If data is small enough, inline it
    inline_data: dict[str, Any] | None = None

    # If data is large, it's in continuation blocks
    has_continuation: bool = False
    total_chunks: int = 1
    total_size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result: dict[str, Any] = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "turn": self.turn,
            "summary": self.summary,
            "has_continuation": self.has_continuation,
            "total_chunks": self.total_chunks,
            "total_size_bytes": self.total_size_bytes,
        }
        if self.inline_data is not None:
            result["inline_data"] = self.inline_data
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventData:
        """Deserialize from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            turn=data.get("turn"),
            summary=data.get("summary", {}),
            inline_data=data.get("inline_data"),
            has_continuation=data.get("has_continuation", False),
            total_chunks=data.get("total_chunks", 1),
            total_size_bytes=data.get("total_size_bytes", 0),
        )


@dataclass
class EventDataChunk:
    """Data for EVENT_DATA continuation block.

    Contains a chunk of large event data.
    """

    event_id: str
    chunk_data: str  # JSON-encoded chunk

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_id": self.event_id,
            "chunk_data": self.chunk_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventDataChunk:
        """Deserialize from dictionary."""
        return cls(
            event_id=data["event_id"],
            chunk_data=data["chunk_data"],
        )


@dataclass
class RewindData:
    """Data for REWIND block.

    Marks that all blocks with sequence > target_sequence
    should be considered "rewound" (logically deleted).
    History is preserved for audit, but current state
    excludes rewound blocks.
    """

    target_sequence: int
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result: dict[str, Any] = {"target_sequence": self.target_sequence}
        if self.reason is not None:
            result["reason"] = self.reason
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RewindData:
        """Deserialize from dictionary."""
        return cls(
            target_sequence=data["target_sequence"],
            reason=data.get("reason"),
        )


@dataclass
class ForkData:
    """Data for FORK block.

    Indicates this session was forked from another session
    at a specific point.
    """

    source_session_id: str
    source_sequence: int  # Fork point

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source_session_id": self.source_session_id,
            "source_sequence": self.source_sequence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForkData:
        """Deserialize from dictionary."""
        return cls(
            source_session_id=data["source_session_id"],
            source_sequence=data["source_sequence"],
        )
