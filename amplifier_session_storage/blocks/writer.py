"""
Block writer for creating session blocks.

Handles creation of blocks including chunking large events
into multiple continuation blocks.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import UTC, datetime
from typing import Any

from .sequence import SequenceAllocator, SimpleSequenceAllocator
from .types import (
    BlockType,
    EventData,
    EventDataChunk,
    MessageData,
    RewindData,
    SessionBlock,
    SessionCreatedData,
    SessionUpdatedData,
)

# Size thresholds for chunking
MAX_INLINE_SIZE = 50_000  # 50KB - inline in event block
CHUNK_SIZE = 300_000  # 300KB per chunk


def _extract_summary(data: dict[str, Any], event_type: str) -> dict[str, Any]:
    """Extract a safe summary from event data.

    Creates a small summary with non-sensitive fields
    that can be used for display without loading full data.
    """
    summary: dict[str, Any] = {"event_type": event_type}

    # Extract common safe fields
    safe_fields = ["status", "model", "provider", "tool_name", "duration_ms", "token_count"]
    for field in safe_fields:
        if field in data:
            summary[field] = data[field]

    # Add size info
    summary["_data_size"] = len(json.dumps(data))

    return summary


class BlockWriter:
    """Creates blocks for session operations.

    Handles all block types and automatically chunks
    large event data into multiple blocks.
    """

    def __init__(
        self,
        session_id: str,
        user_id: str,
        device_id: str,
        sequence_allocator: SequenceAllocator | SimpleSequenceAllocator | None = None,
    ) -> None:
        """Initialize the block writer.

        Args:
            session_id: The session ID for all blocks
            user_id: The user ID for all blocks
            device_id: The device ID for all blocks
            sequence_allocator: Optional custom sequence allocator
        """
        self.session_id = session_id
        self.user_id = user_id
        self.device_id = device_id
        self.allocator = sequence_allocator or SimpleSequenceAllocator()

    def _generate_block_id(self) -> str:
        """Generate a unique block ID."""
        return f"{self.session_id}_blk_{uuid.uuid4().hex[:12]}"

    def _create_block(
        self,
        block_type: BlockType,
        data: dict[str, Any],
        parent_block_id: str | None = None,
        chunk_index: int | None = None,
        is_final_chunk: bool = True,
    ) -> SessionBlock:
        """Create a block with common fields populated."""
        return SessionBlock(
            block_id=self._generate_block_id(),
            session_id=self.session_id,
            user_id=self.user_id,
            sequence=self.allocator.next_sequence(),
            timestamp=datetime.now(UTC),
            device_id=self.device_id,
            block_type=block_type,
            data=data,
            parent_block_id=parent_block_id,
            chunk_index=chunk_index,
            is_final_chunk=is_final_chunk,
        )

    def create_session(
        self,
        project_slug: str,
        name: str | None = None,
        description: str | None = None,
        bundle: str | None = None,
        model: str | None = None,
        visibility: str = "private",
        org_id: str | None = None,
        team_ids: list[str] | None = None,
        tags: list[str] | None = None,
        parent_session_id: str | None = None,
    ) -> SessionBlock:
        """Create a SESSION_CREATED block.

        Args:
            project_slug: The project slug
            name: Optional session name
            description: Optional description
            bundle: Optional bundle name
            model: Optional model name
            visibility: Visibility (private, team, org, public)
            org_id: Optional organization ID
            team_ids: Optional team IDs for team visibility
            tags: Optional tags
            parent_session_id: If forked, the parent session

        Returns:
            The SESSION_CREATED block
        """
        data = SessionCreatedData(
            project_slug=project_slug,
            name=name,
            description=description,
            bundle=bundle,
            model=model,
            visibility=visibility,
            org_id=org_id,
            team_ids=team_ids,
            tags=tags,
            parent_session_id=parent_session_id,
        )
        return self._create_block(BlockType.SESSION_CREATED, data.to_dict())

    def update_session(
        self,
        name: str | None = None,
        description: str | None = None,
        visibility: str | None = None,
        team_ids: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> SessionBlock:
        """Create a SESSION_UPDATED block.

        Only include fields that changed.

        Args:
            name: New name (or None if unchanged)
            description: New description (or None if unchanged)
            visibility: New visibility (or None if unchanged)
            team_ids: New team IDs (or None if unchanged)
            tags: New tags (or None if unchanged)

        Returns:
            The SESSION_UPDATED block
        """
        data = SessionUpdatedData(
            name=name,
            description=description,
            visibility=visibility,
            team_ids=team_ids,
            tags=tags,
        )
        return self._create_block(BlockType.SESSION_UPDATED, data.to_dict())

    def add_message(
        self,
        role: str,
        content: Any,
        turn: int,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
    ) -> SessionBlock:
        """Create a MESSAGE block.

        Args:
            role: Message role (user, assistant, tool, system)
            content: Message content
            turn: Turn number
            tool_calls: Optional tool calls (for assistant messages)
            tool_call_id: Optional tool call ID (for tool messages)

        Returns:
            The MESSAGE block
        """
        data = MessageData(
            role=role,  # type: ignore[arg-type]
            content=content,
            turn=turn,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            timestamp=datetime.now(UTC),
        )
        return self._create_block(BlockType.MESSAGE, data.to_dict())

    def add_event(
        self,
        event_id: str,
        event_type: str,
        data: dict[str, Any],
        turn: int | None = None,
    ) -> list[SessionBlock]:
        """Create EVENT block(s) for an event.

        Automatically chunks large events into multiple blocks.

        Args:
            event_id: Unique event ID
            event_type: Event type string
            data: Event data (may be large)
            turn: Optional turn number

        Returns:
            List of blocks (1 for small events, multiple for large)
        """
        blocks: list[SessionBlock] = []
        serialized = json.dumps(data, sort_keys=True)
        size = len(serialized.encode("utf-8"))

        if size <= MAX_INLINE_SIZE:
            # Small event - single block with inline data
            event_data = EventData(
                event_id=event_id,
                event_type=event_type,
                turn=turn,
                summary=_extract_summary(data, event_type),
                inline_data=data,
                has_continuation=False,
                total_chunks=1,
                total_size_bytes=size,
            )
            blocks.append(self._create_block(BlockType.EVENT, event_data.to_dict()))
        else:
            # Large event - header + continuation chunks
            num_chunks = math.ceil(size / CHUNK_SIZE)

            # Header block
            event_data = EventData(
                event_id=event_id,
                event_type=event_type,
                turn=turn,
                summary=_extract_summary(data, event_type),
                inline_data=None,
                has_continuation=True,
                total_chunks=num_chunks,
                total_size_bytes=size,
            )
            header_block = self._create_block(BlockType.EVENT, event_data.to_dict())
            blocks.append(header_block)

            # Chunk blocks
            for i in range(num_chunks):
                start = i * CHUNK_SIZE
                end = min(start + CHUNK_SIZE, len(serialized))
                chunk_data = serialized[start:end]

                chunk = EventDataChunk(
                    event_id=event_id,
                    chunk_data=chunk_data,
                )
                chunk_block = self._create_block(
                    BlockType.EVENT_DATA,
                    chunk.to_dict(),
                    parent_block_id=header_block.block_id,
                    chunk_index=i,
                    is_final_chunk=(i == num_chunks - 1),
                )
                blocks.append(chunk_block)

        return blocks

    def rewind(
        self,
        target_sequence: int,
        reason: str | None = None,
    ) -> SessionBlock:
        """Create a REWIND block.

        Marks all blocks after target_sequence as logically deleted.

        Args:
            target_sequence: Keep blocks with seq <= this
            reason: Optional reason for rewind

        Returns:
            The REWIND block
        """
        data = RewindData(
            target_sequence=target_sequence,
            reason=reason,
        )
        return self._create_block(BlockType.REWIND, data.to_dict())

    def delete_session(self) -> SessionBlock:
        """Create a SESSION_DELETED block.

        Marks the session as deleted.

        Returns:
            The SESSION_DELETED block
        """
        return self._create_block(
            BlockType.SESSION_DELETED, {"deleted_at": datetime.now(UTC).isoformat()}
        )
