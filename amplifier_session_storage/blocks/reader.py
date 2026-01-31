"""
Session state reader for event-sourced blocks.

Reconstructs session state (metadata, transcript, events)
from a stream of blocks, handling rewind markers and
computing the "current" view of the session.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

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


@dataclass
class SessionMetadata:
    """Reconstructed session metadata from blocks."""

    session_id: str
    user_id: str
    name: str | None = None
    description: str | None = None
    project_slug: str = ""
    bundle: str | None = None
    model: str | None = None
    visibility: str = "private"
    org_id: str | None = None
    team_ids: list[str] | None = None
    tags: list[str] | None = None
    parent_session_id: str | None = None
    created: datetime | None = None
    updated: datetime | None = None
    turn_count: int = 0
    message_count: int = 0
    event_count: int = 0
    last_sequence: int = 0


@dataclass
class TranscriptMessage:
    """A message in the transcript."""

    role: str
    content: Any
    turn: int
    sequence: int  # Block sequence for ordering
    timestamp: datetime
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@dataclass
class EventSummary:
    """Summary of an event (without full data)."""

    event_id: str
    event_type: str
    turn: int | None
    sequence: int
    timestamp: datetime
    summary: dict[str, Any] = field(default_factory=dict)
    has_full_data: bool = False
    total_size_bytes: int = 0


class SessionStateReader:
    """Reads session state by applying blocks.

    Handles rewind markers to compute "current" state
    while preserving full history for audit.
    """

    def compute_current_state(
        self,
        blocks: list[SessionBlock],
    ) -> tuple[SessionMetadata, list[TranscriptMessage], list[EventSummary]]:
        """Compute current state from block stream.

        Process:
        1. Sort blocks by sequence
        2. Find latest REWIND marker (if any)
        3. Filter blocks: keep seq <= rewind.target_sequence
        4. Build state from filtered blocks

        Args:
            blocks: List of blocks (any order)

        Returns:
            Tuple of (metadata, messages, events)
        """
        if not blocks:
            raise ValueError("Cannot compute state from empty block list")

        # Sort by sequence
        sorted_blocks = sorted(blocks, key=lambda b: b.sequence)

        # Find latest rewind (if any)
        rewind_target = self._find_rewind_target(sorted_blocks)

        # Filter to "current" blocks
        current_blocks = sorted_blocks
        if rewind_target is not None:
            current_blocks = [
                b
                for b in sorted_blocks
                if b.sequence <= rewind_target or b.block_type == BlockType.REWIND
            ]

        # Build state
        return self._build_state(current_blocks)

    def get_blocks_since(
        self,
        blocks: list[SessionBlock],
        since_sequence: int,
    ) -> list[SessionBlock]:
        """Get blocks since a sequence number.

        Useful for incremental sync.

        Args:
            blocks: All blocks
            since_sequence: Return blocks with sequence > this

        Returns:
            Filtered and sorted blocks
        """
        filtered = [b for b in blocks if b.sequence > since_sequence]
        return sorted(filtered, key=lambda b: b.sequence)

    def reconstruct_event_data(
        self,
        header_block: SessionBlock,
        chunk_blocks: list[SessionBlock],
    ) -> dict[str, Any]:
        """Reconstruct full event data from header and chunks.

        Args:
            header_block: The EVENT block with metadata
            chunk_blocks: List of EVENT_DATA blocks (chunks)

        Returns:
            The reconstructed event data dictionary
        """
        event_data = EventData.from_dict(header_block.data)

        # If data is inline, return it directly
        if event_data.inline_data is not None:
            return event_data.inline_data

        # Otherwise, reassemble from chunks
        if not event_data.has_continuation:
            return {}

        # Sort chunks by index
        sorted_chunks = sorted(
            chunk_blocks,
            key=lambda b: b.chunk_index if b.chunk_index is not None else 0,
        )

        # Concatenate chunk data
        full_data = ""
        for chunk_block in sorted_chunks:
            chunk = EventDataChunk.from_dict(chunk_block.data)
            full_data += chunk.chunk_data

        # Parse JSON
        return json.loads(full_data)

    def _find_rewind_target(self, sorted_blocks: list[SessionBlock]) -> int | None:
        """Find the rewind target from the latest REWIND block.

        Args:
            sorted_blocks: Blocks sorted by sequence

        Returns:
            Target sequence if rewind exists, None otherwise
        """
        for block in reversed(sorted_blocks):
            if block.block_type == BlockType.REWIND:
                rewind_data = RewindData.from_dict(block.data)
                return rewind_data.target_sequence
        return None

    def _build_state(
        self,
        blocks: list[SessionBlock],
    ) -> tuple[SessionMetadata, list[TranscriptMessage], list[EventSummary]]:
        """Build state from a list of blocks.

        Args:
            blocks: Blocks to process (already filtered)

        Returns:
            Tuple of (metadata, messages, events)
        """
        metadata: SessionMetadata | None = None
        messages: list[TranscriptMessage] = []
        events: list[EventSummary] = []
        max_turn = 0

        for block in blocks:
            if block.block_type == BlockType.SESSION_CREATED:
                created_data = SessionCreatedData.from_dict(block.data)
                metadata = SessionMetadata(
                    session_id=block.session_id,
                    user_id=block.user_id,
                    name=created_data.name,
                    description=created_data.description,
                    project_slug=created_data.project_slug,
                    bundle=created_data.bundle,
                    model=created_data.model,
                    visibility=created_data.visibility,
                    org_id=created_data.org_id,
                    team_ids=created_data.team_ids,
                    tags=created_data.tags,
                    parent_session_id=created_data.parent_session_id,
                    created=block.timestamp,
                    updated=block.timestamp,
                    last_sequence=block.sequence,
                )

            elif block.block_type == BlockType.SESSION_UPDATED:
                if metadata is not None:
                    updated_data = SessionUpdatedData.from_dict(block.data)
                    if updated_data.name is not None:
                        metadata.name = updated_data.name
                    if updated_data.description is not None:
                        metadata.description = updated_data.description
                    if updated_data.visibility is not None:
                        metadata.visibility = updated_data.visibility
                    if updated_data.team_ids is not None:
                        metadata.team_ids = updated_data.team_ids
                    if updated_data.tags is not None:
                        metadata.tags = updated_data.tags
                    metadata.updated = block.timestamp
                    metadata.last_sequence = block.sequence

            elif block.block_type == BlockType.MESSAGE:
                msg_data = MessageData.from_dict(block.data)
                messages.append(
                    TranscriptMessage(
                        role=msg_data.role,
                        content=msg_data.content,
                        turn=msg_data.turn,
                        sequence=block.sequence,
                        timestamp=block.timestamp,
                        tool_calls=msg_data.tool_calls,
                        tool_call_id=msg_data.tool_call_id,
                    )
                )
                max_turn = max(max_turn, msg_data.turn)

            elif block.block_type == BlockType.EVENT:
                event_data = EventData.from_dict(block.data)
                events.append(
                    EventSummary(
                        event_id=event_data.event_id,
                        event_type=event_data.event_type,
                        turn=event_data.turn,
                        sequence=block.sequence,
                        timestamp=block.timestamp,
                        summary=event_data.summary,
                        has_full_data=event_data.inline_data is not None
                        or event_data.has_continuation,
                        total_size_bytes=event_data.total_size_bytes,
                    )
                )

            # Update metadata tracking
            if metadata is not None:
                metadata.last_sequence = max(metadata.last_sequence, block.sequence)
                metadata.updated = block.timestamp

        # Finalize metadata counts
        if metadata is not None:
            metadata.message_count = len(messages)
            metadata.event_count = len(events)
            metadata.turn_count = max_turn

        # Handle case where no SESSION_CREATED block exists
        if metadata is None and blocks:
            first_block = blocks[0]
            metadata = SessionMetadata(
                session_id=first_block.session_id,
                user_id=first_block.user_id,
                created=first_block.timestamp,
                updated=blocks[-1].timestamp,
                message_count=len(messages),
                event_count=len(events),
                turn_count=max_turn,
                last_sequence=blocks[-1].sequence,
            )

        if metadata is None:
            raise ValueError("Cannot build state: no blocks provided")

        return metadata, messages, events
