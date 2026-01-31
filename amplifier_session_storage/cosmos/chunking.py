"""
Large event chunking for Cosmos DB storage.

Cosmos DB has a 2MB document size limit, but we use a conservative 400KB
limit per chunk to ensure efficient storage and retrieval.

This module handles:
- Detection of events that need chunking
- Splitting large events into multiple chunks
- Reassembling chunks back into complete events
"""

import json
import math
from dataclasses import dataclass
from typing import Any

from ..exceptions import ChunkingError

# Chunk size limits
CHUNK_SIZE = 400_000  # 400KB per chunk
MAX_INLINE_SIZE = 400_000  # Events larger than this get chunked
MAX_TOTAL_SIZE = 10_000_000  # 10MB max total event size


@dataclass
class EventChunk:
    """A single chunk of a large event.

    Attributes:
        chunk_id: Unique identifier for this chunk
        event_id: ID of the parent event
        chunk_index: Zero-based index of this chunk
        total_chunks: Total number of chunks for this event
        data: The chunk's data payload (JSON string)
        is_metadata: True if this is the metadata chunk (index 0)
    """

    chunk_id: str
    event_id: str
    chunk_index: int
    total_chunks: int
    data: str
    is_metadata: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.chunk_id,
            "chunk_id": self.chunk_id,
            "event_id": self.event_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "data": self.data,
            "is_metadata": self.is_metadata,
            "_type": "event_chunk",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventChunk":
        """Create from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            event_id=data["event_id"],
            chunk_index=data["chunk_index"],
            total_chunks=data["total_chunks"],
            data=data["data"],
            is_metadata=data.get("is_metadata", False),
        )


def should_chunk(data: dict[str, Any]) -> bool:
    """Determine if event data needs to be chunked.

    Args:
        data: Event data dictionary

    Returns:
        True if the data exceeds MAX_INLINE_SIZE and needs chunking
    """
    try:
        serialized = json.dumps(data)
        return len(serialized.encode("utf-8")) > MAX_INLINE_SIZE
    except (TypeError, ValueError):
        # If we can't serialize, assume it needs chunking
        return True


def get_data_size(data: dict[str, Any]) -> int:
    """Get the serialized size of event data in bytes.

    Args:
        data: Event data dictionary

    Returns:
        Size in bytes
    """
    try:
        return len(json.dumps(data).encode("utf-8"))
    except (TypeError, ValueError):
        return 0


def chunk_event(event_id: str, data: dict[str, Any]) -> list[EventChunk]:
    """Split a large event into multiple chunks.

    The first chunk (index 0) contains metadata about the event,
    including which fields are chunked. Subsequent chunks contain
    the actual data split into CHUNK_SIZE pieces.

    Args:
        event_id: Unique event identifier
        data: Event data to chunk

    Returns:
        List of EventChunk objects

    Raises:
        ChunkingError: If event exceeds MAX_TOTAL_SIZE or cannot be serialized
    """
    try:
        serialized = json.dumps(data)
    except (TypeError, ValueError) as e:
        raise ChunkingError(event_id, f"Cannot serialize event data: {e}") from e

    data_bytes = serialized.encode("utf-8")
    total_size = len(data_bytes)

    if total_size > MAX_TOTAL_SIZE:
        raise ChunkingError(
            event_id, f"Event too large: {total_size} bytes exceeds {MAX_TOTAL_SIZE} limit"
        )

    if total_size <= MAX_INLINE_SIZE:
        # No chunking needed - return single chunk
        return [
            EventChunk(
                chunk_id=f"{event_id}_chunk_0",
                event_id=event_id,
                chunk_index=0,
                total_chunks=1,
                data=serialized,
                is_metadata=True,
            )
        ]

    # Calculate number of chunks needed
    # Reserve space for metadata in each chunk
    effective_chunk_size = CHUNK_SIZE - 200  # 200 bytes for metadata overhead
    num_chunks = math.ceil(total_size / effective_chunk_size)

    chunks: list[EventChunk] = []

    # Create metadata chunk (index 0)
    metadata = {
        "event_id": event_id,
        "total_size": total_size,
        "total_chunks": num_chunks + 1,  # +1 for metadata chunk
        "chunk_size": effective_chunk_size,
        "chunked_at": "data",  # Which field is chunked
    }
    chunks.append(
        EventChunk(
            chunk_id=f"{event_id}_chunk_0",
            event_id=event_id,
            chunk_index=0,
            total_chunks=num_chunks + 1,
            data=json.dumps(metadata),
            is_metadata=True,
        )
    )

    # Split data into chunks
    for i in range(num_chunks):
        start = i * effective_chunk_size
        end = min(start + effective_chunk_size, total_size)
        chunk_data = data_bytes[start:end].decode("utf-8", errors="replace")

        chunks.append(
            EventChunk(
                chunk_id=f"{event_id}_chunk_{i + 1}",
                event_id=event_id,
                chunk_index=i + 1,
                total_chunks=num_chunks + 1,
                data=chunk_data,
                is_metadata=False,
            )
        )

    return chunks


def reassemble_event(chunks: list[EventChunk]) -> dict[str, Any]:
    """Reassemble chunks back into the original event data.

    Args:
        chunks: List of EventChunk objects (must include all chunks)

    Returns:
        Reconstructed event data dictionary

    Raises:
        ChunkingError: If chunks are missing, invalid, or cannot be reassembled
    """
    if not chunks:
        raise ChunkingError("unknown", "No chunks provided")

    # Sort chunks by index
    sorted_chunks = sorted(chunks, key=lambda c: c.chunk_index)

    # Validate we have all chunks
    event_id = sorted_chunks[0].event_id
    total_chunks = sorted_chunks[0].total_chunks

    if len(sorted_chunks) != total_chunks:
        raise ChunkingError(
            event_id, f"Missing chunks: have {len(sorted_chunks)}, expected {total_chunks}"
        )

    # Verify all chunks belong to same event
    for chunk in sorted_chunks:
        if chunk.event_id != event_id:
            raise ChunkingError(event_id, f"Chunk event_id mismatch: {chunk.event_id}")
        if chunk.total_chunks != total_chunks:
            raise ChunkingError(event_id, f"Chunk total_chunks mismatch: {chunk.total_chunks}")

    # Verify chunk indices are sequential
    for i, chunk in enumerate(sorted_chunks):
        if chunk.chunk_index != i:
            raise ChunkingError(event_id, f"Missing chunk at index {i}")

    # Single chunk - just parse and return
    if total_chunks == 1:
        try:
            return json.loads(sorted_chunks[0].data)
        except json.JSONDecodeError as e:
            raise ChunkingError(event_id, f"Invalid JSON in chunk: {e}") from e

    # Multiple chunks - first is metadata, rest is data
    metadata_chunk = sorted_chunks[0]
    if not metadata_chunk.is_metadata:
        raise ChunkingError(event_id, "First chunk is not metadata chunk")

    # Concatenate data chunks
    data_parts = [chunk.data for chunk in sorted_chunks[1:]]
    combined_data = "".join(data_parts)

    try:
        return json.loads(combined_data)
    except json.JSONDecodeError as e:
        raise ChunkingError(event_id, f"Failed to parse reassembled data: {e}") from e


def validate_chunk(chunk: EventChunk) -> bool:
    """Validate a chunk's structure.

    Args:
        chunk: Chunk to validate

    Returns:
        True if valid
    """
    if not chunk.chunk_id:
        return False
    if not chunk.event_id:
        return False
    if chunk.chunk_index < 0:
        return False
    if chunk.total_chunks < 1:
        return False
    if chunk.chunk_index >= chunk.total_chunks:
        return False
    if not chunk.data:
        return False
    return True


def estimate_chunk_count(data_size_bytes: int) -> int:
    """Estimate how many chunks will be needed for data of a given size.

    Args:
        data_size_bytes: Size of data in bytes

    Returns:
        Estimated number of chunks (including metadata chunk if needed)
    """
    if data_size_bytes <= MAX_INLINE_SIZE:
        return 1

    effective_chunk_size = CHUNK_SIZE - 200
    data_chunks = math.ceil(data_size_bytes / effective_chunk_size)
    return data_chunks + 1  # +1 for metadata chunk
