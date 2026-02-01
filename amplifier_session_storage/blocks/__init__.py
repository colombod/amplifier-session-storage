"""
Event-sourced block storage model.

Sessions are stored as streams of immutable blocks, enabling
efficient sync, natural audit trails, and simple conflict resolution.
"""

from .reader import (
    ReconstructedEventSummary,
    ReconstructedMessage,
    ReconstructedMetadata,
    SessionStateReader,
)
from .sequence import SequenceAllocator
from .types import (
    BlockType,
    EventData,
    EventDataChunk,
    ForkData,
    MessageData,
    RewindData,
    SessionBlock,
    SessionCreatedData,
    SessionUpdatedData,
)
from .writer import BlockWriter

__all__ = [
    # Block types
    "BlockType",
    "SessionBlock",
    # Data types
    "SessionCreatedData",
    "SessionUpdatedData",
    "MessageData",
    "EventData",
    "EventDataChunk",
    "RewindData",
    "ForkData",
    # Reconstructed state types (from block stream)
    "ReconstructedMetadata",
    "ReconstructedMessage",
    "ReconstructedEventSummary",
    # Utilities
    "SequenceAllocator",
    "SessionStateReader",
    "BlockWriter",
]
