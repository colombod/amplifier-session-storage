"""ID generation and parsing utilities for session storage.

Centralizes the ID format knowledge so callers never need to
construct or parse document IDs directly.

Transcript IDs: {session_id}_msg_{sequence}
Event IDs: {session_id}_evt_{sequence}

Sequences are 0-indexed, matching the start_sequence used by the upload API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from amplifier_session_storage.backends.base import SyncGapResult

if TYPE_CHECKING:
    from amplifier_session_storage.backends.base import StorageAdmin


def transcript_id(session_id: str, sequence: int) -> str:
    """Generate a transcript document ID."""
    return f"{session_id}_msg_{sequence}"


def event_id(session_id: str, sequence: int) -> str:
    """Generate an event document ID."""
    return f"{session_id}_evt_{sequence}"


def parse_transcript_sequence(doc_id: str) -> int:
    """Extract the sequence number from a transcript document ID.

    Raises ValueError on malformed input.
    """
    try:
        parts = doc_id.rsplit("_msg_", 1)
        if len(parts) != 2 or not parts[1]:
            raise ValueError
        return int(parts[1])
    except (ValueError, IndexError):
        raise ValueError(f"Malformed transcript ID: {doc_id}") from None


def parse_event_sequence(doc_id: str) -> int:
    """Extract the sequence number from an event document ID.

    Raises ValueError on malformed input.
    """
    try:
        parts = doc_id.rsplit("_evt_", 1)
        if len(parts) != 2 or not parts[1]:
            raise ValueError
        return int(parts[1])
    except (ValueError, IndexError):
        raise ValueError(f"Malformed event ID: {doc_id}") from None


async def find_missing_sequences(
    backend: StorageAdmin,
    user_id: str,
    project_slug: str,
    session_id: str,
    transcript_line_count: int | None = None,
    event_line_count: int | None = None,
) -> SyncGapResult:
    """Compare expected sequences against stored IDs, return gaps.

    Args:
        backend: Any StorageAdmin implementation.
        user_id: User identifier.
        project_slug: Project slug.
        session_id: Session identifier.
        transcript_line_count: Number of lines in local transcript.jsonl.
            If None, transcript comparison is skipped.
        event_line_count: Number of lines in local events.jsonl.
            If None, event comparison is skipped.

    Returns:
        SyncGapResult with stored counts and missing sequence numbers.
    """
    transcript_stored_count = 0
    transcript_missing: list[int] = []
    event_stored_count = 0
    event_missing: list[int] = []

    if transcript_line_count is not None:
        stored_ids = await backend.get_stored_transcript_ids(
            user_id,
            project_slug,
            session_id,
        )
        transcript_stored_count = len(stored_ids)
        expected = {transcript_id(session_id, seq) for seq in range(0, transcript_line_count)}
        missing_ids = expected - set(stored_ids)
        transcript_missing = sorted(parse_transcript_sequence(mid) for mid in missing_ids)

    if event_line_count is not None:
        stored_ids = await backend.get_stored_event_ids(
            user_id,
            project_slug,
            session_id,
        )
        event_stored_count = len(stored_ids)
        expected = {event_id(session_id, seq) for seq in range(0, event_line_count)}
        missing_ids = expected - set(stored_ids)
        event_missing = sorted(parse_event_sequence(mid) for mid in missing_ids)

    return SyncGapResult(
        transcript_stored_count=transcript_stored_count,
        transcript_missing_sequences=transcript_missing,
        event_stored_count=event_stored_count,
        event_missing_sequences=event_missing,
    )
