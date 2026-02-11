"""ID generation and parsing utilities for session storage.

Centralizes the ID format knowledge so callers never need to
construct or parse document IDs directly.

Transcript IDs: {session_id}_msg_{sequence}
Event IDs: {session_id}_evt_{sequence}

Sequences are 1-indexed, matching line numbers in the source JSONL files.
"""


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
