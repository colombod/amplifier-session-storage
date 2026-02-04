"""
Load a real Amplifier project into storage backends for testing.

Sanitizes sensitive data (API keys, emails, secrets) while preserving structure.
Useful for testing with real conversation patterns and complex message structures.

Usage:
    python scripts/load_real_project.py <project_path> --backend duckdb --output ./test.db
    python scripts/load_real_project.py <project_path> --backend sqlite --output ./test.sqlite
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from amplifier_session_storage.backends.duckdb import DuckDBBackend, DuckDBConfig
from amplifier_session_storage.backends.sqlite import SQLiteBackend, SQLiteConfig
from amplifier_session_storage.sanitization import (
    sanitize_event,
    sanitize_session_metadata,
    sanitize_transcript_message,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def load_project_to_duckdb(
    project_path: Path,
    output_path: Path,
    user_id: str = "test-user",
    host_id: str = "test-host",
    sanitize: bool = True,
) -> dict:
    """
    Load Amplifier project into DuckDB.

    Args:
        project_path: Path to project directory (contains sessions/)
        output_path: Path for DuckDB file
        user_id: User ID for storage
        host_id: Host ID for storage
        sanitize: Whether to sanitize sensitive data

    Returns:
        Statistics about what was loaded
    """
    config = DuckDBConfig(db_path=str(output_path))

    # Create backend (no embedding provider for now - just load data)
    storage = await DuckDBBackend.create(config=config, embedding_provider=None)

    sessions_dir = project_path / "sessions"
    if not sessions_dir.exists():
        raise ValueError(f"No sessions directory found at {sessions_dir}")

    stats = {
        "sessions_loaded": 0,
        "transcripts_loaded": 0,
        "events_loaded": 0,
        "sanitized": sanitize,
    }

    # Load each session
    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue

        session_id = session_dir.name
        logger.info(f"Loading session: {session_id}")

        # Load metadata
        metadata_file = session_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

            if sanitize:
                metadata = sanitize_session_metadata(metadata)

            await storage.upsert_session_metadata(
                user_id=user_id, host_id=host_id, metadata=metadata
            )
            stats["sessions_loaded"] += 1

        # Load transcript
        transcript_file = session_dir / "transcript.jsonl"
        if transcript_file.exists():
            with open(transcript_file) as f:
                lines = []
                for line in f:
                    message = json.loads(line.strip())
                    if sanitize:
                        message = sanitize_transcript_message(message)
                    lines.append(message)

            if lines:
                # Sync without embeddings for now
                synced = await storage.sync_transcript_lines(
                    user_id=user_id,
                    host_id=host_id,
                    project_slug=metadata.get("project_slug", "unknown"),
                    session_id=session_id,
                    lines=lines,
                    embeddings=None,  # No embeddings yet
                )
                stats["transcripts_loaded"] += synced

        # Load events
        events_file = session_dir / "events.jsonl"
        if events_file.exists():
            with open(events_file) as f:
                events = []
                for line in f:
                    event = json.loads(line.strip())
                    if sanitize:
                        event = sanitize_event(event)
                    events.append(event)

            if events:
                synced = await storage.sync_event_lines(
                    user_id=user_id,
                    host_id=host_id,
                    project_slug=metadata.get("project_slug", "unknown"),
                    session_id=session_id,
                    lines=events,
                )
                stats["events_loaded"] += synced

    await storage.close()
    logger.info(f"Load complete: {stats}")

    return stats


async def load_project_to_sqlite(
    project_path: Path,
    output_path: Path,
    user_id: str = "test-user",
    host_id: str = "test-host",
    sanitize: bool = True,
) -> dict:
    """Load Amplifier project into SQLite (same as DuckDB but different backend)."""
    config = SQLiteConfig(db_path=str(output_path))
    storage = await SQLiteBackend.create(config=config, embedding_provider=None)

    # Same logic as DuckDB
    sessions_dir = project_path / "sessions"
    if not sessions_dir.exists():
        raise ValueError(f"No sessions directory found at {sessions_dir}")

    stats = {
        "sessions_loaded": 0,
        "transcripts_loaded": 0,
        "events_loaded": 0,
        "sanitized": sanitize,
    }

    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue

        session_id = session_dir.name
        logger.info(f"Loading session: {session_id}")

        # Metadata
        metadata_file = session_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

            if sanitize:
                metadata = sanitize_session_metadata(metadata)

            await storage.upsert_session_metadata(
                user_id=user_id, host_id=host_id, metadata=metadata
            )
            stats["sessions_loaded"] += 1

        # Transcript
        transcript_file = session_dir / "transcript.jsonl"
        if transcript_file.exists():
            with open(transcript_file) as f:
                lines = [json.loads(line.strip()) for line in f]

            if sanitize:
                lines = [sanitize_transcript_message(msg) for msg in lines]

            if lines:
                synced = await storage.sync_transcript_lines(
                    user_id=user_id,
                    host_id=host_id,
                    project_slug=metadata.get("project_slug", "unknown"),
                    session_id=session_id,
                    lines=lines,
                    embeddings=None,
                )
                stats["transcripts_loaded"] += synced

        # Events
        events_file = session_dir / "events.jsonl"
        if events_file.exists():
            with open(events_file) as f:
                events = [json.loads(line.strip()) for line in f]

            if sanitize:
                events = [sanitize_event(e) for e in events]

            if events:
                synced = await storage.sync_event_lines(
                    user_id=user_id,
                    host_id=host_id,
                    project_slug=metadata.get("project_slug", "unknown"),
                    session_id=session_id,
                    lines=events,
                )
                stats["events_loaded"] += synced

    await storage.close()
    logger.info(f"Load complete: {stats}")

    return stats


async def main():
    parser = argparse.ArgumentParser(description="Load real Amplifier project into storage backend")
    parser.add_argument("project_path", type=Path, help="Path to Amplifier project directory")
    parser.add_argument("--backend", choices=["duckdb", "sqlite"], required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output database file")
    parser.add_argument("--user-id", default="test-user", help="User ID for storage")
    parser.add_argument("--host-id", default="test-host", help="Host ID for storage")
    parser.add_argument("--no-sanitize", action="store_true", help="Skip sanitization (UNSAFE!)")

    args = parser.parse_args()

    sanitize = not args.no_sanitize

    if not sanitize:
        logger.warning("⚠️  Sanitization disabled - sensitive data will be stored!")

    if args.backend == "duckdb":
        stats = await load_project_to_duckdb(
            args.project_path, args.output, args.user_id, args.host_id, sanitize
        )
    else:
        stats = await load_project_to_sqlite(
            args.project_path, args.output, args.user_id, args.host_id, sanitize
        )

    print("\n" + "=" * 70)
    print("LOAD COMPLETE")
    print("=" * 70)
    print(f"Sessions loaded: {stats['sessions_loaded']}")
    print(f"Transcripts loaded: {stats['transcripts_loaded']}")
    print(f"Events loaded: {stats['events_loaded']}")
    print(f"Sanitized: {stats['sanitized']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
