"""
Load a real Amplifier project into storage backends for testing.

Useful for testing with real conversation patterns and complex message structures.

Usage:
    python scripts/load_real_project.py <project_path> --backend duckdb --output ./test.db
    python scripts/load_real_project.py <project_path> --backend sqlite --output ./test.sqlite
    python scripts/load_real_project.py <project_path> --backend cosmos

For Cosmos, set environment variables:
    AMPLIFIER_COSMOS_ENDPOINT - Cosmos DB endpoint URL
    AMPLIFIER_COSMOS_DATABASE - Database name (use test DB, NOT production!)
    AMPLIFIER_COSMOS_ENABLE_VECTOR - Enable vector search (true/false)

WARNING: For Cosmos, always use a TEST database, never production!
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from amplifier_session_storage.backends.cosmos import CosmosBackend, CosmosConfig
from amplifier_session_storage.backends.duckdb import DuckDBBackend, DuckDBConfig
from amplifier_session_storage.backends.sqlite import SQLiteBackend, SQLiteConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def load_project_to_duckdb(
    project_path: Path,
    output_path: Path,
    user_id: str = "test-user",
    host_id: str = "test-host",
) -> dict:
    """
    Load Amplifier project into DuckDB.

    Args:
        project_path: Path to project directory (contains sessions/)
        output_path: Path for DuckDB file
        user_id: User ID for storage
        host_id: Host ID for storage

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
    }

    # Load each session
    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue

        session_id = session_dir.name
        logger.info(f"Loading session: {session_id}")

        metadata: dict = {}  # Initialize metadata

        # Load metadata
        metadata_file = session_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

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
                    if line.strip():
                        message = json.loads(line.strip())
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
                    if line.strip():
                        event = json.loads(line.strip())
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
    }

    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue

        session_id = session_dir.name
        logger.info(f"Loading session: {session_id}")

        metadata: dict = {}  # Initialize metadata

        # Metadata
        metadata_file = session_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

            await storage.upsert_session_metadata(
                user_id=user_id, host_id=host_id, metadata=metadata
            )
            stats["sessions_loaded"] += 1

        # Transcript
        transcript_file = session_dir / "transcript.jsonl"
        if transcript_file.exists():
            with open(transcript_file) as f:
                lines = [json.loads(line.strip()) for line in f if line.strip()]

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
                events = [json.loads(line.strip()) for line in f if line.strip()]

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


async def load_project_to_cosmos(
    project_path: Path,
    user_id: str = "test-user",
    host_id: str = "test-host",
) -> dict:
    """
    Load Amplifier project into Cosmos DB.

    IMPORTANT: Use a TEST database, never production!

    Args:
        project_path: Path to project directory (contains sessions/)
        user_id: User ID for storage
        host_id: Host ID for storage

    Returns:
        Statistics about what was loaded
    """
    import os

    # Safety check - warn if pointing to production-looking database
    db_name = os.environ.get("AMPLIFIER_COSMOS_DATABASE", "")
    if "prod" in db_name.lower() or db_name == "amplifier-db":
        logger.error("REFUSING to load into production database!")
        logger.error(f"Database name: {db_name}")
        logger.error("Set AMPLIFIER_COSMOS_DATABASE to a test database name.")
        raise ValueError("Cannot load test data into production database")

    config = CosmosConfig.from_env()
    logger.info(f"Connecting to Cosmos DB: {config.endpoint}")
    logger.info(f"Database: {config.database_name}")

    storage = await CosmosBackend.create(config=config, embedding_provider=None)

    sessions_dir = project_path / "sessions"
    if not sessions_dir.exists():
        raise ValueError(f"No sessions directory found at {sessions_dir}")

    stats = {
        "sessions_loaded": 0,
        "transcripts_loaded": 0,
        "events_loaded": 0,
        "backend": "cosmos",
        "database": config.database_name,
    }

    # Load each session
    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue

        session_id = session_dir.name
        logger.info(f"Loading session: {session_id}")

        metadata = {}  # Initialize metadata

        # Load metadata
        metadata_file = session_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

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
                    if line.strip():
                        message = json.loads(line.strip())
                        lines.append(message)

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

        # Load events
        events_file = session_dir / "events.jsonl"
        if events_file.exists():
            with open(events_file) as f:
                events = []
                for line in f:
                    if line.strip():
                        event = json.loads(line.strip())
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


async def main():
    parser = argparse.ArgumentParser(
        description="Load real Amplifier project into storage backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Load into DuckDB
    python scripts/load_real_project.py /path/to/project --backend duckdb --output ./test.db

    # Load into SQLite
    python scripts/load_real_project.py /path/to/project --backend sqlite --output ./test.sqlite

    # Load into Cosmos (uses environment variables)
    AMPLIFIER_COSMOS_ENDPOINT="https://...test..." \\
    AMPLIFIER_COSMOS_DATABASE="amplifier-test-db" \\
    python scripts/load_real_project.py /path/to/project --backend cosmos

WARNING: For Cosmos, always use a TEST database, never production!
        """,
    )
    parser.add_argument("project_path", type=Path, help="Path to Amplifier project directory")
    parser.add_argument("--backend", choices=["duckdb", "sqlite", "cosmos"], required=True)
    parser.add_argument(
        "--output", type=Path, help="Output database file (required for duckdb/sqlite)"
    )
    parser.add_argument("--user-id", default="test-user", help="User ID for storage")
    parser.add_argument("--host-id", default="test-host", help="Host ID for storage")

    args = parser.parse_args()

    # Validate output path for file-based backends
    if args.backend in ["duckdb", "sqlite"] and not args.output:
        parser.error(f"--output is required for {args.backend} backend")

    if args.backend == "duckdb":
        stats = await load_project_to_duckdb(
            args.project_path, args.output, args.user_id, args.host_id
        )
    elif args.backend == "sqlite":
        stats = await load_project_to_sqlite(
            args.project_path, args.output, args.user_id, args.host_id
        )
    else:  # cosmos
        stats = await load_project_to_cosmos(
            args.project_path, args.user_id, args.host_id
        )

    print("\n" + "=" * 70)
    print("LOAD COMPLETE")
    print("=" * 70)
    print(f"Backend: {args.backend}")
    print(f"Sessions loaded: {stats['sessions_loaded']}")
    print(f"Transcripts loaded: {stats['transcripts_loaded']}")
    print(f"Events loaded: {stats['events_loaded']}")
    if args.output:
        print(f"Output: {args.output}")
    if stats.get("database"):
        print(f"Database: {stats['database']}")


if __name__ == "__main__":
    asyncio.run(main())
