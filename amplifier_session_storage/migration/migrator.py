"""
Session migrator for converting legacy sessions to blocks.

Reads existing session files (events.jsonl, transcript.jsonl)
and converts them to the block-based format for storage.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..blocks.sequence import SimpleSequenceAllocator
from ..blocks.types import BlockType, SessionBlock
from ..blocks.writer import BlockWriter
from ..storage.base import BlockStorage
from .types import MigrationBatch, MigrationResult, MigrationStatus, SessionSource


class SessionMigrator:
    """Migrates legacy sessions to block-based storage.

    Reads existing Amplifier session files and converts them
    to the block-based event-sourced format.

    Legacy format:
    - events.jsonl: Stream of all session events
    - transcript.jsonl: Conversation messages
    - state.json: Session state/metadata

    Target format:
    - SESSION_CREATED block (from state.json)
    - MESSAGE blocks (from transcript.jsonl)
    - EVENT blocks (from events.jsonl)
    """

    def __init__(
        self,
        storage: BlockStorage,
        user_id: str,
        device_id: str | None = None,
    ) -> None:
        """Initialize the migrator.

        Args:
            storage: Target storage backend (local or cosmos)
            user_id: User ID for migrated sessions
            device_id: Device ID for blocks (defaults to migration-{timestamp})
        """
        self.storage = storage
        self.user_id = user_id
        self.device_id = device_id or f"migration-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    async def discover_sessions(
        self,
        sessions_path: Path,
        project_filter: str | None = None,
    ) -> list[SessionSource]:
        """Discover sessions available for migration.

        Scans the legacy sessions directory structure:
        {sessions_path}/{project_slug}/{session_id}/

        Args:
            sessions_path: Path to legacy sessions directory
            project_filter: Optional project slug to filter by

        Returns:
            List of discovered session sources
        """
        sources: list[SessionSource] = []

        if not sessions_path.exists():
            return sources

        # Iterate through projects
        for project_dir in sessions_path.iterdir():
            if not project_dir.is_dir():
                continue

            project_slug = project_dir.name

            # Filter by project if specified
            if project_filter and project_slug != project_filter:
                continue

            # Iterate through sessions
            for session_dir in project_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                session_id = session_dir.name

                # Check if this looks like a valid session
                events_file = session_dir / "events.jsonl"
                transcript_file = session_dir / "transcript.jsonl"

                if not events_file.exists() and not transcript_file.exists():
                    continue

                source = SessionSource(
                    session_id=session_id,
                    path=session_dir,
                    project_slug=project_slug,
                )

                # Try to get metadata from state.json
                state_file = session_dir / "state.json"
                if state_file.exists():
                    try:
                        with open(state_file) as f:
                            state = json.load(f)
                            if "created" in state:
                                source.created = datetime.fromisoformat(state["created"])
                            if "updated" in state:
                                source.updated = datetime.fromisoformat(state["updated"])
                    except Exception:
                        pass

                sources.append(source)

        return sources

    async def migrate_session(
        self,
        source: SessionSource,
        skip_if_exists: bool = True,
    ) -> MigrationResult:
        """Migrate a single session to block storage.

        Args:
            source: Session source information
            skip_if_exists: If True, skip if session already exists

        Returns:
            Migration result
        """
        result = MigrationResult(
            session_id=source.session_id,
            status=MigrationStatus.PENDING,
            source_path=str(source.path),
            source_size_bytes=source.total_size_bytes,
        )
        result.started_at = datetime.utcnow()

        try:
            # Check if already migrated
            if skip_if_exists:
                existing = await self.storage.read_blocks(source.session_id, limit=1)
                if existing:
                    result.status = MigrationStatus.SKIPPED
                    result.completed_at = datetime.utcnow()
                    return result

            result.status = MigrationStatus.IN_PROGRESS

            # Create block writer
            allocator = SimpleSequenceAllocator()
            writer = BlockWriter(
                session_id=source.session_id,
                user_id=self.user_id,
                device_id=self.device_id,
                sequence_allocator=allocator,
            )

            blocks: list[SessionBlock] = []

            # 1. Create SESSION_CREATED block
            session_block = await self._create_session_block(writer, source)
            blocks.append(session_block)

            # 2. Process transcript (messages)
            if source.transcript_file and source.transcript_file.exists():
                message_blocks = await self._process_transcript(writer, source.transcript_file)
                blocks.extend(message_blocks)
                result.messages_migrated = len(message_blocks)

            # 3. Process events
            if source.events_file and source.events_file.exists():
                event_blocks = await self._process_events(writer, source.events_file)
                blocks.extend(event_blocks)
                result.events_migrated = len(
                    [b for b in event_blocks if b.block_type == BlockType.EVENT]
                )

            # Write all blocks
            if blocks:
                await self.storage.write_blocks(blocks)

            result.blocks_created = len(blocks)
            result.status = MigrationStatus.COMPLETED

        except Exception as e:
            result.status = MigrationStatus.FAILED
            result.error_message = str(e)
            result.error_details = {"type": type(e).__name__}

        result.completed_at = datetime.utcnow()
        return result

    async def migrate_batch(
        self,
        sources: list[SessionSource],
        skip_if_exists: bool = True,
        on_progress: Any | None = None,
    ) -> MigrationBatch:
        """Migrate multiple sessions.

        Args:
            sources: List of session sources
            skip_if_exists: Skip sessions that already exist
            on_progress: Optional callback(result, index, total)

        Returns:
            Batch migration result
        """
        batch = MigrationBatch(
            total_sessions=len(sources),
            started_at=datetime.utcnow(),
        )

        for i, source in enumerate(sources):
            result = await self.migrate_session(source, skip_if_exists)
            batch.add_result(result)

            if on_progress:
                on_progress(result, i + 1, len(sources))

        batch.completed_at = datetime.utcnow()
        return batch

    async def _create_session_block(
        self,
        writer: BlockWriter,
        source: SessionSource,
    ) -> SessionBlock:
        """Create SESSION_CREATED block from source metadata."""
        # Try to load state.json for metadata
        metadata: dict[str, Any] = {}
        if source.state_file and source.state_file.exists():
            try:
                with open(source.state_file) as f:
                    metadata = json.load(f)
            except Exception:
                pass

        return writer.create_session(
            project_slug=source.project_slug,
            name=metadata.get("name"),
            description=metadata.get("description"),
            bundle=metadata.get("bundle"),
            model=metadata.get("model"),
            visibility=metadata.get("visibility", "private"),
            org_id=metadata.get("org_id"),
            team_ids=metadata.get("team_ids"),
            tags=metadata.get("tags"),
            parent_session_id=metadata.get("parent_session_id"),
        )

    async def _process_transcript(
        self,
        writer: BlockWriter,
        transcript_file: Path,
    ) -> list[SessionBlock]:
        """Process transcript.jsonl into MESSAGE blocks."""
        blocks: list[SessionBlock] = []

        with open(transcript_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)

                    # Extract message fields
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    turn = msg.get("turn", 1)
                    tool_calls = msg.get("tool_calls")
                    tool_call_id = msg.get("tool_call_id")

                    block = writer.add_message(
                        role=role,
                        content=content,
                        turn=turn,
                        tool_calls=tool_calls,
                        tool_call_id=tool_call_id,
                    )
                    blocks.append(block)

                except json.JSONDecodeError:
                    continue

        return blocks

    async def _process_events(
        self,
        writer: BlockWriter,
        events_file: Path,
    ) -> list[SessionBlock]:
        """Process events.jsonl into EVENT blocks.

        Note: Large events are automatically chunked by the writer.
        """
        blocks: list[SessionBlock] = []

        with open(events_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)

                    # Extract event fields
                    event_id = event.get("id", str(uuid.uuid4()))
                    event_type = event.get("type", event.get("event_type", "unknown"))
                    turn = event.get("turn")

                    # The writer handles chunking for large events
                    event_blocks = writer.add_event(
                        event_id=event_id,
                        event_type=event_type,
                        data=event,
                        turn=turn,
                    )
                    blocks.extend(event_blocks)

                except json.JSONDecodeError:
                    continue

        return blocks
