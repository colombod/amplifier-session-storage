"""
Local file-based block storage.

Stores blocks as JSONL files on disk, compatible with
existing Amplifier session file format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os

from ..blocks.reader import SessionStateReader
from ..blocks.types import BlockType, SessionBlock
from .base import (
    AccessDeniedError,
    BlockStorage,
    SessionNotFoundError,
    StorageConfig,
    StorageError,
)


class LocalBlockStorage(BlockStorage):
    """Local file-based block storage.

    Stores each session as a directory containing:
    - blocks.jsonl: Stream of session blocks
    - metadata.json: Quick-access session metadata (cached)

    Directory structure:
    {base_path}/
      {user_id}/
        {project_slug}/
          {session_id}/
            blocks.jsonl
            metadata.json
    """

    def __init__(self, config: StorageConfig) -> None:
        """Initialize local storage.

        Args:
            config: Storage configuration
        """
        self.config = config
        self.user_id = config.user_id

        # Default to ~/.amplifier/sessions
        if config.local_path:
            self.base_path = Path(config.local_path)
        else:
            self.base_path = Path.home() / ".amplifier" / "sessions"

        self._reader = SessionStateReader()

    def _session_dir(self, session_id: str, project_slug: str = "default") -> Path:
        """Get the directory for a session."""
        return self.base_path / self.user_id / project_slug / session_id

    def _blocks_file(self, session_id: str, project_slug: str = "default") -> Path:
        """Get the blocks file path for a session."""
        return self._session_dir(session_id, project_slug) / "blocks.jsonl"

    def _metadata_file(self, session_id: str, project_slug: str = "default") -> Path:
        """Get the metadata file path for a session."""
        return self._session_dir(session_id, project_slug) / "metadata.json"

    async def write_block(self, block: SessionBlock) -> None:
        """Write a single block to storage."""
        await self.write_blocks([block])

    async def write_blocks(self, blocks: list[SessionBlock]) -> None:
        """Write multiple blocks to storage.

        Appends blocks to the JSONL file and updates metadata cache.
        """
        if not blocks:
            return

        # Verify all blocks belong to the same session
        session_id = blocks[0].session_id
        if not all(b.session_id == session_id for b in blocks):
            raise StorageError("All blocks must belong to the same session")

        # Verify ownership
        if not all(b.user_id == self.user_id for b in blocks):
            raise AccessDeniedError("Cannot write blocks for another user")

        # Try to find existing session directory first
        existing_dir = await self._find_session_dir(session_id)
        if existing_dir:
            project_slug = existing_dir.parent.name
        else:
            # Get project_slug from session_created block or default
            project_slug = self._get_project_slug(blocks)

        # Ensure directory exists
        session_dir = self._session_dir(session_id, project_slug)
        await aiofiles.os.makedirs(session_dir, exist_ok=True)

        # Append blocks to JSONL file
        blocks_file = self._blocks_file(session_id, project_slug)
        async with aiofiles.open(blocks_file, "a") as f:
            for block in blocks:
                line = json.dumps(block.to_dict()) + "\n"
                await f.write(line)

        # Update metadata cache
        await self._update_metadata_cache(session_id, project_slug)

    async def read_blocks(
        self,
        session_id: str,
        since_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[SessionBlock]:
        """Read blocks for a session."""
        # Find the session directory
        session_dir = await self._find_session_dir(session_id)
        if session_dir is None:
            return []

        blocks_file = session_dir / "blocks.jsonl"
        if not blocks_file.exists():
            return []

        blocks: list[SessionBlock] = []
        async with aiofiles.open(blocks_file) as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                block = SessionBlock.from_dict(data)

                # Filter by sequence if requested
                if since_sequence is not None and block.sequence <= since_sequence:
                    continue

                blocks.append(block)

                # Apply limit
                if limit is not None and len(blocks) >= limit:
                    break

        # Sort by sequence
        blocks.sort(key=lambda b: b.sequence)
        return blocks

    async def get_latest_sequence(self, session_id: str) -> int:
        """Get the latest sequence number for a session."""
        blocks = await self.read_blocks(session_id)
        if not blocks:
            return 0
        return max(b.sequence for b in blocks)

    async def list_sessions(
        self,
        project_slug: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List sessions for the current user."""
        sessions: list[dict[str, Any]] = []

        user_dir = self.base_path / self.user_id
        if not user_dir.exists():
            return []

        # Iterate through projects
        for project_dir in user_dir.iterdir():
            if not project_dir.is_dir():
                continue

            # Filter by project_slug if specified
            if project_slug is not None and project_dir.name != project_slug:
                continue

            # Iterate through sessions
            for session_dir in project_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                # Read metadata
                metadata = await self._read_metadata(session_dir)
                if metadata:
                    sessions.append(metadata)

        # Sort by updated timestamp (most recent first)
        sessions.sort(
            key=lambda s: s.get("updated", s.get("created", "")),
            reverse=True,
        )

        # Apply pagination
        return sessions[offset : offset + limit]

    async def delete_session(self, session_id: str) -> None:
        """Delete all blocks for a session."""
        session_dir = await self._find_session_dir(session_id)
        if session_dir is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")

        # Remove all files in the directory
        for item in session_dir.iterdir():
            if item.is_file():
                item.unlink()

        # Remove the directory
        session_dir.rmdir()

    async def close(self) -> None:
        """Close storage (no-op for local storage)."""
        pass

    def _get_project_slug(self, blocks: list[SessionBlock]) -> str:
        """Extract project_slug from blocks."""
        for block in blocks:
            if block.block_type == BlockType.SESSION_CREATED:
                return block.data.get("project_slug", "default")
        return "default"

    async def _find_session_dir(self, session_id: str) -> Path | None:
        """Find the directory for a session (searches all projects)."""
        user_dir = self.base_path / self.user_id
        if not user_dir.exists():
            return None

        for project_dir in user_dir.iterdir():
            if not project_dir.is_dir():
                continue

            session_dir = project_dir / session_id
            if session_dir.exists():
                return session_dir

        return None

    async def _update_metadata_cache(
        self,
        session_id: str,
        project_slug: str,
    ) -> None:
        """Update the metadata cache file."""
        blocks = await self.read_blocks(session_id)
        if not blocks:
            return

        try:
            metadata, messages, events = self._reader.compute_current_state(blocks)

            cache_data = {
                "session_id": metadata.session_id,
                "user_id": metadata.user_id,
                "name": metadata.name,
                "description": metadata.description,
                "project_slug": metadata.project_slug or project_slug,
                "bundle": metadata.bundle,
                "model": metadata.model,
                "visibility": metadata.visibility,
                "org_id": metadata.org_id,
                "team_ids": metadata.team_ids,
                "tags": metadata.tags,
                "parent_session_id": metadata.parent_session_id,
                "created": metadata.created.isoformat() if metadata.created else None,
                "updated": metadata.updated.isoformat() if metadata.updated else None,
                "turn_count": metadata.turn_count,
                "message_count": metadata.message_count,
                "event_count": metadata.event_count,
                "last_sequence": metadata.last_sequence,
            }

            metadata_file = self._metadata_file(session_id, project_slug)
            async with aiofiles.open(metadata_file, "w") as f:
                await f.write(json.dumps(cache_data, indent=2))

        except Exception:
            # Metadata cache is optional; don't fail on errors
            pass

    async def _read_metadata(self, session_dir: Path) -> dict[str, Any] | None:
        """Read metadata from cache or compute from blocks."""
        metadata_file = session_dir / "metadata.json"

        # Try cache first
        if metadata_file.exists():
            try:
                async with aiofiles.open(metadata_file) as f:
                    content = await f.read()
                    return json.loads(content)
            except Exception:
                pass

        # Fall back to computing from blocks
        blocks_file = session_dir / "blocks.jsonl"
        if not blocks_file.exists():
            return None

        blocks: list[SessionBlock] = []
        try:
            async with aiofiles.open(blocks_file) as f:
                async for line in f:
                    line = line.strip()
                    if line:
                        blocks.append(SessionBlock.from_dict(json.loads(line)))

            if not blocks:
                return None

            metadata, _, _ = self._reader.compute_current_state(blocks)

            return {
                "session_id": metadata.session_id,
                "user_id": metadata.user_id,
                "name": metadata.name,
                "project_slug": metadata.project_slug,
                "visibility": metadata.visibility,
                "created": metadata.created.isoformat() if metadata.created else None,
                "updated": metadata.updated.isoformat() if metadata.updated else None,
                "turn_count": metadata.turn_count,
                "message_count": metadata.message_count,
            }

        except Exception:
            return None
