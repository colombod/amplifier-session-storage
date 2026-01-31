"""
Hybrid block storage with local + cloud sync.

Provides offline-first operation with background sync to Cosmos DB.
Handles multi-device scenarios with conflict resolution.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

from ..blocks.types import SessionBlock
from .base import (
    BlockStorage,
    SessionNotFoundError,
    StorageConfig,
)
from .cosmos import CosmosBlockStorage
from .local import LocalBlockStorage

logger = logging.getLogger(__name__)


class SyncState(Enum):
    """Sync state for a session."""

    SYNCED = "synced"  # All blocks synced
    PENDING = "pending"  # Has unsynced local blocks
    CONFLICT = "conflict"  # Has conflicts requiring resolution
    ERROR = "error"  # Sync failed


class ConflictResolution(Enum):
    """Strategy for resolving sync conflicts."""

    LOCAL_WINS = "local_wins"  # Local changes take precedence
    REMOTE_WINS = "remote_wins"  # Remote changes take precedence
    MERGE = "merge"  # Merge both (append remote blocks)
    MANUAL = "manual"  # Require manual resolution


class SyncConflict:
    """Represents a sync conflict between local and remote blocks."""

    def __init__(
        self,
        session_id: str,
        local_blocks: list[SessionBlock],
        remote_blocks: list[SessionBlock],
        conflict_sequence: int,
    ) -> None:
        self.session_id = session_id
        self.local_blocks = local_blocks
        self.remote_blocks = remote_blocks
        self.conflict_sequence = conflict_sequence
        self.detected_at = datetime.utcnow()

    def __repr__(self) -> str:
        return (
            f"SyncConflict(session={self.session_id}, "
            f"local={len(self.local_blocks)}, remote={len(self.remote_blocks)}, "
            f"at_seq={self.conflict_sequence})"
        )


class HybridBlockStorage(BlockStorage):
    """Hybrid storage combining local files with Cosmos DB sync.

    Architecture:
    - Writes go to LOCAL first (immediate, always available)
    - Background sync uploads to Cosmos DB when connected
    - Reads prefer local but can fetch from remote if needed
    - Handles offline operation gracefully
    - Resolves conflicts from multi-device usage

    Sync flow:
    1. Write to local storage (immediate return)
    2. Queue block for sync
    3. Background task uploads to Cosmos
    4. On conflict, apply resolution strategy

    Multi-device handling:
    - Each device has unique device_id in blocks
    - Sequence numbers are per-session, not per-device
    - Conflicts detected when same sequence from different devices
    - Resolution strategies: local_wins, remote_wins, merge, manual
    """

    def __init__(
        self,
        config: StorageConfig,
        conflict_resolution: ConflictResolution = ConflictResolution.MERGE,
        sync_interval: float = 5.0,
        on_conflict: Callable[[SyncConflict], None] | None = None,
        on_sync_error: Callable[[Exception], None] | None = None,
    ) -> None:
        """Initialize hybrid storage.

        Args:
            config: Storage configuration
            conflict_resolution: Strategy for resolving conflicts
            sync_interval: Seconds between sync attempts
            on_conflict: Callback when conflict detected (for MANUAL mode)
            on_sync_error: Callback when sync fails
        """
        self.config = config
        self.conflict_resolution = conflict_resolution
        self.sync_interval = sync_interval
        self.on_conflict = on_conflict
        self.on_sync_error = on_sync_error

        # Initialize storage backends
        self._local = LocalBlockStorage(config)
        self._remote: CosmosBlockStorage | None = None
        self._remote_available = False

        # Sync state
        self._sync_queue: asyncio.Queue[SessionBlock] = asyncio.Queue()
        self._sync_task: asyncio.Task[None] | None = None
        self._pending_sessions: set[str] = set()
        self._conflict_sessions: dict[str, SyncConflict] = {}
        self._running = False

        # Track sync state per session
        self._sync_states: dict[str, SyncState] = {}

    async def start(self) -> None:
        """Start the hybrid storage and background sync.

        Call this after initialization to enable cloud sync.
        """
        if self._running:
            return

        self._running = True

        # Try to connect to Cosmos DB
        if self.config.cosmos_endpoint:
            try:
                self._remote = CosmosBlockStorage(self.config)
                # Test connection
                await self._remote.list_sessions(limit=1)
                self._remote_available = True
                logger.info("Connected to Cosmos DB for sync")
            except Exception as e:
                logger.warning(f"Cosmos DB not available, running in offline mode: {e}")
                self._remote_available = False

        # Start background sync task
        self._sync_task = asyncio.create_task(self._sync_loop())

    async def stop(self) -> None:
        """Stop background sync and cleanup."""
        self._running = False

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

    async def write_block(self, block: SessionBlock) -> None:
        """Write a block to local storage and queue for sync."""
        # Always write to local first
        await self._local.write_block(block)

        # Queue for sync if remote available
        if self._remote_available:
            await self._sync_queue.put(block)
            self._pending_sessions.add(block.session_id)
            self._sync_states[block.session_id] = SyncState.PENDING

    async def write_blocks(self, blocks: list[SessionBlock]) -> None:
        """Write multiple blocks to local and queue for sync."""
        if not blocks:
            return

        # Write all to local
        await self._local.write_blocks(blocks)

        # Queue all for sync
        if self._remote_available:
            for block in blocks:
                await self._sync_queue.put(block)
                self._pending_sessions.add(block.session_id)
            self._sync_states[blocks[0].session_id] = SyncState.PENDING

    async def read_blocks(
        self,
        session_id: str,
        since_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[SessionBlock]:
        """Read blocks, merging local and remote if needed."""
        # Get local blocks
        local_blocks = await self._local.read_blocks(session_id, since_sequence, limit)

        # If remote available and we might be missing blocks, check remote
        if self._remote_available and self._remote:
            try:
                # Get latest sequence from remote
                remote_seq = await self._remote.get_latest_sequence(session_id)
                local_seq = await self._local.get_latest_sequence(session_id)

                # If remote has newer blocks, fetch them
                if remote_seq > local_seq:
                    remote_blocks = await self._remote.read_blocks(
                        session_id, since_sequence=local_seq
                    )
                    # Merge remote blocks into local
                    if remote_blocks:
                        await self._merge_remote_blocks(session_id, remote_blocks)
                        # Re-read local to get merged result
                        local_blocks = await self._local.read_blocks(
                            session_id, since_sequence, limit
                        )

            except Exception as e:
                logger.warning(f"Failed to sync from remote: {e}")

        return local_blocks

    async def get_latest_sequence(self, session_id: str) -> int:
        """Get latest sequence, checking both local and remote."""
        local_seq = await self._local.get_latest_sequence(session_id)

        if self._remote_available and self._remote:
            try:
                remote_seq = await self._remote.get_latest_sequence(session_id)
                return max(local_seq, remote_seq)
            except Exception:
                pass

        return local_seq

    async def list_sessions(
        self,
        project_slug: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List sessions from local storage.

        Remote sessions are synced on first access.
        """
        # For now, just return local sessions
        # TODO: Merge with remote session list
        return await self._local.list_sessions(project_slug, limit, offset)

    async def delete_session(self, session_id: str) -> None:
        """Delete session from both local and remote."""
        # Delete from local
        await self._local.delete_session(session_id)

        # Queue delete for remote
        if self._remote_available and self._remote:
            try:
                await self._remote.delete_session(session_id)
            except SessionNotFoundError:
                pass  # Already deleted remotely
            except Exception as e:
                logger.warning(f"Failed to delete session from remote: {e}")

        # Clean up sync state
        self._pending_sessions.discard(session_id)
        self._conflict_sessions.pop(session_id, None)
        self._sync_states.pop(session_id, None)

    async def close(self) -> None:
        """Stop sync and close all connections."""
        await self.stop()
        await self._local.close()
        if self._remote:
            await self._remote.close()

    # Sync state methods

    def get_sync_state(self, session_id: str) -> SyncState:
        """Get sync state for a session."""
        return self._sync_states.get(session_id, SyncState.SYNCED)

    def get_pending_sessions(self) -> set[str]:
        """Get sessions with pending sync."""
        return self._pending_sessions.copy()

    def get_conflicts(self) -> dict[str, SyncConflict]:
        """Get sessions with unresolved conflicts."""
        return self._conflict_sessions.copy()

    async def resolve_conflict(
        self,
        session_id: str,
        resolution: ConflictResolution,
    ) -> None:
        """Manually resolve a conflict for a session."""
        conflict = self._conflict_sessions.get(session_id)
        if not conflict:
            return

        await self._apply_resolution(conflict, resolution)
        del self._conflict_sessions[session_id]
        self._sync_states[session_id] = SyncState.PENDING

    async def force_sync(self, session_id: str | None = None) -> None:
        """Force immediate sync for a session or all pending sessions."""
        if session_id:
            await self._sync_session(session_id)
        else:
            for sid in list(self._pending_sessions):
                await self._sync_session(sid)

    # Private sync methods

    async def _sync_loop(self) -> None:
        """Background sync loop."""
        while self._running:
            try:
                # Process queued blocks
                blocks_to_sync: list[SessionBlock] = []
                while not self._sync_queue.empty():
                    try:
                        block = self._sync_queue.get_nowait()
                        blocks_to_sync.append(block)
                    except asyncio.QueueEmpty:
                        break

                if blocks_to_sync and self._remote_available and self._remote:
                    # Group by session
                    by_session: dict[str, list[SessionBlock]] = {}
                    for block in blocks_to_sync:
                        by_session.setdefault(block.session_id, []).append(block)

                    # Sync each session
                    for session_id, session_blocks in by_session.items():
                        try:
                            await self._sync_blocks_to_remote(session_id, session_blocks)
                            self._pending_sessions.discard(session_id)
                            if session_id not in self._conflict_sessions:
                                self._sync_states[session_id] = SyncState.SYNCED
                        except Exception as e:
                            logger.error(f"Sync failed for {session_id}: {e}")
                            self._sync_states[session_id] = SyncState.ERROR
                            if self.on_sync_error:
                                self.on_sync_error(e)

                await asyncio.sleep(self.sync_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(self.sync_interval)

    async def _sync_session(self, session_id: str) -> None:
        """Sync a specific session."""
        if not self._remote_available or not self._remote:
            return

        # Get local blocks
        local_blocks = await self._local.read_blocks(session_id)
        if not local_blocks:
            return

        # Get remote sequence to find blocks that need syncing
        remote_seq = await self._remote.get_latest_sequence(session_id)

        # Find blocks that need syncing
        blocks_to_push = [b for b in local_blocks if b.sequence > remote_seq]

        if blocks_to_push:
            await self._sync_blocks_to_remote(session_id, blocks_to_push)

    async def _sync_blocks_to_remote(
        self,
        session_id: str,
        blocks: list[SessionBlock],
    ) -> None:
        """Push blocks to remote, handling conflicts."""
        if not self._remote:
            return

        for block in blocks:
            try:
                # Check for conflict (same sequence exists remotely)
                existing = await self._remote.read_blocks(
                    session_id,
                    since_sequence=block.sequence - 1,
                    limit=1,
                )

                if existing and existing[0].sequence == block.sequence:
                    # Conflict detected
                    if existing[0].device_id != block.device_id:
                        await self._handle_conflict(session_id, block, existing[0])
                        continue

                # No conflict, write block
                await self._remote.write_block(block)

            except Exception as e:
                logger.error(f"Failed to sync block {block.block_id}: {e}")
                raise

    async def _handle_conflict(
        self,
        session_id: str,
        local_block: SessionBlock,
        remote_block: SessionBlock,
    ) -> None:
        """Handle a sync conflict."""
        logger.warning(
            f"Conflict detected in {session_id} at sequence {local_block.sequence}: "
            f"local device={local_block.device_id}, remote device={remote_block.device_id}"
        )

        conflict = SyncConflict(
            session_id=session_id,
            local_blocks=[local_block],
            remote_blocks=[remote_block],
            conflict_sequence=local_block.sequence,
        )

        if self.conflict_resolution == ConflictResolution.MANUAL:
            # Store conflict for manual resolution
            self._conflict_sessions[session_id] = conflict
            self._sync_states[session_id] = SyncState.CONFLICT
            if self.on_conflict:
                self.on_conflict(conflict)
        else:
            # Auto-resolve
            await self._apply_resolution(conflict, self.conflict_resolution)

    async def _apply_resolution(
        self,
        conflict: SyncConflict,
        resolution: ConflictResolution,
    ) -> None:
        """Apply a resolution strategy to a conflict."""
        if resolution == ConflictResolution.LOCAL_WINS:
            # Overwrite remote with local
            if self._remote:
                for block in conflict.local_blocks:
                    await self._remote.write_block(block)

        elif resolution == ConflictResolution.REMOTE_WINS:
            # Overwrite local with remote
            for block in conflict.remote_blocks:
                await self._local.write_block(block)

        elif resolution == ConflictResolution.MERGE:
            # Keep both by giving remote blocks new sequence numbers
            await self._merge_conflicting_blocks(conflict)

    async def _merge_conflicting_blocks(self, conflict: SyncConflict) -> None:
        """Merge conflicting blocks by appending remote as new sequences."""
        if not self._remote:
            return

        # Get the latest sequence after the conflict point
        latest_seq = await self.get_latest_sequence(conflict.session_id)

        # Create copies of remote blocks with new sequences
        for i, remote_block in enumerate(conflict.remote_blocks):
            new_seq = latest_seq + i + 1

            # Create a new block with incremented sequence
            merged_block = SessionBlock(
                block_id=f"{remote_block.block_id}-merged-{new_seq}",
                session_id=remote_block.session_id,
                user_id=remote_block.user_id,
                sequence=new_seq,
                timestamp=remote_block.timestamp,
                device_id=remote_block.device_id,
                block_type=remote_block.block_type,
                data=remote_block.data,
            )

            # Write merged block to both local and remote
            await self._local.write_block(merged_block)
            await self._remote.write_block(merged_block)

        # Now write local blocks to remote
        for block in conflict.local_blocks:
            await self._remote.write_block(block)

    async def _merge_remote_blocks(
        self,
        session_id: str,
        remote_blocks: list[SessionBlock],
    ) -> None:
        """Merge remote blocks into local storage."""
        for block in remote_blocks:
            try:
                # Check if we already have this sequence
                existing = await self._local.read_blocks(
                    session_id,
                    since_sequence=block.sequence - 1,
                    limit=1,
                )

                if existing and existing[0].sequence == block.sequence:
                    # Same sequence exists - check if it's the same block
                    if existing[0].block_id == block.block_id:
                        continue  # Already have it
                    # Different block at same sequence - conflict
                    await self._handle_conflict(session_id, existing[0], block)
                else:
                    # New block, add to local
                    await self._local.write_block(block)

            except Exception as e:
                logger.error(f"Failed to merge remote block: {e}")
