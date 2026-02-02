"""Tests for sync filtering in hybrid storage.

The sync filter allows host applications to control which sessions
get synced to Cosmos DB.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from amplifier_session_storage.blocks.types import BlockType, SessionBlock
from amplifier_session_storage.storage.base import StorageConfig
from amplifier_session_storage.storage.hybrid import (
    HybridBlockStorage,
    SyncPolicy,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def base_config(temp_dir):
    """Create a base storage config."""
    return StorageConfig(
        user_id="test-user",
        local_path=temp_dir,
        enable_sync=True,
        cosmos_endpoint="https://test.documents.azure.com:443/",
        cosmos_database="test-db",
        cosmos_container="test-container",
    )


def create_test_block(
    session_id: str,
    sequence: int = 1,
    block_type: BlockType = BlockType.SESSION_CREATED,
    data: dict[str, Any] | None = None,
) -> SessionBlock:
    """Create a test session block."""
    import uuid

    return SessionBlock(
        block_id=str(uuid.uuid4()),
        session_id=session_id,
        user_id="test-user",
        sequence=sequence,
        timestamp=datetime.now(UTC),
        device_id="test-device",
        block_type=block_type,
        data=data or {"test": "data"},
    )


class TestSyncPolicy:
    """Tests for SyncPolicy enum and basic filtering."""

    def test_sync_all_policy(self):
        """SYNC_ALL syncs everything."""
        assert SyncPolicy.SYNC_ALL.value == "sync_all"

    def test_sync_none_policy(self):
        """SYNC_NONE syncs nothing."""
        assert SyncPolicy.SYNC_NONE.value == "sync_none"

    def test_sync_filter_policy(self):
        """SYNC_FILTER uses custom filter function."""
        assert SyncPolicy.SYNC_FILTER.value == "sync_filter"


class TestSyncFilterFunction:
    """Tests for custom sync filter functions."""

    def test_filter_by_bundle(self):
        """Filter sessions by bundle name."""

        def only_production_bundles(session_id: str, metadata: dict[str, Any]) -> bool:
            bundle = metadata.get("bundle", "")
            return bundle.startswith("prod-")

        # Should sync production bundles
        assert only_production_bundles("s1", {"bundle": "prod-app"}) is True
        assert only_production_bundles("s2", {"bundle": "prod-api"}) is True

        # Should NOT sync dev/test bundles
        assert only_production_bundles("s3", {"bundle": "dev-app"}) is False
        assert only_production_bundles("s4", {"bundle": "test-suite"}) is False

    def test_filter_by_project(self):
        """Filter sessions by project slug."""

        def only_shared_projects(session_id: str, metadata: dict[str, Any]) -> bool:
            project = metadata.get("project_slug", "")
            return project in ["team-shared", "org-shared"]

        assert only_shared_projects("s1", {"project_slug": "team-shared"}) is True
        assert only_shared_projects("s2", {"project_slug": "personal"}) is False

    def test_filter_by_metadata_flag(self):
        """Filter sessions by explicit sync flag in metadata."""

        def check_sync_flag(session_id: str, metadata: dict[str, Any]) -> bool:
            # Default to True if not specified
            return metadata.get("sync_to_cloud", True)

        # Default behavior - sync
        assert check_sync_flag("s1", {}) is True
        assert check_sync_flag("s2", {"bundle": "test"}) is True

        # Explicit opt-out
        assert check_sync_flag("s3", {"sync_to_cloud": False}) is False

        # Explicit opt-in
        assert check_sync_flag("s4", {"sync_to_cloud": True}) is True

    def test_filter_by_turn_count(self):
        """Only sync sessions with minimum turns."""

        def min_turns_filter(session_id: str, metadata: dict[str, Any]) -> bool:
            turn_count = metadata.get("turn_count", 0)
            return turn_count >= 3  # Only sync sessions with 3+ turns

        assert min_turns_filter("s1", {"turn_count": 5}) is True
        assert min_turns_filter("s2", {"turn_count": 3}) is True
        assert min_turns_filter("s3", {"turn_count": 2}) is False
        assert min_turns_filter("s4", {"turn_count": 0}) is False

    def test_combined_filter(self):
        """Combine multiple filter conditions."""

        def combined_filter(session_id: str, metadata: dict[str, Any]) -> bool:
            # Must be production bundle
            bundle = metadata.get("bundle", "")
            if not bundle.startswith("prod-"):
                return False

            # Must not be explicitly disabled
            if metadata.get("sync_to_cloud") is False:
                return False

            # Must have at least 2 turns
            turn_count = metadata.get("turn_count", 0)
            if turn_count < 2:
                return False

            return True

        # Passes all conditions
        assert (
            combined_filter(
                "s1",
                {
                    "bundle": "prod-app",
                    "turn_count": 5,
                },
            )
            is True
        )

        # Fails bundle check
        assert (
            combined_filter(
                "s2",
                {
                    "bundle": "dev-app",
                    "turn_count": 5,
                },
            )
            is False
        )

        # Fails sync flag check
        assert (
            combined_filter(
                "s3",
                {
                    "bundle": "prod-app",
                    "turn_count": 5,
                    "sync_to_cloud": False,
                },
            )
            is False
        )

        # Fails turn count check
        assert (
            combined_filter(
                "s4",
                {
                    "bundle": "prod-app",
                    "turn_count": 1,
                },
            )
            is False
        )


class TestHybridStorageSyncFilter:
    """Tests for sync filtering in HybridBlockStorage."""

    @pytest.fixture
    def mock_remote(self):
        """Create a mock remote storage."""
        mock = AsyncMock()
        mock.write_block = AsyncMock()
        mock.write_blocks = AsyncMock()
        mock.list_sessions = AsyncMock(return_value=[])
        mock.get_latest_sequence = AsyncMock(return_value=0)
        return mock

    @pytest.mark.asyncio
    async def test_sync_all_policy_syncs_everything(self, base_config, mock_remote):
        """With SYNC_ALL policy, all sessions are synced."""
        storage = HybridBlockStorage(
            base_config,
            sync_policy=SyncPolicy.SYNC_ALL,
        )
        storage._remote = mock_remote
        storage._remote_available = True

        block = create_test_block("session-1")
        await storage.write_block(block)

        # Block should be queued for sync
        assert storage._sync_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_sync_none_policy_syncs_nothing(self, base_config, mock_remote):
        """With SYNC_NONE policy, no sessions are synced."""
        storage = HybridBlockStorage(
            base_config,
            sync_policy=SyncPolicy.SYNC_NONE,
        )
        storage._remote = mock_remote
        storage._remote_available = True

        block = create_test_block("session-1")
        await storage.write_block(block)

        # Block should NOT be queued for sync
        assert storage._sync_queue.qsize() == 0

        # But it should still be written locally
        local_blocks = await storage._local.read_blocks("session-1")
        assert len(local_blocks) == 1

    @pytest.mark.asyncio
    async def test_custom_filter_function(self, base_config, mock_remote):
        """Custom filter function controls which sessions sync."""

        def only_important_sessions(session_id: str, metadata: dict[str, Any]) -> bool:
            return metadata.get("important", False)

        storage = HybridBlockStorage(
            base_config,
            sync_policy=SyncPolicy.SYNC_FILTER,
            sync_filter=only_important_sessions,
        )
        storage._remote = mock_remote
        storage._remote_available = True

        # Register metadata for sessions
        storage.set_session_metadata("session-important", {"important": True})
        storage.set_session_metadata("session-normal", {"important": False})

        # Write blocks
        block1 = create_test_block("session-important")
        block2 = create_test_block("session-normal")

        await storage.write_block(block1)
        await storage.write_block(block2)

        # Only important session should be queued
        assert storage._sync_queue.qsize() == 1
        queued_block = await storage._sync_queue.get()
        assert queued_block.session_id == "session-important"

    @pytest.mark.asyncio
    async def test_mark_session_local_only(self, base_config, mock_remote):
        """Sessions can be marked as local-only at runtime."""
        storage = HybridBlockStorage(
            base_config,
            sync_policy=SyncPolicy.SYNC_ALL,
        )
        storage._remote = mock_remote
        storage._remote_available = True

        # Mark session as local-only
        storage.mark_session_local_only("session-private")

        # Write to local-only session
        block = create_test_block("session-private")
        await storage.write_block(block)

        # Should NOT be queued for sync
        assert storage._sync_queue.qsize() == 0

        # But should still be written locally
        local_blocks = await storage._local.read_blocks("session-private")
        assert len(local_blocks) == 1

    @pytest.mark.asyncio
    async def test_unmark_session_local_only(self, base_config, mock_remote):
        """Sessions can be unmarked from local-only."""
        storage = HybridBlockStorage(
            base_config,
            sync_policy=SyncPolicy.SYNC_ALL,
        )
        storage._remote = mock_remote
        storage._remote_available = True

        # Mark then unmark
        storage.mark_session_local_only("session-1")
        storage.unmark_session_local_only("session-1")

        # Write block
        block = create_test_block("session-1")
        await storage.write_block(block)

        # Should be queued for sync
        assert storage._sync_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_is_session_local_only(self, base_config):
        """Check if session is marked as local-only."""
        storage = HybridBlockStorage(base_config)

        assert storage.is_session_local_only("session-1") is False

        storage.mark_session_local_only("session-1")
        assert storage.is_session_local_only("session-1") is True

        storage.unmark_session_local_only("session-1")
        assert storage.is_session_local_only("session-1") is False

    @pytest.mark.asyncio
    async def test_filter_receives_current_metadata(self, base_config, mock_remote):
        """Filter function receives current session metadata."""
        received_metadata: list[dict] = []

        def capture_metadata(session_id: str, metadata: dict[str, Any]) -> bool:
            received_metadata.append(metadata.copy())
            return True

        storage = HybridBlockStorage(
            base_config,
            sync_policy=SyncPolicy.SYNC_FILTER,
            sync_filter=capture_metadata,
        )
        storage._remote = mock_remote
        storage._remote_available = True

        # Set metadata before writing
        storage.set_session_metadata(
            "session-1",
            {
                "bundle": "test-bundle",
                "model": "test-model",
                "turn_count": 5,
            },
        )

        block = create_test_block("session-1")
        await storage.write_block(block)

        # Filter should have received the metadata
        assert len(received_metadata) == 1
        assert received_metadata[0]["bundle"] == "test-bundle"
        assert received_metadata[0]["model"] == "test-model"
        assert received_metadata[0]["turn_count"] == 5

    @pytest.mark.asyncio
    async def test_update_metadata_affects_future_blocks(self, base_config, mock_remote):
        """Updated metadata is used for subsequent blocks."""

        def sync_if_complete(session_id: str, metadata: dict[str, Any]) -> bool:
            return metadata.get("status") == "complete"

        storage = HybridBlockStorage(
            base_config,
            sync_policy=SyncPolicy.SYNC_FILTER,
            sync_filter=sync_if_complete,
        )
        storage._remote = mock_remote
        storage._remote_available = True

        # Start with incomplete status
        storage.set_session_metadata("session-1", {"status": "active"})

        block1 = create_test_block("session-1", sequence=1)
        await storage.write_block(block1)

        # Not queued yet
        assert storage._sync_queue.qsize() == 0

        # Update status to complete
        storage.set_session_metadata("session-1", {"status": "complete"})

        block2 = create_test_block("session-1", sequence=2)
        await storage.write_block(block2)

        # Now should be queued
        assert storage._sync_queue.qsize() == 1


class TestSyncFilterWithBatches:
    """Tests for sync filtering with batch writes.

    Note: write_blocks requires all blocks to be from the same session.
    For multi-session scenarios, use write_block individually.
    """

    @pytest.fixture
    def mock_remote(self):
        """Create a mock remote storage."""
        mock = AsyncMock()
        mock.write_blocks = AsyncMock()
        mock.list_sessions = AsyncMock(return_value=[])
        return mock

    @pytest.mark.asyncio
    async def test_batch_write_respects_filter(self, base_config, mock_remote):
        """Batch writes respect sync filter."""

        def only_important(session_id: str, metadata: dict[str, Any]) -> bool:
            return metadata.get("important", False)

        storage = HybridBlockStorage(
            base_config,
            sync_policy=SyncPolicy.SYNC_FILTER,
            sync_filter=only_important,
        )
        storage._remote = mock_remote
        storage._remote_available = True

        # Mark session as important
        storage.set_session_metadata("session-a", {"important": True})

        # Write batch (all same session as required by write_blocks)
        blocks = [
            create_test_block("session-a", sequence=1),
            create_test_block("session-a", sequence=2),
            create_test_block("session-a", sequence=3),
        ]

        await storage.write_blocks(blocks)

        # All blocks should be queued (session is important)
        assert storage._sync_queue.qsize() == 3

    @pytest.mark.asyncio
    async def test_batch_write_filtered_out(self, base_config, mock_remote):
        """Batch writes can be completely filtered out."""

        def only_important(session_id: str, metadata: dict[str, Any]) -> bool:
            return metadata.get("important", False)

        storage = HybridBlockStorage(
            base_config,
            sync_policy=SyncPolicy.SYNC_FILTER,
            sync_filter=only_important,
        )
        storage._remote = mock_remote
        storage._remote_available = True

        # Session is NOT important
        storage.set_session_metadata("session-b", {"important": False})

        # Write batch
        blocks = [
            create_test_block("session-b", sequence=1),
            create_test_block("session-b", sequence=2),
        ]

        await storage.write_blocks(blocks)

        # Nothing should be queued
        assert storage._sync_queue.qsize() == 0

        # But should be written locally
        local = await storage._local.read_blocks("session-b")
        assert len(local) == 2

    @pytest.mark.asyncio
    async def test_batch_write_with_sync_none(self, base_config, mock_remote):
        """Batch writes with SYNC_NONE policy."""
        storage = HybridBlockStorage(
            base_config,
            sync_policy=SyncPolicy.SYNC_NONE,
        )
        storage._remote = mock_remote
        storage._remote_available = True

        # Single session batch
        blocks = [
            create_test_block("session-1", sequence=1),
            create_test_block("session-1", sequence=2),
        ]

        await storage.write_blocks(blocks)

        # Nothing should be queued
        assert storage._sync_queue.qsize() == 0

        # But should be written locally
        local = await storage._local.read_blocks("session-1")
        assert len(local) == 2

    @pytest.mark.asyncio
    async def test_individual_writes_filter_per_session(self, base_config, mock_remote):
        """Individual write_block calls filter correctly per session."""

        def only_session_a(session_id: str, metadata: dict[str, Any]) -> bool:
            return session_id == "session-a"

        storage = HybridBlockStorage(
            base_config,
            sync_policy=SyncPolicy.SYNC_FILTER,
            sync_filter=only_session_a,
        )
        storage._remote = mock_remote
        storage._remote_available = True

        # Write blocks individually (different sessions)
        await storage.write_block(create_test_block("session-a", sequence=1))
        await storage.write_block(create_test_block("session-b", sequence=1))
        await storage.write_block(create_test_block("session-a", sequence=2))

        # Only session-a blocks should be queued (2 blocks)
        assert storage._sync_queue.qsize() == 2

        # Verify both are from session-a
        queued = []
        while not storage._sync_queue.empty():
            queued.append(await storage._sync_queue.get())

        assert all(b.session_id == "session-a" for b in queued)
