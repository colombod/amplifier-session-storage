"""Tests for storage module."""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from amplifier_session_storage.blocks import BlockWriter
from amplifier_session_storage.storage import LocalBlockStorage, StorageConfig
from amplifier_session_storage.storage.base import AccessDeniedError, SessionNotFoundError


class TestStorageConfig:
    """Tests for StorageConfig."""

    def test_minimal_config(self) -> None:
        """Test creating config with minimal fields."""
        config = StorageConfig(user_id="user-123")

        assert config.user_id == "user-123"
        assert config.org_id is None
        assert config.enable_sync is False
        assert config.cosmos_database == "amplifier-db"

    def test_full_config(self) -> None:
        """Test creating config with all fields."""
        config = StorageConfig(
            user_id="user-123",
            org_id="org-456",
            enable_sync=True,
            cosmos_endpoint="https://example.cosmos.azure.com",
            cosmos_key="secret-key",
            cosmos_database="mydb",
            cosmos_container="mycontainer",
            local_path="/tmp/sessions",
        )

        assert config.user_id == "user-123"
        assert config.org_id == "org-456"
        assert config.enable_sync is True
        assert config.cosmos_endpoint == "https://example.cosmos.azure.com"
        assert config.local_path == "/tmp/sessions"


class TestLocalBlockStorage:
    """Tests for LocalBlockStorage."""

    @pytest.fixture
    async def temp_dir(self) -> AsyncIterator[Path]:
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    async def storage(self, temp_dir: Path) -> AsyncIterator[LocalBlockStorage]:
        """Create a local storage instance for tests."""
        config = StorageConfig(
            user_id="test-user",
            local_path=str(temp_dir),
        )
        storage = LocalBlockStorage(config)
        yield storage
        await storage.close()

    @pytest.fixture
    def writer(self) -> BlockWriter:
        """Create a block writer for tests."""
        return BlockWriter(
            session_id="sess-test",
            user_id="test-user",
            device_id="dev-test",
        )

    async def test_write_and_read_blocks(
        self, storage: LocalBlockStorage, writer: BlockWriter
    ) -> None:
        """Test writing and reading blocks."""
        # Create blocks
        blocks = [
            writer.create_session(project_slug="test-project", name="Test Session"),
            writer.add_message(role="user", content="Hello", turn=1),
            writer.add_message(role="assistant", content="Hi there!", turn=1),
        ]

        # Write blocks
        await storage.write_blocks(blocks)

        # Read blocks back
        read_blocks = await storage.read_blocks("sess-test")

        assert len(read_blocks) == 3
        assert read_blocks[0].data["project_slug"] == "test-project"
        assert read_blocks[1].data["content"] == "Hello"
        assert read_blocks[2].data["content"] == "Hi there!"

    async def test_write_single_block(
        self, storage: LocalBlockStorage, writer: BlockWriter
    ) -> None:
        """Test writing a single block."""
        block = writer.create_session(project_slug="test-project")

        await storage.write_block(block)

        read_blocks = await storage.read_blocks("sess-test")
        assert len(read_blocks) == 1

    async def test_read_blocks_since_sequence(
        self, storage: LocalBlockStorage, writer: BlockWriter
    ) -> None:
        """Test reading blocks since a sequence."""
        blocks = [
            writer.create_session(project_slug="test-project"),  # seq 1
            writer.add_message(role="user", content="Msg 1", turn=1),  # seq 2
            writer.add_message(role="assistant", content="Msg 2", turn=1),  # seq 3
            writer.add_message(role="user", content="Msg 3", turn=2),  # seq 4
        ]
        await storage.write_blocks(blocks)

        # Read only blocks after sequence 2
        read_blocks = await storage.read_blocks("sess-test", since_sequence=2)

        assert len(read_blocks) == 2
        assert read_blocks[0].sequence == 3
        assert read_blocks[1].sequence == 4

    async def test_read_blocks_with_limit(
        self, storage: LocalBlockStorage, writer: BlockWriter
    ) -> None:
        """Test reading blocks with limit."""
        blocks = [
            writer.create_session(project_slug="test-project"),
            writer.add_message(role="user", content="Msg 1", turn=1),
            writer.add_message(role="assistant", content="Msg 2", turn=1),
            writer.add_message(role="user", content="Msg 3", turn=2),
        ]
        await storage.write_blocks(blocks)

        read_blocks = await storage.read_blocks("sess-test", limit=2)

        assert len(read_blocks) == 2

    async def test_get_latest_sequence(
        self, storage: LocalBlockStorage, writer: BlockWriter
    ) -> None:
        """Test getting latest sequence number."""
        # No blocks yet
        seq = await storage.get_latest_sequence("sess-test")
        assert seq == 0

        # Write some blocks
        blocks = [
            writer.create_session(project_slug="test-project"),
            writer.add_message(role="user", content="Hello", turn=1),
        ]
        await storage.write_blocks(blocks)

        seq = await storage.get_latest_sequence("sess-test")
        assert seq == 2

    async def test_list_sessions(self, storage: LocalBlockStorage) -> None:
        """Test listing sessions."""
        # Create multiple sessions
        for i in range(3):
            writer = BlockWriter(
                session_id=f"sess-{i}",
                user_id="test-user",
                device_id="dev-test",
            )
            block = writer.create_session(
                project_slug="test-project",
                name=f"Session {i}",
            )
            await storage.write_block(block)

        sessions = await storage.list_sessions()

        assert len(sessions) == 3
        session_ids = [s["session_id"] for s in sessions]
        assert "sess-0" in session_ids
        assert "sess-1" in session_ids
        assert "sess-2" in session_ids

    async def test_list_sessions_by_project(self, storage: LocalBlockStorage) -> None:
        """Test listing sessions filtered by project."""
        # Create sessions in different projects
        for project in ["project-a", "project-b"]:
            writer = BlockWriter(
                session_id=f"sess-{project}",
                user_id="test-user",
                device_id="dev-test",
            )
            block = writer.create_session(project_slug=project)
            await storage.write_block(block)

        # List only project-a sessions
        sessions = await storage.list_sessions(project_slug="project-a")

        assert len(sessions) == 1
        assert sessions[0]["project_slug"] == "project-a"

    async def test_delete_session(self, storage: LocalBlockStorage, writer: BlockWriter) -> None:
        """Test deleting a session."""
        # Create session
        blocks = [
            writer.create_session(project_slug="test-project"),
            writer.add_message(role="user", content="Hello", turn=1),
        ]
        await storage.write_blocks(blocks)

        # Verify it exists
        read_blocks = await storage.read_blocks("sess-test")
        assert len(read_blocks) == 2

        # Delete it
        await storage.delete_session("sess-test")

        # Verify it's gone
        read_blocks = await storage.read_blocks("sess-test")
        assert len(read_blocks) == 0

    async def test_delete_nonexistent_session(self, storage: LocalBlockStorage) -> None:
        """Test deleting a session that doesn't exist."""
        with pytest.raises(SessionNotFoundError):
            await storage.delete_session("nonexistent")

    async def test_access_denied_for_other_user(self, temp_dir: Path) -> None:
        """Test that writing blocks for another user is denied."""
        config = StorageConfig(
            user_id="user-a",
            local_path=str(temp_dir),
        )
        storage = LocalBlockStorage(config)

        # Try to write blocks owned by a different user
        writer = BlockWriter(
            session_id="sess-test",
            user_id="user-b",  # Different user!
            device_id="dev-test",
        )
        block = writer.create_session(project_slug="test")

        with pytest.raises(AccessDeniedError):
            await storage.write_block(block)

        await storage.close()

    async def test_read_empty_session(self, storage: LocalBlockStorage) -> None:
        """Test reading blocks from a non-existent session."""
        blocks = await storage.read_blocks("nonexistent-session")
        assert blocks == []

    async def test_metadata_cache_created(
        self, storage: LocalBlockStorage, writer: BlockWriter, temp_dir: Path
    ) -> None:
        """Test that metadata cache file is created."""
        block = writer.create_session(
            project_slug="test-project",
            name="Test Session",
        )
        await storage.write_block(block)

        # Check metadata file exists
        metadata_path = temp_dir / "test-user" / "test-project" / "sess-test" / "metadata.json"
        assert metadata_path.exists()

    async def test_incremental_writes(
        self, storage: LocalBlockStorage, writer: BlockWriter
    ) -> None:
        """Test that blocks can be written incrementally."""
        # Write first block
        block1 = writer.create_session(project_slug="test-project")
        await storage.write_block(block1)

        # Write more blocks later
        block2 = writer.add_message(role="user", content="Hello", turn=1)
        await storage.write_block(block2)

        block3 = writer.add_message(role="assistant", content="Hi!", turn=1)
        await storage.write_block(block3)

        # Read all blocks
        blocks = await storage.read_blocks("sess-test")
        assert len(blocks) == 3
        assert blocks[0].sequence == 1
        assert blocks[1].sequence == 2
        assert blocks[2].sequence == 3

    async def test_blocks_sorted_by_sequence(self, storage: LocalBlockStorage) -> None:
        """Test that read blocks are always sorted by sequence."""
        # Write blocks with non-sequential sequences (simulating sync)
        writer = BlockWriter(
            session_id="sess-test",
            user_id="test-user",
            device_id="dev-test",
        )

        # Write blocks
        block1 = writer.create_session(project_slug="test-project")
        block2 = writer.add_message(role="user", content="First", turn=1)
        block3 = writer.add_message(role="assistant", content="Second", turn=1)

        # Write in shuffled order
        await storage.write_block(block3)
        await storage.write_block(block1)
        await storage.write_block(block2)

        # Read should be sorted
        blocks = await storage.read_blocks("sess-test")
        sequences = [b.sequence for b in blocks]
        assert sequences == sorted(sequences)
