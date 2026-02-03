"""
Tests for CosmosFileStorage.

These tests verify the CosmosFileStorage class works correctly with Azure Cosmos DB.
Requires AMPLIFIER_COSMOS_* environment variables to be set.
"""

import os
import uuid
from datetime import UTC, datetime

import pytest

from amplifier_session_storage import CosmosFileConfig, CosmosFileStorage

# Skip all tests if Cosmos environment not configured
pytestmark = pytest.mark.skipif(
    not os.environ.get("AMPLIFIER_COSMOS_ENDPOINT"),
    reason="AMPLIFIER_COSMOS_ENDPOINT not set",
)


@pytest.fixture
def config() -> CosmosFileConfig:
    """Create config from environment."""
    return CosmosFileConfig.from_env()


@pytest.fixture
def test_ids() -> dict[str, str]:
    """Generate unique test IDs."""
    unique = uuid.uuid4().hex[:8]
    return {
        "user_id": f"test-user-{unique}",
        "host_id": f"test-host-{unique}",
        "project_slug": f"test-project-{unique}",
        "session_id": f"test-session-{unique}",
    }


@pytest.fixture
async def storage(config: CosmosFileConfig):  # type: ignore[misc]
    """Create and initialize storage."""
    storage = CosmosFileStorage(config)
    await storage.initialize()
    yield storage
    await storage.close()


class TestCosmosFileStorage:
    """Test CosmosFileStorage operations."""

    @pytest.mark.asyncio
    async def test_connection(self, storage: CosmosFileStorage) -> None:
        """Test connection verification."""
        result = await storage.verify_connection()
        assert result is True

    @pytest.mark.asyncio
    async def test_upsert_session_metadata(
        self, storage: CosmosFileStorage, test_ids: dict[str, str]
    ) -> None:
        """Test upserting session metadata with host_id."""
        metadata = {
            "session_id": test_ids["session_id"],
            "project_slug": test_ids["project_slug"],
            "bundle": "test-bundle",
            "created": datetime.now(UTC).isoformat(),
            "turn_count": 5,
        }

        # Upsert metadata
        await storage.upsert_session_metadata(
            user_id=test_ids["user_id"],
            host_id=test_ids["host_id"],
            metadata=metadata,
        )

        # Verify we can retrieve it
        result = await storage.get_session_metadata(
            user_id=test_ids["user_id"],
            session_id=test_ids["session_id"],
        )

        assert result is not None
        assert result["session_id"] == test_ids["session_id"]
        assert result["user_id"] == test_ids["user_id"]
        assert result["host_id"] == test_ids["host_id"]
        assert result["bundle"] == "test-bundle"
        assert result["turn_count"] == 5

        # Cleanup
        await storage.delete_session(
            user_id=test_ids["user_id"],
            project_slug=test_ids["project_slug"],
            session_id=test_ids["session_id"],
        )

    @pytest.mark.asyncio
    async def test_sync_transcript_lines(
        self, storage: CosmosFileStorage, test_ids: dict[str, str]
    ) -> None:
        """Test syncing transcript lines with host_id."""
        # First create session metadata
        metadata = {
            "session_id": test_ids["session_id"],
            "project_slug": test_ids["project_slug"],
        }
        await storage.upsert_session_metadata(
            user_id=test_ids["user_id"],
            host_id=test_ids["host_id"],
            metadata=metadata,
        )

        # Sync transcript lines
        lines = [
            {
                "role": "user",
                "content": "Hello",
                "turn": 0,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            {
                "role": "assistant",
                "content": "Hi there!",
                "turn": 0,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            {
                "role": "user",
                "content": "How are you?",
                "turn": 1,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        ]

        synced = await storage.sync_transcript_lines(
            user_id=test_ids["user_id"],
            host_id=test_ids["host_id"],
            project_slug=test_ids["project_slug"],
            session_id=test_ids["session_id"],
            lines=lines,
            start_sequence=0,
        )

        assert synced == 3

        # Verify count
        count = await storage.get_transcript_count(
            user_id=test_ids["user_id"],
            project_slug=test_ids["project_slug"],
            session_id=test_ids["session_id"],
        )
        assert count == 3

        # Verify we can retrieve them
        retrieved = await storage.get_transcript_lines(
            user_id=test_ids["user_id"],
            project_slug=test_ids["project_slug"],
            session_id=test_ids["session_id"],
        )
        assert len(retrieved) == 3
        # Verify host_id is present
        assert all(r["host_id"] == test_ids["host_id"] for r in retrieved)

        # Cleanup
        await storage.delete_session(
            user_id=test_ids["user_id"],
            project_slug=test_ids["project_slug"],
            session_id=test_ids["session_id"],
        )

    @pytest.mark.asyncio
    async def test_sync_event_lines(
        self, storage: CosmosFileStorage, test_ids: dict[str, str]
    ) -> None:
        """Test syncing event lines with host_id."""
        # First create session metadata
        metadata = {
            "session_id": test_ids["session_id"],
            "project_slug": test_ids["project_slug"],
        }
        await storage.upsert_session_metadata(
            user_id=test_ids["user_id"],
            host_id=test_ids["host_id"],
            metadata=metadata,
        )

        # Sync event lines
        lines = [
            {
                "event": "session.start",
                "ts": datetime.now(UTC).isoformat(),
                "lvl": "info",
                "turn": 0,
            },
            {
                "event": "llm.request",
                "ts": datetime.now(UTC).isoformat(),
                "lvl": "debug",
                "turn": 0,
                "data": {"model": "test"},
            },
            {
                "event": "llm.response",
                "ts": datetime.now(UTC).isoformat(),
                "lvl": "debug",
                "turn": 0,
            },
        ]

        synced = await storage.sync_event_lines(
            user_id=test_ids["user_id"],
            host_id=test_ids["host_id"],
            project_slug=test_ids["project_slug"],
            session_id=test_ids["session_id"],
            lines=lines,
            start_sequence=0,
        )

        assert synced == 3

        # Verify count
        count = await storage.get_event_count(
            user_id=test_ids["user_id"],
            project_slug=test_ids["project_slug"],
            session_id=test_ids["session_id"],
        )
        assert count == 3

        # Cleanup
        await storage.delete_session(
            user_id=test_ids["user_id"],
            project_slug=test_ids["project_slug"],
            session_id=test_ids["session_id"],
        )

    @pytest.mark.asyncio
    async def test_incremental_sync(
        self, storage: CosmosFileStorage, test_ids: dict[str, str]
    ) -> None:
        """Test incremental sync - only new lines should be synced."""
        # Create session
        metadata = {
            "session_id": test_ids["session_id"],
            "project_slug": test_ids["project_slug"],
        }
        await storage.upsert_session_metadata(
            user_id=test_ids["user_id"],
            host_id=test_ids["host_id"],
            metadata=metadata,
        )

        # Sync first batch
        lines_batch1 = [
            {"role": "user", "content": "Hello", "turn": 0},
            {"role": "assistant", "content": "Hi!", "turn": 0},
        ]
        await storage.sync_transcript_lines(
            user_id=test_ids["user_id"],
            host_id=test_ids["host_id"],
            project_slug=test_ids["project_slug"],
            session_id=test_ids["session_id"],
            lines=lines_batch1,
            start_sequence=0,
        )

        # Check count after first batch
        count1 = await storage.get_transcript_count(
            user_id=test_ids["user_id"],
            project_slug=test_ids["project_slug"],
            session_id=test_ids["session_id"],
        )
        assert count1 == 2

        # Sync second batch (incremental)
        lines_batch2 = [
            {"role": "user", "content": "What's up?", "turn": 1},
        ]
        await storage.sync_transcript_lines(
            user_id=test_ids["user_id"],
            host_id=test_ids["host_id"],
            project_slug=test_ids["project_slug"],
            session_id=test_ids["session_id"],
            lines=lines_batch2,
            start_sequence=2,  # Continue from where we left off
        )

        # Check count after second batch
        count2 = await storage.get_transcript_count(
            user_id=test_ids["user_id"],
            project_slug=test_ids["project_slug"],
            session_id=test_ids["session_id"],
        )
        assert count2 == 3

        # Cleanup
        await storage.delete_session(
            user_id=test_ids["user_id"],
            project_slug=test_ids["project_slug"],
            session_id=test_ids["session_id"],
        )

    @pytest.mark.asyncio
    async def test_delete_session(
        self, storage: CosmosFileStorage, test_ids: dict[str, str]
    ) -> None:
        """Test deleting a session and all its data."""
        # Create session with data
        metadata = {
            "session_id": test_ids["session_id"],
            "project_slug": test_ids["project_slug"],
        }
        await storage.upsert_session_metadata(
            user_id=test_ids["user_id"],
            host_id=test_ids["host_id"],
            metadata=metadata,
        )

        await storage.sync_transcript_lines(
            user_id=test_ids["user_id"],
            host_id=test_ids["host_id"],
            project_slug=test_ids["project_slug"],
            session_id=test_ids["session_id"],
            lines=[{"role": "user", "content": "Test", "turn": 0}],
        )

        await storage.sync_event_lines(
            user_id=test_ids["user_id"],
            host_id=test_ids["host_id"],
            project_slug=test_ids["project_slug"],
            session_id=test_ids["session_id"],
            lines=[{"event": "test", "ts": datetime.now(UTC).isoformat()}],
        )

        # Delete session
        deleted = await storage.delete_session(
            user_id=test_ids["user_id"],
            project_slug=test_ids["project_slug"],
            session_id=test_ids["session_id"],
        )
        assert deleted is True

        # Verify metadata is gone
        result = await storage.get_session_metadata(
            user_id=test_ids["user_id"],
            session_id=test_ids["session_id"],
        )
        assert result is None

        # Verify transcript is gone
        count = await storage.get_transcript_count(
            user_id=test_ids["user_id"],
            project_slug=test_ids["project_slug"],
            session_id=test_ids["session_id"],
        )
        assert count == 0

        # Verify events are gone
        event_count = await storage.get_event_count(
            user_id=test_ids["user_id"],
            project_slug=test_ids["project_slug"],
            session_id=test_ids["session_id"],
        )
        assert event_count == 0


class TestCosmosFileConfig:
    """Test CosmosFileConfig."""

    def test_from_env(self) -> None:
        """Test creating config from environment."""
        config = CosmosFileConfig.from_env()
        assert config.endpoint is not None
        assert config.database_name is not None

    def test_partition_key_format(self) -> None:
        """Test partition key format."""
        pk = CosmosFileStorage.make_partition_key("user1", "project1", "session1")
        assert pk == "user1|project1|session1"
