"""
Integration tests for Cosmos DB block storage.

These tests require a real Cosmos DB connection and are marked as integration tests.
Run with: pytest -m integration tests/test_cosmos_integration.py

Environment variables required:
- COSMOS_ENDPOINT: Cosmos DB endpoint URL

Authentication (one of):
- COSMOS_KEY: Cosmos DB account key (for key-based auth)
- Azure identity: DefaultAzureCredential (for identity-based auth)

Optional:
- COSMOS_DATABASE: Database name (default: amplifier-db)
- COSMOS_CONTAINER: Container name (default: items)
"""

import os
import uuid

import pytest

from amplifier_session_storage import (
    BlockWriter,
    CosmosAuthMethod,
    CosmosBlockStorage,
    StorageConfig,
)


def _cosmos_available() -> bool:
    """Check if Cosmos DB is available (either key or identity auth)."""
    if not os.environ.get("COSMOS_ENDPOINT"):
        return False
    # If key is set, use that
    if os.environ.get("COSMOS_KEY"):
        return True
    # Otherwise try identity-based auth
    try:
        from azure.identity import DefaultAzureCredential

        DefaultAzureCredential()
        return True
    except Exception:
        return False


# Skip all tests in this module if Cosmos is not configured
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _cosmos_available(),
        reason="COSMOS_ENDPOINT not set or no valid auth available",
    ),
]


@pytest.fixture
def test_user_id():
    """Generate unique user ID for test isolation."""
    return f"test-user-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_session_id():
    """Generate unique session ID."""
    return f"test-session-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def storage_config(test_user_id):
    """Create storage configuration for tests."""
    key = os.environ.get("COSMOS_KEY")
    auth_method = CosmosAuthMethod.KEY if key else CosmosAuthMethod.DEFAULT_CREDENTIAL

    return StorageConfig(
        user_id=test_user_id,
        cosmos_endpoint=os.environ["COSMOS_ENDPOINT"],
        cosmos_key=key,
        cosmos_auth_method=auth_method,
        cosmos_database=os.environ.get("COSMOS_DATABASE", "amplifier-db"),
        cosmos_container=os.environ.get("COSMOS_CONTAINER", "items"),
    )


@pytest.fixture
def storage(storage_config):
    """Create CosmosBlockStorage instance."""
    return CosmosBlockStorage(storage_config)


@pytest.fixture
def block_writer(test_session_id, test_user_id):
    """Create BlockWriter for creating test blocks."""
    return BlockWriter(
        session_id=test_session_id,
        user_id=test_user_id,
        device_id="test-device",
    )


class TestCosmosBlockCRUD:
    """Tests for block CRUD operations."""

    @pytest.mark.asyncio
    async def test_write_and_read_session_block(self, storage, block_writer, test_session_id):
        """Write a session block and read it back."""
        # Create session
        block = block_writer.create_session(
            project_slug="test-project",
            name="Test Session",
            description="Integration test session",
        )

        # Write
        await storage.write_block(block)

        # Read back
        blocks = await storage.read_blocks(test_session_id)
        assert len(blocks) >= 1

        # Find our block
        session_block = next((b for b in blocks if b.block_type.value == "session_created"), None)
        assert session_block is not None
        assert session_block.session_id == test_session_id

        # Cleanup
        await storage.delete_session(test_session_id)

    @pytest.mark.asyncio
    async def test_write_message_block(self, storage, block_writer, test_session_id):
        """Write message blocks to a session."""
        # Create session first
        session_block = block_writer.create_session(
            project_slug="test-project",
            name="Message Test",
        )
        await storage.write_block(session_block)

        # Add messages
        msg1 = block_writer.add_message(role="user", content="Hello!", turn=1)
        await storage.write_block(msg1)

        msg2 = block_writer.add_message(role="assistant", content="Hi there!", turn=1)
        await storage.write_block(msg2)

        # Read back
        blocks = await storage.read_blocks(test_session_id)
        message_blocks = [b for b in blocks if b.block_type.value == "message"]
        assert len(message_blocks) == 2

        # Cleanup
        await storage.delete_session(test_session_id)

    @pytest.mark.asyncio
    async def test_list_sessions(self, storage, block_writer, test_session_id):
        """List sessions returns created session."""
        # Create session
        block = block_writer.create_session(
            project_slug="test-project",
            name="List Test Session",
        )
        await storage.write_block(block)

        # List sessions
        sessions = await storage.list_sessions()
        session_ids = [s["session_id"] for s in sessions]
        assert test_session_id in session_ids

        # Cleanup
        await storage.delete_session(test_session_id)

    @pytest.mark.asyncio
    async def test_delete_session(self, storage, block_writer, test_session_id):
        """Delete session removes all blocks."""
        # Create session with message
        session_block = block_writer.create_session(
            project_slug="test-project",
            name="Delete Test",
        )
        await storage.write_block(session_block)

        msg = block_writer.add_message(role="user", content="Test", turn=1)
        await storage.write_block(msg)

        # Verify blocks exist
        blocks = await storage.read_blocks(test_session_id)
        assert len(blocks) >= 2

        # Delete
        await storage.delete_session(test_session_id)

        # Verify deleted
        blocks_after = await storage.read_blocks(test_session_id)
        assert len(blocks_after) == 0

    @pytest.mark.asyncio
    async def test_get_latest_sequence(self, storage, block_writer, test_session_id):
        """Get latest sequence number for a session."""
        # Create session
        block = block_writer.create_session(project_slug="test-project")
        await storage.write_block(block)

        # Add messages
        for i in range(3):
            msg = block_writer.add_message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
                turn=i // 2 + 1,
            )
            await storage.write_block(msg)

        # Check sequence
        seq = await storage.get_latest_sequence(test_session_id)
        assert seq >= 4  # session + 3 messages

        # Cleanup
        await storage.delete_session(test_session_id)


class TestCosmosUserIsolation:
    """Tests for user data isolation."""

    @pytest.mark.asyncio
    async def test_user_cannot_read_other_users_session(self, storage_config):
        """User A cannot read User B's session."""
        user_a_id = f"user-a-{uuid.uuid4().hex[:8]}"
        user_b_id = f"user-b-{uuid.uuid4().hex[:8]}"
        session_id = f"isolated-session-{uuid.uuid4().hex[:8]}"

        # User A creates a session
        config_a = StorageConfig(
            user_id=user_a_id,
            cosmos_endpoint=storage_config.cosmos_endpoint,
            cosmos_key=storage_config.cosmos_key,
            cosmos_auth_method=storage_config.cosmos_auth_method,
            cosmos_database=storage_config.cosmos_database,
            cosmos_container=storage_config.cosmos_container,
        )
        storage_a = CosmosBlockStorage(config_a)

        writer_a = BlockWriter(session_id=session_id, user_id=user_a_id, device_id="device-a")
        block = writer_a.create_session(project_slug="private", name="User A Session")
        await storage_a.write_block(block)

        # User B tries to read
        config_b = StorageConfig(
            user_id=user_b_id,
            cosmos_endpoint=storage_config.cosmos_endpoint,
            cosmos_key=storage_config.cosmos_key,
            cosmos_auth_method=storage_config.cosmos_auth_method,
            cosmos_database=storage_config.cosmos_database,
            cosmos_container=storage_config.cosmos_container,
        )
        storage_b = CosmosBlockStorage(config_b)

        # User B should not see User A's session
        blocks = await storage_b.read_blocks(session_id)
        assert len(blocks) == 0  # Empty because different user

        # Cleanup
        await storage_a.delete_session(session_id)

    @pytest.mark.asyncio
    async def test_list_sessions_only_returns_own(self, storage_config):
        """List sessions only returns user's own sessions."""
        user_a_id = f"user-a-{uuid.uuid4().hex[:8]}"
        user_b_id = f"user-b-{uuid.uuid4().hex[:8]}"
        session_a = f"session-a-{uuid.uuid4().hex[:8]}"
        session_b = f"session-b-{uuid.uuid4().hex[:8]}"

        # Create storage for each user
        config_a = StorageConfig(
            user_id=user_a_id,
            cosmos_endpoint=storage_config.cosmos_endpoint,
            cosmos_key=storage_config.cosmos_key,
            cosmos_auth_method=storage_config.cosmos_auth_method,
            cosmos_database=storage_config.cosmos_database,
            cosmos_container=storage_config.cosmos_container,
        )
        storage_a = CosmosBlockStorage(config_a)

        config_b = StorageConfig(
            user_id=user_b_id,
            cosmos_endpoint=storage_config.cosmos_endpoint,
            cosmos_key=storage_config.cosmos_key,
            cosmos_auth_method=storage_config.cosmos_auth_method,
            cosmos_database=storage_config.cosmos_database,
            cosmos_container=storage_config.cosmos_container,
        )
        storage_b = CosmosBlockStorage(config_b)

        # Each user creates a session
        writer_a = BlockWriter(session_id=session_a, user_id=user_a_id, device_id="device-a")
        await storage_a.write_block(
            writer_a.create_session(project_slug="test", name="A's Session")
        )

        writer_b = BlockWriter(session_id=session_b, user_id=user_b_id, device_id="device-b")
        await storage_b.write_block(
            writer_b.create_session(project_slug="test", name="B's Session")
        )

        # User A only sees their session
        sessions_a = await storage_a.list_sessions()
        session_ids_a = [s["session_id"] for s in sessions_a]
        assert session_a in session_ids_a
        assert session_b not in session_ids_a

        # User B only sees their session
        sessions_b = await storage_b.list_sessions()
        session_ids_b = [s["session_id"] for s in sessions_b]
        assert session_b in session_ids_b
        assert session_a not in session_ids_b

        # Cleanup
        await storage_a.delete_session(session_a)
        await storage_b.delete_session(session_b)


class TestCosmosEventBlocks:
    """Tests for event block handling."""

    @pytest.mark.asyncio
    async def test_write_event_block(self, storage, block_writer, test_session_id):
        """Write event blocks to a session."""
        # Create session
        await storage.write_block(block_writer.create_session(project_slug="test-project"))

        # Add event
        event_blocks = block_writer.add_event(
            event_id="evt-001",
            event_type="llm:request",
            turn=1,
            data={"model": "claude-sonnet", "message_count": 5},
        )
        for block in event_blocks:
            await storage.write_block(block)

        # Read back
        blocks = await storage.read_blocks(test_session_id)
        event_blocks = [b for b in blocks if b.block_type.value == "event"]
        assert len(event_blocks) == 1

        # Cleanup
        await storage.delete_session(test_session_id)

    @pytest.mark.asyncio
    async def test_large_event_chunking(self, storage, block_writer, test_session_id):
        """Large events should be automatically chunked."""
        # Create session
        await storage.write_block(block_writer.create_session(project_slug="test-project"))

        # Create a large event (>100KB to trigger chunking)
        large_content = "x" * (150 * 1024)  # 150KB
        event_blocks = block_writer.add_event(
            event_id="evt-large-001",
            event_type="tool:result",
            turn=1,
            data={"result": large_content},
        )
        for block in event_blocks:
            await storage.write_block(block)

        # Read back - should have chunked blocks
        blocks = await storage.read_blocks(test_session_id)

        # At least the session block + event (possibly chunked)
        assert len(blocks) >= 2

        # Cleanup
        await storage.delete_session(test_session_id)


class TestCosmosRewind:
    """Tests for session rewind functionality."""

    @pytest.mark.asyncio
    async def test_rewind_to_sequence(self, storage, block_writer, test_session_id):
        """Rewind session removes blocks after specified sequence."""
        # Create session with multiple messages
        await storage.write_block(block_writer.create_session(project_slug="test-project"))

        # Add 5 messages
        for i in range(5):
            msg = block_writer.add_message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
                turn=i // 2 + 1,
            )
            await storage.write_block(msg)

        # Get blocks before rewind
        blocks_before = await storage.read_blocks(test_session_id)
        assert len(blocks_before) == 6  # session + 5 messages

        # Rewind to sequence 3 (session + 2 messages)
        await storage.rewind_to_sequence(test_session_id, 3)

        # Check blocks after rewind
        blocks_after = await storage.read_blocks(test_session_id)
        assert len(blocks_after) == 3

        # Cleanup
        await storage.delete_session(test_session_id)


class TestCosmosSchemaMapping:
    """Tests that verify schema mapping between local and Cosmos."""

    @pytest.mark.asyncio
    async def test_session_block_has_required_fields(self, storage, block_writer, test_session_id):
        """Session blocks have all required schema fields."""
        block = block_writer.create_session(
            project_slug="test-project",
            name="Schema Test",
            description="Testing schema fields",
            bundle="bundle:foundation",
            model="claude-sonnet",
        )
        await storage.write_block(block)

        # Read back
        blocks = await storage.read_blocks(test_session_id)
        session_block = blocks[0]

        # Verify required fields exist
        assert session_block.session_id == test_session_id
        assert session_block.user_id == block_writer.user_id
        assert session_block.device_id == block_writer.device_id
        assert session_block.sequence >= 1
        assert session_block.timestamp is not None
        assert session_block.block_type is not None

        # Cleanup
        await storage.delete_session(test_session_id)

    @pytest.mark.asyncio
    async def test_message_block_has_required_fields(self, storage, block_writer, test_session_id):
        """Message blocks have all required schema fields."""
        await storage.write_block(block_writer.create_session(project_slug="test-project"))

        msg = block_writer.add_message(
            role="user",
            content="Test message content",
            turn=1,
        )
        await storage.write_block(msg)

        # Read back
        blocks = await storage.read_blocks(test_session_id)
        msg_block = next(b for b in blocks if b.block_type.value == "message")

        # Verify message data
        assert msg_block.data is not None
        data = msg_block.data
        assert data.get("role") == "user"
        assert data.get("content") == "Test message content"
        assert data.get("turn") == 1

        # Cleanup
        await storage.delete_session(test_session_id)
