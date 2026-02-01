"""Tests for blocks module."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from amplifier_session_storage.blocks import (
    BlockType,
    BlockWriter,
    EventData,
    MessageData,
    RewindData,
    SequenceAllocator,
    SessionBlock,
    SessionCreatedData,
    SessionStateReader,
    SessionUpdatedData,
)
from amplifier_session_storage.blocks.sequence import SimpleSequenceAllocator


class TestSessionBlock:
    """Tests for SessionBlock dataclass."""

    def test_create_block(self) -> None:
        """Test creating a basic block."""
        now = datetime.now(UTC)
        block = SessionBlock(
            block_id="blk-123",
            session_id="sess-456",
            user_id="user-789",
            sequence=1,
            timestamp=now,
            device_id="dev-001",
            block_type=BlockType.MESSAGE,
            data={"role": "user", "content": "Hello"},
        )

        assert block.block_id == "blk-123"
        assert block.session_id == "sess-456"
        assert block.sequence == 1
        assert block.block_type == BlockType.MESSAGE
        assert block.checksum is not None
        assert block.size_bytes > 0

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        now = datetime.now(UTC)
        block = SessionBlock(
            block_id="blk-123",
            session_id="sess-456",
            user_id="user-789",
            sequence=1,
            timestamp=now,
            device_id="dev-001",
            block_type=BlockType.MESSAGE,
            data={"role": "user", "content": "Hello"},
        )

        data = block.to_dict()

        assert data["id"] == "blk-123"
        assert data["block_id"] == "blk-123"
        assert data["session_id"] == "sess-456"
        assert data["user_id_session_id"] == "user-789_sess-456"
        assert data["sequence"] == 1
        assert data["block_type"] == "message"

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        now = datetime.now(UTC)
        data = {
            "id": "blk-123",
            "block_id": "blk-123",
            "session_id": "sess-456",
            "user_id": "user-789",
            "sequence": 1,
            "timestamp": now.isoformat(),
            "device_id": "dev-001",
            "block_type": "message",
            "data": {"role": "user", "content": "Hello"},
            "checksum": "abc123",
            "size_bytes": 100,
        }

        block = SessionBlock.from_dict(data)

        assert block.block_id == "blk-123"
        assert block.sequence == 1
        assert block.block_type == BlockType.MESSAGE

    def test_checksum_computed(self) -> None:
        """Test that checksum is computed on creation."""
        now = datetime.now(UTC)
        block1 = SessionBlock(
            block_id="blk-1",
            session_id="sess-1",
            user_id="user-1",
            sequence=1,
            timestamp=now,
            device_id="dev-1",
            block_type=BlockType.MESSAGE,
            data={"content": "Hello"},
        )

        block2 = SessionBlock(
            block_id="blk-2",
            session_id="sess-1",
            user_id="user-1",
            sequence=2,
            timestamp=now,
            device_id="dev-1",
            block_type=BlockType.MESSAGE,
            data={"content": "Hello"},
        )

        # Same data = same checksum
        assert block1.checksum == block2.checksum

        block3 = SessionBlock(
            block_id="blk-3",
            session_id="sess-1",
            user_id="user-1",
            sequence=3,
            timestamp=now,
            device_id="dev-1",
            block_type=BlockType.MESSAGE,
            data={"content": "Different"},
        )

        # Different data = different checksum
        assert block1.checksum != block3.checksum


class TestDataTypes:
    """Tests for data type classes."""

    def test_session_created_data(self) -> None:
        """Test SessionCreatedData roundtrip."""
        original = SessionCreatedData(
            project_slug="my-project",
            name="Test Session",
            visibility="team",
            org_id="org-123",
            team_ids=["team-a", "team-b"],
            tags=["important"],
        )

        data = original.to_dict()
        restored = SessionCreatedData.from_dict(data)

        assert restored.project_slug == original.project_slug
        assert restored.name == original.name
        assert restored.visibility == original.visibility
        assert restored.org_id == original.org_id
        assert restored.team_ids == original.team_ids
        assert restored.tags == original.tags

    def test_session_updated_data(self) -> None:
        """Test SessionUpdatedData only includes set fields."""
        data = SessionUpdatedData(name="New Name", visibility="public")

        serialized = data.to_dict()

        assert "name" in serialized
        assert "visibility" in serialized
        assert "description" not in serialized
        assert "tags" not in serialized

    def test_message_data(self) -> None:
        """Test MessageData roundtrip."""
        original = MessageData(
            role="assistant",
            content="Hello, how can I help?",
            turn=1,
            tool_calls=[{"id": "call_1", "type": "function"}],
        )

        data = original.to_dict()
        restored = MessageData.from_dict(data)

        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.turn == original.turn
        assert restored.tool_calls == original.tool_calls

    def test_event_data_small(self) -> None:
        """Test EventData for small events."""
        event = EventData(
            event_id="evt-123",
            event_type="llm:response",
            turn=1,
            summary={"status": "success"},
            inline_data={"response": "Hello"},
            has_continuation=False,
        )

        data = event.to_dict()
        restored = EventData.from_dict(data)

        assert restored.event_id == event.event_id
        assert restored.inline_data == event.inline_data
        assert restored.has_continuation is False

    def test_event_data_large(self) -> None:
        """Test EventData for large events (with continuation)."""
        event = EventData(
            event_id="evt-456",
            event_type="llm:response",
            turn=1,
            summary={"status": "success", "_data_size": 500000},
            inline_data=None,
            has_continuation=True,
            total_chunks=3,
            total_size_bytes=500000,
        )

        data = event.to_dict()
        restored = EventData.from_dict(data)

        assert restored.has_continuation is True
        assert restored.total_chunks == 3
        assert restored.inline_data is None

    def test_rewind_data(self) -> None:
        """Test RewindData roundtrip."""
        original = RewindData(
            target_sequence=10,
            reason="User requested rewind",
        )

        data = original.to_dict()
        restored = RewindData.from_dict(data)

        assert restored.target_sequence == original.target_sequence
        assert restored.reason == original.reason


class TestSequenceAllocator:
    """Tests for sequence allocation."""

    def test_simple_allocator(self) -> None:
        """Test SimpleSequenceAllocator increments correctly."""
        alloc = SimpleSequenceAllocator(start=1)

        assert alloc.next_sequence() == 1
        assert alloc.next_sequence() == 2
        assert alloc.next_sequence() == 3
        assert alloc.get_current() == 3

    def test_simple_allocator_custom_start(self) -> None:
        """Test SimpleSequenceAllocator with custom start."""
        alloc = SimpleSequenceAllocator(start=100)

        assert alloc.next_sequence() == 100
        assert alloc.next_sequence() == 101

    def test_sequence_allocator_local_sequences(self) -> None:
        """Test SequenceAllocator generates device-local sequences."""
        alloc = SequenceAllocator(device_id="dev-001")

        seq1 = alloc.next_sequence()
        seq2 = alloc.next_sequence()

        assert seq1 == 1  # base=0, local=1 → 0*1000+1=1
        assert seq2 == 2  # base=0, local=2 → 0*1000+2=2

    def test_sequence_allocator_after_sync(self) -> None:
        """Test SequenceAllocator after setting base from sync."""
        alloc = SequenceAllocator(device_id="dev-001")

        # Generate some local sequences
        alloc.next_sequence()
        alloc.next_sequence()

        # Simulate sync - cloud says latest is sequence 5
        alloc.set_base_sequence(5)

        # Next sequence should be based on 5
        seq = alloc.next_sequence()
        assert seq == 5001  # base=5, local=1 → 5*1000+1=5001


class TestBlockWriter:
    """Tests for BlockWriter."""

    def test_create_session_block(self) -> None:
        """Test creating a session block."""
        writer = BlockWriter(
            session_id="sess-123",
            user_id="user-456",
            device_id="dev-001",
        )

        block = writer.create_session(
            project_slug="my-project",
            name="Test Session",
            visibility="private",
        )

        assert block.session_id == "sess-123"
        assert block.user_id == "user-456"
        assert block.block_type == BlockType.SESSION_CREATED
        assert block.data["project_slug"] == "my-project"
        assert block.data["name"] == "Test Session"
        assert block.sequence == 1

    def test_update_session_block(self) -> None:
        """Test creating an update block."""
        writer = BlockWriter(
            session_id="sess-123",
            user_id="user-456",
            device_id="dev-001",
        )

        block = writer.update_session(
            name="New Name",
            visibility="team",
            team_ids=["team-a"],
        )

        assert block.block_type == BlockType.SESSION_UPDATED
        assert block.data["name"] == "New Name"
        assert block.data["visibility"] == "team"
        assert block.data["team_ids"] == ["team-a"]

    def test_add_message_block(self) -> None:
        """Test creating a message block."""
        writer = BlockWriter(
            session_id="sess-123",
            user_id="user-456",
            device_id="dev-001",
        )

        block = writer.add_message(
            role="user",
            content="Hello, world!",
            turn=1,
        )

        assert block.block_type == BlockType.MESSAGE
        assert block.data["role"] == "user"
        assert block.data["content"] == "Hello, world!"
        assert block.data["turn"] == 1

    def test_add_small_event(self) -> None:
        """Test creating a small event (single block)."""
        writer = BlockWriter(
            session_id="sess-123",
            user_id="user-456",
            device_id="dev-001",
        )

        blocks = writer.add_event(
            event_id="evt-001",
            event_type="llm:response",
            data={"response": "Hello"},
            turn=1,
        )

        assert len(blocks) == 1
        assert blocks[0].block_type == BlockType.EVENT
        assert blocks[0].data["inline_data"] == {"response": "Hello"}
        assert blocks[0].data["has_continuation"] is False

    def test_add_large_event(self) -> None:
        """Test creating a large event (multiple blocks)."""
        writer = BlockWriter(
            session_id="sess-123",
            user_id="user-456",
            device_id="dev-001",
        )

        # Create data larger than MAX_INLINE_SIZE (50KB)
        large_content = "x" * 100_000  # 100KB

        blocks = writer.add_event(
            event_id="evt-002",
            event_type="llm:response",
            data={"content": large_content},
            turn=1,
        )

        # Should have header + 1 chunk (100KB < 300KB chunk size)
        assert len(blocks) >= 2

        # First block is header
        assert blocks[0].block_type == BlockType.EVENT
        assert blocks[0].data["has_continuation"] is True
        assert blocks[0].data.get("inline_data") is None  # Not included when None

        # Subsequent blocks are chunks
        for chunk_block in blocks[1:]:
            assert chunk_block.block_type == BlockType.EVENT_DATA
            assert chunk_block.parent_block_id == blocks[0].block_id

        # Last chunk should be marked as final
        assert blocks[-1].is_final_chunk is True

    def test_rewind_block(self) -> None:
        """Test creating a rewind block."""
        writer = BlockWriter(
            session_id="sess-123",
            user_id="user-456",
            device_id="dev-001",
        )

        block = writer.rewind(target_sequence=5, reason="User requested")

        assert block.block_type == BlockType.REWIND
        assert block.data["target_sequence"] == 5
        assert block.data["reason"] == "User requested"

    def test_sequence_incrementing(self) -> None:
        """Test that sequences increment correctly."""
        writer = BlockWriter(
            session_id="sess-123",
            user_id="user-456",
            device_id="dev-001",
        )

        block1 = writer.create_session(project_slug="test")
        block2 = writer.add_message(role="user", content="Hello", turn=1)
        block3 = writer.add_message(role="assistant", content="Hi", turn=1)

        assert block1.sequence == 1
        assert block2.sequence == 2
        assert block3.sequence == 3


class TestSessionStateReader:
    """Tests for SessionStateReader."""

    def _create_blocks(self) -> list[SessionBlock]:
        """Helper to create a set of test blocks."""
        writer = BlockWriter(
            session_id="sess-test",
            user_id="user-test",
            device_id="dev-test",
        )

        blocks = [
            writer.create_session(
                project_slug="test-project",
                name="Test Session",
                visibility="private",
            ),
            writer.add_message(role="user", content="Hello", turn=1),
            writer.add_message(role="assistant", content="Hi there!", turn=1),
            writer.add_message(role="user", content="How are you?", turn=2),
        ]

        # Add a small event
        event_blocks = writer.add_event(
            event_id="evt-1",
            event_type="llm:response",
            data={"status": "success"},
            turn=1,
        )
        blocks.extend(event_blocks)

        return blocks

    def test_compute_basic_state(self) -> None:
        """Test computing state from basic blocks."""
        blocks = self._create_blocks()
        reader = SessionStateReader()

        metadata, messages, events = reader.compute_current_state(blocks)

        assert metadata.session_id == "sess-test"
        assert metadata.name == "Test Session"
        assert metadata.project_slug == "test-project"
        assert metadata.message_count == 3
        assert metadata.event_count == 1
        assert metadata.turn_count == 2

        assert len(messages) == 3
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"

        assert len(events) == 1
        assert events[0].event_type == "llm:response"

    def test_compute_state_with_updates(self) -> None:
        """Test that updates are applied."""
        writer = BlockWriter(
            session_id="sess-test",
            user_id="user-test",
            device_id="dev-test",
        )

        blocks = [
            writer.create_session(
                project_slug="test",
                name="Original Name",
                visibility="private",
            ),
            writer.update_session(name="Updated Name"),
            writer.update_session(visibility="team", team_ids=["team-a"]),
        ]

        reader = SessionStateReader()
        metadata, _, _ = reader.compute_current_state(blocks)

        assert metadata.name == "Updated Name"
        assert metadata.visibility == "team"
        assert metadata.team_ids == ["team-a"]

    def test_compute_state_with_rewind(self) -> None:
        """Test that rewind filters out blocks after target."""
        writer = BlockWriter(
            session_id="sess-test",
            user_id="user-test",
            device_id="dev-test",
        )

        blocks = [
            writer.create_session(project_slug="test"),  # seq=1
            writer.add_message(role="user", content="Message 1", turn=1),  # seq=2
            writer.add_message(role="assistant", content="Reply 1", turn=1),  # seq=3
            writer.add_message(role="user", content="Message 2", turn=2),  # seq=4
            writer.add_message(role="assistant", content="Reply 2", turn=2),  # seq=5
            writer.rewind(target_sequence=3, reason="Undo last exchange"),  # seq=6
        ]

        reader = SessionStateReader()
        metadata, messages, _ = reader.compute_current_state(blocks)

        # Should only have messages up to seq=3
        assert len(messages) == 2
        assert messages[0].content == "Message 1"
        assert messages[1].content == "Reply 1"
        assert metadata.turn_count == 1

    def test_get_blocks_since(self) -> None:
        """Test getting blocks since a sequence."""
        blocks = self._create_blocks()
        reader = SessionStateReader()

        new_blocks = reader.get_blocks_since(blocks, since_sequence=2)

        # Should return all blocks with sequence > 2
        assert all(b.sequence > 2 for b in new_blocks)
        assert len(new_blocks) < len(blocks)

    def test_reconstruct_event_data_inline(self) -> None:
        """Test reconstructing inline event data."""
        writer = BlockWriter(
            session_id="sess-test",
            user_id="user-test",
            device_id="dev-test",
        )

        event_blocks = writer.add_event(
            event_id="evt-1",
            event_type="test",
            data={"key": "value"},
        )

        reader = SessionStateReader()
        data = reader.reconstruct_event_data(event_blocks[0], [])

        assert data == {"key": "value"}

    def test_reconstruct_event_data_chunked(self) -> None:
        """Test reconstructing chunked event data."""
        writer = BlockWriter(
            session_id="sess-test",
            user_id="user-test",
            device_id="dev-test",
        )

        # Create large event
        large_data = {"content": "x" * 100_000}
        event_blocks = writer.add_event(
            event_id="evt-large",
            event_type="test",
            data=large_data,
        )

        reader = SessionStateReader()
        header = event_blocks[0]
        chunks = event_blocks[1:]

        reconstructed = reader.reconstruct_event_data(header, chunks)

        assert reconstructed == large_data

    def test_empty_blocks_raises(self) -> None:
        """Test that empty block list raises error."""
        reader = SessionStateReader()

        with pytest.raises(ValueError, match="empty"):
            reader.compute_current_state([])

    def test_out_of_order_blocks(self) -> None:
        """Test that out-of-order blocks are sorted correctly."""
        writer = BlockWriter(
            session_id="sess-test",
            user_id="user-test",
            device_id="dev-test",
        )

        # Create blocks in order
        block1 = writer.create_session(project_slug="test")
        block2 = writer.add_message(role="user", content="First", turn=1)
        block3 = writer.add_message(role="assistant", content="Second", turn=1)

        # Pass them out of order
        blocks = [block3, block1, block2]

        reader = SessionStateReader()
        _, messages, _ = reader.compute_current_state(blocks)

        # Should be in correct sequence order
        assert messages[0].content == "First"
        assert messages[1].content == "Second"
