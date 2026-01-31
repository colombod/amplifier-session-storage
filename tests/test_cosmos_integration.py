"""
Integration tests for Cosmos DB storage.

These tests require a real Cosmos DB connection and are marked as integration tests.
Run with: pytest -m integration tests/test_cosmos_integration.py

Environment variables required:
- COSMOS_ENDPOINT: Cosmos DB endpoint URL
- COSMOS_KEY: Cosmos DB account key

Optional:
- COSMOS_DATABASE: Database name (default: amplifier-sessions-test)
"""

import os
import uuid
from datetime import UTC, datetime

import pytest

# Skip all tests in this module if Cosmos is not configured
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("COSMOS_ENDPOINT") or not os.environ.get("COSMOS_KEY"),
        reason="COSMOS_ENDPOINT or COSMOS_KEY not set - skipping Cosmos integration tests",
    ),
]


@pytest.fixture(scope="module")
def cosmos_config():
    """Get Cosmos configuration from environment."""
    from amplifier_session_storage.cosmos.client import CosmosConfig

    return CosmosConfig(
        endpoint=os.environ["COSMOS_ENDPOINT"],
        key=os.environ["COSMOS_KEY"],
        database_name=os.environ.get("COSMOS_DATABASE", "amplifier-sessions-test"),
    )


@pytest.fixture
async def storage(cosmos_config):
    """Create a CosmosDBStorage instance for tests."""
    from amplifier_session_storage.cosmos.storage import CosmosDBStorage

    storage = await CosmosDBStorage.create(cosmos_config)
    yield storage
    await storage.close()


@pytest.fixture
def test_user_id():
    """Generate unique user ID for test isolation."""
    return f"test-user-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_session_id():
    """Generate unique session ID."""
    return f"test-session-{uuid.uuid4().hex[:8]}"


class TestCosmosSessionCRUD:
    """Tests for session CRUD operations."""

    async def test_create_session(self, storage, test_user_id, test_session_id):
        """Create a session and verify it's stored."""
        from amplifier_session_storage.protocol import SessionMetadata

        metadata = SessionMetadata(
            session_id=test_session_id,
            user_id=test_user_id,
            project_slug="test-project",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
            name="Test Session",
            bundle="bundle:foundation",
            model="claude-sonnet",
        )

        created = await storage.create_session(metadata)

        assert created.session_id == test_session_id
        assert created.user_id == test_user_id
        assert created.name == "Test Session"

        # Cleanup
        await storage.delete_session(test_user_id, test_session_id)

    async def test_get_session(self, storage, test_user_id, test_session_id):
        """Get a session by ID."""
        from amplifier_session_storage.protocol import SessionMetadata

        # Create session first
        metadata = SessionMetadata(
            session_id=test_session_id,
            user_id=test_user_id,
            project_slug="test-project",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
        )
        await storage.create_session(metadata)

        # Get session
        retrieved = await storage.get_session(test_user_id, test_session_id)

        assert retrieved is not None
        assert retrieved.session_id == test_session_id
        assert retrieved.project_slug == "test-project"

        # Cleanup
        await storage.delete_session(test_user_id, test_session_id)

    async def test_get_nonexistent_session(self, storage, test_user_id):
        """Get a session that doesn't exist returns None."""
        retrieved = await storage.get_session(test_user_id, "nonexistent-session")
        assert retrieved is None

    async def test_update_session(self, storage, test_user_id, test_session_id):
        """Update session metadata."""
        from amplifier_session_storage.protocol import SessionMetadata

        # Create session
        metadata = SessionMetadata(
            session_id=test_session_id,
            user_id=test_user_id,
            project_slug="test-project",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
            name="Original Name",
        )
        await storage.create_session(metadata)

        # Update
        metadata.name = "Updated Name"
        metadata.turn_count = 5
        updated = await storage.update_session(metadata)

        assert updated.name == "Updated Name"
        assert updated.turn_count == 5

        # Verify persistence
        retrieved = await storage.get_session(test_user_id, test_session_id)
        assert retrieved.name == "Updated Name"

        # Cleanup
        await storage.delete_session(test_user_id, test_session_id)

    async def test_delete_session(self, storage, test_user_id, test_session_id):
        """Delete a session."""
        from amplifier_session_storage.protocol import SessionMetadata

        # Create session
        metadata = SessionMetadata(
            session_id=test_session_id,
            user_id=test_user_id,
            project_slug="test-project",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
        )
        await storage.create_session(metadata)

        # Delete
        result = await storage.delete_session(test_user_id, test_session_id)
        assert result is True

        # Verify deleted
        retrieved = await storage.get_session(test_user_id, test_session_id)
        assert retrieved is None

    async def test_delete_nonexistent_session(self, storage, test_user_id):
        """Delete a session that doesn't exist returns False."""
        result = await storage.delete_session(test_user_id, "nonexistent-session")
        assert result is False


class TestCosmosUserIsolation:
    """Tests for user isolation - CRITICAL security requirement."""

    async def test_user_cannot_access_other_users_session(self, storage):
        """User A cannot access User B's session."""
        from amplifier_session_storage.protocol import SessionMetadata

        user_a = f"user-a-{uuid.uuid4().hex[:8]}"
        user_b = f"user-b-{uuid.uuid4().hex[:8]}"
        session_id = f"session-{uuid.uuid4().hex[:8]}"

        # User A creates session
        metadata = SessionMetadata(
            session_id=session_id,
            user_id=user_a,
            project_slug="test-project",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
        )
        await storage.create_session(metadata)

        # User B tries to access - should return None
        retrieved = await storage.get_session(user_b, session_id)
        assert retrieved is None

        # Cleanup
        await storage.delete_session(user_a, session_id)

    async def test_list_sessions_only_returns_own(self, storage):
        """list_sessions only returns user's own sessions."""
        from amplifier_session_storage.protocol import SessionMetadata, SessionQuery

        user_a = f"user-a-{uuid.uuid4().hex[:8]}"
        user_b = f"user-b-{uuid.uuid4().hex[:8]}"

        # Create sessions for both users
        for user, prefix in [(user_a, "a"), (user_b, "b")]:
            for i in range(2):
                metadata = SessionMetadata(
                    session_id=f"session-{prefix}-{i}-{uuid.uuid4().hex[:8]}",
                    user_id=user,
                    project_slug="test-project",
                    created=datetime.now(UTC),
                    updated=datetime.now(UTC),
                )
                await storage.create_session(metadata)

        # User A lists sessions
        query = SessionQuery(user_id=user_a)
        sessions = await storage.list_sessions(query)

        # Should only see user A's sessions
        for session in sessions:
            assert session.user_id == user_a

        # Cleanup
        for session in sessions:
            await storage.delete_session(user_a, session.session_id)

        query_b = SessionQuery(user_id=user_b)
        sessions_b = await storage.list_sessions(query_b)
        for session in sessions_b:
            await storage.delete_session(user_b, session.session_id)


class TestCosmosTranscript:
    """Tests for transcript message operations."""

    async def test_append_and_get_transcript(self, storage, test_user_id, test_session_id):
        """Append messages and retrieve transcript."""
        from amplifier_session_storage.protocol import SessionMetadata, TranscriptMessage

        # Create session first
        metadata = SessionMetadata(
            session_id=test_session_id,
            user_id=test_user_id,
            project_slug="test-project",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
        )
        await storage.create_session(metadata)

        # Append messages
        msg1 = TranscriptMessage(
            sequence=0,
            role="user",
            content="Hello",
            timestamp=datetime.now(UTC),
            turn=1,
        )
        msg2 = TranscriptMessage(
            sequence=0,  # Will be assigned by storage
            role="assistant",
            content="Hi there!",
            timestamp=datetime.now(UTC),
            turn=1,
        )

        await storage.append_message(test_user_id, test_session_id, msg1)
        await storage.append_message(test_user_id, test_session_id, msg2)

        # Get transcript
        transcript = await storage.get_transcript(test_user_id, test_session_id)

        assert len(transcript) == 2
        assert transcript[0].role == "user"
        assert transcript[0].content == "Hello"
        assert transcript[1].role == "assistant"
        assert transcript[1].content == "Hi there!"

        # Cleanup
        await storage.delete_session(test_user_id, test_session_id)

    async def test_get_transcript_for_turn(self, storage, test_user_id, test_session_id):
        """Get messages for a specific turn."""
        from amplifier_session_storage.protocol import SessionMetadata, TranscriptMessage

        # Create session
        metadata = SessionMetadata(
            session_id=test_session_id,
            user_id=test_user_id,
            project_slug="test-project",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
        )
        await storage.create_session(metadata)

        # Add messages across turns
        for turn in [1, 1, 2, 2, 3]:
            msg = TranscriptMessage(
                sequence=0,
                role="user" if turn % 2 else "assistant",
                content=f"Turn {turn} message",
                timestamp=datetime.now(UTC),
                turn=turn,
            )
            await storage.append_message(test_user_id, test_session_id, msg)

        # Get turn 2 only
        turn_2_messages = await storage.get_transcript_for_turn(
            test_user_id, test_session_id, turn=2
        )

        assert len(turn_2_messages) == 2
        for msg in turn_2_messages:
            assert msg.turn == 2

        # Cleanup
        await storage.delete_session(test_user_id, test_session_id)


class TestCosmosEvents:
    """Tests for event operations with projection enforcement."""

    async def test_append_event(self, storage, test_user_id, test_session_id):
        """Append an event and verify summary is returned."""
        from amplifier_session_storage.protocol import SessionMetadata

        # Create session
        metadata = SessionMetadata(
            session_id=test_session_id,
            user_id=test_user_id,
            project_slug="test-project",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
        )
        await storage.create_session(metadata)

        # Append event
        event_id = f"event-{uuid.uuid4().hex[:8]}"
        data = {
            "model": "claude-sonnet",
            "duration_ms": 1234,
            "content": "This is a large content field that should NOT be in summary",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

        summary = await storage.append_event(
            test_user_id, test_session_id, event_id, "llm:response", data, turn=1
        )

        # Summary should have safe fields only
        assert summary.event_id == event_id
        assert summary.event_type == "llm:response"
        assert "model" in summary.summary
        assert "content" not in summary.summary  # CRITICAL: content excluded

        # Cleanup
        await storage.delete_session(test_user_id, test_session_id)

    async def test_query_events_never_returns_full_data(
        self, storage, test_user_id, test_session_id
    ):
        """CRITICAL: query_events must NEVER return full event data."""
        from amplifier_session_storage.protocol import EventQuery, SessionMetadata

        # Create session
        metadata = SessionMetadata(
            session_id=test_session_id,
            user_id=test_user_id,
            project_slug="test-project",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
        )
        await storage.create_session(metadata)

        # Add event with large data
        large_content = "x" * 10000  # 10KB of data
        await storage.append_event(
            test_user_id,
            test_session_id,
            f"event-{uuid.uuid4().hex[:8]}",
            "llm:response",
            {"content": large_content, "model": "test"},
            turn=1,
        )

        # Query events
        query = EventQuery(session_id=test_session_id, user_id=test_user_id)
        events = await storage.query_events(query)

        assert len(events) == 1
        event = events[0]

        # CRITICAL ASSERTION: EventSummary does NOT have a 'data' attribute
        assert not hasattr(event, "data")
        assert not hasattr(event, "content")

        # Summary should have safe fields only
        assert event.event_type == "llm:response"
        assert event.data_size_bytes > 0

        # Cleanup
        await storage.delete_session(test_user_id, test_session_id)

    async def test_get_event_data_returns_full_data(self, storage, test_user_id, test_session_id):
        """get_event_data is the ONLY method that returns full data."""
        from amplifier_session_storage.protocol import SessionMetadata

        # Create session
        metadata = SessionMetadata(
            session_id=test_session_id,
            user_id=test_user_id,
            project_slug="test-project",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
        )
        await storage.create_session(metadata)

        # Add event
        event_id = f"event-{uuid.uuid4().hex[:8]}"
        original_data = {
            "content": "Full content here",
            "model": "test",
            "usage": {"input_tokens": 100},
        }
        await storage.append_event(
            test_user_id, test_session_id, event_id, "llm:response", original_data
        )

        # Get full data
        full_data = await storage.get_event_data(test_user_id, test_session_id, event_id)

        assert full_data is not None
        assert full_data["content"] == "Full content here"
        assert full_data["model"] == "test"

        # Cleanup
        await storage.delete_session(test_user_id, test_session_id)


class TestCosmosEventChunking:
    """Tests for large event chunking."""

    async def test_large_event_is_chunked(self, storage, test_user_id, test_session_id):
        """Events larger than 400KB should be chunked."""
        from amplifier_session_storage.protocol import SessionMetadata

        # Create session
        metadata = SessionMetadata(
            session_id=test_session_id,
            user_id=test_user_id,
            project_slug="test-project",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
        )
        await storage.create_session(metadata)

        # Create large event (500KB+)
        event_id = f"event-{uuid.uuid4().hex[:8]}"
        large_content = "x" * (500 * 1024)  # 500KB
        large_data = {"content": large_content, "model": "test"}

        summary = await storage.append_event(
            test_user_id, test_session_id, event_id, "llm:response", large_data
        )

        assert summary.data_size_bytes > 400 * 1024

        # Should still be retrievable
        full_data = await storage.get_event_data(test_user_id, test_session_id, event_id)

        assert full_data is not None
        assert full_data["content"] == large_content

        # Cleanup
        await storage.delete_session(test_user_id, test_session_id)


class TestCosmosRewind:
    """Tests for rewind operations."""

    async def test_rewind_to_turn(self, storage, test_user_id, test_session_id):
        """Rewind session to a specific turn."""
        from amplifier_session_storage.protocol import SessionMetadata, TranscriptMessage

        # Create session
        metadata = SessionMetadata(
            session_id=test_session_id,
            user_id=test_user_id,
            project_slug="test-project",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
        )
        await storage.create_session(metadata)

        # Add messages for 3 turns
        for turn in range(1, 4):
            # User message
            user_msg = TranscriptMessage(
                sequence=0,
                role="user",
                content=f"Turn {turn} user",
                timestamp=datetime.now(UTC),
                turn=turn,
            )
            await storage.append_message(test_user_id, test_session_id, user_msg)
            # Assistant message
            assistant_msg = TranscriptMessage(
                sequence=0,
                role="assistant",
                content=f"Turn {turn} assistant",
                timestamp=datetime.now(UTC),
                turn=turn,
            )
            await storage.append_message(test_user_id, test_session_id, assistant_msg)

        # Rewind to turn 1
        result = await storage.rewind_to_turn(test_user_id, test_session_id, turn=1)

        assert result.success
        assert result.messages_removed == 4  # Turns 2 and 3 (2 messages each)
        assert result.new_turn_count == 1

        # Verify only turn 1 remains
        transcript = await storage.get_transcript(test_user_id, test_session_id)
        assert len(transcript) == 2
        for msg in transcript:
            assert msg.turn == 1

        # Cleanup
        await storage.delete_session(test_user_id, test_session_id)


class TestCosmosSchemaMapping:
    """Tests verifying schema mapping between disk and Cosmos."""

    async def test_session_metadata_fields_match_schema(
        self, storage, test_user_id, test_session_id
    ):
        """Verify SessionMetadata has all required fields from schema."""
        from amplifier_session_storage.protocol import SessionMetadata, SessionVisibility

        # Create session with all fields
        metadata = SessionMetadata(
            session_id=test_session_id,
            user_id=test_user_id,
            project_slug="test-project",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
            name="Test Session",
            description="Test description",
            bundle="bundle:foundation",
            model="claude-sonnet",
            turn_count=5,
            message_count=10,
            event_count=20,
            parent_id="parent-123",
            visibility=SessionVisibility.PRIVATE,
            tags=["test", "integration"],
        )

        created = await storage.create_session(metadata)

        # Verify all fields persisted
        assert created.session_id == test_session_id
        assert created.user_id == test_user_id
        assert created.project_slug == "test-project"
        assert created.name == "Test Session"
        assert created.description == "Test description"
        assert created.bundle == "bundle:foundation"
        assert created.model == "claude-sonnet"
        assert created.turn_count == 5
        assert created.message_count == 10
        assert created.event_count == 20
        assert created.parent_id == "parent-123"
        assert created.visibility == SessionVisibility.PRIVATE
        assert created.tags == ["test", "integration"]

        # Cleanup
        await storage.delete_session(test_user_id, test_session_id)

    async def test_transcript_message_fields_match_schema(
        self, storage, test_user_id, test_session_id
    ):
        """Verify TranscriptMessage has all required fields."""
        from amplifier_session_storage.protocol import SessionMetadata, TranscriptMessage

        # Create session
        metadata = SessionMetadata(
            session_id=test_session_id,
            user_id=test_user_id,
            project_slug="test-project",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
        )
        await storage.create_session(metadata)

        # Create message with all fields
        now = datetime.now(UTC)
        msg = TranscriptMessage(
            sequence=0,
            role="assistant",
            content="Hello",
            timestamp=now,
            turn=1,
            tool_calls=[{"id": "call_123", "name": "test_tool"}],
        )

        await storage.append_message(test_user_id, test_session_id, msg)

        # Retrieve and verify
        transcript = await storage.get_transcript(test_user_id, test_session_id)
        assert len(transcript) == 1

        retrieved = transcript[0]
        assert retrieved.role == "assistant"
        assert retrieved.content == "Hello"
        assert retrieved.turn == 1
        assert retrieved.tool_calls == [{"id": "call_123", "name": "test_tool"}]

        # Cleanup
        await storage.delete_session(test_user_id, test_session_id)
