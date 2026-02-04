"""
Schema validation tests - verify storage backends match Amplifier CLI structure.

Ensures that DuckDB and SQLite backends create schemas compatible with
Amplifier's session storage format from ~/.amplifier/projects.
"""

from datetime import UTC, datetime

import pytest

from amplifier_session_storage.backends.duckdb import DuckDBBackend, DuckDBConfig
from amplifier_session_storage.backends.sqlite import SQLiteBackend, SQLiteConfig


class TestDuckDBSchemaCompatibility:
    """Verify DuckDB schema matches Amplifier session structure."""

    @pytest.mark.asyncio
    async def test_session_metadata_schema(self, embedding_provider):
        """Session metadata matches Amplifier's metadata.json structure."""
        storage = await DuckDBBackend.create(
            config=DuckDBConfig(db_path=":memory:"), embedding_provider=embedding_provider
        )

        # Amplifier metadata.json structure
        metadata = {
            "session_id": "sess_abc123",
            "project_slug": "my-project",
            "bundle": "foundation",
            "created": datetime.now(UTC).isoformat(),
            "updated": datetime.now(UTC).isoformat(),
            "turn_count": 5,
        }

        # Store
        await storage.upsert_session_metadata(
            user_id="user-123", host_id="laptop-01", metadata=metadata
        )

        # Retrieve
        retrieved = await storage.get_session_metadata(user_id="user-123", session_id="sess_abc123")

        # Verify retrieval succeeded
        assert retrieved is not None

        # Verify all Amplifier fields present
        assert retrieved["session_id"] == "sess_abc123"
        assert retrieved["project_slug"] == "my-project"
        assert retrieved["bundle"] == "foundation"
        assert "created" in retrieved
        assert "turn_count" in retrieved
        assert retrieved["turn_count"] == 5

        await storage.close()

    @pytest.mark.asyncio
    async def test_transcript_line_schema(self, embedding_provider):
        """Transcript lines match Amplifier's transcript.jsonl structure."""
        storage = await DuckDBBackend.create(
            config=DuckDBConfig(db_path=":memory:"), embedding_provider=embedding_provider
        )

        # Amplifier transcript.jsonl line structure
        lines = [
            {
                "role": "user",
                "content": "How do I use this feature?",
                "turn": 0,
                "ts": "2024-01-15T10:00:00Z",
            },
            {
                "role": "assistant",
                "content": "You can use it by...",
                "turn": 0,
                "ts": "2024-01-15T10:00:01Z",
            },
        ]

        # Store
        synced = await storage.sync_transcript_lines(
            user_id="user-123",
            host_id="laptop-01",
            project_slug="my-project",
            session_id="sess_abc",
            lines=lines,
            start_sequence=0,
        )

        assert synced == 2

        # Retrieve
        retrieved = await storage.get_transcript_lines(
            user_id="user-123", project_slug="my-project", session_id="sess_abc"
        )

        # Verify Amplifier fields present
        assert len(retrieved) == 2
        assert retrieved[0]["role"] == "user"
        assert retrieved[0]["content"] == "How do I use this feature?"
        assert retrieved[0]["turn"] == 0
        assert "ts" in retrieved[0]

        assert retrieved[1]["role"] == "assistant"
        assert retrieved[1]["turn"] == 0

        await storage.close()

    @pytest.mark.asyncio
    async def test_event_line_schema(self, embedding_provider):
        """Event lines match Amplifier's events.jsonl structure."""
        storage = await DuckDBBackend.create(
            config=DuckDBConfig(db_path=":memory:"), embedding_provider=embedding_provider
        )

        # Amplifier events.jsonl line structure
        events = [
            {
                "event": "session.start",
                "ts": "2024-01-15T10:00:00Z",
                "lvl": "info",
                "turn": 0,
            },
            {
                "event": "llm.request",
                "ts": "2024-01-15T10:00:01Z",
                "lvl": "debug",
                "turn": 0,
                "data": {"model": "gpt-4", "prompt_tokens": 100},
            },
        ]

        # Store
        synced = await storage.sync_event_lines(
            user_id="user-123",
            host_id="laptop-01",
            project_slug="my-project",
            session_id="sess_abc",
            lines=events,
            start_sequence=0,
        )

        assert synced == 2

        # Retrieve
        retrieved = await storage.get_event_lines(
            user_id="user-123", project_slug="my-project", session_id="sess_abc"
        )

        # Verify Amplifier fields present
        assert len(retrieved) == 2
        assert retrieved[0]["event"] == "session.start"
        # Timestamp field exists (DuckDB may return datetime object or string)
        assert "ts" in retrieved[0]
        assert retrieved[0]["lvl"] == "info"
        assert retrieved[0]["turn"] == 0

        assert retrieved[1]["event"] == "llm.request"
        assert retrieved[1]["lvl"] == "debug"

        await storage.close()

    @pytest.mark.asyncio
    async def test_incremental_sync(self, embedding_provider):
        """Incremental sync works like Amplifier's append-only JSONL."""
        storage = await DuckDBBackend.create(
            config=DuckDBConfig(db_path=":memory:"), embedding_provider=embedding_provider
        )

        # First batch (like first few turns)
        batch1 = [
            {"role": "user", "content": "First message", "turn": 0},
            {"role": "assistant", "content": "First response", "turn": 0},
        ]

        await storage.sync_transcript_lines(
            user_id="user-123",
            host_id="laptop-01",
            project_slug="my-project",
            session_id="sess_abc",
            lines=batch1,
            start_sequence=0,
        )

        # Second batch (like next few turns)
        batch2 = [
            {"role": "user", "content": "Second message", "turn": 1},
            {"role": "assistant", "content": "Second response", "turn": 1},
        ]

        await storage.sync_transcript_lines(
            user_id="user-123",
            host_id="laptop-01",
            project_slug="my-project",
            session_id="sess_abc",
            lines=batch2,
            start_sequence=2,  # Continue sequence
        )

        # Get all
        all_messages = await storage.get_transcript_lines(
            user_id="user-123", project_slug="my-project", session_id="sess_abc"
        )

        # Should have all 4 messages in sequence
        assert len(all_messages) == 4
        assert all_messages[0]["sequence"] == 0
        assert all_messages[1]["sequence"] == 1
        assert all_messages[2]["sequence"] == 2
        assert all_messages[3]["sequence"] == 3

        await storage.close()


class TestSQLiteSchemaCompatibility:
    """Verify SQLite schema matches Amplifier session structure."""

    @pytest.mark.asyncio
    async def test_session_metadata_schema(self, embedding_provider):
        """Session metadata matches Amplifier's metadata.json structure."""
        storage = await SQLiteBackend.create(
            config=SQLiteConfig(db_path=":memory:"), embedding_provider=embedding_provider
        )

        metadata = {
            "session_id": "sess_xyz789",
            "project_slug": "test-project",
            "bundle": "custom-bundle",
            "created": datetime.now(UTC).isoformat(),
            "turn_count": 10,
        }

        await storage.upsert_session_metadata(
            user_id="user-456", host_id="workstation-02", metadata=metadata
        )

        retrieved = await storage.get_session_metadata(user_id="user-456", session_id="sess_xyz789")

        # Verify structure matches
        assert retrieved["session_id"] == "sess_xyz789"
        assert retrieved["project_slug"] == "test-project"
        assert retrieved["bundle"] == "custom-bundle"
        assert retrieved["turn_count"] == 10

        await storage.close()

    @pytest.mark.asyncio
    async def test_full_session_workflow(self, embedding_provider):
        """Complete session workflow matches Amplifier pattern."""
        storage = await SQLiteBackend.create(
            config=SQLiteConfig(db_path=":memory:"), embedding_provider=embedding_provider
        )

        session_id = "sess_workflow_test"
        project_slug = "my-app"

        # 1. Create session metadata
        await storage.upsert_session_metadata(
            user_id="user-123",
            host_id="dev-laptop",
            metadata={
                "session_id": session_id,
                "project_slug": project_slug,
                "bundle": "foundation",
                "created": "2024-01-15T10:00:00Z",
                "turn_count": 0,
            },
        )

        # 2. Add first turn (user + assistant)
        turn0_lines = [
            {"role": "user", "content": "Hello", "turn": 0, "ts": "2024-01-15T10:00:00Z"},
            {"role": "assistant", "content": "Hi!", "turn": 0, "ts": "2024-01-15T10:00:01Z"},
        ]

        await storage.sync_transcript_lines(
            user_id="user-123",
            host_id="dev-laptop",
            project_slug=project_slug,
            session_id=session_id,
            lines=turn0_lines,
            start_sequence=0,
        )

        # 3. Add events for turn 0
        turn0_events = [
            {"event": "session.start", "ts": "2024-01-15T10:00:00Z", "lvl": "info", "turn": 0},
            {"event": "llm.request", "ts": "2024-01-15T10:00:01Z", "lvl": "debug", "turn": 0},
        ]

        await storage.sync_event_lines(
            user_id="user-123",
            host_id="dev-laptop",
            project_slug=project_slug,
            session_id=session_id,
            lines=turn0_events,
            start_sequence=0,
        )

        # 4. Update session metadata (turn_count incremented)
        await storage.upsert_session_metadata(
            user_id="user-123",
            host_id="dev-laptop",
            metadata={
                "session_id": session_id,
                "project_slug": project_slug,
                "bundle": "foundation",
                "created": "2024-01-15T10:00:00Z",
                "turn_count": 1,
            },
        )

        # 5. Verify we can retrieve everything
        session = await storage.get_session_metadata("user-123", session_id)
        transcripts = await storage.get_transcript_lines("user-123", project_slug, session_id)
        events = await storage.get_event_lines("user-123", project_slug, session_id)

        # Verify complete session
        assert session["turn_count"] == 1
        assert len(transcripts) == 2
        assert len(events) == 2
        assert transcripts[0]["role"] == "user"
        assert events[0]["event"] == "session.start"

        await storage.close()


class TestSchemaFieldMapping:
    """Verify field mappings between Amplifier CLI and storage backends."""

    @pytest.mark.asyncio
    async def test_metadata_field_mapping(self, embedding_provider):
        """All Amplifier metadata fields are preserved."""
        storage = await DuckDBBackend.create(
            config=DuckDBConfig(db_path=":memory:"), embedding_provider=embedding_provider
        )

        # Complete Amplifier metadata structure
        amplifier_metadata = {
            "session_id": "sess_test",
            "project_slug": "amplifier-core",
            "bundle": "foundation",
            "created": "2024-01-15T10:00:00.123456+00:00",
            "updated": "2024-01-15T11:30:00.123456+00:00",
            "turn_count": 42,
            "model": "claude-3-5-sonnet",  # Extra field
            "provider": "anthropic",  # Extra field
        }

        await storage.upsert_session_metadata(
            user_id="user-123", host_id="laptop-01", metadata=amplifier_metadata
        )

        retrieved = await storage.get_session_metadata("user-123", "sess_test")

        # All fields should be preserved (stored in metadata JSON column)
        assert retrieved["session_id"] == amplifier_metadata["session_id"]
        assert retrieved["project_slug"] == amplifier_metadata["project_slug"]
        assert retrieved["bundle"] == amplifier_metadata["bundle"]
        assert retrieved["created"] == amplifier_metadata["created"]
        assert retrieved["updated"] == amplifier_metadata["updated"]
        assert retrieved["turn_count"] == amplifier_metadata["turn_count"]
        assert retrieved["model"] == amplifier_metadata["model"]
        assert retrieved["provider"] == amplifier_metadata["provider"]

        await storage.close()

    @pytest.mark.asyncio
    async def test_transcript_field_mapping(self, embedding_provider):
        """All Amplifier transcript fields are preserved."""
        storage = await SQLiteBackend.create(
            config=SQLiteConfig(db_path=":memory:"), embedding_provider=embedding_provider
        )

        # Amplifier transcript.jsonl line structure
        transcript_lines = [
            {
                "role": "user",
                "content": "What is the meaning of life?",
                "turn": 0,
                "ts": "2024-01-15T10:00:00.123456+00:00",
                "timestamp": "2024-01-15T10:00:00.123456+00:00",  # Alternate field
            }
        ]

        await storage.sync_transcript_lines(
            user_id="user-123",
            host_id="laptop-01",
            project_slug="philosophy",
            session_id="sess_deep",
            lines=transcript_lines,
        )

        retrieved = await storage.get_transcript_lines(
            user_id="user-123", project_slug="philosophy", session_id="sess_deep"
        )

        # Verify retrieval worked and has correct data
        assert retrieved is not None
        assert len(retrieved) == 1
        assert retrieved[0]["role"] == "user"
        assert retrieved[0]["content"] == "What is the meaning of life?"
        assert retrieved[0]["turn"] == 0
        assert "ts" in retrieved[0]

        await storage.close()

    @pytest.mark.asyncio
    async def test_event_field_mapping(self, embedding_provider):
        """All Amplifier event fields are preserved."""
        storage = await DuckDBBackend.create(
            config=DuckDBConfig(db_path=":memory:"), embedding_provider=embedding_provider
        )

        # Amplifier events.jsonl line structure
        event_lines = [
            {
                "event": "tool.call",
                "ts": "2024-01-15T10:00:00Z",
                "lvl": "debug",
                "turn": 0,
                "data": {
                    "tool": "bash",
                    "command": "ls -la",
                    "result": "total 123...",
                },
            }
        ]

        await storage.sync_event_lines(
            user_id="user-123",
            host_id="laptop-01",
            project_slug="automation",
            session_id="sess_tools",
            lines=event_lines,
        )

        retrieved = await storage.get_event_lines(
            user_id="user-123", project_slug="automation", session_id="sess_tools"
        )

        # Verify fields
        assert len(retrieved) == 1
        assert retrieved[0]["event"] == "tool.call"
        assert retrieved[0]["lvl"] == "debug"
        assert retrieved[0]["turn"] == 0
        assert "ts" in retrieved[0]

        await storage.close()
