"""
Integration tests for CosmosFileStorage and HybridFileStorage.

These tests require a real Cosmos DB connection. Set the following environment variables:
- AMPLIFIER_COSMOS_ENDPOINT
- AMPLIFIER_COSMOS_DATABASE
- AMPLIFIER_COSMOS_AUTH_METHOD (default_credential or key)
- AMPLIFIER_COSMOS_KEY (only if auth_method=key)

Run with: pytest tests/test_file_storage_integration.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pytest

# Check if cosmos dependencies are available
try:
    import importlib.util

    COSMOS_AVAILABLE = importlib.util.find_spec("azure.cosmos") is not None
except ImportError:
    COSMOS_AVAILABLE = False

# Check if environment is configured
COSMOS_CONFIGURED = bool(os.environ.get("AMPLIFIER_COSMOS_ENDPOINT"))

# Skip all tests if cosmos not available
pytestmark = [
    pytest.mark.skipif(not COSMOS_AVAILABLE, reason="azure-cosmos not installed"),
    pytest.mark.skipif(not COSMOS_CONFIGURED, reason="Cosmos DB not configured"),
    pytest.mark.integration,
]


@pytest.fixture
def test_user_id() -> str:
    """Generate a unique user ID for test isolation."""
    return f"test-user-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_project_slug() -> str:
    """Generate a unique project slug for test isolation."""
    return f"test-project-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_session_id() -> str:
    """Generate a unique session ID."""
    return f"test-session-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def sample_metadata(test_session_id: str, test_project_slug: str) -> dict:
    """Sample session metadata matching CLI format."""
    now = datetime.now(UTC).isoformat()
    return {
        "session_id": test_session_id,
        "project_slug": test_project_slug,
        "name": "Test Session",
        "description": "Integration test session",
        "bundle": "test-bundle",
        "model": "test-model",
        "created": now,
        "updated": now,
        "turn_count": 0,
        "message_count": 0,
        "event_count": 0,
    }


@pytest.fixture
def sample_transcript_lines() -> list[dict]:
    """Sample transcript messages matching CLI format."""
    now = datetime.now(UTC).isoformat()
    return [
        {
            "role": "user",
            "content": "Hello, this is a test message",
            "timestamp": now,
            "turn": 1,
        },
        {
            "role": "assistant",
            "content": "Hello! I'm responding to your test.",
            "timestamp": now,
            "turn": 1,
        },
        {
            "role": "user",
            "content": "Can you help me with something?",
            "timestamp": now,
            "turn": 2,
        },
        {
            "role": "assistant",
            "content": "Of course! What do you need help with?",
            "timestamp": now,
            "turn": 2,
        },
    ]


@pytest.fixture
def sample_event_lines() -> list[dict]:
    """Sample events matching CLI format."""
    now = datetime.now(UTC).isoformat()
    return [
        {
            "ts": now,
            "lvl": "INFO",
            "event": "session:start",
            "turn": 0,
            "data": {"bundle": "test-bundle", "model": "test-model"},
        },
        {
            "ts": now,
            "lvl": "DEBUG",
            "event": "llm:request",
            "turn": 1,
            "data": {"model": "test-model", "message_count": 1},
        },
        {
            "ts": now,
            "lvl": "INFO",
            "event": "llm:response",
            "turn": 1,
            "data": {
                "model": "test-model",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        },
    ]


class TestCosmosFileStorage:
    """Tests for CosmosFileStorage."""

    @pytest.fixture
    async def cosmos_storage(self):
        """Create and initialize CosmosFileStorage."""
        from amplifier_session_storage.cosmos.file_storage import (
            CosmosFileConfig,
            CosmosFileStorage,
        )

        config = CosmosFileConfig.from_env()
        storage = CosmosFileStorage(config)
        await storage.initialize()

        yield storage

        # Cleanup
        await storage.close()

    @pytest.mark.asyncio
    async def test_connection(self, cosmos_storage):
        """Test that we can connect to Cosmos DB."""
        assert cosmos_storage._initialized is True

        # Verify connection works
        result = await cosmos_storage.verify_connection()
        assert result is True

    @pytest.mark.asyncio
    async def test_session_metadata_crud(
        self,
        cosmos_storage,
        test_user_id: str,
        test_session_id: str,
        test_project_slug: str,
        sample_metadata: dict,
    ):
        """Test session metadata create/read/update/delete."""
        # Create
        await cosmos_storage.upsert_session_metadata(test_user_id, sample_metadata)

        # Read
        result = await cosmos_storage.get_session_metadata(test_user_id, test_session_id)
        assert result is not None
        assert result["session_id"] == test_session_id
        assert result["project_slug"] == test_project_slug
        assert result["name"] == "Test Session"

        # Update
        sample_metadata["name"] = "Updated Test Session"
        sample_metadata["turn_count"] = 5
        await cosmos_storage.upsert_session_metadata(test_user_id, sample_metadata)

        result = await cosmos_storage.get_session_metadata(test_user_id, test_session_id)
        assert result["name"] == "Updated Test Session"
        assert result["turn_count"] == 5

        # List sessions
        sessions = await cosmos_storage.list_sessions(test_user_id)
        session_ids = [s["session_id"] for s in sessions]
        assert test_session_id in session_ids

        # List by project
        sessions = await cosmos_storage.list_sessions(test_user_id, project_slug=test_project_slug)
        assert len(sessions) >= 1
        assert all(s["project_slug"] == test_project_slug for s in sessions)

        # Delete
        deleted = await cosmos_storage.delete_session(
            test_user_id, test_project_slug, test_session_id
        )
        assert deleted is True

        # Verify deleted
        result = await cosmos_storage.get_session_metadata(test_user_id, test_session_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_transcript_sync(
        self,
        cosmos_storage,
        test_user_id: str,
        test_session_id: str,
        test_project_slug: str,
        sample_transcript_lines: list[dict],
    ):
        """Test transcript line sync."""
        # Sync transcript lines
        synced = await cosmos_storage.sync_transcript_lines(
            user_id=test_user_id,
            project_slug=test_project_slug,
            session_id=test_session_id,
            lines=sample_transcript_lines,
            start_sequence=0,
        )
        assert synced == len(sample_transcript_lines)

        # Read back
        lines = await cosmos_storage.get_transcript_lines(
            user_id=test_user_id,
            project_slug=test_project_slug,
            session_id=test_session_id,
        )
        assert len(lines) == len(sample_transcript_lines)

        # Verify content
        for i, line in enumerate(lines):
            assert line["role"] == sample_transcript_lines[i]["role"]
            assert line["content"] == sample_transcript_lines[i]["content"]
            assert line["sequence"] == i

        # Test incremental sync
        new_messages = [
            {
                "role": "user",
                "content": "New message after initial sync",
                "timestamp": datetime.now(UTC).isoformat(),
                "turn": 3,
            }
        ]
        synced = await cosmos_storage.sync_transcript_lines(
            user_id=test_user_id,
            project_slug=test_project_slug,
            session_id=test_session_id,
            lines=new_messages,
            start_sequence=len(sample_transcript_lines),
        )
        assert synced == 1

        # Verify incremental read
        lines = await cosmos_storage.get_transcript_lines(
            user_id=test_user_id,
            project_slug=test_project_slug,
            session_id=test_session_id,
            after_sequence=len(sample_transcript_lines) - 1,
        )
        assert len(lines) == 1
        assert lines[0]["content"] == "New message after initial sync"

        # Get count
        count = await cosmos_storage.get_transcript_count(
            user_id=test_user_id,
            project_slug=test_project_slug,
            session_id=test_session_id,
        )
        assert count == len(sample_transcript_lines) + 1

        # Cleanup
        await cosmos_storage.delete_session(test_user_id, test_project_slug, test_session_id)

    @pytest.mark.asyncio
    async def test_event_sync(
        self,
        cosmos_storage,
        test_user_id: str,
        test_session_id: str,
        test_project_slug: str,
        sample_event_lines: list[dict],
    ):
        """Test event line sync."""
        # Sync events
        synced = await cosmos_storage.sync_event_lines(
            user_id=test_user_id,
            project_slug=test_project_slug,
            session_id=test_session_id,
            lines=sample_event_lines,
            start_sequence=0,
        )
        assert synced == len(sample_event_lines)

        # Read back summaries
        events = await cosmos_storage.get_event_lines(
            user_id=test_user_id,
            project_slug=test_project_slug,
            session_id=test_session_id,
        )
        assert len(events) == len(sample_event_lines)

        # Verify event types
        event_types = [e["event"] for e in events]
        assert "session:start" in event_types
        assert "llm:request" in event_types
        assert "llm:response" in event_types

        # Get count
        count = await cosmos_storage.get_event_count(
            user_id=test_user_id,
            project_slug=test_project_slug,
            session_id=test_session_id,
        )
        assert count == len(sample_event_lines)

        # Cleanup
        await cosmos_storage.delete_session(test_user_id, test_project_slug, test_session_id)

    @pytest.mark.asyncio
    async def test_large_event_truncation(
        self,
        cosmos_storage,
        test_user_id: str,
        test_session_id: str,
        test_project_slug: str,
    ):
        """Test that large events are stored with truncation."""
        # Create a large event (>400KB)
        large_data = {"content": "x" * (500 * 1024)}  # 500KB
        large_event = {
            "ts": datetime.now(UTC).isoformat(),
            "lvl": "DEBUG",
            "event": "large:event",
            "turn": 1,
            "data": large_data,
        }

        # Sync
        synced = await cosmos_storage.sync_event_lines(
            user_id=test_user_id,
            project_slug=test_project_slug,
            session_id=test_session_id,
            lines=[large_event],
            start_sequence=0,
        )
        assert synced == 1

        # Read back - should be truncated
        events = await cosmos_storage.get_event_lines(
            user_id=test_user_id,
            project_slug=test_project_slug,
            session_id=test_session_id,
        )
        assert len(events) == 1
        assert events[0]["data_truncated"] is True
        assert events[0]["data_size_bytes"] > 400 * 1024

        # Cleanup
        await cosmos_storage.delete_session(test_user_id, test_project_slug, test_session_id)


class TestHybridFileStorage:
    """Tests for HybridFileStorage."""

    @pytest.fixture
    def temp_base_path(self) -> Generator[Path, None, None]:
        """Create a temporary directory for local storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def local_session_files(
        self,
        temp_base_path: Path,
        test_project_slug: str,
        test_session_id: str,
        sample_metadata: dict,
        sample_transcript_lines: list[dict],
        sample_event_lines: list[dict],
    ) -> Path:
        """Create local session files matching CLI format."""
        session_dir = temp_base_path / test_project_slug / "sessions" / test_session_id
        session_dir.mkdir(parents=True)

        # Write metadata.json
        metadata_path = session_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(sample_metadata, f)

        # Write transcript.jsonl
        transcript_path = session_dir / "transcript.jsonl"
        with open(transcript_path, "w") as f:
            for line in sample_transcript_lines:
                f.write(json.dumps(line) + "\n")

        # Write events.jsonl
        events_path = session_dir / "events.jsonl"
        with open(events_path, "w") as f:
            for line in sample_event_lines:
                f.write(json.dumps(line) + "\n")

        return session_dir

    @pytest.fixture
    async def hybrid_storage(self, temp_base_path: Path, test_user_id: str):
        """Create and initialize HybridFileStorage."""
        from amplifier_session_storage.cosmos.file_storage import CosmosFileConfig
        from amplifier_session_storage.hybrid.file_storage import (
            HybridFileStorage,
            HybridFileStorageConfig,
        )

        config = HybridFileStorageConfig(
            base_path=temp_base_path,
            cosmos_config=CosmosFileConfig.from_env(),
            user_id=test_user_id,
            exclusion_patterns=[],
        )
        storage = HybridFileStorage(config)
        await storage.initialize()

        yield storage

        await storage.close()

    @pytest.mark.asyncio
    async def test_hybrid_initialization(self, hybrid_storage):
        """Test hybrid storage initializes with cloud available."""
        assert hybrid_storage.cloud_available is True

    @pytest.mark.asyncio
    async def test_sync_session(
        self,
        hybrid_storage,
        local_session_files: Path,
        test_project_slug: str,
        test_session_id: str,
        sample_transcript_lines: list[dict],
        sample_event_lines: list[dict],
    ):
        """Test syncing a local session to cloud."""
        # Sync the session
        result = await hybrid_storage.sync_session(test_project_slug, test_session_id)

        assert result.success is True
        assert result.sessions_synced == 1
        assert result.messages_synced == len(sample_transcript_lines)
        assert result.events_synced == len(sample_event_lines)
        assert result.skipped_excluded == 0
        assert len(result.errors) == 0

        # Verify in cloud
        cloud_session = await hybrid_storage.get_cloud_session(test_session_id)
        assert cloud_session is not None
        assert cloud_session["session_id"] == test_session_id
        assert cloud_session["project_slug"] == test_project_slug

        # Cleanup
        if hybrid_storage._cosmos:
            await hybrid_storage._cosmos.delete_session(
                hybrid_storage.config.user_id, test_project_slug, test_session_id
            )

    @pytest.mark.asyncio
    async def test_incremental_sync(
        self,
        hybrid_storage,
        local_session_files: Path,
        test_project_slug: str,
        test_session_id: str,
        sample_transcript_lines: list[dict],
    ):
        """Test that incremental sync only syncs new data."""
        # First sync
        result1 = await hybrid_storage.sync_session(test_project_slug, test_session_id)
        assert result1.success is True
        initial_messages = result1.messages_synced

        # Add more messages to local file
        transcript_path = local_session_files / "transcript.jsonl"
        with open(transcript_path, "a") as f:
            new_message = {
                "role": "user",
                "content": "This is a new message",
                "timestamp": datetime.now(UTC).isoformat(),
                "turn": 3,
            }
            f.write(json.dumps(new_message) + "\n")

        # Second sync - should only sync new message
        result2 = await hybrid_storage.sync_session(test_project_slug, test_session_id)
        assert result2.success is True
        assert result2.messages_synced == 1  # Only the new message

        # Verify sync state
        state = hybrid_storage.get_sync_state(test_project_slug, test_session_id)
        assert state["transcript"] == initial_messages + 1

        # Cleanup
        if hybrid_storage._cosmos:
            await hybrid_storage._cosmos.delete_session(
                hybrid_storage.config.user_id, test_project_slug, test_session_id
            )

    @pytest.mark.asyncio
    async def test_exclusion_patterns(
        self,
        temp_base_path: Path,
        test_user_id: str,
        test_project_slug: str,
        test_session_id: str,
        sample_metadata: dict,
    ):
        """Test that exclusion patterns prevent sync."""
        from amplifier_session_storage.cosmos.file_storage import CosmosFileConfig
        from amplifier_session_storage.hybrid.file_storage import (
            HybridFileStorage,
            HybridFileStorageConfig,
        )

        # Create storage with exclusion pattern
        config = HybridFileStorageConfig(
            base_path=temp_base_path,
            cosmos_config=CosmosFileConfig.from_env(),
            user_id=test_user_id,
            exclusion_patterns=["test-*"],  # Exclude test projects
        )

        # Verify pattern matching
        assert config.is_excluded(test_project_slug) is True
        assert config.is_excluded("prod-project") is False

        # Create local session
        session_dir = temp_base_path / test_project_slug / "sessions" / test_session_id
        session_dir.mkdir(parents=True)
        with open(session_dir / "metadata.json", "w") as f:
            json.dump(sample_metadata, f)

        # Try to sync - should be skipped
        storage = HybridFileStorage(config)
        await storage.initialize()

        result = await storage.sync_session(test_project_slug, test_session_id)
        assert result.skipped_excluded == 1
        assert result.sessions_synced == 0

        await storage.close()

    @pytest.mark.asyncio
    async def test_sync_project(
        self,
        hybrid_storage,
        temp_base_path: Path,
        test_project_slug: str,
        sample_metadata: dict,
    ):
        """Test syncing all sessions in a project."""
        # Create multiple sessions
        session_ids = []
        for i in range(3):
            session_id = f"session-{i}-{uuid.uuid4().hex[:8]}"
            session_ids.append(session_id)

            session_dir = temp_base_path / test_project_slug / "sessions" / session_id
            session_dir.mkdir(parents=True)

            metadata = {**sample_metadata, "session_id": session_id}
            with open(session_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)

        # Sync project
        result = await hybrid_storage.sync_project(test_project_slug)
        assert result.success is True
        assert result.sessions_synced == 3

        # List from cloud
        cloud_sessions = await hybrid_storage.list_cloud_sessions(project_slug=test_project_slug)
        cloud_ids = [s["session_id"] for s in cloud_sessions]
        for session_id in session_ids:
            assert session_id in cloud_ids

        # Cleanup
        if hybrid_storage._cosmos:
            for session_id in session_ids:
                await hybrid_storage._cosmos.delete_session(
                    hybrid_storage.config.user_id, test_project_slug, session_id
                )

    @pytest.mark.asyncio
    async def test_force_resync(
        self,
        hybrid_storage,
        local_session_files: Path,
        test_project_slug: str,
        test_session_id: str,
        sample_transcript_lines: list[dict],
    ):
        """Test force resync ignores sync state."""
        # First sync
        result1 = await hybrid_storage.sync_session(test_project_slug, test_session_id)
        assert result1.messages_synced == len(sample_transcript_lines)

        # Second sync without force - should sync nothing
        result2 = await hybrid_storage.sync_session(test_project_slug, test_session_id)
        assert result2.messages_synced == 0

        # Force resync - should sync everything again
        result3 = await hybrid_storage.sync_session(test_project_slug, test_session_id, force=True)
        assert result3.messages_synced == len(sample_transcript_lines)

        # Cleanup
        if hybrid_storage._cosmos:
            await hybrid_storage._cosmos.delete_session(
                hybrid_storage.config.user_id, test_project_slug, test_session_id
            )

    @pytest.mark.asyncio
    async def test_reset_sync_state(
        self,
        hybrid_storage,
        local_session_files: Path,
        test_project_slug: str,
        test_session_id: str,
    ):
        """Test resetting sync state."""
        # Sync to establish state
        await hybrid_storage.sync_session(test_project_slug, test_session_id)
        state = hybrid_storage.get_sync_state(test_project_slug, test_session_id)
        assert state["transcript"] > 0

        # Reset state
        hybrid_storage.reset_sync_state(test_project_slug, test_session_id)
        state = hybrid_storage.get_sync_state(test_project_slug, test_session_id)
        assert state["transcript"] == 0
        assert state["events"] == 0

        # Cleanup
        if hybrid_storage._cosmos:
            await hybrid_storage._cosmos.delete_session(
                hybrid_storage.config.user_id, test_project_slug, test_session_id
            )

    @pytest.mark.asyncio
    async def test_exclusion_pattern_management(self, hybrid_storage):
        """Test adding/removing exclusion patterns."""
        # Initially empty
        patterns = hybrid_storage.list_exclusion_patterns()
        assert len(patterns) == 0

        # Add patterns
        hybrid_storage.add_exclusion_pattern("temp-*")
        hybrid_storage.add_exclusion_pattern("*-scratch")

        patterns = hybrid_storage.list_exclusion_patterns()
        assert "temp-*" in patterns
        assert "*-scratch" in patterns

        # Test matching
        assert hybrid_storage.config.is_excluded("temp-project") is True
        assert hybrid_storage.config.is_excluded("my-scratch") is True
        assert hybrid_storage.config.is_excluded("production") is False

        # Remove pattern
        hybrid_storage.remove_exclusion_pattern("temp-*")
        patterns = hybrid_storage.list_exclusion_patterns()
        assert "temp-*" not in patterns
        assert hybrid_storage.config.is_excluded("temp-project") is False


class TestLocalOnlyMode:
    """Tests for local-only mode when cloud is unavailable."""

    @pytest.fixture
    async def local_only_storage(self, temp_base_path: Path, test_user_id: str):
        """Create HybridFileStorage without cloud config."""
        from amplifier_session_storage.hybrid.file_storage import (
            HybridFileStorage,
            HybridFileStorageConfig,
        )

        config = HybridFileStorageConfig(
            base_path=temp_base_path,
            cosmos_config=None,  # No cloud config
            user_id=test_user_id,
        )
        storage = HybridFileStorage(config)
        await storage.initialize()

        yield storage

        await storage.close()

    @pytest.fixture
    def temp_base_path(self) -> Generator[Path, None, None]:
        """Create a temporary directory for local storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_local_only_initialization(self, local_only_storage):
        """Test storage initializes in local-only mode."""
        assert local_only_storage.cloud_available is False
        assert local_only_storage._cosmos is None

    @pytest.mark.asyncio
    async def test_sync_fails_gracefully_local_only(
        self,
        local_only_storage,
        temp_base_path: Path,
        test_project_slug: str,
        test_session_id: str,
        sample_metadata: dict,
    ):
        """Test sync returns appropriate error in local-only mode."""
        # Create local session
        session_dir = temp_base_path / test_project_slug / "sessions" / test_session_id
        session_dir.mkdir(parents=True)
        with open(session_dir / "metadata.json", "w") as f:
            json.dump(sample_metadata, f)

        # Try to sync
        result = await local_only_storage.sync_session(test_project_slug, test_session_id)
        assert result.success is False
        assert "not available" in result.errors[0]

    @pytest.mark.asyncio
    async def test_cloud_queries_return_empty(self, local_only_storage):
        """Test cloud queries return empty results in local-only mode."""
        sessions = await local_only_storage.list_cloud_sessions()
        assert sessions == []

        session = await local_only_storage.get_cloud_session("nonexistent")
        assert session is None


# Fixtures that need to be shared
@pytest.fixture
def temp_base_path() -> Generator[Path, None, None]:
    """Create a temporary directory for local storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
