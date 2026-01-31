"""Tests for migration module."""

from __future__ import annotations

import json
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from amplifier_session_storage.migration import (
    MigrationResult,
    MigrationStatus,
    SessionMigrator,
    SessionSource,
)
from amplifier_session_storage.storage import LocalBlockStorage, StorageConfig


class TestSessionSource:
    """Tests for SessionSource."""

    def test_discover_files(self, tmp_path: Path) -> None:
        """Test that files are discovered on init."""
        session_dir = tmp_path / "project" / "sess-123"
        session_dir.mkdir(parents=True)

        # Create mock session files
        events_file = session_dir / "events.jsonl"
        events_file.write_text('{"type": "test"}\n')

        transcript_file = session_dir / "transcript.jsonl"
        transcript_file.write_text('{"role": "user", "content": "Hello"}\n')

        source = SessionSource(
            session_id="sess-123",
            path=session_dir,
            project_slug="project",
        )

        assert source.events_file == events_file
        assert source.transcript_file == transcript_file
        assert source.total_size_bytes > 0

    def test_missing_files(self, tmp_path: Path) -> None:
        """Test handling of missing files."""
        session_dir = tmp_path / "project" / "sess-123"
        session_dir.mkdir(parents=True)

        source = SessionSource(
            session_id="sess-123",
            path=session_dir,
            project_slug="project",
        )

        assert source.events_file is None
        assert source.transcript_file is None
        assert source.total_size_bytes == 0


class TestMigrationResult:
    """Tests for MigrationResult."""

    def test_duration_calculation(self) -> None:
        """Test duration calculation."""
        from datetime import datetime

        result = MigrationResult(
            session_id="sess-123",
            status=MigrationStatus.COMPLETED,
        )
        result.started_at = datetime(2024, 1, 1, 12, 0, 0)
        result.completed_at = datetime(2024, 1, 1, 12, 0, 5)

        assert result.duration_seconds == 5.0

    def test_duration_none_when_incomplete(self) -> None:
        """Test duration is None when not completed."""
        result = MigrationResult(
            session_id="sess-123",
            status=MigrationStatus.IN_PROGRESS,
        )

        assert result.duration_seconds is None

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        result = MigrationResult(
            session_id="sess-123",
            status=MigrationStatus.COMPLETED,
            blocks_created=10,
            messages_migrated=5,
            events_migrated=3,
        )

        data = result.to_dict()

        assert data["session_id"] == "sess-123"
        assert data["status"] == "completed"
        assert data["blocks_created"] == 10


class TestSessionMigrator:
    """Tests for SessionMigrator."""

    @pytest.fixture
    async def storage_dir(self) -> AsyncIterator[Path]:
        """Create a temporary directory for storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    async def legacy_sessions_dir(self) -> AsyncIterator[Path]:
        """Create a temporary directory with legacy sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    async def storage(self, storage_dir: Path) -> AsyncIterator[LocalBlockStorage]:
        """Create a local storage instance."""
        config = StorageConfig(
            user_id="test-user",
            local_path=str(storage_dir),
        )
        storage = LocalBlockStorage(config)
        yield storage
        await storage.close()

    @pytest.fixture
    def migrator(self, storage: LocalBlockStorage) -> SessionMigrator:
        """Create a migrator instance."""
        return SessionMigrator(
            storage=storage,
            user_id="test-user",
            device_id="test-device",
        )

    def _create_legacy_session(
        self,
        base_path: Path,
        project_slug: str,
        session_id: str,
        messages: list[dict] | None = None,
        events: list[dict] | None = None,
        state: dict | None = None,
    ) -> Path:
        """Helper to create a legacy session directory."""
        session_dir = base_path / project_slug / session_id
        session_dir.mkdir(parents=True)

        # Create transcript.jsonl
        if messages:
            transcript_file = session_dir / "transcript.jsonl"
            with open(transcript_file, "w") as f:
                for msg in messages:
                    f.write(json.dumps(msg) + "\n")

        # Create events.jsonl
        if events:
            events_file = session_dir / "events.jsonl"
            with open(events_file, "w") as f:
                for event in events:
                    f.write(json.dumps(event) + "\n")

        # Create state.json
        if state:
            state_file = session_dir / "state.json"
            with open(state_file, "w") as f:
                json.dump(state, f)

        return session_dir

    async def test_discover_sessions(
        self,
        migrator: SessionMigrator,
        legacy_sessions_dir: Path,
    ) -> None:
        """Test discovering legacy sessions."""
        # Create some legacy sessions
        self._create_legacy_session(
            legacy_sessions_dir,
            "project-a",
            "sess-1",
            messages=[{"role": "user", "content": "Hello"}],
        )
        self._create_legacy_session(
            legacy_sessions_dir,
            "project-a",
            "sess-2",
            messages=[{"role": "user", "content": "Hi"}],
        )
        self._create_legacy_session(
            legacy_sessions_dir,
            "project-b",
            "sess-3",
            events=[{"type": "test"}],
        )

        sources = await migrator.discover_sessions(legacy_sessions_dir)

        assert len(sources) == 3
        session_ids = [s.session_id for s in sources]
        assert "sess-1" in session_ids
        assert "sess-2" in session_ids
        assert "sess-3" in session_ids

    async def test_discover_sessions_with_filter(
        self,
        migrator: SessionMigrator,
        legacy_sessions_dir: Path,
    ) -> None:
        """Test discovering sessions with project filter."""
        self._create_legacy_session(
            legacy_sessions_dir,
            "project-a",
            "sess-1",
            messages=[{"role": "user", "content": "Hello"}],
        )
        self._create_legacy_session(
            legacy_sessions_dir,
            "project-b",
            "sess-2",
            messages=[{"role": "user", "content": "Hi"}],
        )

        sources = await migrator.discover_sessions(
            legacy_sessions_dir,
            project_filter="project-a",
        )

        assert len(sources) == 1
        assert sources[0].session_id == "sess-1"
        assert sources[0].project_slug == "project-a"

    async def test_migrate_session_with_messages(
        self,
        migrator: SessionMigrator,
        storage: LocalBlockStorage,
        legacy_sessions_dir: Path,
    ) -> None:
        """Test migrating a session with messages."""
        self._create_legacy_session(
            legacy_sessions_dir,
            "test-project",
            "sess-test",
            messages=[
                {"role": "user", "content": "Hello", "turn": 1},
                {"role": "assistant", "content": "Hi there!", "turn": 1},
                {"role": "user", "content": "How are you?", "turn": 2},
            ],
            state={
                "name": "Test Session",
                "created": "2024-01-01T12:00:00",
            },
        )

        source = SessionSource(
            session_id="sess-test",
            path=legacy_sessions_dir / "test-project" / "sess-test",
            project_slug="test-project",
        )

        result = await migrator.migrate_session(source)

        assert result.status == MigrationStatus.COMPLETED
        assert result.messages_migrated == 3
        assert result.blocks_created >= 4  # 1 session + 3 messages

        # Verify blocks were written
        blocks = await storage.read_blocks("sess-test")
        assert len(blocks) >= 4

    async def test_migrate_session_with_events(
        self,
        migrator: SessionMigrator,
        storage: LocalBlockStorage,
        legacy_sessions_dir: Path,
    ) -> None:
        """Test migrating a session with events."""
        self._create_legacy_session(
            legacy_sessions_dir,
            "test-project",
            "sess-test",
            events=[
                {"id": "evt-1", "type": "llm:request", "turn": 1},
                {"id": "evt-2", "type": "llm:response", "turn": 1},
            ],
        )

        source = SessionSource(
            session_id="sess-test",
            path=legacy_sessions_dir / "test-project" / "sess-test",
            project_slug="test-project",
        )

        result = await migrator.migrate_session(source)

        assert result.status == MigrationStatus.COMPLETED
        assert result.events_migrated == 2

    async def test_migrate_session_skip_existing(
        self,
        migrator: SessionMigrator,
        storage: LocalBlockStorage,
        legacy_sessions_dir: Path,
    ) -> None:
        """Test that existing sessions are skipped."""
        # First migration
        self._create_legacy_session(
            legacy_sessions_dir,
            "test-project",
            "sess-test",
            messages=[{"role": "user", "content": "Hello", "turn": 1}],
        )

        source = SessionSource(
            session_id="sess-test",
            path=legacy_sessions_dir / "test-project" / "sess-test",
            project_slug="test-project",
        )

        result1 = await migrator.migrate_session(source)
        assert result1.status == MigrationStatus.COMPLETED

        # Second migration should be skipped
        result2 = await migrator.migrate_session(source, skip_if_exists=True)
        assert result2.status == MigrationStatus.SKIPPED

    async def test_migrate_session_force_overwrite(
        self,
        migrator: SessionMigrator,
        storage: LocalBlockStorage,
        legacy_sessions_dir: Path,
    ) -> None:
        """Test migrating even if session exists."""
        self._create_legacy_session(
            legacy_sessions_dir,
            "test-project",
            "sess-test",
            messages=[{"role": "user", "content": "Hello", "turn": 1}],
        )

        source = SessionSource(
            session_id="sess-test",
            path=legacy_sessions_dir / "test-project" / "sess-test",
            project_slug="test-project",
        )

        # First migration
        result1 = await migrator.migrate_session(source)
        assert result1.status == MigrationStatus.COMPLETED

        # Second migration with force
        result2 = await migrator.migrate_session(source, skip_if_exists=False)
        assert result2.status == MigrationStatus.COMPLETED

    async def test_migrate_batch(
        self,
        migrator: SessionMigrator,
        storage: LocalBlockStorage,
        legacy_sessions_dir: Path,
    ) -> None:
        """Test batch migration."""
        # Create multiple sessions
        for i in range(3):
            self._create_legacy_session(
                legacy_sessions_dir,
                "test-project",
                f"sess-{i}",
                messages=[{"role": "user", "content": f"Hello {i}", "turn": 1}],
            )

        sources = await migrator.discover_sessions(legacy_sessions_dir)
        batch = await migrator.migrate_batch(sources)

        assert batch.total_sessions == 3
        assert batch.completed == 3
        assert batch.failed == 0
        assert batch.success_rate == 100.0

    async def test_migrate_batch_with_progress(
        self,
        migrator: SessionMigrator,
        legacy_sessions_dir: Path,
    ) -> None:
        """Test batch migration with progress callback."""
        for i in range(2):
            self._create_legacy_session(
                legacy_sessions_dir,
                "test-project",
                f"sess-{i}",
                messages=[{"role": "user", "content": f"Hello {i}", "turn": 1}],
            )

        progress_calls: list[tuple[MigrationResult, int, int]] = []

        def on_progress(result: MigrationResult, index: int, total: int) -> None:
            progress_calls.append((result, index, total))

        sources = await migrator.discover_sessions(legacy_sessions_dir)
        await migrator.migrate_batch(sources, on_progress=on_progress)

        assert len(progress_calls) == 2
        assert progress_calls[0][1] == 1  # First call: index 1
        assert progress_calls[1][1] == 2  # Second call: index 2
        assert progress_calls[0][2] == 2  # Total is 2
        assert progress_calls[1][2] == 2

    async def test_migrate_empty_session(
        self,
        migrator: SessionMigrator,
        storage: LocalBlockStorage,
        legacy_sessions_dir: Path,
    ) -> None:
        """Test migrating a session with no messages or events."""
        session_dir = legacy_sessions_dir / "test-project" / "sess-empty"
        session_dir.mkdir(parents=True)

        # Create empty files
        (session_dir / "events.jsonl").write_text("")
        (session_dir / "transcript.jsonl").write_text("")

        source = SessionSource(
            session_id="sess-empty",
            path=session_dir,
            project_slug="test-project",
        )

        result = await migrator.migrate_session(source)

        assert result.status == MigrationStatus.COMPLETED
        assert result.messages_migrated == 0
        assert result.events_migrated == 0
        assert result.blocks_created == 1  # Just the session block

    async def test_migrate_session_with_metadata(
        self,
        migrator: SessionMigrator,
        storage: LocalBlockStorage,
        legacy_sessions_dir: Path,
    ) -> None:
        """Test that session metadata is preserved."""
        self._create_legacy_session(
            legacy_sessions_dir,
            "test-project",
            "sess-test",
            messages=[{"role": "user", "content": "Hello", "turn": 1}],
            state={
                "name": "My Test Session",
                "description": "A test session",
                "bundle": "test-bundle",
                "model": "test-model",
                "visibility": "team",
                "org_id": "org-123",
                "team_ids": ["team-a", "team-b"],
                "tags": ["important", "test"],
            },
        )

        source = SessionSource(
            session_id="sess-test",
            path=legacy_sessions_dir / "test-project" / "sess-test",
            project_slug="test-project",
        )

        await migrator.migrate_session(source)

        # Check the SESSION_CREATED block
        blocks = await storage.read_blocks("sess-test")
        session_block = blocks[0]

        assert session_block.data["name"] == "My Test Session"
        assert session_block.data["description"] == "A test session"
        assert session_block.data["bundle"] == "test-bundle"
        assert session_block.data["visibility"] == "team"
        assert session_block.data["org_id"] == "org-123"
        assert session_block.data["team_ids"] == ["team-a", "team-b"]
