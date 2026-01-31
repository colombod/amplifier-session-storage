"""
Tests for SessionStore compatibility with amplifier-app-cli.

These tests verify that the SessionStore class is a drop-in replacement
for the amplifier-app-cli SessionStore and maintains compatibility with
the session-analyst agent expectations.
"""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from amplifier_session_storage import EventsLog, SessionStore, read_events_summary
from amplifier_session_storage.exceptions import SessionNotFoundError, SessionValidationError
from amplifier_session_storage.local.session_store import extract_session_mode, is_top_level_session


class TestIsTopLevelSession:
    """Tests for is_top_level_session helper."""

    def test_top_level_uuid(self):
        """Top-level sessions are UUIDs without underscores."""
        assert is_top_level_session("abc123-def456-789") is True

    def test_spawned_session(self):
        """Spawned sub-sessions have parent_id_agent format."""
        assert is_top_level_session("abc123_foundation:explorer") is False

    def test_multiple_underscores(self):
        """Multiple underscores still means spawned."""
        assert is_top_level_session("abc_def_ghi") is False


class TestExtractSessionMode:
    """Tests for extract_session_mode helper."""

    def test_bundle_prefix(self):
        """Extracts bundle name from bundle: prefix."""
        metadata = {"bundle": "bundle:foundation"}
        bundle, _ = extract_session_mode(metadata)
        assert bundle == "foundation"

    def test_bundle_without_prefix(self):
        """Returns bundle name as-is if no prefix."""
        metadata = {"bundle": "foundation"}
        bundle, _ = extract_session_mode(metadata)
        assert bundle == "foundation"

    def test_unknown_bundle(self):
        """Returns None for unknown bundle."""
        metadata = {"bundle": "unknown"}
        bundle, _ = extract_session_mode(metadata)
        assert bundle is None

    def test_missing_bundle(self):
        """Returns None for missing bundle key."""
        metadata = {}
        bundle, _ = extract_session_mode(metadata)
        assert bundle is None


class TestSessionStore:
    """Tests for SessionStore class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def store(self, temp_dir):
        """Create a SessionStore instance."""
        return SessionStore(base_dir=temp_dir)

    def test_init_creates_directory(self, temp_dir):
        """SessionStore creates base directory on init."""
        base_dir = temp_dir / "sessions"
        assert not base_dir.exists()
        SessionStore(base_dir=base_dir)
        assert base_dir.exists()

    def test_init_default_path(self):
        """SessionStore uses default path if not specified."""
        store = SessionStore(project_slug="test-project")
        expected = Path.home() / ".amplifier" / "projects" / "test-project" / "sessions"
        assert store.base_dir == expected

    def test_save_creates_session_dir(self, store, temp_dir):
        """Save creates session directory."""
        session_id = "test-session-123"
        store.save(session_id, [], {"session_id": session_id})
        assert (temp_dir / session_id).is_dir()

    def test_save_creates_transcript_jsonl(self, store, temp_dir):
        """Save creates transcript.jsonl file."""
        session_id = "test-session-123"
        transcript = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        store.save(session_id, transcript, {"session_id": session_id})

        transcript_file = temp_dir / session_id / "transcript.jsonl"
        assert transcript_file.exists()

        # Verify JSONL format
        with open(transcript_file) as f:
            lines = f.readlines()
        assert len(lines) == 2

        msg1 = json.loads(lines[0])
        assert msg1["role"] == "user"
        assert msg1["content"] == "Hello"
        assert "timestamp" in msg1

    def test_save_creates_metadata_json(self, store, temp_dir):
        """Save creates metadata.json file."""
        session_id = "test-session-123"
        metadata = {
            "session_id": session_id,
            "bundle": "foundation",
            "model": "claude-sonnet",
        }
        store.save(session_id, [], metadata)

        metadata_file = temp_dir / session_id / "metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            loaded = json.load(f)
        assert loaded["session_id"] == session_id
        assert loaded["bundle"] == "foundation"

    def test_save_skips_system_messages(self, store, temp_dir):
        """Save skips system and developer role messages."""
        session_id = "test-session-123"
        transcript = [
            {"role": "system", "content": "You are helpful."},
            {"role": "developer", "content": "Internal note."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        store.save(session_id, transcript, {"session_id": session_id})

        transcript_file = temp_dir / session_id / "transcript.jsonl"
        with open(transcript_file) as f:
            lines = f.readlines()

        # Only user and assistant messages
        assert len(lines) == 2
        msg1 = json.loads(lines[0])
        assert msg1["role"] == "user"

    def test_save_creates_backup(self, store, temp_dir):
        """Save creates backup file on update."""
        session_id = "test-session-123"

        # First save
        store.save(session_id, [{"role": "user", "content": "First"}], {"session_id": session_id})

        # Second save
        store.save(session_id, [{"role": "user", "content": "Second"}], {"session_id": session_id})

        # Backup should exist
        backup_file = temp_dir / session_id / "transcript.jsonl.backup"
        assert backup_file.exists()

    def test_load_returns_transcript_and_metadata(self, store):
        """Load returns tuple of transcript and metadata."""
        session_id = "test-session-123"
        transcript = [{"role": "user", "content": "Hello"}]
        metadata = {"session_id": session_id, "bundle": "test"}

        store.save(session_id, transcript, metadata)
        loaded_transcript, loaded_metadata = store.load(session_id)

        assert len(loaded_transcript) == 1
        assert loaded_transcript[0]["content"] == "Hello"
        assert loaded_metadata["bundle"] == "test"

    def test_load_nonexistent_session_raises(self, store):
        """Load raises SessionNotFoundError for nonexistent session."""
        with pytest.raises(SessionNotFoundError):
            store.load("nonexistent-session")

    def test_load_recovers_from_backup(self, store, temp_dir):
        """Load recovers from backup if main file is corrupted."""
        session_id = "test-session-123"
        # First save creates the session
        store.save(session_id, [{"role": "user", "content": "Good"}], {"session_id": session_id})
        # Second save creates a backup of the first save
        store.save(session_id, [{"role": "user", "content": "Good"}], {"session_id": session_id})

        # Verify backup exists
        backup_file = temp_dir / session_id / "transcript.jsonl.backup"
        assert backup_file.exists(), "Backup file should exist after second save"

        # Corrupt the main file
        transcript_file = temp_dir / session_id / "transcript.jsonl"
        transcript_file.write_text("not valid json{{{")

        # Should recover from backup
        loaded_transcript, _ = store.load(session_id)
        assert len(loaded_transcript) == 1
        assert loaded_transcript[0]["content"] == "Good"

    def test_exists_returns_true_for_existing(self, store):
        """Exists returns True for existing session."""
        session_id = "test-session-123"
        store.save(session_id, [], {"session_id": session_id})
        assert store.exists(session_id) is True

    def test_exists_returns_false_for_nonexistent(self, store):
        """Exists returns False for nonexistent session."""
        assert store.exists("nonexistent") is False

    def test_exists_returns_false_for_invalid_id(self, store):
        """Exists returns False for invalid session ID."""
        assert store.exists("") is False
        assert store.exists("../escape") is False

    def test_list_sessions_sorted_by_mtime(self, store, temp_dir):
        """List sessions returns IDs sorted by modification time."""
        import time

        store.save("session-1", [], {"session_id": "session-1"})
        time.sleep(0.01)
        store.save("session-2", [], {"session_id": "session-2"})
        time.sleep(0.01)
        store.save("session-3", [], {"session_id": "session-3"})

        sessions = store.list_sessions()
        # Newest first
        assert sessions[0] == "session-3"
        assert sessions[-1] == "session-1"

    def test_list_sessions_filters_spawned(self, store):
        """List sessions filters out spawned sub-sessions by default."""
        store.save("main-session", [], {"session_id": "main-session"})
        store.save("main-session_explorer", [], {"session_id": "main-session_explorer"})

        sessions = store.list_sessions(top_level_only=True)
        assert "main-session" in sessions
        assert "main-session_explorer" not in sessions

    def test_list_sessions_includes_spawned_when_requested(self, store):
        """List sessions includes spawned sessions when requested."""
        store.save("main-session", [], {"session_id": "main-session"})
        store.save("main-session_explorer", [], {"session_id": "main-session_explorer"})

        sessions = store.list_sessions(top_level_only=False)
        assert "main-session" in sessions
        assert "main-session_explorer" in sessions

    def test_find_session_exact_match(self, store):
        """Find session returns exact match."""
        session_id = "abc123-def456"
        store.save(session_id, [], {"session_id": session_id})

        found = store.find_session(session_id)
        assert found == session_id

    def test_find_session_prefix_match(self, store):
        """Find session returns match by prefix."""
        session_id = "abc123-def456-ghi789"
        store.save(session_id, [], {"session_id": session_id})

        found = store.find_session("abc123")
        assert found == session_id

    def test_find_session_not_found(self, store):
        """Find session raises SessionNotFoundError if no match."""
        store.save("xyz789", [], {"session_id": "xyz789"})

        with pytest.raises(SessionNotFoundError):
            store.find_session("abc")

    def test_find_session_ambiguous(self, store):
        """Find session raises SessionValidationError if ambiguous."""
        store.save("abc123-one", [], {"session_id": "abc123-one"})
        store.save("abc123-two", [], {"session_id": "abc123-two"})

        with pytest.raises(SessionValidationError) as exc_info:
            store.find_session("abc123")
        assert "ambiguous" in str(exc_info.value).lower()

    def test_get_metadata_without_loading_transcript(self, store):
        """Get metadata loads only metadata, not transcript."""
        session_id = "test-session"
        transcript = [{"role": "user", "content": "Hello"} for _ in range(100)]
        metadata = {"session_id": session_id, "bundle": "test"}

        store.save(session_id, transcript, metadata)

        loaded_metadata = store.get_metadata(session_id)
        assert loaded_metadata["bundle"] == "test"

    def test_update_metadata_merges_updates(self, store):
        """Update metadata merges new fields into existing."""
        session_id = "test-session"
        store.save(session_id, [], {"session_id": session_id, "bundle": "test"})

        updated = store.update_metadata(session_id, {"turn_count": 5})
        assert updated["bundle"] == "test"
        assert updated["turn_count"] == 5

    def test_save_config_snapshot(self, store, temp_dir):
        """Save config snapshot creates config.md file."""
        session_id = "test-session"
        store.save(session_id, [], {"session_id": session_id})

        config = {"bundle": "foundation", "model": "claude-sonnet"}
        store.save_config_snapshot(session_id, config)

        config_file = temp_dir / session_id / "config.md"
        assert config_file.exists()
        content = config_file.read_text()
        assert "---" in content  # YAML frontmatter
        assert "bundle: foundation" in content

    def test_delete_session_removes_directory(self, store, temp_dir):
        """Delete session removes session directory."""
        session_id = "test-session"
        store.save(session_id, [], {"session_id": session_id})
        assert (temp_dir / session_id).exists()

        result = store.delete_session(session_id)
        assert result is True
        assert not (temp_dir / session_id).exists()

    def test_delete_nonexistent_session(self, store):
        """Delete nonexistent session returns False."""
        result = store.delete_session("nonexistent")
        assert result is False

    def test_validate_session_id_rejects_path_traversal(self, store):
        """Validate session ID rejects path traversal attempts."""
        with pytest.raises(SessionValidationError):
            store.save("../escape", [], {})

        with pytest.raises(SessionValidationError):
            store.save("foo/bar", [], {})

        with pytest.raises(SessionValidationError):
            store.save("..", [], {})

    def test_get_session_dir(self, store, temp_dir):
        """Get session dir returns correct path."""
        session_id = "test-session"
        expected = temp_dir / session_id
        assert store.get_session_dir(session_id) == expected


class TestEventsLog:
    """Tests for EventsLog class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_context_manager(self, temp_dir):
        """EventsLog works as context manager."""
        session_dir = temp_dir / "test-session"
        session_dir.mkdir()

        with EventsLog(session_dir) as log:
            log.append({"event": "test:event", "data": {"key": "value"}})

        events_file = session_dir / "events.jsonl"
        assert events_file.exists()

    def test_append_event(self, temp_dir):
        """Append adds event to log file."""
        session_dir = temp_dir / "test-session"

        with EventsLog(session_dir, "test-session") as log:
            log.append({"event": "session:start", "data": {"bundle": "test"}})

        events_file = session_dir / "events.jsonl"
        with open(events_file) as f:
            event = json.loads(f.readline())

        assert event["event"] == "session:start"
        assert event["session_id"] == "test-session"
        assert event["data"]["bundle"] == "test"
        assert "ts" in event
        assert event["lvl"] == "INFO"

    def test_append_session_start(self, temp_dir):
        """Helper method for session:start event."""
        session_dir = temp_dir / "test-session"

        with EventsLog(session_dir) as log:
            log.append_session_start(bundle="foundation", model="claude-sonnet")

        events_file = session_dir / "events.jsonl"
        with open(events_file) as f:
            event = json.loads(f.readline())

        assert event["event"] == "session:start"
        assert event["data"]["bundle"] == "foundation"
        assert event["data"]["model"] == "claude-sonnet"

    def test_append_llm_request(self, temp_dir):
        """Helper method for llm:request event."""
        session_dir = temp_dir / "test-session"

        with EventsLog(session_dir) as log:
            log.append_llm_request(model="claude-sonnet", message_count=5)

        events_file = session_dir / "events.jsonl"
        with open(events_file) as f:
            event = json.loads(f.readline())

        assert event["event"] == "llm:request"
        assert event["data"]["model"] == "claude-sonnet"
        assert event["data"]["message_count"] == 5

    def test_append_error(self, temp_dir):
        """Helper method for error events."""
        session_dir = temp_dir / "test-session"

        with EventsLog(session_dir) as log:
            log.append_error("api_error", "Connection timeout")

        events_file = session_dir / "events.jsonl"
        with open(events_file) as f:
            event = json.loads(f.readline())

        assert event["event"] == "error"
        assert event["lvl"] == "ERROR"
        assert event["data"]["error_type"] == "api_error"
        assert event["data"]["message"] == "Connection timeout"

    def test_event_count(self, temp_dir):
        """Event count tracks number of events written."""
        session_dir = temp_dir / "test-session"

        with EventsLog(session_dir) as log:
            assert log.event_count == 0
            log.append({"event": "test:one"})
            assert log.event_count == 1
            log.append({"event": "test:two"})
            assert log.event_count == 2

    def test_requires_event_key(self, temp_dir):
        """Append raises ValueError if event key missing."""
        session_dir = temp_dir / "test-session"

        with EventsLog(session_dir) as log:
            with pytest.raises(ValueError):
                log.append({"data": "no event key"})


class TestReadEventsSummary:
    """Tests for read_events_summary function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_nonexistent_file(self, temp_dir):
        """Returns empty summary for nonexistent file."""
        summary = read_events_summary(temp_dir / "nonexistent.jsonl")
        assert summary["total_events"] == 0
        assert summary["event_types"] == {}

    def test_counts_events(self, temp_dir):
        """Counts total events and by type."""
        events_file = temp_dir / "events.jsonl"
        with open(events_file, "w") as f:
            f.write(json.dumps({"event": "session:start", "ts": "2025-01-01T00:00:00Z"}) + "\n")
            f.write(json.dumps({"event": "llm:request", "ts": "2025-01-01T00:00:01Z"}) + "\n")
            f.write(json.dumps({"event": "llm:response", "ts": "2025-01-01T00:00:02Z"}) + "\n")
            f.write(json.dumps({"event": "llm:request", "ts": "2025-01-01T00:00:03Z"}) + "\n")

        summary = read_events_summary(events_file)
        assert summary["total_events"] == 4
        assert summary["event_types"]["session:start"] == 1
        assert summary["event_types"]["llm:request"] == 2
        assert summary["event_types"]["llm:response"] == 1

    def test_tracks_timestamps(self, temp_dir):
        """Tracks first and last timestamps."""
        events_file = temp_dir / "events.jsonl"
        with open(events_file, "w") as f:
            f.write(json.dumps({"event": "first", "ts": "2025-01-01T00:00:00Z"}) + "\n")
            f.write(json.dumps({"event": "middle", "ts": "2025-01-01T00:00:30Z"}) + "\n")
            f.write(json.dumps({"event": "last", "ts": "2025-01-01T00:01:00Z"}) + "\n")

        summary = read_events_summary(events_file)
        assert summary["first_timestamp"] == "2025-01-01T00:00:00Z"
        assert summary["last_timestamp"] == "2025-01-01T00:01:00Z"

    def test_collects_errors(self, temp_dir):
        """Collects error event summaries."""
        events_file = temp_dir / "events.jsonl"
        with open(events_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "event": "error",
                        "lvl": "ERROR",
                        "ts": "2025-01-01T00:00:00Z",
                        "data": {"message": "Test error message"},
                    }
                )
                + "\n"
            )

        summary = read_events_summary(events_file)
        assert len(summary["errors"]) == 1
        assert summary["errors"][0]["message"] == "Test error message"


class TestSessionAnalystCompatibility:
    """
    Tests verifying session-analyst agent compatibility.

    The session-analyst agent expects:
    - Sessions at ~/.amplifier/projects/{project}/sessions/{session_id}/
    - metadata.json with session_id, created, bundle, model, turn_count
    - transcript.jsonl with role, content, timestamp per line
    - events.jsonl with ts, event, session_id, data per line
    """

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def store(self, temp_dir):
        """Create a SessionStore instance."""
        return SessionStore(base_dir=temp_dir)

    def test_file_structure_matches_expected(self, store, temp_dir):
        """Verify file structure matches session-analyst expectations."""
        session_id = "test-session-uuid"
        transcript = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        metadata = {
            "session_id": session_id,
            "created": datetime.now(UTC).isoformat(),
            "bundle": "bundle:foundation",
            "model": "claude-sonnet-4-20250514",
            "turn_count": 1,
        }

        store.save(session_id, transcript, metadata)

        # Verify directory structure
        session_dir = temp_dir / session_id
        assert session_dir.is_dir()
        assert (session_dir / "metadata.json").exists()
        assert (session_dir / "transcript.jsonl").exists()

    def test_metadata_has_required_fields(self, store, temp_dir):
        """Verify metadata.json has fields session-analyst expects."""
        session_id = "test-session"
        metadata = {
            "session_id": session_id,
            "created": "2025-01-31T12:00:00.000Z",
            "bundle": "bundle:foundation",
            "model": "claude-sonnet",
            "turn_count": 5,
        }

        store.save(session_id, [], metadata)

        with open(temp_dir / session_id / "metadata.json") as f:
            loaded = json.load(f)

        # session-analyst uses these fields for filtering
        assert "session_id" in loaded
        assert "created" in loaded
        assert "bundle" in loaded

    def test_transcript_jsonl_format(self, store, temp_dir):
        """Verify transcript.jsonl format matches expectations."""
        session_id = "test-session"
        transcript = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]

        store.save(session_id, transcript, {"session_id": session_id})

        with open(temp_dir / session_id / "transcript.jsonl") as f:
            lines = f.readlines()

        for line in lines:
            msg = json.loads(line)
            # session-analyst expects these fields
            assert "role" in msg
            assert "content" in msg
            assert "timestamp" in msg

    def test_events_jsonl_format(self, temp_dir):
        """Verify events.jsonl format matches expectations."""
        session_dir = temp_dir / "test-session"
        session_dir.mkdir()

        with EventsLog(session_dir, "test-session") as log:
            log.append_session_start(bundle="foundation")
            log.append_llm_request(model="claude-sonnet", message_count=2)

        with open(session_dir / "events.jsonl") as f:
            for line in f:
                event = json.loads(line)
                # session-analyst expects these fields
                assert "ts" in event
                assert "event" in event
                assert "session_id" in event

    def test_can_search_by_project_path(self, temp_dir):
        """Sessions can be found via project path pattern."""
        # Simulate ~/.amplifier/projects/myproject/sessions/
        project_dir = temp_dir / "projects" / "myproject" / "sessions"
        store = SessionStore(base_dir=project_dir)

        store.save("session-1", [], {"session_id": "session-1"})
        store.save("session-2", [], {"session_id": "session-2"})

        # session-analyst uses glob patterns like:
        # ~/.amplifier/projects/*/sessions/*
        sessions = list(project_dir.glob("*"))
        assert len(sessions) == 2
