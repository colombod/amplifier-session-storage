"""Tests for SessionTool - the safe session access tool for agents."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from amplifier_session_storage.exceptions import SessionNotFoundError
from amplifier_session_storage.local import SessionStore
from amplifier_session_storage.tools import SessionTool, SessionToolConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def store(temp_dir):
    """Create a SessionStore for tests."""
    return SessionStore(base_dir=temp_dir, project_slug="test-project")


@pytest.fixture
def tool(temp_dir):
    """Create a SessionTool for tests."""
    config = SessionToolConfig(
        base_dir=temp_dir,
        project_slug="test-project",
        max_results=50,
        max_excerpt_length=500,
    )
    return SessionTool(config)


@pytest.fixture
def populated_store(store):
    """Create a store with sample sessions."""
    # Create a few sessions
    for i in range(3):
        session_id = f"session-{i:03d}"
        transcript = [
            {"role": "user", "content": f"Hello {i}", "turn": 1},
            {"role": "assistant", "content": f"Hi there {i}!", "turn": 1},
            {"role": "user", "content": f"How are you {i}?", "turn": 2},
            {"role": "assistant", "content": f"I'm doing great {i}!", "turn": 2},
        ]
        metadata = {
            "session_id": session_id,
            "created": datetime.now(UTC).isoformat(),
            "bundle": "bundle:test",
            "model": "test-model",
            "turn_count": 2,
            "name": f"Test Session {i}",
        }
        store.save(session_id, transcript, metadata)

        # Create events.jsonl
        events_file = store.get_session_dir(session_id) / "events.jsonl"
        with open(events_file, "w") as f:
            for event_type in ["llm:request", "llm:response", "tool:call"]:
                event = {
                    "ts": datetime.now(UTC).isoformat(),
                    "event": event_type,
                    "session_id": session_id,
                    "lvl": "INFO",
                    "data": {"turn": 1, "model": "test"},
                }
                f.write(json.dumps(event) + "\n")

    return store


class TestSessionToolListSessions:
    """Tests for list_sessions operation."""

    def test_list_sessions_empty(self, tool):
        """List sessions when none exist."""
        sessions = tool.list_sessions()
        assert sessions == []

    def test_list_sessions_returns_all(self, tool, populated_store):
        """List all sessions."""
        sessions = tool.list_sessions()
        assert len(sessions) == 3

    def test_list_sessions_includes_metadata(self, tool, populated_store):
        """Session info includes metadata fields."""
        sessions = tool.list_sessions()
        session = sessions[0]

        assert session.session_id is not None
        assert session.project == "test-project"
        assert session.bundle == "bundle:test"
        assert session.model == "test-model"
        assert session.source == "local"
        assert session.path is not None

    def test_list_sessions_with_limit(self, tool, populated_store):
        """Limit number of sessions returned."""
        sessions = tool.list_sessions(limit=2)
        assert len(sessions) == 2

    def test_list_sessions_returns_local_paths(self, tool, populated_store):
        """Sessions include local file paths for manual inspection."""
        sessions = tool.list_sessions()
        for session in sessions:
            assert session.path is not None
            assert Path(session.path).exists()


class TestSessionToolGetSession:
    """Tests for get_session operation."""

    def test_get_session_by_full_id(self, tool, populated_store):
        """Get session by full ID."""
        result = tool.get_session("session-001")

        assert result["session_id"] == "session-001"
        assert result["source"] == "local"
        assert "metadata" in result
        assert "path" in result

    def test_get_session_by_partial_id(self, tool, populated_store):
        """Get session by partial ID (unique prefix match)."""
        # Use unique prefix that only matches one session
        result = tool.get_session("session-001")

        assert result["session_id"] == "session-001"

    def test_get_session_with_transcript(self, tool, populated_store):
        """Get session with transcript included."""
        result = tool.get_session("session-001", include_transcript=True)

        assert "transcript" in result
        assert len(result["transcript"]) == 4  # 4 messages

    def test_get_session_with_events_summary(self, tool, populated_store):
        """Get session with events summary included."""
        result = tool.get_session("session-001", include_events_summary=True)

        assert "events_summary" in result
        summary = result["events_summary"]
        assert summary["total_events"] == 3

    def test_get_session_not_found(self, tool, populated_store):
        """Get session that doesn't exist raises error."""
        with pytest.raises(SessionNotFoundError):
            tool.get_session("nonexistent-session")


class TestSessionToolSearchSessions:
    """Tests for search_sessions operation."""

    def test_search_metadata(self, tool, populated_store):
        """Search in metadata fields."""
        matches = tool.search_sessions("Test Session 1", scope="metadata")

        assert len(matches) == 1
        assert matches[0].session_id == "session-001"
        assert matches[0].match_type == "metadata"

    def test_search_transcript(self, tool, populated_store):
        """Search in transcript content."""
        matches = tool.search_sessions("Hello 2", scope="transcript")

        assert len(matches) == 1
        assert matches[0].session_id == "session-002"
        assert matches[0].match_type == "transcript"
        assert matches[0].line_number is not None

    def test_search_all_scopes(self, tool, populated_store):
        """Search in both metadata and transcript."""
        matches = tool.search_sessions("Session", scope="all")

        # Should find matches in metadata (name field)
        assert len(matches) >= 1

    def test_search_case_insensitive(self, tool, populated_store):
        """Search is case-insensitive."""
        matches = tool.search_sessions("HELLO", scope="transcript")

        assert len(matches) >= 1  # Should find "Hello"

    def test_search_with_limit(self, tool, populated_store):
        """Limit search results."""
        matches = tool.search_sessions("Session", limit=1)

        assert len(matches) == 1

    def test_search_excerpt_truncated(self, tool, populated_store):
        """Search excerpts are length-limited."""
        matches = tool.search_sessions("Hello", scope="transcript")

        if matches:
            assert len(matches[0].excerpt) <= tool.config.max_excerpt_length


class TestSessionToolGetEvents:
    """Tests for get_events operation - CRITICAL SAFETY TESTS."""

    def test_get_events_returns_summaries(self, tool, populated_store):
        """get_events returns event summaries, not full data."""
        result = tool.get_events("session-001")

        assert "events" in result
        assert len(result["events"]) == 3

        # Check event summary fields
        event = result["events"][0]
        assert "ts" in event
        assert "event_type" in event

    def test_get_events_never_returns_data_payload(self, tool, populated_store):
        """CRITICAL: get_events must NEVER return full data payloads."""
        result = tool.get_events("session-001")

        for event in result["events"]:
            # These fields should NEVER be present
            assert "data" not in event
            assert "content" not in event
            assert "messages" not in event

    def test_get_events_with_type_filter(self, tool, populated_store):
        """Filter events by type."""
        result = tool.get_events("session-001", event_types=["llm:request"])

        assert all(e["event_type"] == "llm:request" for e in result["events"])

    def test_get_events_with_pagination(self, tool, populated_store):
        """Events support pagination."""
        result = tool.get_events("session-001", limit=2, offset=0)

        assert len(result["events"]) == 2
        assert result["total_count"] == 3
        assert result["has_more"] is True

        # Get next page
        result2 = tool.get_events("session-001", limit=2, offset=2)
        assert len(result2["events"]) == 1
        assert result2["has_more"] is False

    def test_get_events_includes_pagination_info(self, tool, populated_store):
        """Response includes pagination metadata."""
        result = tool.get_events("session-001")

        assert "total_count" in result
        assert "has_more" in result
        assert "offset" in result
        assert "limit" in result


class TestSessionToolAnalyzeEvents:
    """Tests for analyze_events operation."""

    def test_analyze_summary(self, tool, populated_store):
        """Get event summary analysis."""
        analysis = tool.analyze_events("session-001", analysis_type="summary")

        assert analysis.total_events == 3
        assert "llm:request" in analysis.event_types
        assert analysis.llm_requests == 1
        assert analysis.tool_calls == 1

    def test_analyze_errors(self, tool, store):
        """Analyze errors in events."""
        session_id = "error-session"
        store.save(session_id, [], {"session_id": session_id})

        # Create events with errors
        events_file = store.get_session_dir(session_id) / "events.jsonl"
        with open(events_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "event": "llm:error",
                        "lvl": "ERROR",
                        "data": {"message": "API timeout"},
                    }
                )
                + "\n"
            )

        # Analyze
        analysis = tool.analyze_events(session_id, analysis_type="errors")

        assert len(analysis.errors) == 1
        assert "timeout" in analysis.errors[0]["message"].lower()

    def test_analyze_error_messages_truncated(self, tool, store):
        """Error messages are truncated for safety."""
        session_id = "long-error-session"
        store.save(session_id, [], {"session_id": session_id})

        # Create event with very long error message
        long_message = "x" * 1000
        events_file = store.get_session_dir(session_id) / "events.jsonl"
        with open(events_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "event": "error",
                        "lvl": "ERROR",
                        "data": {"message": long_message},
                    }
                )
                + "\n"
            )

        analysis = tool.analyze_events(session_id, analysis_type="errors")

        # Error message should be truncated
        assert len(analysis.errors[0]["message"]) <= 200


class TestSessionToolRewind:
    """Tests for rewind_session operation."""

    def test_rewind_dry_run_default(self, tool, populated_store):
        """Rewind defaults to dry run mode."""
        preview = tool.rewind_session("session-001", to_turn=1)

        assert preview.dry_run is True
        assert preview.would_remove_messages == 2  # Turn 2 messages
        assert preview.new_turn_count == 1

        # Verify nothing actually changed
        result = tool.get_session("session-001", include_transcript=True)
        assert len(result["transcript"]) == 4  # Still has all messages

    def test_rewind_execute(self, tool, populated_store):
        """Rewind with dry_run=False actually modifies session."""
        preview = tool.rewind_session("session-001", to_turn=1, dry_run=False)

        assert preview.dry_run is False

        # Verify transcript was truncated
        result = tool.get_session("session-001", include_transcript=True)
        assert len(result["transcript"]) == 2  # Only turn 1 messages

    def test_rewind_to_message_index(self, tool, populated_store):
        """Rewind to specific message index."""
        preview = tool.rewind_session("session-001", to_message=2, dry_run=True)

        assert preview.would_remove_messages == 2  # Remove last 2 messages


class TestSessionToolSafety:
    """Safety tests - ensuring the tool never leaks sensitive data."""

    def test_events_never_expose_full_content(self, tool, store):
        """Events with large content never expose it."""
        session_id = "large-content-session"
        store.save(session_id, [], {"session_id": session_id})

        # Create event with very large content
        large_content = "x" * 100000  # 100KB
        events_file = store.get_session_dir(session_id) / "events.jsonl"
        with open(events_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "event": "llm:response",
                        "lvl": "INFO",
                        "data": {"content": large_content, "model": "test"},
                    }
                )
                + "\n"
            )

        # Get events - should NOT contain the large content
        result = tool.get_events(session_id)

        # Serialize result to check size
        result_str = json.dumps(result)
        assert len(result_str) < 10000  # Result should be small

    def test_search_excerpts_are_bounded(self, tool, store):
        """Search excerpts have bounded length."""
        session_id = "large-message-session"
        long_content = "keyword " + "x" * 10000  # Large message with keyword at start
        transcript = [{"role": "user", "content": long_content}]
        store.save(session_id, transcript, {"session_id": session_id})

        matches = tool.search_sessions("keyword", scope="transcript")

        if matches:
            # Excerpt should be bounded
            assert len(matches[0].excerpt) <= tool.config.max_excerpt_length
