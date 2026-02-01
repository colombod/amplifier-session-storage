"""Tests for the Amplifier tool module wrapper."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from amplifier_session_storage.local import SessionStore
from amplifier_session_storage.tool_module import SessionToolModule, ToolResult, create_tool, mount


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def tool_module(temp_dir):
    """Create a SessionToolModule for tests."""
    return SessionToolModule(
        base_dir=temp_dir,
        project_slug="test-project",
    )


@pytest.fixture
def populated_store(temp_dir):
    """Create a store with sample sessions and return the tool module."""
    store = SessionStore(base_dir=temp_dir, project_slug="test-project")

    # Create sample sessions
    for i in range(2):
        session_id = f"session-{i:03d}"
        transcript = [
            {"role": "user", "content": f"Hello {i}", "turn": 1},
            {"role": "assistant", "content": f"Hi there {i}!", "turn": 1},
        ]
        metadata = {
            "session_id": session_id,
            "created": datetime.now(UTC).isoformat(),
            "bundle": "bundle:test",
            "model": "test-model",
            "turn_count": 1,
            "name": f"Test Session {i}",
        }
        store.save(session_id, transcript, metadata)

        # Create events
        events_file = store.get_session_dir(session_id) / "events.jsonl"
        with open(events_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "event": "llm:request",
                        "lvl": "INFO",
                        "data": {"turn": 1},
                    }
                )
                + "\n"
            )

    return SessionToolModule(base_dir=temp_dir, project_slug="test-project")


class TestToolModuleInterface:
    """Tests for the tool module interface (what Amplifier expects)."""

    def test_has_name(self, tool_module):
        """Tool has a name property."""
        assert tool_module.name == "session"

    def test_has_description(self, tool_module):
        """Tool has a description property."""
        assert isinstance(tool_module.description, str)
        assert len(tool_module.description) > 0

    def test_has_schema(self, tool_module):
        """Tool has a JSON schema property."""
        schema = tool_module.schema
        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "operation" in schema["properties"]

    def test_has_async_execute_method(self, tool_module):
        """Tool has an async execute method."""
        import asyncio

        assert callable(tool_module.execute)
        assert asyncio.iscoroutinefunction(tool_module.execute)


class TestToolModuleOperations:
    """Tests for tool module operations via execute()."""

    @pytest.mark.asyncio
    async def test_list_sessions(self, populated_store):
        """list_sessions operation works."""
        result = await populated_store.execute({"operation": "list_sessions"})

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output["operation"] == "list_sessions"
        assert result.output["count"] == 2
        assert len(result.output["sessions"]) == 2

    @pytest.mark.asyncio
    async def test_get_session(self, populated_store):
        """get_session operation works."""
        result = await populated_store.execute(
            {
                "operation": "get_session",
                "session_id": "session-000",
            }
        )

        assert result.success is True
        assert result.output["operation"] == "get_session"
        assert result.output["session_id"] == "session-000"
        assert "metadata" in result.output

    @pytest.mark.asyncio
    async def test_get_session_with_transcript(self, populated_store):
        """get_session with transcript included."""
        result = await populated_store.execute(
            {
                "operation": "get_session",
                "session_id": "session-000",
                "include_transcript": True,
            }
        )

        assert result.success is True
        assert "transcript" in result.output
        assert len(result.output["transcript"]) == 2

    @pytest.mark.asyncio
    async def test_get_session_missing_id(self, populated_store):
        """get_session requires session_id."""
        result = await populated_store.execute({"operation": "get_session"})

        assert result.success is False
        assert result.error is not None
        assert "session_id" in result.error

    @pytest.mark.asyncio
    async def test_search_sessions(self, populated_store):
        """search_sessions operation works."""
        result = await populated_store.execute(
            {
                "operation": "search_sessions",
                "query": "Hello",
            }
        )

        assert result.success is True
        assert result.output["operation"] == "search_sessions"
        assert result.output["query"] == "Hello"
        assert result.output["count"] >= 1

    @pytest.mark.asyncio
    async def test_search_sessions_missing_query(self, populated_store):
        """search_sessions requires query."""
        result = await populated_store.execute({"operation": "search_sessions"})

        assert result.success is False
        assert result.error is not None
        assert "query" in result.error

    @pytest.mark.asyncio
    async def test_get_events(self, populated_store):
        """get_events operation works."""
        result = await populated_store.execute(
            {
                "operation": "get_events",
                "session_id": "session-000",
            }
        )

        assert result.success is True
        assert result.output["operation"] == "get_events"
        assert "events" in result.output
        assert len(result.output["events"]) >= 1

    @pytest.mark.asyncio
    async def test_get_events_missing_id(self, populated_store):
        """get_events requires session_id."""
        result = await populated_store.execute({"operation": "get_events"})

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_analyze_events(self, populated_store):
        """analyze_events operation works."""
        result = await populated_store.execute(
            {
                "operation": "analyze_events",
                "session_id": "session-000",
            }
        )

        assert result.success is True
        assert result.output["operation"] == "analyze_events"
        assert "total_events" in result.output

    @pytest.mark.asyncio
    async def test_rewind_session_dry_run(self, populated_store):
        """rewind_session in dry_run mode."""
        result = await populated_store.execute(
            {
                "operation": "rewind_session",
                "session_id": "session-000",
                "to_turn": 0,
                "dry_run": True,
            }
        )

        assert result.success is True
        assert result.output["operation"] == "rewind_session"
        assert result.output["dry_run"] is True

    @pytest.mark.asyncio
    async def test_rewind_session_missing_target(self, populated_store):
        """rewind_session requires to_turn or to_message."""
        result = await populated_store.execute(
            {
                "operation": "rewind_session",
                "session_id": "session-000",
            }
        )

        assert result.success is False
        assert result.error is not None
        assert "to_turn" in result.error or "to_message" in result.error

    @pytest.mark.asyncio
    async def test_unknown_operation(self, tool_module):
        """Unknown operation returns error with available operations."""
        result = await tool_module.execute({"operation": "unknown_op"})

        assert result.success is False
        assert result.error is not None
        assert "available_operations" in result.output


class TestCreateToolFactory:
    """Tests for the create_tool factory function."""

    def test_create_tool_default_config(self, temp_dir):
        """create_tool with no config works."""
        tool = create_tool(base_dir=temp_dir)

        assert isinstance(tool, SessionToolModule)
        assert tool.name == "session"

    def test_create_tool_with_config(self, temp_dir):
        """create_tool with custom config works."""
        tool = create_tool(
            base_dir=temp_dir,
            project_slug="custom-project",
            max_results=25,
        )

        assert isinstance(tool, SessionToolModule)


class TestMountFunction:
    """Tests for the mount() entry point."""

    def test_mount_with_no_args(self):
        """mount() works with no arguments."""
        tool = mount()
        assert isinstance(tool, SessionToolModule)
        assert tool.name == "session"

    def test_mount_with_config(self, temp_dir):
        """mount() accepts config dict."""
        tool = mount(config={"base_dir": temp_dir, "project_slug": "test"})
        assert isinstance(tool, SessionToolModule)

    def test_mount_with_coordinator(self, temp_dir):
        """mount() accepts coordinator (for protocol compliance)."""
        # Coordinator is unused but accepted for protocol compliance
        tool = mount(coordinator=None, config={"base_dir": temp_dir})
        assert isinstance(tool, SessionToolModule)


class TestToolModuleSafety:
    """Safety tests for the tool module."""

    @pytest.mark.asyncio
    async def test_get_events_never_returns_data_field(self, populated_store):
        """get_events results never contain 'data' field."""
        result = await populated_store.execute(
            {
                "operation": "get_events",
                "session_id": "session-000",
            }
        )

        assert result.success is True
        for event in result.output.get("events", []):
            assert "data" not in event
            assert "content" not in event

    @pytest.mark.asyncio
    async def test_error_handling_returns_tool_result(self, populated_store):
        """Errors are returned as ToolResult, not raised."""
        result = await populated_store.execute(
            {
                "operation": "get_session",
                "session_id": "nonexistent",
            }
        )

        # Should return ToolResult with success=False, not raise
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error is not None


class TestToolResultContract:
    """Tests for the ToolResult contract."""

    def test_tool_result_success(self):
        """ToolResult for successful operations."""
        result = ToolResult(success=True, output={"data": "test"})
        assert result.success is True
        assert result.output == {"data": "test"}
        assert result.error is None

    def test_tool_result_failure(self):
        """ToolResult for failed operations."""
        result = ToolResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_tool_result_to_dict(self):
        """ToolResult converts to dict correctly."""
        result = ToolResult(success=True, output={"key": "value"})
        d = result.to_dict()
        assert d["success"] is True
        assert d["output"] == {"key": "value"}
        assert "error" not in d

    def test_tool_result_to_dict_with_error(self):
        """ToolResult with error converts to dict correctly."""
        result = ToolResult(success=False, error="Failed")
        d = result.to_dict()
        assert d["success"] is False
        assert d["error"] == "Failed"
