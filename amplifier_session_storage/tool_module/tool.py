"""
Amplifier tool module implementation for session operations.

This module provides the `session` tool that can be registered with Amplifier
agents for safe, structured access to session data.

Implements the Amplifier Tool protocol:
- name: str property
- description: str property
- async execute(input: dict[str, Any]) -> ToolResult
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..tools import SessionTool, SessionToolConfig


@dataclass
class ToolResult:
    """Result from tool execution, matching Amplifier's ToolResult contract."""

    success: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {"success": self.success}
        if self.success:
            result["output"] = self.output
        if self.error:
            result["error"] = self.error
        return result


class SessionToolModule:
    """
    Amplifier tool module that wraps SessionTool.

    This class provides the interface expected by Amplifier's tool loading
    system, exposing session operations as tool functions.
    """

    def __init__(
        self,
        base_dir: Path | str | None = None,
        project_slug: str = "default",
        enable_cloud: bool = False,
        max_results: int = 50,
        max_excerpt_length: int = 500,
    ):
        """Initialize the session tool module.

        Args:
            base_dir: Base directory for local session storage.
            project_slug: Project slug for session organization.
            enable_cloud: Whether to include cloud sessions.
            max_results: Default maximum results for operations.
            max_excerpt_length: Maximum length of excerpts.
        """
        config = SessionToolConfig(
            base_dir=Path(base_dir) if base_dir else None,
            project_slug=project_slug,
            enable_cloud=enable_cloud,
            max_results=max_results,
            max_excerpt_length=max_excerpt_length,
        )
        self._tool = SessionTool(config)

    @property
    def name(self) -> str:
        """Tool name for Amplifier registration."""
        return "session"

    @property
    def description(self) -> str:
        """Tool description for Amplifier."""
        return (
            "Safe, structured access to Amplifier session data. "
            "Use for analyzing, searching, and managing sessions without "
            "risk of context overflow from large event payloads."
        )

    @property
    def schema(self) -> dict[str, Any]:
        """JSON Schema for the tool's input parameters."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "Operation to perform",
                    "enum": [
                        "list_sessions",
                        "get_session",
                        "search_sessions",
                        "get_events",
                        "analyze_events",
                        "rewind_session",
                    ],
                },
                # list_sessions parameters
                "project": {
                    "type": "string",
                    "description": "Filter by project slug",
                },
                "date_range": {
                    "type": "string",
                    "description": "Filter by date range: 'today', 'last_week', or 'YYYY-MM-DD:YYYY-MM-DD'",
                },
                "top_level_only": {
                    "type": "boolean",
                    "description": "Exclude spawned sub-sessions (default: true)",
                    "default": True,
                },
                # Common parameters
                "session_id": {
                    "type": "string",
                    "description": "Session ID (full or unique partial)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results",
                },
                # get_session parameters
                "include_transcript": {
                    "type": "boolean",
                    "description": "Include conversation messages",
                    "default": False,
                },
                "include_events_summary": {
                    "type": "boolean",
                    "description": "Include event statistics",
                    "default": False,
                },
                # search_sessions parameters
                "query": {
                    "type": "string",
                    "description": "Search term (case-insensitive)",
                },
                "scope": {
                    "type": "string",
                    "description": "Where to search",
                    "enum": ["metadata", "transcript", "all"],
                    "default": "all",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Context lines around transcript matches",
                    "default": 2,
                },
                # get_events parameters
                "event_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by event types (e.g., ['llm:request', 'tool:call'])",
                },
                "errors_only": {
                    "type": "boolean",
                    "description": "Only return error events",
                    "default": False,
                },
                "offset": {
                    "type": "integer",
                    "description": "Skip first N results (for pagination)",
                    "default": 0,
                },
                # analyze_events parameters
                "analysis_type": {
                    "type": "string",
                    "description": "Type of analysis",
                    "enum": ["summary", "errors", "timeline", "usage"],
                    "default": "summary",
                },
                # rewind_session parameters
                "to_turn": {
                    "type": "integer",
                    "description": "Rewind to after this turn number",
                },
                "to_message": {
                    "type": "integer",
                    "description": "Rewind to after this message index",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Preview only, don't execute (default: true)",
                    "default": True,
                },
            },
            "required": ["operation"],
        }

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Execute a session tool operation.

        Implements the Amplifier Tool protocol.

        Args:
            input: Operation parameters from the tool call.

        Returns:
            ToolResult with success status and output/error.
        """
        operation = input.get("operation")

        try:
            if operation == "list_sessions":
                output = self._list_sessions(input)
            elif operation == "get_session":
                output = self._get_session(input)
            elif operation == "search_sessions":
                output = self._search_sessions(input)
            elif operation == "get_events":
                output = self._get_events(input)
            elif operation == "analyze_events":
                output = self._analyze_events(input)
            elif operation == "rewind_session":
                output = self._rewind_session(input)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown operation: {operation}",
                    output={
                        "available_operations": [
                            "list_sessions",
                            "get_session",
                            "search_sessions",
                            "get_events",
                            "analyze_events",
                            "rewind_session",
                        ]
                    },
                )

            # Check if operation returned an error
            if "error" in output:
                return ToolResult(success=False, error=output["error"], output=output)

            return ToolResult(success=True, output=output)

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _list_sessions(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute list_sessions operation."""
        sessions = self._tool.list_sessions(
            project=params.get("project"),
            date_range=params.get("date_range"),
            top_level_only=params.get("top_level_only", True),
            limit=params.get("limit"),
        )

        return {
            "operation": "list_sessions",
            "count": len(sessions),
            "sessions": [asdict(s) for s in sessions],
        }

    def _get_session(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute get_session operation."""
        session_id = params.get("session_id")
        if not session_id:
            return {"error": "session_id is required"}

        try:
            result = self._tool.get_session(
                session_id=session_id,
                include_transcript=params.get("include_transcript", False),
                include_events_summary=params.get("include_events_summary", False),
            )
            result["operation"] = "get_session"
            return result
        except Exception as e:
            return {"error": str(e), "operation": "get_session"}

    def _search_sessions(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute search_sessions operation."""
        query = params.get("query")
        if not query:
            return {"error": "query is required"}

        matches = self._tool.search_sessions(
            query=query,
            scope=params.get("scope", "all"),
            project=params.get("project"),
            limit=params.get("limit"),
            context_lines=params.get("context_lines", 2),
        )

        return {
            "operation": "search_sessions",
            "query": query,
            "count": len(matches),
            "matches": [asdict(m) for m in matches],
        }

    def _get_events(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute get_events operation."""
        session_id = params.get("session_id")
        if not session_id:
            return {"error": "session_id is required"}

        try:
            result = self._tool.get_events(
                session_id=session_id,
                event_types=params.get("event_types"),
                errors_only=params.get("errors_only", False),
                limit=params.get("limit", 100),
                offset=params.get("offset", 0),
            )
            result["operation"] = "get_events"
            return result
        except Exception as e:
            return {"error": str(e), "operation": "get_events"}

    def _analyze_events(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute analyze_events operation."""
        session_id = params.get("session_id")
        if not session_id:
            return {"error": "session_id is required"}

        try:
            analysis = self._tool.analyze_events(
                session_id=session_id,
                analysis_type=params.get("analysis_type", "summary"),
            )
            result = asdict(analysis)
            result["operation"] = "analyze_events"
            return result
        except Exception as e:
            return {"error": str(e), "operation": "analyze_events"}

    def _rewind_session(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute rewind_session operation."""
        session_id = params.get("session_id")
        if not session_id:
            return {"error": "session_id is required"}

        if params.get("to_turn") is None and params.get("to_message") is None:
            return {"error": "Either to_turn or to_message is required"}

        try:
            preview = self._tool.rewind_session(
                session_id=session_id,
                to_turn=params.get("to_turn"),
                to_message=params.get("to_message"),
                dry_run=params.get("dry_run", True),
            )
            result = asdict(preview)
            result["operation"] = "rewind_session"
            return result
        except Exception as e:
            return {"error": str(e), "operation": "rewind_session"}


def create_tool(**config: Any) -> SessionToolModule:
    """Factory function for creating the session tool.

    This function is called by Amplifier's tool loading system.

    Args:
        **config: Tool configuration from the behavior definition.

    Returns:
        Configured SessionToolModule instance.
    """
    return SessionToolModule(
        base_dir=config.get("base_dir"),
        project_slug=config.get("project_slug", "default"),
        enable_cloud=config.get("enable_cloud", False),
        max_results=config.get("max_results", 50),
        max_excerpt_length=config.get("max_excerpt_length", 500),
    )


def mount(coordinator: Any = None, config: dict[str, Any] | None = None) -> SessionToolModule:
    """Standard Amplifier module entry point.

    This function follows the Amplifier module protocol for tool loading.

    Args:
        coordinator: Amplifier coordinator instance (unused, for protocol compliance).
        config: Tool configuration dictionary.

    Returns:
        Configured SessionToolModule instance.
    """
    config = config or {}
    return create_tool(**config)
