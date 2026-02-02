"""
Incremental facets update logic.

This module provides pure functions for updating SessionFacets based on
individual events or blocks. The updater is designed to be:

1. Pure - No side effects, returns new/mutated facets
2. Incremental - Process one event at a time, O(1) per event
3. Idempotent - Same event processed twice gives same result

Usage:
    updater = FacetsUpdater()
    facets = updater.process_event(facets, event_type, event_data, timestamp)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .types import SessionFacets, WorkflowPattern

# Known file operation tools
FILE_READ_TOOLS = frozenset({"read_file", "glob", "grep"})
FILE_WRITE_TOOLS = frozenset({"write_file"})
FILE_EDIT_TOOLS = frozenset({"edit_file"})
FILE_OPERATION_TOOLS = FILE_READ_TOOLS | FILE_WRITE_TOOLS | FILE_EDIT_TOOLS

# Known delegation tools
DELEGATION_TOOLS = frozenset({"delegate", "task"})

# Known recipe tools
RECIPE_TOOLS = frozenset({"recipes"})

# Programming language detection patterns
LANGUAGE_PATTERNS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"```(?:python|py)\b", re.IGNORECASE),
    "javascript": re.compile(r"```(?:javascript|js)\b", re.IGNORECASE),
    "typescript": re.compile(r"```(?:typescript|ts)\b", re.IGNORECASE),
    "rust": re.compile(r"```rust\b", re.IGNORECASE),
    "go": re.compile(r"```go\b", re.IGNORECASE),
    "java": re.compile(r"```java\b", re.IGNORECASE),
    "csharp": re.compile(r"```(?:csharp|c#|cs)\b", re.IGNORECASE),
    "cpp": re.compile(r"```(?:cpp|c\+\+)\b", re.IGNORECASE),
    "ruby": re.compile(r"```(?:ruby|rb)\b", re.IGNORECASE),
    "shell": re.compile(r"```(?:bash|sh|shell|zsh)\b", re.IGNORECASE),
    "sql": re.compile(r"```sql\b", re.IGNORECASE),
    "yaml": re.compile(r"```(?:yaml|yml)\b", re.IGNORECASE),
    "json": re.compile(r"```json\b", re.IGNORECASE),
    "html": re.compile(r"```html\b", re.IGNORECASE),
    "css": re.compile(r"```css\b", re.IGNORECASE),
}


@dataclass
class UpdateResult:
    """Result of a facet update operation."""

    facets: SessionFacets
    fields_updated: list[str]


class FacetsUpdater:
    """Incrementally updates session facets based on events.

    This class provides methods to update facets from:
    - Individual events (tool_result, assistant_response, etc.)
    - Session creation data
    - Message data

    All methods are designed to be called incrementally as events
    are processed, building up the facets over time.
    """

    def __init__(self) -> None:
        """Initialize the updater."""
        pass

    def initialize_from_session_created(
        self,
        facets: SessionFacets,
        data: dict[str, Any],
        timestamp: datetime,
    ) -> SessionFacets:
        """Initialize facets from SESSION_CREATED block data.

        Args:
            facets: Facets to update (typically fresh instance)
            data: SessionCreatedData as dict
            timestamp: Block timestamp

        Returns:
            Updated facets with initial configuration
        """
        facets.bundle = data.get("bundle")
        facets.initial_model = data.get("model")

        # Extract provider from model name if possible
        model = data.get("model", "")
        if model:
            facets.initial_provider = self._extract_provider_from_model(model)
            if model not in facets.models_used:
                facets.models_used.append(model)
            if facets.initial_provider and facets.initial_provider not in facets.providers_used:
                facets.providers_used.append(facets.initial_provider)

        facets.first_event_at = timestamp
        facets.last_event_at = timestamp

        return facets

    def process_event(
        self,
        facets: SessionFacets,
        event_type: str,
        data: dict[str, Any],
        timestamp: datetime,
    ) -> SessionFacets:
        """Process a single event and update facets.

        This is the main entry point for incremental updates.
        Routes to specific handlers based on event_type.

        Args:
            facets: Current facets state
            event_type: Type of event (e.g., "tool_result", "assistant_response")
            data: Event data/summary
            timestamp: Event timestamp

        Returns:
            Updated facets
        """
        # Update timing
        if facets.first_event_at is None:
            facets.first_event_at = timestamp
        facets.last_event_at = timestamp

        # Route to specific handler
        match event_type:
            case "tool_result":
                self._process_tool_result(facets, data, timestamp)
            case "tool_call":
                self._process_tool_call(facets, data)
            case "assistant_response" | "assistant_message":
                self._process_assistant_response(facets, data, timestamp)
            case "user_message" | "user_prompt":
                self._process_user_message(facets, data)
            case "error" | "tool_error":
                self._process_error(facets, data)
            case "delegate_spawn" | "session_spawn":
                self._process_delegation(facets, data)
            case "recipe_start" | "recipe_execute":
                self._process_recipe_start(facets, data)
            case "recipe_complete" | "recipe_end":
                self._process_recipe_complete(facets, data)

        # Increment events processed
        facets.events_processed += 1

        return facets

    def process_message(
        self,
        facets: SessionFacets,
        role: str,
        content: Any,
        turn: int,
        timestamp: datetime,
    ) -> SessionFacets:
        """Process a transcript message and update facets.

        Args:
            facets: Current facets state
            role: Message role (user, assistant, tool, system)
            content: Message content
            turn: Turn number
            timestamp: Message timestamp

        Returns:
            Updated facets
        """
        # Update timing
        if facets.first_event_at is None:
            facets.first_event_at = timestamp
        facets.last_event_at = timestamp

        # Update turn tracking
        if turn > facets.max_turn:
            facets.max_turn = turn

        # Update message counts
        if role == "user":
            facets.user_message_count += 1
        elif role == "assistant":
            facets.assistant_message_count += 1
            # Check for code blocks in assistant content
            if isinstance(content, str):
                self._detect_code_blocks(facets, content)

        return facets

    def finalize(self, facets: SessionFacets) -> SessionFacets:
        """Finalize facets computation.

        Call this after processing all events to:
        - Compute derived fields
        - Determine workflow pattern
        - Calculate durations

        Args:
            facets: Facets to finalize

        Returns:
            Finalized facets
        """
        # Compute total tokens
        facets.total_tokens = facets.total_input_tokens + facets.total_output_tokens

        # Determine workflow pattern
        facets.workflow_pattern = self._determine_workflow_pattern(facets)

        # Compute duration if we have timing data
        if facets.first_event_at and facets.last_event_at:
            delta = facets.last_event_at - facets.first_event_at
            facets.total_duration_ms = int(delta.total_seconds() * 1000)

        # Update metadata
        facets.last_computed = datetime.now(UTC)
        facets.is_stale = False

        return facets

    # =========================================================================
    # Private Event Handlers
    # =========================================================================

    def _process_tool_result(
        self,
        facets: SessionFacets,
        data: dict[str, Any],
        timestamp: datetime,
    ) -> None:
        """Process tool_result event."""
        tool_name = data.get("tool_name") or data.get("name")

        if tool_name:
            # Track unique tools
            if tool_name not in facets.tools_used:
                facets.tools_used.append(tool_name)

            facets.tool_call_count += 1

            # Track file operations
            if tool_name in FILE_OPERATION_TOOLS:
                facets.has_file_operations = True
                if tool_name in FILE_READ_TOOLS:
                    facets.files_read += 1
                elif tool_name in FILE_WRITE_TOOLS:
                    facets.files_written += 1
                elif tool_name in FILE_EDIT_TOOLS:
                    facets.files_edited += 1

            # Track delegations from tool calls
            if tool_name in DELEGATION_TOOLS:
                facets.has_child_sessions = True
                facets.child_session_count += 1  # Count ALL delegations
                agent = data.get("agent")
                if agent and agent not in facets.agents_delegated_to:
                    facets.agents_delegated_to.append(agent)  # Track unique agents

            # Track recipe usage from tool calls
            if tool_name in RECIPE_TOOLS:
                operation = data.get("operation")
                if operation == "execute":
                    facets.has_recipes = True
                    recipe_path = data.get("recipe_path")
                    if recipe_path and recipe_path not in facets.recipe_names:
                        facets.recipe_names.append(recipe_path)

        # Track errors in tool results
        if data.get("has_error") or data.get("is_error") or data.get("error"):
            facets.has_errors = True
            facets.error_count += 1
            error_type = data.get("error_type", "tool_error")
            if error_type not in facets.error_types:
                facets.error_types.append(error_type)

    def _process_tool_call(self, facets: SessionFacets, data: dict[str, Any]) -> None:
        """Process tool_call event (before result)."""
        tool_name = data.get("tool_name") or data.get("name")

        if tool_name and tool_name not in facets.tools_used:
            facets.tools_used.append(tool_name)

    def _process_assistant_response(
        self,
        facets: SessionFacets,
        data: dict[str, Any],
        timestamp: datetime,
    ) -> None:
        """Process assistant_response event."""
        # Track model usage
        model = data.get("model")
        if model and model not in facets.models_used:
            facets.models_used.append(model)

        # Track provider usage
        provider = data.get("provider") or self._extract_provider_from_model(model or "")
        if provider and provider not in facets.providers_used:
            facets.providers_used.append(provider)

        # Track token usage
        usage = data.get("usage", {})
        if isinstance(usage, dict):
            facets.total_input_tokens += usage.get("input_tokens", 0)
            facets.total_output_tokens += usage.get("output_tokens", 0)

        # Track duration
        duration_ms = data.get("duration_ms", 0)
        facets.active_duration_ms += duration_ms

        # Check for code blocks
        if data.get("has_code_blocks"):
            facets.has_code_blocks = True

    def _process_user_message(self, facets: SessionFacets, data: dict[str, Any]) -> None:
        """Process user_message event."""
        facets.user_message_count += 1

        turn = data.get("turn", 0)
        if turn > facets.max_turn:
            facets.max_turn = turn

    def _process_error(self, facets: SessionFacets, data: dict[str, Any]) -> None:
        """Process error event."""
        facets.has_errors = True
        facets.error_count += 1

        error_type = data.get("error_type") or data.get("type") or "unknown"
        if error_type not in facets.error_types:
            facets.error_types.append(error_type)

    def _process_delegation(self, facets: SessionFacets, data: dict[str, Any]) -> None:
        """Process delegation/spawn event."""
        facets.has_child_sessions = True
        facets.child_session_count += 1

        agent = data.get("agent")
        if agent and agent not in facets.agents_delegated_to:
            facets.agents_delegated_to.append(agent)

    def _process_recipe_start(self, facets: SessionFacets, data: dict[str, Any]) -> None:
        """Process recipe start event."""
        facets.has_recipes = True

        recipe_name = data.get("recipe_name") or data.get("recipe_path")
        if recipe_name and recipe_name not in facets.recipe_names:
            facets.recipe_names.append(recipe_name)

    def _process_recipe_complete(self, facets: SessionFacets, data: dict[str, Any]) -> None:
        """Process recipe complete event."""
        # Recipe already tracked in start, just note completion
        pass

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _extract_provider_from_model(self, model: str) -> str | None:
        """Extract provider name from model string.

        Examples:
            "claude-3-opus" -> "anthropic"
            "gpt-4" -> "openai"
            "llama-2" -> "meta"
        """
        model_lower = model.lower()

        if "claude" in model_lower:
            return "anthropic"
        if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
            return "openai"
        if "llama" in model_lower:
            return "meta"
        if "gemini" in model_lower:
            return "google"
        if "mistral" in model_lower or "mixtral" in model_lower:
            return "mistral"
        if "deepseek" in model_lower:
            return "deepseek"

        return None

    def _detect_code_blocks(self, facets: SessionFacets, content: str) -> None:
        """Detect code blocks and languages in content."""
        if "```" in content:
            facets.has_code_blocks = True

            for language, pattern in LANGUAGE_PATTERNS.items():
                if pattern.search(content):
                    if language not in facets.languages_detected:
                        facets.languages_detected.append(language)

    def _determine_workflow_pattern(self, facets: SessionFacets) -> str:
        """Determine the workflow pattern based on facets."""
        if facets.has_recipes:
            return WorkflowPattern.RECIPE_DRIVEN.value

        if facets.has_child_sessions or facets.child_session_count > 0:
            return WorkflowPattern.MULTI_AGENT.value

        # High interaction ratio suggests interactive exploration
        if facets.user_message_count > 10 and facets.tool_call_count < facets.user_message_count:
            return WorkflowPattern.INTERACTIVE.value

        return WorkflowPattern.SINGLE_AGENT.value
