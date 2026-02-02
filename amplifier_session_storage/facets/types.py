"""
Session facets data types for efficient server-side filtering.

Facets are denormalized aggregates computed from session events that enable
efficient queries like "find sessions that used tool X" without loading all events.

Design Principles:
1. Facets are computed separately from event writes (keeps writes fast)
2. Facets can be rebuilt from events at any time (eventual consistency)
3. Facets are stored alongside session metadata for single-query filtering
4. All facet fields are indexed in Cosmos DB for server-side filtering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal


class WorkflowPattern(Enum):
    """Detected workflow patterns in a session."""

    SINGLE_AGENT = "single_agent"  # No delegation
    MULTI_AGENT = "multi_agent"  # Uses delegate tool
    RECIPE_DRIVEN = "recipe_driven"  # Uses recipe execution
    INTERACTIVE = "interactive"  # High user turn count, exploratory


@dataclass
class SessionFacets:
    """Aggregated queryable facets for a session.

    These facets enable efficient server-side filtering in Cosmos DB
    without loading full event streams. Facets are computed from events
    and can be refreshed on-demand or periodically.

    Update Strategy:
    - Facets are NOT updated on every event write (keeps writes fast)
    - Computed on: session close, explicit refresh, periodic background job
    - Can be rebuilt from events at any time

    Query Examples:
    - "Find sessions using bundle X" -> facets.bundle = "X"
    - "Find sessions that used delegate tool" -> "delegate" in facets.tools_used
    - "Find multi-agent sessions" -> facets.workflow_pattern = "multi_agent"
    """

    # =========================================================================
    # Configuration (from SESSION_CREATED block, immutable)
    # =========================================================================

    bundle: str | None = None
    initial_model: str | None = None
    initial_provider: str | None = None

    # =========================================================================
    # Tool Usage (aggregated from tool_result events)
    # =========================================================================

    tools_used: list[str] = field(default_factory=list)
    tool_call_count: int = 0

    # =========================================================================
    # Model/Provider Usage (aggregated from assistant_response events)
    # =========================================================================

    models_used: list[str] = field(default_factory=list)
    providers_used: list[str] = field(default_factory=list)

    # =========================================================================
    # Token Metrics (aggregated from usage data)
    # =========================================================================

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0

    # =========================================================================
    # Error Tracking
    # =========================================================================

    has_errors: bool = False
    error_count: int = 0
    error_types: list[str] = field(default_factory=list)

    # =========================================================================
    # Delegation Patterns (from delegate/spawn events)
    # =========================================================================

    has_child_sessions: bool = False
    child_session_count: int = 0
    agents_delegated_to: list[str] = field(default_factory=list)

    # =========================================================================
    # Recipe/Workflow Patterns
    # =========================================================================

    has_recipes: bool = False
    recipe_names: list[str] = field(default_factory=list)
    workflow_pattern: str | None = None  # WorkflowPattern value

    # =========================================================================
    # Content Indicators (for search optimization)
    # =========================================================================

    has_file_operations: bool = False
    has_code_blocks: bool = False
    languages_detected: list[str] = field(default_factory=list)

    # File operations detail
    files_read: int = 0
    files_written: int = 0
    files_edited: int = 0

    # =========================================================================
    # Timing Metrics
    # =========================================================================

    first_event_at: datetime | None = None
    last_event_at: datetime | None = None
    total_duration_ms: int = 0
    active_duration_ms: int = 0  # Excludes idle time

    # =========================================================================
    # Conversation Metrics
    # =========================================================================

    user_message_count: int = 0
    assistant_message_count: int = 0
    max_turn: int = 0

    # =========================================================================
    # Facets Metadata
    # =========================================================================

    # Version for schema migration
    facets_version: int = 1

    # When facets were last computed
    last_computed: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Number of events processed when facets were computed
    events_processed: int = 0

    # Whether facets are potentially stale (more events since computation)
    is_stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage.

        All datetime fields are converted to ISO format strings.
        All enums are converted to their string values.
        """
        return {
            # Configuration
            "bundle": self.bundle,
            "initial_model": self.initial_model,
            "initial_provider": self.initial_provider,
            # Tool usage
            "tools_used": self.tools_used,
            "tool_call_count": self.tool_call_count,
            # Model/Provider
            "models_used": self.models_used,
            "providers_used": self.providers_used,
            # Tokens
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            # Errors
            "has_errors": self.has_errors,
            "error_count": self.error_count,
            "error_types": self.error_types,
            # Delegation
            "has_child_sessions": self.has_child_sessions,
            "child_session_count": self.child_session_count,
            "agents_delegated_to": self.agents_delegated_to,
            # Recipes/Workflow
            "has_recipes": self.has_recipes,
            "recipe_names": self.recipe_names,
            "workflow_pattern": self.workflow_pattern,
            # Content
            "has_file_operations": self.has_file_operations,
            "has_code_blocks": self.has_code_blocks,
            "languages_detected": self.languages_detected,
            "files_read": self.files_read,
            "files_written": self.files_written,
            "files_edited": self.files_edited,
            # Timing
            "first_event_at": (self.first_event_at.isoformat() if self.first_event_at else None),
            "last_event_at": (self.last_event_at.isoformat() if self.last_event_at else None),
            "total_duration_ms": self.total_duration_ms,
            "active_duration_ms": self.active_duration_ms,
            # Conversation
            "user_message_count": self.user_message_count,
            "assistant_message_count": self.assistant_message_count,
            "max_turn": self.max_turn,
            # Metadata
            "facets_version": self.facets_version,
            "last_computed": self.last_computed.isoformat(),
            "events_processed": self.events_processed,
            "is_stale": self.is_stale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionFacets:
        """Deserialize from dictionary."""
        # Parse datetime fields
        first_event_at = None
        if data.get("first_event_at"):
            first_event_at = datetime.fromisoformat(data["first_event_at"])

        last_event_at = None
        if data.get("last_event_at"):
            last_event_at = datetime.fromisoformat(data["last_event_at"])

        last_computed = datetime.now(UTC)
        if data.get("last_computed"):
            last_computed = datetime.fromisoformat(data["last_computed"])

        return cls(
            # Configuration
            bundle=data.get("bundle"),
            initial_model=data.get("initial_model"),
            initial_provider=data.get("initial_provider"),
            # Tool usage
            tools_used=data.get("tools_used", []),
            tool_call_count=data.get("tool_call_count", 0),
            # Model/Provider
            models_used=data.get("models_used", []),
            providers_used=data.get("providers_used", []),
            # Tokens
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            # Errors
            has_errors=data.get("has_errors", False),
            error_count=data.get("error_count", 0),
            error_types=data.get("error_types", []),
            # Delegation
            has_child_sessions=data.get("has_child_sessions", False),
            child_session_count=data.get("child_session_count", 0),
            agents_delegated_to=data.get("agents_delegated_to", []),
            # Recipes/Workflow
            has_recipes=data.get("has_recipes", False),
            recipe_names=data.get("recipe_names", []),
            workflow_pattern=data.get("workflow_pattern"),
            # Content
            has_file_operations=data.get("has_file_operations", False),
            has_code_blocks=data.get("has_code_blocks", False),
            languages_detected=data.get("languages_detected", []),
            files_read=data.get("files_read", 0),
            files_written=data.get("files_written", 0),
            files_edited=data.get("files_edited", 0),
            # Timing
            first_event_at=first_event_at,
            last_event_at=last_event_at,
            total_duration_ms=data.get("total_duration_ms", 0),
            active_duration_ms=data.get("active_duration_ms", 0),
            # Conversation
            user_message_count=data.get("user_message_count", 0),
            assistant_message_count=data.get("assistant_message_count", 0),
            max_turn=data.get("max_turn", 0),
            # Metadata
            facets_version=data.get("facets_version", 1),
            last_computed=last_computed,
            events_processed=data.get("events_processed", 0),
            is_stale=data.get("is_stale", False),
        )

    def mark_stale(self) -> None:
        """Mark facets as potentially stale (more events added since computation)."""
        self.is_stale = True

    def merge_lists(self, field_name: str, new_items: list[str]) -> None:
        """Add unique items to a list field.

        Helper method for adding items without duplicates.
        """
        current: list[str] = getattr(self, field_name)
        for item in new_items:
            if item and item not in current:
                current.append(item)


@dataclass
class FacetQuery:
    """Query parameters for facet-based session filtering.

    All fields are optional. Multiple fields combine with AND logic.
    List fields use ANY-match semantics (e.g., tools_used matches if
    ANY of the specified tools were used).

    For ALL-match semantics on lists, use the *_all variants.
    """

    # =========================================================================
    # Basic Filters (from SessionMetadata)
    # =========================================================================

    user_id: str | None = None
    session_id: str | None = None  # Filter by specific session
    parent_id: str | None = None  # Filter by parent session (for child sessions)
    project_slug: str | None = None
    name_contains: str | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    updated_after: datetime | None = None
    updated_before: datetime | None = None
    tags: list[str] | None = None  # Sessions with ALL these tags

    # =========================================================================
    # Configuration Filters
    # =========================================================================

    bundle: str | None = None
    bundles: list[str] | None = None  # ANY of these bundles

    model: str | None = None
    models: list[str] | None = None  # ANY of these models

    provider: str | None = None
    providers: list[str] | None = None  # ANY of these providers

    # =========================================================================
    # Tool Filters
    # =========================================================================

    tool_used: str | None = None  # Sessions that used this tool
    tools_used: list[str] | None = None  # ANY of these tools
    tools_used_all: list[str] | None = None  # ALL of these tools

    # =========================================================================
    # Error Filters
    # =========================================================================

    has_errors: bool | None = None
    min_error_count: int | None = None
    error_type: str | None = None

    # =========================================================================
    # Delegation Filters
    # =========================================================================

    has_child_sessions: bool | None = None
    min_child_sessions: int | None = None
    agent_delegated_to: str | None = None
    agents_delegated_to: list[str] | None = None  # ANY of these agents

    # =========================================================================
    # Recipe/Workflow Filters
    # =========================================================================

    has_recipes: bool | None = None
    recipe_name: str | None = None
    workflow_pattern: str | None = None  # WorkflowPattern value

    # =========================================================================
    # Token Filters
    # =========================================================================

    min_tokens: int | None = None
    max_tokens: int | None = None
    min_input_tokens: int | None = None
    min_output_tokens: int | None = None

    # =========================================================================
    # Duration Filters
    # =========================================================================

    min_duration_ms: int | None = None
    max_duration_ms: int | None = None

    # =========================================================================
    # Turn Count Filters
    # =========================================================================

    min_turns: int | None = None
    max_turns: int | None = None

    # =========================================================================
    # Content Filters
    # =========================================================================

    has_file_operations: bool | None = None
    has_code_blocks: bool | None = None
    language_detected: str | None = None

    # =========================================================================
    # Pagination & Ordering
    # =========================================================================

    limit: int = 100
    offset: int = 0
    order_by: Literal["created", "updated", "tokens", "duration", "error_count"] = "updated"
    order_desc: bool = True

    def has_facet_filters(self) -> bool:
        """Check if any facet-specific filters are set.

        Returns True if any filter beyond basic SessionQuery fields is set.
        """
        facet_fields = [
            self.bundle,
            self.bundles,
            self.model,
            self.models,
            self.provider,
            self.providers,
            self.tool_used,
            self.tools_used,
            self.tools_used_all,
            self.has_errors,
            self.min_error_count,
            self.error_type,
            self.has_child_sessions,
            self.min_child_sessions,
            self.agent_delegated_to,
            self.agents_delegated_to,
            self.has_recipes,
            self.recipe_name,
            self.workflow_pattern,
            self.min_tokens,
            self.max_tokens,
            self.min_input_tokens,
            self.min_output_tokens,
            self.min_duration_ms,
            self.max_duration_ms,
            self.has_file_operations,
            self.has_code_blocks,
            self.language_detected,
        ]
        return any(f is not None for f in facet_fields)
