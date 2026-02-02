"""Tests for facets module - TDD approach.

These tests define the expected behavior of the facets system:
1. SessionFacets - data type and serialization
2. FacetsUpdater - incremental event processing
3. FacetsRebuilder - full reconstruction from blocks
4. FacetQuery - query building and validation
5. Cosmos query translation - FacetQuery → SQL
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from amplifier_session_storage.blocks import BlockType, SessionBlock
from amplifier_session_storage.facets import (
    FacetQuery,
    FacetsRebuilder,
    FacetsUpdater,
    SessionFacets,
    WorkflowPattern,
    rebuild_facets,
)

# =============================================================================
# SessionFacets Tests
# =============================================================================


class TestSessionFacets:
    """Tests for SessionFacets dataclass."""

    def test_create_empty_facets(self) -> None:
        """Test creating empty facets with defaults."""
        facets = SessionFacets()

        assert facets.bundle is None
        assert facets.tools_used == []
        assert facets.tool_call_count == 0
        assert facets.total_tokens == 0
        assert facets.has_errors is False
        assert facets.workflow_pattern is None

    def test_create_facets_with_values(self) -> None:
        """Test creating facets with explicit values."""
        facets = SessionFacets(
            bundle="amplifier-dev",
            tools_used=["delegate", "read_file"],
            tool_call_count=5,
            total_input_tokens=1000,
            total_output_tokens=2000,
            has_errors=True,
            error_count=2,
        )

        assert facets.bundle == "amplifier-dev"
        assert facets.tools_used == ["delegate", "read_file"]
        assert facets.tool_call_count == 5
        assert facets.total_input_tokens == 1000
        assert facets.total_output_tokens == 2000
        assert facets.has_errors is True
        assert facets.error_count == 2

    def test_to_dict_serialization(self) -> None:
        """Test serialization to dictionary."""
        now = datetime.now(UTC)
        facets = SessionFacets(
            bundle="test-bundle",
            tools_used=["bash", "grep"],
            first_event_at=now,
            last_event_at=now,
        )

        data = facets.to_dict()

        assert data["bundle"] == "test-bundle"
        assert data["tools_used"] == ["bash", "grep"]
        assert data["first_event_at"] == now.isoformat()
        assert data["last_event_at"] == now.isoformat()
        assert "facets_version" in data

    def test_from_dict_deserialization(self) -> None:
        """Test deserialization from dictionary."""
        now = datetime.now(UTC)
        data = {
            "bundle": "test-bundle",
            "tools_used": ["bash", "grep"],
            "tool_call_count": 10,
            "first_event_at": now.isoformat(),
            "last_event_at": now.isoformat(),
            "workflow_pattern": "multi_agent",
        }

        facets = SessionFacets.from_dict(data)

        assert facets.bundle == "test-bundle"
        assert facets.tools_used == ["bash", "grep"]
        assert facets.tool_call_count == 10
        assert facets.workflow_pattern == "multi_agent"

    def test_round_trip_serialization(self) -> None:
        """Test that to_dict -> from_dict preserves all data."""
        now = datetime.now(UTC)
        original = SessionFacets(
            bundle="amplifier-dev",
            initial_model="claude-sonnet-4",
            initial_provider="anthropic",
            tools_used=["delegate", "read_file", "write_file"],
            tool_call_count=15,
            models_used=["claude-sonnet-4", "claude-haiku"],
            providers_used=["anthropic"],
            total_input_tokens=5000,
            total_output_tokens=10000,
            total_tokens=15000,
            has_errors=True,
            error_count=2,
            error_types=["ValidationError", "TimeoutError"],
            has_child_sessions=True,
            child_session_count=3,
            agents_delegated_to=["foundation:explorer", "foundation:bug-hunter"],
            has_recipes=True,
            recipe_names=["code-review"],
            workflow_pattern="multi_agent",
            has_file_operations=True,
            files_read=10,
            files_written=5,
            files_edited=3,
            first_event_at=now,
            last_event_at=now + timedelta(hours=1),
            user_message_count=20,
            assistant_message_count=19,
            max_turn=20,
        )

        data = original.to_dict()
        restored = SessionFacets.from_dict(data)

        # Check key fields
        assert restored.bundle == original.bundle
        assert restored.initial_model == original.initial_model
        assert restored.tools_used == original.tools_used
        assert restored.tool_call_count == original.tool_call_count
        assert restored.total_tokens == original.total_tokens
        assert restored.has_errors == original.has_errors
        assert restored.error_types == original.error_types
        assert restored.agents_delegated_to == original.agents_delegated_to
        assert restored.workflow_pattern == original.workflow_pattern
        assert restored.files_read == original.files_read

    def test_merge_lists_adds_unique_items(self) -> None:
        """Test that merge_lists adds only unique items."""
        facets = SessionFacets(tools_used=["bash", "grep"])

        facets.merge_lists("tools_used", ["grep", "read_file", "bash"])

        assert sorted(facets.tools_used) == ["bash", "grep", "read_file"]

    def test_mark_stale(self) -> None:
        """Test marking facets as stale."""
        facets = SessionFacets()
        assert facets.is_stale is False

        facets.mark_stale()

        assert facets.is_stale is True


# =============================================================================
# FacetsUpdater Tests
# =============================================================================


class TestFacetsUpdater:
    """Tests for FacetsUpdater incremental processing."""

    def test_initialize_from_session_created(self) -> None:
        """Test initialization from session created data."""
        updater = FacetsUpdater()
        facets = SessionFacets()
        now = datetime.now(UTC)

        data = {
            "bundle": "amplifier-dev",
            "model": "claude-sonnet-4",
        }

        facets = updater.initialize_from_session_created(facets, data, now)

        assert facets.bundle == "amplifier-dev"
        assert facets.initial_model == "claude-sonnet-4"
        assert facets.initial_provider == "anthropic"
        assert "claude-sonnet-4" in facets.models_used
        assert "anthropic" in facets.providers_used
        assert facets.first_event_at == now

    def test_process_tool_result_tracks_tools(self) -> None:
        """Test that tool_result events track tool usage."""
        updater = FacetsUpdater()
        facets = SessionFacets()
        now = datetime.now(UTC)

        facets = updater.process_event(facets, "tool_result", {"tool_name": "read_file"}, now)
        facets = updater.process_event(facets, "tool_result", {"tool_name": "write_file"}, now)
        facets = updater.process_event(
            facets, "tool_result", {"tool_name": "read_file"}, now
        )  # Duplicate

        assert sorted(facets.tools_used) == ["read_file", "write_file"]
        assert facets.tool_call_count == 3

    def test_process_tool_result_tracks_file_operations(self) -> None:
        """Test that file operation tools are tracked separately."""
        updater = FacetsUpdater()
        facets = SessionFacets()
        now = datetime.now(UTC)

        facets = updater.process_event(facets, "tool_result", {"tool_name": "read_file"}, now)
        facets = updater.process_event(facets, "tool_result", {"tool_name": "read_file"}, now)
        facets = updater.process_event(facets, "tool_result", {"tool_name": "write_file"}, now)
        facets = updater.process_event(facets, "tool_result", {"tool_name": "edit_file"}, now)
        facets = updater.process_event(facets, "tool_result", {"tool_name": "glob"}, now)

        assert facets.has_file_operations is True
        assert facets.files_read == 3  # read_file x2 + glob
        assert facets.files_written == 1
        assert facets.files_edited == 1

    def test_process_delegate_tracks_agents(self) -> None:
        """Test that delegate tool tracks agents."""
        updater = FacetsUpdater()
        facets = SessionFacets()
        now = datetime.now(UTC)

        facets = updater.process_event(
            facets,
            "tool_result",
            {"tool_name": "delegate", "agent": "foundation:explorer"},
            now,
        )
        facets = updater.process_event(
            facets,
            "tool_result",
            {"tool_name": "delegate", "agent": "foundation:bug-hunter"},
            now,
        )
        facets = updater.process_event(
            facets,
            "tool_result",
            {"tool_name": "delegate", "agent": "foundation:explorer"},
            now,
        )  # Duplicate

        assert facets.has_child_sessions is True
        assert facets.child_session_count == 3
        assert sorted(facets.agents_delegated_to) == [
            "foundation:bug-hunter",
            "foundation:explorer",
        ]

    def test_process_assistant_response_tracks_tokens(self) -> None:
        """Test that assistant_response tracks token usage."""
        updater = FacetsUpdater()
        facets = SessionFacets()
        now = datetime.now(UTC)

        facets = updater.process_event(
            facets,
            "assistant_response",
            {
                "model": "claude-sonnet-4",
                "usage": {"input_tokens": 100, "output_tokens": 200},
                "duration_ms": 500,
            },
            now,
        )
        facets = updater.process_event(
            facets,
            "assistant_response",
            {
                "model": "claude-haiku",
                "usage": {"input_tokens": 50, "output_tokens": 100},
                "duration_ms": 200,
            },
            now,
        )

        assert facets.total_input_tokens == 150
        assert facets.total_output_tokens == 300
        assert facets.active_duration_ms == 700
        assert sorted(facets.models_used) == ["claude-haiku", "claude-sonnet-4"]
        assert facets.providers_used == ["anthropic"]  # Auto-detected

    def test_process_error_tracks_errors(self) -> None:
        """Test that error events are tracked."""
        updater = FacetsUpdater()
        facets = SessionFacets()
        now = datetime.now(UTC)

        facets = updater.process_event(facets, "error", {"error_type": "ValidationError"}, now)
        facets = updater.process_event(facets, "error", {"error_type": "TimeoutError"}, now)
        facets = updater.process_event(
            facets, "error", {"error_type": "ValidationError"}, now
        )  # Duplicate type

        assert facets.has_errors is True
        assert facets.error_count == 3
        assert sorted(facets.error_types) == ["TimeoutError", "ValidationError"]

    def test_process_recipe_tracks_recipes(self) -> None:
        """Test that recipe events are tracked."""
        updater = FacetsUpdater()
        facets = SessionFacets()
        now = datetime.now(UTC)

        facets = updater.process_event(
            facets,
            "tool_result",
            {
                "tool_name": "recipes",
                "operation": "execute",
                "recipe_path": "code-review.yaml",
            },
            now,
        )

        assert facets.has_recipes is True
        assert "code-review.yaml" in facets.recipe_names

    def test_process_user_message_tracks_turns(self) -> None:
        """Test that user messages track turn count."""
        updater = FacetsUpdater()
        facets = SessionFacets()
        now = datetime.now(UTC)

        facets = updater.process_event(facets, "user_message", {"turn": 1}, now)
        facets = updater.process_event(facets, "user_message", {"turn": 2}, now)
        facets = updater.process_event(facets, "user_message", {"turn": 5}, now)

        assert facets.user_message_count == 3
        assert facets.max_turn == 5

    def test_finalize_computes_totals(self) -> None:
        """Test that finalize computes derived values."""
        updater = FacetsUpdater()
        now = datetime.now(UTC)
        facets = SessionFacets(
            total_input_tokens=1000,
            total_output_tokens=2000,
            first_event_at=now,
            last_event_at=now + timedelta(minutes=5),
        )

        facets = updater.finalize(facets)

        assert facets.total_tokens == 3000
        assert facets.total_duration_ms == 5 * 60 * 1000
        assert facets.is_stale is False

    def test_finalize_detects_workflow_pattern_single_agent(self) -> None:
        """Test workflow pattern detection for single agent."""
        updater = FacetsUpdater()
        facets = SessionFacets(
            user_message_count=3,
            tool_call_count=10,
        )

        facets = updater.finalize(facets)

        assert facets.workflow_pattern == WorkflowPattern.SINGLE_AGENT.value

    def test_finalize_detects_workflow_pattern_multi_agent(self) -> None:
        """Test workflow pattern detection for multi-agent."""
        updater = FacetsUpdater()
        facets = SessionFacets(
            has_child_sessions=True,
            child_session_count=2,
        )

        facets = updater.finalize(facets)

        assert facets.workflow_pattern == WorkflowPattern.MULTI_AGENT.value

    def test_finalize_detects_workflow_pattern_recipe_driven(self) -> None:
        """Test workflow pattern detection for recipes."""
        updater = FacetsUpdater()
        facets = SessionFacets(
            has_recipes=True,
            recipe_names=["code-review.yaml"],
        )

        facets = updater.finalize(facets)

        assert facets.workflow_pattern == WorkflowPattern.RECIPE_DRIVEN.value

    def test_finalize_detects_workflow_pattern_interactive(self) -> None:
        """Test workflow pattern detection for interactive sessions."""
        updater = FacetsUpdater()
        facets = SessionFacets(
            user_message_count=15,  # High interaction
            tool_call_count=5,  # Low tool usage relative to messages
        )

        facets = updater.finalize(facets)

        assert facets.workflow_pattern == WorkflowPattern.INTERACTIVE.value

    def test_provider_detection_openai(self) -> None:
        """Test provider detection for OpenAI models."""
        updater = FacetsUpdater()
        facets = SessionFacets()
        now = datetime.now(UTC)

        facets = updater.process_event(
            facets,
            "assistant_response",
            {"model": "gpt-4o", "usage": {"input_tokens": 100, "output_tokens": 200}},
            now,
        )

        assert "openai" in facets.providers_used

    def test_provider_detection_google(self) -> None:
        """Test provider detection for Google models."""
        updater = FacetsUpdater()
        facets = SessionFacets()
        now = datetime.now(UTC)

        facets = updater.process_event(
            facets,
            "assistant_response",
            {
                "model": "gemini-pro",
                "usage": {"input_tokens": 100, "output_tokens": 200},
            },
            now,
        )

        assert "google" in facets.providers_used


# =============================================================================
# FacetsRebuilder Tests
# =============================================================================


class TestFacetsRebuilder:
    """Tests for FacetsRebuilder full reconstruction."""

    def _create_block(
        self,
        block_type: BlockType,
        data: dict,
        sequence: int,
        timestamp: datetime | None = None,
    ) -> SessionBlock:
        """Helper to create test blocks."""
        return SessionBlock(
            block_id=f"blk-{sequence}",
            session_id="test-session",
            user_id="test-user",
            sequence=sequence,
            timestamp=timestamp or datetime.now(UTC),
            device_id="test-device",
            block_type=block_type,
            data=data,
        )

    @pytest.mark.asyncio
    async def test_rebuild_from_empty_blocks(self) -> None:
        """Test rebuilding facets from empty block list."""
        rebuilder = FacetsRebuilder()

        result = await rebuilder.rebuild_session_facets([])

        assert result.success is True
        assert result.blocks_processed == 0
        assert isinstance(result.facets, SessionFacets)

    @pytest.mark.asyncio
    async def test_rebuild_from_session_created(self) -> None:
        """Test rebuilding facets from session created block."""
        rebuilder = FacetsRebuilder()
        blocks = [
            self._create_block(
                BlockType.SESSION_CREATED,
                {"bundle": "amplifier-dev", "model": "claude-sonnet-4"},
                sequence=1,
            )
        ]

        result = await rebuilder.rebuild_session_facets(blocks)

        assert result.success is True
        assert result.facets.bundle == "amplifier-dev"
        assert result.facets.initial_model == "claude-sonnet-4"

    @pytest.mark.asyncio
    async def test_rebuild_from_events(self) -> None:
        """Test rebuilding facets from event blocks."""
        rebuilder = FacetsRebuilder()
        now = datetime.now(UTC)
        blocks = [
            self._create_block(
                BlockType.SESSION_CREATED,
                {"bundle": "test"},
                sequence=1,
                timestamp=now,
            ),
            self._create_block(
                BlockType.EVENT,
                {"event_type": "tool_result", "summary": {"tool_name": "read_file"}},
                sequence=2,
                timestamp=now + timedelta(seconds=1),
            ),
            self._create_block(
                BlockType.EVENT,
                {"event_type": "tool_result", "summary": {"tool_name": "write_file"}},
                sequence=3,
                timestamp=now + timedelta(seconds=2),
            ),
            self._create_block(
                BlockType.EVENT,
                {
                    "event_type": "assistant_response",
                    "summary": {
                        "model": "claude-sonnet-4",
                        "usage": {"input_tokens": 100, "output_tokens": 200},
                    },
                },
                sequence=4,
                timestamp=now + timedelta(seconds=3),
            ),
        ]

        result = await rebuilder.rebuild_session_facets(blocks)

        assert result.success is True
        assert result.events_processed == 3
        assert sorted(result.facets.tools_used) == ["read_file", "write_file"]
        assert result.facets.total_input_tokens == 100
        assert result.facets.total_output_tokens == 200

    @pytest.mark.asyncio
    async def test_rebuild_from_messages(self) -> None:
        """Test rebuilding facets from message blocks."""
        rebuilder = FacetsRebuilder()
        now = datetime.now(UTC)
        blocks = [
            self._create_block(
                BlockType.MESSAGE,
                {"role": "user", "content": "Hello", "turn": 1},
                sequence=1,
                timestamp=now,
            ),
            self._create_block(
                BlockType.MESSAGE,
                {"role": "assistant", "content": "Hi there!", "turn": 1},
                sequence=2,
                timestamp=now + timedelta(seconds=1),
            ),
            self._create_block(
                BlockType.MESSAGE,
                {"role": "user", "content": "How are you?", "turn": 2},
                sequence=3,
                timestamp=now + timedelta(seconds=2),
            ),
        ]

        result = await rebuilder.rebuild_session_facets(blocks)

        assert result.success is True
        assert result.messages_processed == 3
        assert result.facets.user_message_count == 2
        assert result.facets.assistant_message_count == 1
        assert result.facets.max_turn == 2

    @pytest.mark.asyncio
    async def test_rebuild_convenience_function(self) -> None:
        """Test the rebuild_facets convenience function."""
        blocks = [
            self._create_block(
                BlockType.SESSION_CREATED,
                {"bundle": "test-bundle"},
                sequence=1,
            )
        ]

        facets = await rebuild_facets(blocks)

        assert facets.bundle == "test-bundle"

    def test_verify_facets_accurate(self) -> None:
        """Test facets verification when accurate."""
        rebuilder = FacetsRebuilder()
        existing = SessionFacets(
            bundle="test",
            tool_call_count=5,
            tools_used=["read_file", "write_file"],
        )
        rebuilt = SessionFacets(
            bundle="test",
            tool_call_count=5,
            tools_used=["write_file", "read_file"],  # Different order, same content
        )

        result = rebuilder.verify_facets(existing, rebuilt)

        assert result["is_accurate"] is True
        assert result["discrepancy_count"] == 0

    def test_verify_facets_with_discrepancies(self) -> None:
        """Test facets verification when discrepancies exist."""
        rebuilder = FacetsRebuilder()
        existing = SessionFacets(
            bundle="test",
            tool_call_count=5,
            tools_used=["read_file"],
        )
        rebuilt = SessionFacets(
            bundle="test",
            tool_call_count=10,  # Different
            tools_used=["read_file", "write_file"],  # Missing one
        )

        result = rebuilder.verify_facets(existing, rebuilt)

        assert result["is_accurate"] is False
        assert result["discrepancy_count"] == 2


# =============================================================================
# FacetQuery Tests
# =============================================================================


class TestFacetQuery:
    """Tests for FacetQuery query building."""

    def test_create_basic_query(self) -> None:
        """Test creating a basic query."""
        query = FacetQuery(user_id="user-123")

        assert query.user_id == "user-123"
        assert query.limit == 100
        assert query.order_by == "updated"
        assert query.order_desc is True

    def test_has_facet_filters_false_for_basic_query(self) -> None:
        """Test that basic queries don't have facet filters."""
        query = FacetQuery(
            user_id="user-123",
            project_slug="my-project",
            created_after=datetime.now(UTC),
        )

        assert query.has_facet_filters() is False

    def test_has_facet_filters_true_for_bundle(self) -> None:
        """Test that bundle filter is detected."""
        query = FacetQuery(user_id="user-123", bundle="amplifier-dev")

        assert query.has_facet_filters() is True

    def test_has_facet_filters_true_for_tool_used(self) -> None:
        """Test that tool_used filter is detected."""
        query = FacetQuery(user_id="user-123", tool_used="delegate")

        assert query.has_facet_filters() is True

    def test_has_facet_filters_true_for_has_errors(self) -> None:
        """Test that has_errors filter is detected."""
        query = FacetQuery(user_id="user-123", has_errors=True)

        assert query.has_facet_filters() is True

    def test_has_facet_filters_true_for_min_tokens(self) -> None:
        """Test that min_tokens filter is detected."""
        query = FacetQuery(user_id="user-123", min_tokens=1000)

        assert query.has_facet_filters() is True

    def test_has_facet_filters_true_for_workflow_pattern(self) -> None:
        """Test that workflow_pattern filter is detected."""
        query = FacetQuery(user_id="user-123", workflow_pattern="multi_agent")

        assert query.has_facet_filters() is True

    def test_complex_query(self) -> None:
        """Test a complex query with multiple filters."""
        query = FacetQuery(
            user_id="user-123",
            bundle="amplifier-dev",
            tool_used="delegate",
            has_errors=False,
            workflow_pattern="multi_agent",
            min_tokens=1000,
            max_tokens=100000,
            limit=50,
            order_by="tokens",
        )

        assert query.has_facet_filters() is True
        assert query.bundle == "amplifier-dev"
        assert query.tool_used == "delegate"
        assert query.has_errors is False
        assert query.min_tokens == 1000
        assert query.max_tokens == 100000


# =============================================================================
# WorkflowPattern Tests
# =============================================================================


class TestWorkflowPattern:
    """Tests for WorkflowPattern enum."""

    def test_enum_values(self) -> None:
        """Test that all workflow patterns have correct values."""
        assert WorkflowPattern.SINGLE_AGENT.value == "single_agent"
        assert WorkflowPattern.MULTI_AGENT.value == "multi_agent"
        assert WorkflowPattern.RECIPE_DRIVEN.value == "recipe_driven"
        assert WorkflowPattern.INTERACTIVE.value == "interactive"

    def test_enum_from_string(self) -> None:
        """Test creating enum from string value."""
        pattern = WorkflowPattern("multi_agent")
        assert pattern == WorkflowPattern.MULTI_AGENT
