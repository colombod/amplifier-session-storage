"""
Full facets reconstruction from event streams.

This module provides functions to rebuild SessionFacets from scratch
by processing all events in a session. Use for:

1. Backfilling facets for existing sessions without facets
2. Recovering from corrupted or stale facets
3. Periodic refresh to ensure accuracy
4. Migration when facets schema changes

The rebuilder uses the same FacetsUpdater logic but processes
all events in sequence to produce complete, accurate facets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from ..blocks.types import BlockType, SessionBlock
from .types import SessionFacets
from .updater import FacetsUpdater

if TYPE_CHECKING:
    pass


# =============================================================================
# Storage Protocol (to avoid circular imports)
# =============================================================================


class BlockReader(Protocol):
    """Protocol for reading blocks from storage."""

    async def get_blocks(
        self,
        user_id: str,
        session_id: str,
        block_types: list[BlockType] | None = None,
        limit: int | None = None,
    ) -> list[SessionBlock]:
        """Get blocks for a session."""
        ...


class SessionReader(Protocol):
    """Protocol for reading session metadata."""

    async def get_session(
        self,
        user_id: str,
        session_id: str,
    ) -> Any:
        """Get session metadata."""
        ...


# =============================================================================
# Rebuild Results
# =============================================================================


@dataclass
class RebuildResult:
    """Result of a facets rebuild operation."""

    facets: SessionFacets
    blocks_processed: int
    events_processed: int
    messages_processed: int
    duration_ms: int
    errors: list[str]

    @property
    def success(self) -> bool:
        """Whether the rebuild completed without errors."""
        return len(self.errors) == 0


@dataclass
class BackfillProgress:
    """Progress tracking for batch backfill operations."""

    total_sessions: int
    processed_sessions: int
    successful: int
    failed: int
    skipped: int
    current_session_id: str | None = None

    @property
    def percent_complete(self) -> float:
        """Percentage of sessions processed."""
        if self.total_sessions == 0:
            return 100.0
        return (self.processed_sessions / self.total_sessions) * 100


@dataclass
class BackfillResult:
    """Result of a batch backfill operation."""

    total_sessions: int
    successful: int
    failed: int
    skipped: int
    duration_ms: int
    failed_session_ids: list[str]


# =============================================================================
# Rebuilder Class
# =============================================================================


class FacetsRebuilder:
    """Rebuilds session facets from event streams.

    This class provides methods to:
    - Rebuild facets for a single session
    - Backfill facets for multiple sessions
    - Verify facets accuracy against event stream
    """

    def __init__(self, updater: FacetsUpdater | None = None) -> None:
        """Initialize the rebuilder.

        Args:
            updater: Optional FacetsUpdater instance. Creates one if not provided.
        """
        self.updater = updater or FacetsUpdater()

    async def rebuild_session_facets(
        self,
        blocks: list[SessionBlock],
        session_metadata: dict[str, Any] | None = None,
    ) -> RebuildResult:
        """Rebuild facets from a list of session blocks.

        Processes all blocks in sequence, extracting events and messages
        to build complete facets. This is a pure function that doesn't
        read from or write to storage.

        Args:
            blocks: List of SessionBlock objects for the session
            session_metadata: Optional session metadata dict for initial values

        Returns:
            RebuildResult with computed facets and statistics
        """
        start_time = datetime.now(UTC)
        errors: list[str] = []

        # Initialize fresh facets
        facets = SessionFacets()

        # Counters
        events_processed = 0
        messages_processed = 0

        # Sort blocks by sequence to ensure correct ordering
        sorted_blocks = sorted(blocks, key=lambda b: b.sequence)

        for block in sorted_blocks:
            try:
                if block.block_type == BlockType.SESSION_CREATED:
                    # Initialize from session creation data
                    self.updater.initialize_from_session_created(
                        facets, block.data, block.timestamp
                    )

                elif block.block_type == BlockType.MESSAGE:
                    # Process message
                    role = block.data.get("role", "")
                    content = block.data.get("content", "")
                    turn = block.data.get("turn", 0)

                    self.updater.process_message(facets, role, content, turn, block.timestamp)
                    messages_processed += 1

                elif block.block_type == BlockType.EVENT:
                    # Process event
                    event_type = block.data.get("event_type", "")
                    summary = block.data.get("summary", {})

                    # Merge inline_data into summary if present
                    if block.data.get("inline_data"):
                        summary = {**summary, **block.data["inline_data"]}

                    self.updater.process_event(facets, event_type, summary, block.timestamp)
                    events_processed += 1

                elif block.block_type == BlockType.SESSION_UPDATED:
                    # Session updates might change tags, etc.
                    # Currently facets don't track these, but could in future
                    pass

            except Exception as e:
                errors.append(
                    f"Error processing block {block.block_id} (type={block.block_type.value}): {e}"
                )

        # Apply any session metadata that wasn't in blocks
        if session_metadata:
            if not facets.bundle and session_metadata.get("bundle"):
                facets.bundle = session_metadata["bundle"]
            if not facets.initial_model and session_metadata.get("model"):
                facets.initial_model = session_metadata["model"]

        # Finalize facets
        self.updater.finalize(facets)

        # Calculate duration
        end_time = datetime.now(UTC)
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        return RebuildResult(
            facets=facets,
            blocks_processed=len(sorted_blocks),
            events_processed=events_processed,
            messages_processed=messages_processed,
            duration_ms=duration_ms,
            errors=errors,
        )

    async def rebuild_from_storage(
        self,
        storage: BlockReader,
        user_id: str,
        session_id: str,
        session_metadata: dict[str, Any] | None = None,
    ) -> RebuildResult:
        """Rebuild facets by reading blocks from storage.

        Convenience method that reads blocks from storage and delegates
        to rebuild_session_facets.

        Args:
            storage: Storage implementation with get_blocks method
            user_id: User ID for access control
            session_id: Session to rebuild facets for
            session_metadata: Optional metadata (will read from storage if not provided)

        Returns:
            RebuildResult with computed facets
        """
        # Read all blocks for the session
        blocks = await storage.get_blocks(
            user_id=user_id,
            session_id=session_id,
            block_types=None,  # Get all types
            limit=None,  # No limit
        )

        return await self.rebuild_session_facets(blocks, session_metadata)

    async def backfill_sessions(
        self,
        storage: BlockReader,
        user_id: str,
        session_ids: list[str],
        skip_if_exists: bool = True,
        on_progress: Any | None = None,
    ) -> BackfillResult:
        """Backfill facets for multiple sessions.

        Processes sessions in sequence, rebuilding facets for each.
        Useful for migrating existing sessions to facets.

        Args:
            storage: Storage implementation
            user_id: User ID for access control
            session_ids: List of session IDs to process
            skip_if_exists: Skip sessions that already have facets
            on_progress: Optional callback(BackfillProgress) for progress updates

        Returns:
            BackfillResult with statistics
        """
        start_time = datetime.now(UTC)

        progress = BackfillProgress(
            total_sessions=len(session_ids),
            processed_sessions=0,
            successful=0,
            failed=0,
            skipped=0,
        )

        failed_session_ids: list[str] = []

        for session_id in session_ids:
            progress.current_session_id = session_id

            try:
                # Rebuild facets
                result = await self.rebuild_from_storage(
                    storage=storage,
                    user_id=user_id,
                    session_id=session_id,
                )

                if result.success:
                    progress.successful += 1
                else:
                    progress.failed += 1
                    failed_session_ids.append(session_id)

            except Exception:
                progress.failed += 1
                failed_session_ids.append(session_id)

            progress.processed_sessions += 1

            # Report progress
            if on_progress:
                on_progress(progress)

        end_time = datetime.now(UTC)
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        return BackfillResult(
            total_sessions=len(session_ids),
            successful=progress.successful,
            failed=progress.failed,
            skipped=progress.skipped,
            duration_ms=duration_ms,
            failed_session_ids=failed_session_ids,
        )

    def verify_facets(
        self,
        existing: SessionFacets,
        rebuilt: SessionFacets,
    ) -> dict[str, Any]:
        """Compare existing facets with rebuilt facets.

        Useful for verifying facet accuracy or detecting drift.

        Args:
            existing: Currently stored facets
            rebuilt: Freshly rebuilt facets

        Returns:
            Dict with comparison results and any discrepancies
        """
        discrepancies: list[dict[str, Any]] = []

        # Compare key fields
        fields_to_check = [
            ("bundle", str),
            ("tool_call_count", int),
            ("total_input_tokens", int),
            ("total_output_tokens", int),
            ("error_count", int),
            ("child_session_count", int),
            ("has_errors", bool),
            ("has_child_sessions", bool),
            ("has_recipes", bool),
            ("workflow_pattern", str),
        ]

        for field_name, _ in fields_to_check:
            existing_value = getattr(existing, field_name)
            rebuilt_value = getattr(rebuilt, field_name)

            if existing_value != rebuilt_value:
                discrepancies.append(
                    {
                        "field": field_name,
                        "existing": existing_value,
                        "rebuilt": rebuilt_value,
                    }
                )

        # Compare list fields (order-independent)
        list_fields = [
            "tools_used",
            "models_used",
            "providers_used",
            "error_types",
            "agents_delegated_to",
            "recipe_names",
            "languages_detected",
        ]

        for field_name in list_fields:
            existing_set = set(getattr(existing, field_name))
            rebuilt_set = set(getattr(rebuilt, field_name))

            if existing_set != rebuilt_set:
                discrepancies.append(
                    {
                        "field": field_name,
                        "existing": sorted(existing_set),
                        "rebuilt": sorted(rebuilt_set),
                        "missing": sorted(rebuilt_set - existing_set),
                        "extra": sorted(existing_set - rebuilt_set),
                    }
                )

        return {
            "is_accurate": len(discrepancies) == 0,
            "discrepancy_count": len(discrepancies),
            "discrepancies": discrepancies,
        }


# =============================================================================
# Convenience Functions
# =============================================================================


async def rebuild_facets(
    blocks: list[SessionBlock],
    session_metadata: dict[str, Any] | None = None,
) -> SessionFacets:
    """Convenience function to rebuild facets from blocks.

    Args:
        blocks: List of session blocks
        session_metadata: Optional session metadata

    Returns:
        Rebuilt SessionFacets
    """
    rebuilder = FacetsRebuilder()
    result = await rebuilder.rebuild_session_facets(blocks, session_metadata)
    return result.facets
