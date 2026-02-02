"""
Facets service for computing and storing session facets.

This module provides a service layer that coordinates:
- Computing facets from session blocks
- Storing facets with session metadata
- Handling refresh triggers (on-demand, session close, batch)

The service is designed to be used by storage backends and
external tools like session analyst.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from ..blocks.types import SessionBlock
from .rebuilder import FacetsRebuilder
from .types import SessionFacets
from .updater import FacetsUpdater

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


# =============================================================================
# Storage Protocol (to avoid circular imports)
# =============================================================================


class FacetsStorageProtocol(Protocol):
    """Protocol for storage backends that support facets."""

    async def read_blocks(
        self,
        session_id: str,
        since_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[SessionBlock]:
        """Read blocks for a session."""
        ...

    async def update_session_facets(
        self,
        session_id: str,
        facets: SessionFacets,
    ) -> None:
        """Update the facets for a session."""
        ...


# =============================================================================
# Service Results
# =============================================================================


@dataclass
class RefreshResult:
    """Result of a facets refresh operation."""

    session_id: str
    facets: SessionFacets
    blocks_processed: int
    duration_ms: int
    was_stale: bool
    error: str | None = None

    @property
    def success(self) -> bool:
        """Whether the refresh completed without errors."""
        return self.error is None


@dataclass
class BatchRefreshResult:
    """Result of a batch refresh operation."""

    total_sessions: int
    successful: int
    failed: int
    skipped: int
    duration_ms: int
    results: list[RefreshResult]

    @property
    def success_rate(self) -> float:
        """Percentage of successful refreshes."""
        if self.total_sessions == 0:
            return 100.0
        return (self.successful / self.total_sessions) * 100


# =============================================================================
# Facets Service
# =============================================================================


class FacetsService:
    """Service for computing and managing session facets.

    This service provides methods to:
    - Refresh facets for a single session (full rebuild from blocks)
    - Batch refresh facets for multiple sessions
    - Check if facets are stale and need refresh
    - Verify facets accuracy

    Usage:
        service = FacetsService()

        # Refresh facets for a session
        result = await service.refresh_session_facets(storage, session_id)

        # Batch refresh
        result = await service.batch_refresh(storage, session_ids)
    """

    def __init__(
        self,
        updater: FacetsUpdater | None = None,
        rebuilder: FacetsRebuilder | None = None,
    ) -> None:
        """Initialize the service.

        Args:
            updater: Optional FacetsUpdater instance
            rebuilder: Optional FacetsRebuilder instance
        """
        self.updater = updater or FacetsUpdater()
        self.rebuilder = rebuilder or FacetsRebuilder(self.updater)

    async def refresh_session_facets(
        self,
        storage: FacetsStorageProtocol,
        session_id: str,
        existing_facets: SessionFacets | None = None,
        force: bool = False,
    ) -> RefreshResult:
        """Refresh facets for a single session.

        Rebuilds facets from the session's blocks and optionally
        updates storage.

        Args:
            storage: Storage backend with read_blocks method
            session_id: Session to refresh facets for
            existing_facets: Current facets (to check staleness)
            force: If True, refresh even if not stale

        Returns:
            RefreshResult with computed facets and statistics
        """
        start_time = datetime.now(UTC)
        was_stale = existing_facets.is_stale if existing_facets else True

        # Check if refresh is needed
        if not force and existing_facets and not existing_facets.is_stale:
            return RefreshResult(
                session_id=session_id,
                facets=existing_facets,
                blocks_processed=0,
                duration_ms=0,
                was_stale=False,
            )

        try:
            # Read all blocks for the session
            blocks = await storage.read_blocks(session_id)

            # Rebuild facets
            rebuild_result = await self.rebuilder.rebuild_session_facets(blocks)

            # Calculate duration
            end_time = datetime.now(UTC)
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            if rebuild_result.errors:
                logger.warning(
                    "Facets rebuild had errors for session %s: %s",
                    session_id,
                    rebuild_result.errors,
                )

            return RefreshResult(
                session_id=session_id,
                facets=rebuild_result.facets,
                blocks_processed=rebuild_result.blocks_processed,
                duration_ms=duration_ms,
                was_stale=was_stale,
                error="; ".join(rebuild_result.errors) if rebuild_result.errors else None,
            )

        except Exception as e:
            end_time = datetime.now(UTC)
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            logger.error("Failed to refresh facets for session %s: %s", session_id, e)

            return RefreshResult(
                session_id=session_id,
                facets=existing_facets or SessionFacets(),
                blocks_processed=0,
                duration_ms=duration_ms,
                was_stale=was_stale,
                error=str(e),
            )

    async def refresh_and_store(
        self,
        storage: FacetsStorageProtocol,
        session_id: str,
        existing_facets: SessionFacets | None = None,
        force: bool = False,
    ) -> RefreshResult:
        """Refresh facets and store them back to storage.

        Combines refresh_session_facets with storage update.

        Args:
            storage: Storage backend with read_blocks and update_session_facets
            session_id: Session to refresh
            existing_facets: Current facets (to check staleness)
            force: If True, refresh even if not stale

        Returns:
            RefreshResult with computed facets
        """
        result = await self.refresh_session_facets(storage, session_id, existing_facets, force)

        if result.success and result.blocks_processed > 0:
            try:
                await storage.update_session_facets(session_id, result.facets)
            except Exception as e:
                logger.error("Failed to store facets for session %s: %s", session_id, e)
                # Return result with store error appended
                return RefreshResult(
                    session_id=result.session_id,
                    facets=result.facets,
                    blocks_processed=result.blocks_processed,
                    duration_ms=result.duration_ms,
                    was_stale=result.was_stale,
                    error=f"{result.error or ''}; Store failed: {e}".strip("; "),
                )

        return result

    async def batch_refresh(
        self,
        storage: FacetsStorageProtocol,
        session_ids: list[str],
        existing_facets_map: dict[str, SessionFacets] | None = None,
        force: bool = False,
        store_results: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> BatchRefreshResult:
        """Batch refresh facets for multiple sessions.

        Args:
            storage: Storage backend
            session_ids: List of session IDs to refresh
            existing_facets_map: Map of session_id -> existing facets
            force: If True, refresh even if not stale
            store_results: If True, store updated facets
            on_progress: Optional callback(processed, total) for progress

        Returns:
            BatchRefreshResult with statistics
        """
        start_time = datetime.now(UTC)
        existing_map = existing_facets_map or {}
        results: list[RefreshResult] = []
        successful = 0
        failed = 0
        skipped = 0

        for i, session_id in enumerate(session_ids):
            existing = existing_map.get(session_id)

            if store_results:
                result = await self.refresh_and_store(storage, session_id, existing, force)
            else:
                result = await self.refresh_session_facets(storage, session_id, existing, force)

            results.append(result)

            if result.error:
                failed += 1
            elif result.blocks_processed == 0 and not result.was_stale:
                skipped += 1
            else:
                successful += 1

            if on_progress:
                on_progress(i + 1, len(session_ids))

        end_time = datetime.now(UTC)
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        return BatchRefreshResult(
            total_sessions=len(session_ids),
            successful=successful,
            failed=failed,
            skipped=skipped,
            duration_ms=duration_ms,
            results=results,
        )

    def check_freshness(
        self,
        facets: SessionFacets | None,
        max_age_seconds: int = 3600,
    ) -> bool:
        """Check if facets are fresh (not stale).

        Args:
            facets: Facets to check
            max_age_seconds: Maximum age before considered stale

        Returns:
            True if facets are fresh, False if stale or missing
        """
        if facets is None:
            return False

        if facets.is_stale:
            return False

        if facets.last_computed is None:
            return False

        age = (datetime.now(UTC) - facets.last_computed).total_seconds()
        return age < max_age_seconds


# =============================================================================
# Convenience Functions
# =============================================================================


async def compute_facets_from_blocks(blocks: list[SessionBlock]) -> SessionFacets:
    """Compute facets from a list of blocks.

    Convenience function for external tools that have blocks but
    don't need the full service.

    Args:
        blocks: List of session blocks

    Returns:
        Computed SessionFacets
    """
    rebuilder = FacetsRebuilder()
    result = await rebuilder.rebuild_session_facets(blocks)
    return result.facets


def compute_facets_incrementally(
    facets: SessionFacets,
    event_type: str,
    data: dict[str, Any],
    timestamp: datetime,
) -> SessionFacets:
    """Update facets incrementally from a single event.

    Convenience function for real-time facet updates during
    session execution.

    Args:
        facets: Current facets state
        event_type: Type of event
        data: Event data
        timestamp: Event timestamp

    Returns:
        Updated SessionFacets
    """
    updater = FacetsUpdater()
    return updater.process_event(facets, event_type, data, timestamp)
