"""
Session facets module for efficient server-side filtering.

This module provides:
- SessionFacets: Denormalized aggregates for queryable session attributes
- FacetQuery: Query parameters for facet-based filtering
- FacetsUpdater: Incremental facet updates from events
- FacetsRebuilder: Full reconstruction from event streams

Usage:
    from amplifier_session_storage.facets import (
        SessionFacets,
        FacetQuery,
        FacetsUpdater,
        FacetsRebuilder,
        rebuild_facets,
    )

    # Incremental updates during session
    updater = FacetsUpdater()
    facets = SessionFacets()
    facets = updater.process_event(facets, "tool_result", event_data, timestamp)

    # Full rebuild from blocks
    facets = await rebuild_facets(blocks)

    # Query with facet filters
    query = FacetQuery(
        user_id="user-123",
        bundle="amplifier-dev",
        tool_used="delegate",
        has_errors=False,
    )
"""

from .indexes import (
    ARRAY_INDEX_PATHS,
    RECOMMENDED_COMPOSITE_INDEXES,
    CompositeIndex,
    IndexPath,
    generate_index_cli_commands,
    generate_index_policy,
)
from .query_builder import (
    CosmosQuery,
    FacetQueryBuilder,
    build_cosmos_count_query,
    build_cosmos_query,
)
from .rebuilder import (
    BackfillProgress,
    BackfillResult,
    FacetsRebuilder,
    RebuildResult,
    rebuild_facets,
)
from .service import (
    BatchRefreshResult,
    FacetsService,
    FacetsStorageProtocol,
    RefreshResult,
    compute_facets_from_blocks,
    compute_facets_incrementally,
)
from .types import FacetQuery, SessionFacets, WorkflowPattern
from .updater import FacetsUpdater, UpdateResult

__all__ = [
    # Types
    "SessionFacets",
    "FacetQuery",
    "WorkflowPattern",
    # Updater
    "FacetsUpdater",
    "UpdateResult",
    # Rebuilder
    "FacetsRebuilder",
    "RebuildResult",
    "BackfillProgress",
    "BackfillResult",
    "rebuild_facets",
    # Service
    "FacetsService",
    "FacetsStorageProtocol",
    "RefreshResult",
    "BatchRefreshResult",
    "compute_facets_from_blocks",
    "compute_facets_incrementally",
    # Query Builder
    "FacetQueryBuilder",
    "CosmosQuery",
    "build_cosmos_query",
    "build_cosmos_count_query",
    # Indexes
    "IndexPath",
    "CompositeIndex",
    "RECOMMENDED_COMPOSITE_INDEXES",
    "ARRAY_INDEX_PATHS",
    "generate_index_policy",
    "generate_index_cli_commands",
]
