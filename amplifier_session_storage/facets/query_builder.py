"""
Cosmos DB query builder for FacetQuery.

Translates FacetQuery objects into Cosmos SQL queries for efficient
server-side filtering. Supports:
- Basic metadata filters (user_id, project_slug, created_after, etc.)
- Facet-based filters (bundle, tools_used, has_errors, tokens, etc.)
- Ordering and pagination
- Parameterized queries for security

The builder generates queries that leverage Cosmos DB's composite indexes
for optimal performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .types import FacetQuery


@dataclass
class CosmosQuery:
    """A parameterized Cosmos DB query.

    Attributes:
        sql: The SQL query string with @parameter placeholders
        parameters: List of parameter dictionaries for the query
        projection: Fields to return (SELECT clause)
        is_count_query: Whether this is a COUNT query
    """

    sql: str
    parameters: list[dict[str, Any]] = field(default_factory=list)
    projection: list[str] = field(default_factory=list)
    is_count_query: bool = False

    def __str__(self) -> str:
        """Return formatted query for debugging."""
        param_str = ", ".join(f"{p['name']}={p['value']!r}" for p in self.parameters)
        return f"{self.sql}\nParameters: {param_str}"


class FacetQueryBuilder:
    """Builds Cosmos SQL queries from FacetQuery objects.

    This builder generates parameterized queries that:
    - Filter on metadata fields (user_id, project_slug, timestamps)
    - Filter on facet fields (bundle, tools_used, workflow_pattern, etc.)
    - Support complex conditions (IN, ARRAY_CONTAINS, range comparisons)
    - Order results and handle pagination

    Usage:
        builder = FacetQueryBuilder()
        query = builder.build(facet_query)
        # Execute: container.query_items(query.sql, parameters=query.parameters)

    Note:
        All facet fields are assumed to be stored under a 'facets' property
        in the Cosmos document: c.facets.bundle, c.facets.tools_used, etc.
    """

    # Document type for session metadata documents
    SESSION_DOC_TYPE = "session_metadata"

    # Base SELECT fields for session queries
    BASE_PROJECTION = [
        "c.id",
        "c.session_id",
        "c.user_id",
        "c.project_slug",
        "c.created",
        "c.updated",
        "c.name",
        "c.bundle",
        "c.model",
        "c.turn_count",
        "c.event_count",
        "c.facets",
    ]

    def __init__(self, facets_path: str = "c.facets") -> None:
        """Initialize the query builder.

        Args:
            facets_path: Path to facets in document (default: c.facets)
        """
        self.facets_path = facets_path
        self._param_counter = 0

    def _next_param(self, prefix: str = "p") -> str:
        """Generate a unique parameter name."""
        self._param_counter += 1
        return f"@{prefix}{self._param_counter}"

    def _reset_params(self) -> None:
        """Reset parameter counter for a new query."""
        self._param_counter = 0

    def build(self, query: FacetQuery) -> CosmosQuery:
        """Build a Cosmos SQL query from a FacetQuery.

        Args:
            query: The FacetQuery to translate

        Returns:
            CosmosQuery with SQL and parameters
        """
        self._reset_params()
        conditions: list[str] = []
        parameters: list[dict[str, Any]] = []

        # Always filter by document type
        conditions.append(f"c.doc_type = '{self.SESSION_DOC_TYPE}'")

        # Build metadata conditions
        self._add_metadata_conditions(query, conditions, parameters)

        # Build facet conditions
        self._add_facet_conditions(query, conditions, parameters)

        # Build WHERE clause
        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Build ORDER BY clause
        order_clause = self._build_order_clause(query)

        # Build SELECT clause
        select_clause = ", ".join(self.BASE_PROJECTION)

        # Build full query with pagination
        sql = f"SELECT {select_clause} FROM c WHERE {where_clause} {order_clause}"

        # Add OFFSET/LIMIT for pagination
        if query.offset > 0:
            offset_param = self._next_param("offset")
            limit_param = self._next_param("limit")
            sql += f" OFFSET {offset_param} LIMIT {limit_param}"
            parameters.append({"name": offset_param, "value": query.offset})
            parameters.append({"name": limit_param, "value": query.limit})
        else:
            limit_param = self._next_param("limit")
            sql += f" OFFSET 0 LIMIT {limit_param}"
            parameters.append({"name": limit_param, "value": query.limit})

        return CosmosQuery(
            sql=sql,
            parameters=parameters,
            projection=self.BASE_PROJECTION,
        )

    def build_count(self, query: FacetQuery) -> CosmosQuery:
        """Build a COUNT query from a FacetQuery.

        Args:
            query: The FacetQuery to translate

        Returns:
            CosmosQuery for counting matching documents
        """
        self._reset_params()
        conditions: list[str] = []
        parameters: list[dict[str, Any]] = []

        # Always filter by document type
        conditions.append(f"c.doc_type = '{self.SESSION_DOC_TYPE}'")

        # Build conditions (same as regular query, without ordering/pagination)
        self._add_metadata_conditions(query, conditions, parameters)
        self._add_facet_conditions(query, conditions, parameters)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT VALUE COUNT(1) FROM c WHERE {where_clause}"

        return CosmosQuery(
            sql=sql,
            parameters=parameters,
            is_count_query=True,
        )

    def _add_metadata_conditions(
        self,
        query: FacetQuery,
        conditions: list[str],
        parameters: list[dict[str, Any]],
    ) -> None:
        """Add metadata-based conditions to the query."""
        # User ID filter (required for most queries)
        if query.user_id:
            param = self._next_param("user")
            conditions.append(f"c.user_id = {param}")
            parameters.append({"name": param, "value": query.user_id})

        # Project slug filter
        if query.project_slug:
            param = self._next_param("proj")
            conditions.append(f"c.project_slug = {param}")
            parameters.append({"name": param, "value": query.project_slug})

        # Session ID filter
        if query.session_id:
            param = self._next_param("sess")
            conditions.append(f"c.session_id = {param}")
            parameters.append({"name": param, "value": query.session_id})

        # Parent ID filter
        if query.parent_id:
            param = self._next_param("parent")
            conditions.append(f"c.parent_id = {param}")
            parameters.append({"name": param, "value": query.parent_id})

        # Created timestamp filters
        if query.created_after:
            param = self._next_param("created_after")
            conditions.append(f"c.created >= {param}")
            parameters.append({"name": param, "value": self._format_datetime(query.created_after)})

        if query.created_before:
            param = self._next_param("created_before")
            conditions.append(f"c.created <= {param}")
            parameters.append({"name": param, "value": self._format_datetime(query.created_before)})

        # Updated timestamp filters
        if query.updated_after:
            param = self._next_param("updated_after")
            conditions.append(f"c.updated >= {param}")
            parameters.append({"name": param, "value": self._format_datetime(query.updated_after)})

        if query.updated_before:
            param = self._next_param("updated_before")
            conditions.append(f"c.updated <= {param}")
            parameters.append({"name": param, "value": self._format_datetime(query.updated_before)})

        # Tags filter (ARRAY_CONTAINS)
        if query.tags:
            for tag in query.tags:
                param = self._next_param("tag")
                conditions.append(f"ARRAY_CONTAINS(c.tags, {param})")
                parameters.append({"name": param, "value": tag})

    def _add_facet_conditions(
        self,
        query: FacetQuery,
        conditions: list[str],
        parameters: list[dict[str, Any]],
    ) -> None:
        """Add facet-based conditions to the query."""
        fp = self.facets_path  # Shorthand

        # Bundle filter
        if query.bundle:
            param = self._next_param("bundle")
            conditions.append(f"{fp}.bundle = {param}")
            parameters.append({"name": param, "value": query.bundle})

        # Model filter
        if query.model:
            param = self._next_param("model")
            # Check either initial_model or models_used array
            conditions.append(
                f"({fp}.initial_model = {param} OR ARRAY_CONTAINS({fp}.models_used, {param}))"
            )
            parameters.append({"name": param, "value": query.model})

        # Provider filter
        if query.provider:
            param = self._next_param("provider")
            conditions.append(
                f"({fp}.initial_provider = {param} OR ARRAY_CONTAINS({fp}.providers_used, {param}))"
            )
            parameters.append({"name": param, "value": query.provider})

        # Tool used filter (ARRAY_CONTAINS)
        if query.tool_used:
            param = self._next_param("tool")
            conditions.append(f"ARRAY_CONTAINS({fp}.tools_used, {param})")
            parameters.append({"name": param, "value": query.tool_used})

        # Agent delegated to filter
        if query.agent_delegated_to:
            param = self._next_param("agent")
            conditions.append(f"ARRAY_CONTAINS({fp}.agents_delegated_to, {param})")
            parameters.append({"name": param, "value": query.agent_delegated_to})

        # Boolean filters
        if query.has_errors is not None:
            param = self._next_param("errors")
            conditions.append(f"{fp}.has_errors = {param}")
            parameters.append({"name": param, "value": query.has_errors})

        if query.has_child_sessions is not None:
            param = self._next_param("children")
            conditions.append(f"{fp}.has_child_sessions = {param}")
            parameters.append({"name": param, "value": query.has_child_sessions})

        if query.has_recipes is not None:
            param = self._next_param("recipes")
            conditions.append(f"{fp}.has_recipes = {param}")
            parameters.append({"name": param, "value": query.has_recipes})

        if query.has_file_operations is not None:
            param = self._next_param("fileops")
            conditions.append(f"{fp}.has_file_operations = {param}")
            parameters.append({"name": param, "value": query.has_file_operations})

        # Workflow pattern filter
        if query.workflow_pattern:
            param = self._next_param("pattern")
            conditions.append(f"{fp}.workflow_pattern = {param}")
            parameters.append({"name": param, "value": query.workflow_pattern})

        # Token range filters
        if query.min_tokens is not None:
            param = self._next_param("min_tok")
            conditions.append(f"{fp}.total_tokens >= {param}")
            parameters.append({"name": param, "value": query.min_tokens})

        if query.max_tokens is not None:
            param = self._next_param("max_tok")
            conditions.append(f"{fp}.total_tokens <= {param}")
            parameters.append({"name": param, "value": query.max_tokens})

        # Turn count range filters
        if query.min_turns is not None:
            param = self._next_param("min_turns")
            conditions.append(f"{fp}.max_turn >= {param}")
            parameters.append({"name": param, "value": query.min_turns})

        if query.max_turns is not None:
            param = self._next_param("max_turns")
            conditions.append(f"{fp}.max_turn <= {param}")
            parameters.append({"name": param, "value": query.max_turns})

        # Recipe name filter
        if query.recipe_name:
            param = self._next_param("recipe")
            conditions.append(f"ARRAY_CONTAINS({fp}.recipe_names, {param})")
            parameters.append({"name": param, "value": query.recipe_name})

        # Error type filter
        if query.error_type:
            param = self._next_param("err_type")
            conditions.append(f"ARRAY_CONTAINS({fp}.error_types, {param})")
            parameters.append({"name": param, "value": query.error_type})

    def _build_order_clause(self, query: FacetQuery) -> str:
        """Build the ORDER BY clause."""
        order_field = query.order_by or "updated"
        direction = "DESC" if query.order_desc else "ASC"

        # Map logical field names to document paths
        field_mapping = {
            "created": "c.created",
            "updated": "c.updated",
            "tokens": f"{self.facets_path}.total_tokens",
            "turns": f"{self.facets_path}.max_turn",
            "duration": f"{self.facets_path}.total_duration_ms",
            "name": "c.name",
        }

        order_path = field_mapping.get(order_field, f"c.{order_field}")
        return f"ORDER BY {order_path} {direction}"

    def _format_datetime(self, dt: datetime) -> str:
        """Format datetime for Cosmos DB query."""
        return dt.isoformat()


# =============================================================================
# Convenience Functions
# =============================================================================


def build_cosmos_query(query: FacetQuery) -> CosmosQuery:
    """Build a Cosmos query from a FacetQuery.

    Convenience function for simple use cases.

    Args:
        query: The FacetQuery to translate

    Returns:
        CosmosQuery ready for execution
    """
    builder = FacetQueryBuilder()
    return builder.build(query)


def build_cosmos_count_query(query: FacetQuery) -> CosmosQuery:
    """Build a Cosmos COUNT query from a FacetQuery.

    Args:
        query: The FacetQuery to translate

    Returns:
        CosmosQuery for counting
    """
    builder = FacetQueryBuilder()
    return builder.build_count(query)
