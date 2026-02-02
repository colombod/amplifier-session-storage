"""
Cosmos DB index configuration for facet queries.

This module defines the recommended composite indexes for efficient
facet-based queries. These indexes should be configured on the Cosmos DB
container to optimize query performance.

Index Design Principles:
1. Composite indexes support equality + range queries in order
2. Array fields (tools_used, etc.) benefit from single-field indexes
3. Most queries filter by user_id first, so it's the leading field
4. Ordering fields should be included for ORDER BY optimization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class IndexPath:
    """A single path in a composite index."""

    path: str
    order: Literal["ascending", "descending"] = "ascending"

    def to_dict(self) -> dict[str, str]:
        """Convert to Cosmos index specification format."""
        return {"path": self.path, "order": self.order}


@dataclass
class CompositeIndex:
    """A composite index definition for Cosmos DB."""

    name: str
    paths: list[IndexPath]
    description: str = ""

    def to_cosmos_spec(self) -> list[dict[str, str]]:
        """Convert to Cosmos DB compositeIndexes format."""
        return [p.to_dict() for p in self.paths]


# =============================================================================
# Recommended Indexes for Facet Queries
# =============================================================================

# User sessions by update time (most common query pattern)
INDEX_USER_SESSIONS_BY_UPDATED = CompositeIndex(
    name="user_sessions_by_updated",
    description="List user's sessions ordered by last update",
    paths=[
        IndexPath("/user_id"),
        IndexPath("/doc_type"),
        IndexPath("/updated", "descending"),
    ],
)

# User sessions by creation time
INDEX_USER_SESSIONS_BY_CREATED = CompositeIndex(
    name="user_sessions_by_created",
    description="List user's sessions ordered by creation time",
    paths=[
        IndexPath("/user_id"),
        IndexPath("/doc_type"),
        IndexPath("/created", "descending"),
    ],
)

# User sessions by project and update time
INDEX_USER_PROJECT_SESSIONS = CompositeIndex(
    name="user_project_sessions",
    description="List user's sessions in a project",
    paths=[
        IndexPath("/user_id"),
        IndexPath("/project_slug"),
        IndexPath("/doc_type"),
        IndexPath("/updated", "descending"),
    ],
)

# Sessions by bundle (for bundle analysis)
INDEX_USER_BUNDLE_SESSIONS = CompositeIndex(
    name="user_bundle_sessions",
    description="Find sessions using a specific bundle",
    paths=[
        IndexPath("/user_id"),
        IndexPath("/facets/bundle"),
        IndexPath("/doc_type"),
        IndexPath("/updated", "descending"),
    ],
)

# Sessions by workflow pattern
INDEX_USER_WORKFLOW_SESSIONS = CompositeIndex(
    name="user_workflow_sessions",
    description="Find sessions by workflow pattern",
    paths=[
        IndexPath("/user_id"),
        IndexPath("/facets/workflow_pattern"),
        IndexPath("/doc_type"),
        IndexPath("/updated", "descending"),
    ],
)

# Sessions with errors
INDEX_USER_ERROR_SESSIONS = CompositeIndex(
    name="user_error_sessions",
    description="Find sessions with errors",
    paths=[
        IndexPath("/user_id"),
        IndexPath("/facets/has_errors"),
        IndexPath("/doc_type"),
        IndexPath("/updated", "descending"),
    ],
)

# Sessions by token usage (for cost analysis)
INDEX_USER_SESSIONS_BY_TOKENS = CompositeIndex(
    name="user_sessions_by_tokens",
    description="Order sessions by token usage",
    paths=[
        IndexPath("/user_id"),
        IndexPath("/doc_type"),
        IndexPath("/facets/total_tokens", "descending"),
    ],
)

# Multi-agent sessions
INDEX_USER_MULTIAGENT_SESSIONS = CompositeIndex(
    name="user_multiagent_sessions",
    description="Find sessions with child sessions",
    paths=[
        IndexPath("/user_id"),
        IndexPath("/facets/has_child_sessions"),
        IndexPath("/doc_type"),
        IndexPath("/updated", "descending"),
    ],
)

# Sessions with recipes
INDEX_USER_RECIPE_SESSIONS = CompositeIndex(
    name="user_recipe_sessions",
    description="Find sessions that used recipes",
    paths=[
        IndexPath("/user_id"),
        IndexPath("/facets/has_recipes"),
        IndexPath("/doc_type"),
        IndexPath("/updated", "descending"),
    ],
)

# Child sessions by parent
INDEX_CHILD_SESSIONS = CompositeIndex(
    name="child_sessions_by_parent",
    description="Find child sessions for a parent session",
    paths=[
        IndexPath("/parent_id"),
        IndexPath("/doc_type"),
        IndexPath("/created", "ascending"),
    ],
)

# =============================================================================
# All Recommended Indexes
# =============================================================================

RECOMMENDED_COMPOSITE_INDEXES: list[CompositeIndex] = [
    INDEX_USER_SESSIONS_BY_UPDATED,
    INDEX_USER_SESSIONS_BY_CREATED,
    INDEX_USER_PROJECT_SESSIONS,
    INDEX_USER_BUNDLE_SESSIONS,
    INDEX_USER_WORKFLOW_SESSIONS,
    INDEX_USER_ERROR_SESSIONS,
    INDEX_USER_SESSIONS_BY_TOKENS,
    INDEX_USER_MULTIAGENT_SESSIONS,
    INDEX_USER_RECIPE_SESSIONS,
    INDEX_CHILD_SESSIONS,
]

# =============================================================================
# Single-Field Indexes for Array Contains
# =============================================================================

# These paths should be indexed for ARRAY_CONTAINS queries
ARRAY_INDEX_PATHS: list[str] = [
    "/facets/tools_used/*",
    "/facets/models_used/*",
    "/facets/providers_used/*",
    "/facets/agents_delegated_to/*",
    "/facets/recipe_names/*",
    "/facets/error_types/*",
    "/facets/languages_detected/*",
    "/tags/*",
]


# =============================================================================
# Index Policy Generation
# =============================================================================


def generate_index_policy(
    include_spatial: bool = False,
    exclude_large_fields: bool = True,
) -> dict[str, Any]:
    """Generate a complete Cosmos DB index policy.

    Args:
        include_spatial: Whether to include spatial indexes
        exclude_large_fields: Whether to exclude potentially large fields

    Returns:
        A dictionary suitable for Cosmos DB container indexing policy
    """
    # Excluded paths (large content fields that shouldn't be indexed)
    excluded_paths: list[dict[str, str]] = []
    if exclude_large_fields:
        excluded_paths = [
            {"path": "/data/*"},  # Event data can be huge
            {"path": "/content/*"},  # Message content
            {"path": "/transcript/*"},  # Full transcript
            {"path": '/"_etag"/?'},  # System field
        ]

    # Build composite indexes specification
    composite_indexes: list[list[dict[str, str]]] = [
        idx.to_cosmos_spec() for idx in RECOMMENDED_COMPOSITE_INDEXES
    ]

    policy: dict[str, Any] = {
        "indexingMode": "consistent",
        "automatic": True,
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": excluded_paths,
        "compositeIndexes": composite_indexes,
    }

    if include_spatial:
        policy["spatialIndexes"] = []

    return policy


def generate_index_cli_commands(
    database_name: str = "amplifier-db",
    container_name: str = "items",
) -> str:
    """Generate Azure CLI commands to create the indexes.

    Args:
        database_name: Cosmos DB database name
        container_name: Container name

    Returns:
        Azure CLI commands as a string
    """
    import json

    policy = generate_index_policy()
    policy_json = json.dumps(policy, indent=2)

    return f"""# Update Cosmos DB container indexing policy
# First, export the current policy:
az cosmosdb sql container show \\
  --resource-group <resource-group> \\
  --account-name <account-name> \\
  --database-name {database_name} \\
  --name {container_name} \\
  --query "resource.indexingPolicy"

# Then update with new policy:
az cosmosdb sql container update \\
  --resource-group <resource-group> \\
  --account-name <account-name> \\
  --database-name {database_name} \\
  --name {container_name} \\
  --idx @indexing-policy.json

# Save this policy to indexing-policy.json:
cat > indexing-policy.json << 'EOF'
{policy_json}
EOF
"""


def print_index_summary() -> None:
    """Print a summary of recommended indexes for documentation."""
    print("=" * 70)
    print("RECOMMENDED COMPOSITE INDEXES FOR FACET QUERIES")
    print("=" * 70)
    print()

    for idx in RECOMMENDED_COMPOSITE_INDEXES:
        print(f"Index: {idx.name}")
        print(f"  Description: {idx.description}")
        print("  Paths:")
        for p in idx.paths:
            print(f"    - {p.path} ({p.order})")
        print()

    print("=" * 70)
    print("ARRAY INDEX PATHS (for ARRAY_CONTAINS queries)")
    print("=" * 70)
    print()
    for path in ARRAY_INDEX_PATHS:
        print(f"  - {path}")
    print()


if __name__ == "__main__":
    print_index_summary()
    print()
    print("=" * 70)
    print("AZURE CLI COMMANDS")
    print("=" * 70)
    print(generate_index_cli_commands())
