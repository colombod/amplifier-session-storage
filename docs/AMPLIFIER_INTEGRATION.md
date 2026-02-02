# Amplifier Integration Guide

This guide explains how to integrate `amplifier-session-storage` with Amplifier applications and expose session capabilities to agents like session-analyst.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Amplifier Application (e.g., amplifier-app-cli)                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Session Manager                                                        ││
│  │  - Creates AmplifierSession instances                                   ││
│  │  - Configures storage backend                                           ││
│  │  - Registers tools with sessions                                        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Session Storage Library (this package)                                 ││
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               ││
│  │  │ Local Storage │  │ Cosmos Storage│  │ Hybrid Storage│               ││
│  │  └───────────────┘  └───────────────┘  └───────────────┘               ││
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               ││
│  │  │  SessionTool  │  │    Facets     │  │ Query Builder │               ││
│  │  └───────────────┘  └───────────────┘  └───────────────┘               ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Agents (via delegation)                                                ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         ││
│  │  │ session-analyst │  │ stories-bundle  │  │  custom agents  │         ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

## Integration Approaches

### Approach 1: Behavior Composition (Recommended)

The cleanest integration is via behavior composition. Agents that need session access compose the `session-storage` behavior.

**In your bundle or behavior file:**

```yaml
bundle:
  name: my-agent
  version: "1.0.0"

includes:
  - session-storage:behaviors/session-storage  # Compose the behavior

# Or inline the behavior reference
behaviors:
  - session-storage
```

**What this provides:**
- The `session` tool with all operations (list, get, search, query, analyze, rewind)
- Context instructions for safe session handling
- Automatic safety guarantees (bounded outputs, pagination)

### Approach 2: Direct Tool Registration

For applications that need more control, register `SessionTool` directly with your tool registry.

**In your application code:**

```python
from amplifier_session_storage.tools import SessionTool, SessionToolConfig
from amplifier_session_storage.storage import HybridBlockStorage
from amplifier_session_storage.facets import FacetsService

# Configure storage backend
config = SessionToolConfig(
    base_dir=Path("~/.amplifier/projects/default/sessions").expanduser(),
    project_slug="default",
    enable_cloud=True,  # Enable Cosmos DB backend
    max_results=50,
)

# Create the tool
session_tool = SessionTool(config)

# Register with your tool registry
tool_registry.register("session", session_tool)
```

### Approach 3: Block Storage Integration

For full control over storage operations, integrate at the block storage level.

```python
from amplifier_session_storage.storage import HybridBlockStorage, StorageConfig
from amplifier_session_storage.facets import FacetsService, FacetQuery

# Configure hybrid storage (local + Cosmos)
storage_config = StorageConfig(
    local_dir=Path("~/.amplifier/sessions"),
    cosmos_connection_string=os.environ.get("COSMOS_CONNECTION_STRING"),
    cosmos_database="amplifier-db",
    cosmos_container="sessions",
    sync_mode="write_through",  # or "write_behind", "read_through"
)

storage = HybridBlockStorage(storage_config)
facets_service = FacetsService(storage)

# Query sessions with facets (server-side filtering for Cosmos)
query = FacetQuery(
    user_id=current_user_id,
    bundle="amplifier-dev",
    has_errors=True,
    created_after=datetime.now() - timedelta(days=7),
)

sessions = await storage.query_sessions(query)
```

## Exposing Capabilities to Agents

### How Session-Analyst Discovers Capabilities

When session-analyst is spawned, it receives capabilities through:

1. **Behavior Composition** - The `session-storage` behavior provides context and tools
2. **Tool Discovery** - The `session` tool appears in available tools
3. **Context Instructions** - Guidelines for safe usage are injected

**Example agent delegation:**

```python
# In root session
result = delegate(
    agent="foundation:session-analyst",
    instruction="Find all sessions that had errors in the last week",
)
```

**What session-analyst can do:**

```
# List recent sessions
session list_sessions date_range=last_week

# Query by facets (server-side filtering)
session query_sessions has_errors=true created_after=2025-01-25T00:00:00Z

# Get detailed session info
session get_session session_id=abc123 include_events_summary=true

# Search for specific content
session search_sessions query="authentication failed" scope=transcript
```

### How Custom Agents Access Sessions

Any agent can access sessions by composing the behavior:

**my-custom-agent.md:**
```yaml
---
meta:
  name: my-custom-agent
  description: An agent that analyzes session patterns

includes:
  - session-storage:behaviors/session-storage

context:
  - my-custom-agent:context/analysis-instructions.md
---

# My Custom Agent

Analyzes session patterns using the session tool.
```

## Facet-Based Querying

The facets system enables efficient server-side filtering, especially important for remote storage.

### Available Filters

| Filter | Description | Local | Cosmos |
|--------|-------------|-------|--------|
| `bundle` | Bundle name | ✓ | ✓ (indexed) |
| `model` | Model used | ✓ | ✓ (indexed) |
| `provider` | Provider name | ✓ | ✓ (indexed) |
| `tool_used` | Tool that was used | ✓ | ✓ (ARRAY_CONTAINS) |
| `agent_delegated_to` | Agent delegated to | ✓ | ✓ (ARRAY_CONTAINS) |
| `has_errors` | Has errors | ✓ | ✓ (indexed) |
| `has_child_sessions` | Multi-agent session | ✓ | ✓ (indexed) |
| `has_recipes` | Used recipes | ✓ | ✓ (indexed) |
| `min_tokens` / `max_tokens` | Token range | ✓ | ✓ (indexed) |
| `workflow_pattern` | Detected pattern | ✓ | ✓ (indexed) |
| `created_after` / `created_before` | Date range | ✓ | ✓ (indexed) |

### Query Examples

**Find multi-agent sessions:**
```
session query_sessions has_child_sessions=true
```

**Find expensive sessions (high token usage):**
```
session query_sessions min_tokens=50000 limit=20
```

**Find sessions using specific tools:**
```
session query_sessions tool_used=delegate bundle=amplifier-dev
```

**Find sessions with errors from last week:**
```
session query_sessions has_errors=true created_after=2025-01-25T00:00:00Z
```

## Cosmos DB Index Configuration

For optimal query performance with Cosmos DB, configure these indexes:

```python
from amplifier_session_storage.facets import generate_index_policy, generate_index_cli_commands

# Generate index policy
policy = generate_index_policy()

# Generate Azure CLI commands
cli_commands = generate_index_cli_commands(
    database_name="amplifier-db",
    container_name="sessions",
)
print(cli_commands)
```

**Recommended composite indexes:**
- User sessions by updated time (most common query)
- User sessions by bundle
- User sessions by workflow pattern
- User error sessions
- User sessions by token usage
- Child sessions by parent

## Safety Guarantees

The session tool provides several safety guarantees:

1. **Never returns full event payloads** - Events can be 100k+ tokens
2. **Pagination for large result sets** - Default limit of 50 items
3. **Truncated excerpts** - Search results are length-bounded (500 chars default)
4. **Dry-run by default** - Rewind operations preview first
5. **User isolation** - Can only access own sessions

These guarantees prevent agents from accidentally overflowing their context window.

## Configuration Reference

### SessionToolConfig

```python
@dataclass
class SessionToolConfig:
    base_dir: Path | None = None          # Local storage directory
    project_slug: str = "default"          # Project for session organization
    enable_cloud: bool = False             # Enable Cosmos DB backend
    max_results: int = 50                  # Default limit for queries
    max_excerpt_length: int = 500          # Max excerpt length in search
```

### FacetQuery

See `amplifier_session_storage/facets/types.py` for full field documentation.

Key fields:
- Basic: `user_id`, `session_id`, `parent_id`, `project_slug`
- Configuration: `bundle`, `model`, `provider`
- Tools: `tool_used`, `tools_used`, `tools_used_all`
- Status: `has_errors`, `has_child_sessions`, `has_recipes`
- Tokens: `min_tokens`, `max_tokens`
- Dates: `created_after`, `created_before`, `updated_after`, `updated_before`
- Pagination: `limit`, `offset`, `order_by`, `order_desc`

## Migration from File-Based Access

If agents previously used file tools for session access:

| Old Approach | New Approach |
|--------------|--------------|
| `grep -r "keyword" ~/.amplifier/` | `session search_sessions query="keyword"` |
| `cat metadata.json` | `session get_session session_id=X` |
| `jq '.event' events.jsonl` | `session get_events session_id=X` |
| `find . -name "*.jsonl"` | `session list_sessions` |
| Manual date filtering | `session query_sessions created_after=...` |
| Grepping for bundles | `session query_sessions bundle=X` |

The new approach is:
- **Safer** - No risk of reading 100k+ token lines
- **Simpler** - No need to know file paths or formats
- **Cloud-ready** - Same API works with Cosmos DB
- **Efficient** - Server-side filtering for remote storage
