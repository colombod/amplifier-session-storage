# Amplifier Session Storage

A library for syncing Amplifier CLI session data to Azure Cosmos DB.

## Overview

This library provides `CosmosFileStorage` - a storage backend that mirrors the Amplifier CLI's local session file format to Azure Cosmos DB. It enables syncing session data from multiple machines to a central cloud store while maintaining complete data parity.

**Key Features:**
- **CLI-compatible format**: Mirrors `metadata.json`, `transcript.jsonl`, and `events.jsonl` structure
- **Multi-device tracking**: Every document includes `user_id` and `host_id` for origin tracking
- **Incremental sync**: Supports delta sync via sequence numbers
- **Separate containers**: Sessions, transcripts, and events in dedicated containers for efficient queries

## Installation

```bash
# Basic installation
uv pip install git+https://github.com/colombod/amplifier-session-storage

# Development setup
git clone https://github.com/colombod/amplifier-session-storage
cd amplifier-session-storage
uv sync --all-extras
uv run pytest tests/ -v
```

## Quick Start

```python
from amplifier_session_storage import CosmosFileStorage, CosmosFileConfig

# Create config from environment variables
config = CosmosFileConfig.from_env()

# Use as async context manager
async with CosmosFileStorage(config) as storage:
    # Upsert session metadata
    await storage.upsert_session_metadata(
        user_id="user-123",
        host_id="laptop-01",
        metadata={
            "session_id": "sess-abc",
            "project_slug": "my-project",
            "bundle": "foundation",
            "created": "2024-01-15T10:00:00Z",
            "turn_count": 5,
        },
    )

    # Sync transcript lines (incremental)
    await storage.sync_transcript_lines(
        user_id="user-123",
        host_id="laptop-01",
        project_slug="my-project",
        session_id="sess-abc",
        lines=[
            {"role": "user", "content": "Hello", "turn": 0},
            {"role": "assistant", "content": "Hi!", "turn": 0},
        ],
        start_sequence=0,
    )

    # Sync event lines (incremental)
    await storage.sync_event_lines(
        user_id="user-123",
        host_id="laptop-01",
        project_slug="my-project",
        session_id="sess-abc",
        lines=[
            {"event": "session.start", "ts": "2024-01-15T10:00:00Z", "lvl": "info"},
            {"event": "llm.request", "ts": "2024-01-15T10:00:01Z", "lvl": "debug"},
        ],
        start_sequence=0,
    )
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AMPLIFIER_COSMOS_ENDPOINT` | Cosmos DB endpoint URL | **Required** |
| `AMPLIFIER_COSMOS_DATABASE` | Database name | `amplifier-db` |
| `AMPLIFIER_COSMOS_AUTH_METHOD` | `default_credential` or `key` | `default_credential` |
| `AMPLIFIER_COSMOS_KEY` | Cosmos DB key (only if auth_method=key) | - |

### Authentication Methods

#### Azure AD (Recommended)

Uses `DefaultAzureCredential` which automatically tries:
- Azure CLI credentials (`az login`)
- Environment variables
- Managed Identity
- Visual Studio Code credentials

```python
config = CosmosFileConfig(
    endpoint="https://your-account.documents.azure.com:443/",
    database_name="amplifier-db",
    auth_method="default_credential",
)
```

**Required RBAC role**: `Cosmos DB Built-in Data Contributor`

```bash
# Assign role to your user
az cosmosdb sql role assignment create \
    --account-name your-cosmos-account \
    --resource-group your-resource-group \
    --role-definition-name "Cosmos DB Built-in Data Contributor" \
    --principal-id $(az ad signed-in-user show --query id -o tsv) \
    --scope "/"
```

#### Key-Based (Development)

```python
config = CosmosFileConfig(
    endpoint="https://your-account.documents.azure.com:443/",
    database_name="amplifier-db",
    auth_method="key",
    key="your-cosmos-key",
)
```

## Data Model

### Container Structure

The storage creates three containers with optimized partition keys:

| Container | Partition Key | Contents |
|-----------|---------------|----------|
| `sessions` | `/user_id` | Session metadata (one doc per session) |
| `transcripts` | `/partition_key` | Transcript messages (one doc per message) |
| `events` | `/partition_key` | Events (one doc per event) |

### Document Schema

#### Session Metadata

```json
{
    "id": "sess-abc",
    "user_id": "user-123",
    "host_id": "laptop-01",
    "session_id": "sess-abc",
    "project_slug": "my-project",
    "bundle": "foundation",
    "created": "2024-01-15T10:00:00Z",
    "turn_count": 5,
    "_type": "session",
    "synced_at": "2024-01-15T12:00:00Z"
}
```

#### Transcript Message

```json
{
    "id": "sess-abc_msg_0",
    "partition_key": "user-123|my-project|sess-abc",
    "user_id": "user-123",
    "host_id": "laptop-01",
    "project_slug": "my-project",
    "session_id": "sess-abc",
    "sequence": 0,
    "role": "user",
    "content": "Hello",
    "turn": 0,
    "_type": "transcript_message",
    "synced_at": "2024-01-15T12:00:00Z"
}
```

#### Event

```json
{
    "id": "sess-abc_evt_0",
    "partition_key": "user-123|my-project|sess-abc",
    "user_id": "user-123",
    "host_id": "laptop-01",
    "project_slug": "my-project",
    "session_id": "sess-abc",
    "sequence": 0,
    "event": "session.start",
    "ts": "2024-01-15T10:00:00Z",
    "lvl": "info",
    "data_truncated": false,
    "data_size_bytes": 150,
    "_type": "event",
    "synced_at": "2024-01-15T12:00:00Z"
}
```

### Large Event Handling

Events larger than 400KB are stored with `data_truncated: true` and only summary fields preserved:
- `event`, `ts`, `lvl`, `turn`
- `data_size_bytes` (original size)

This prevents Cosmos DB document size limit issues while preserving queryable metadata.

## API Reference

### CosmosFileStorage

```python
class CosmosFileStorage:
    """Cosmos DB storage that mirrors CLI file format."""

    async def initialize() -> None
    async def close() -> None
    async def verify_connection() -> bool

    # Session metadata
    async def upsert_session_metadata(user_id, host_id, metadata) -> None
    async def get_session_metadata(user_id, session_id) -> dict | None
    async def list_sessions(user_id, project_slug=None, limit=100) -> list[dict]
    async def delete_session(user_id, project_slug, session_id) -> bool

    # Transcript
    async def sync_transcript_lines(user_id, host_id, project_slug, session_id, lines, start_sequence=0) -> int
    async def get_transcript_lines(user_id, project_slug, session_id, after_sequence=-1) -> list[dict]
    async def get_transcript_count(user_id, project_slug, session_id) -> int

    # Events
    async def sync_event_lines(user_id, host_id, project_slug, session_id, lines, start_sequence=0) -> int
    async def get_event_lines(user_id, project_slug, session_id, after_sequence=-1) -> list[dict]
    async def get_event_count(user_id, project_slug, session_id) -> int

    # Utilities
    @staticmethod
    def make_partition_key(user_id, project_slug, session_id) -> str
```

### CosmosFileConfig

```python
@dataclass
class CosmosFileConfig:
    endpoint: str              # Cosmos DB endpoint URL
    database_name: str         # Database name (default: "amplifier-db")
    auth_method: str           # "default_credential" or "key"
    key: str | None            # Cosmos key (only if auth_method="key")

    @classmethod
    def from_env(cls) -> CosmosFileConfig
```

## Incremental Sync Pattern

The storage supports efficient incremental sync via sequence numbers:

```python
# Get current sync status
transcript_count = await storage.get_transcript_count(user_id, project_slug, session_id)
event_count = await storage.get_event_count(user_id, project_slug, session_id)

# Sync only new lines (after existing ones)
new_transcript_lines = local_transcript[transcript_count:]
new_event_lines = local_events[event_count:]

if new_transcript_lines:
    await storage.sync_transcript_lines(
        user_id, host_id, project_slug, session_id,
        lines=new_transcript_lines,
        start_sequence=transcript_count,
    )

if new_event_lines:
    await storage.sync_event_lines(
        user_id, host_id, project_slug, session_id,
        lines=new_event_lines,
        start_sequence=event_count,
    )
```

## Identity Management

The library includes identity utilities for consistent user/device tracking:

```python
from amplifier_session_storage import IdentityContext

# Initialize identity context
IdentityContext.initialize()

# Get user and host IDs
user_id = IdentityContext.get_user_id()
host_id = IdentityContext.get_host_id()
```

## Development

```bash
# Install development dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/ -v

# Run tests with Cosmos DB (requires environment setup)
export AMPLIFIER_COSMOS_ENDPOINT="https://your-account.documents.azure.com:443/"
uv run pytest tests/test_cosmos_file_storage.py -v

# Linting and formatting
uv run ruff check .
uv run ruff format .

# Type checking
uv run pyright
```

## License

MIT
