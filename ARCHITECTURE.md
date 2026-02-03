# Architecture

This document describes the architecture of the Amplifier Session Storage library.

## Overview

The library provides a simple, focused solution for syncing Amplifier CLI session data to Azure Cosmos DB. It mirrors the CLI's local file format (metadata.json, transcript.jsonl, events.jsonl) to cloud storage with complete data parity.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Amplifier CLI (Local)                        │
│                                                                 │
│  ~/.amplifier/projects/{project}/sessions/{session}/            │
│  ├── metadata.json      → Session metadata                      │
│  ├── transcript.jsonl   → Conversation messages                 │
│  └── events.jsonl       → Session events                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Session Sync Daemon
                              │ (reads local files)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CosmosFileStorage                            │
│                                                                 │
│  - upsert_session_metadata()                                    │
│  - sync_transcript_lines()                                      │
│  - sync_event_lines()                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Azure SDK
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Azure Cosmos DB                              │
│                                                                 │
│  Database: amplifier-db                                         │
│  ├── sessions      (partition: /user_id)                        │
│  ├── transcripts   (partition: /partition_key)                  │
│  └── events        (partition: /partition_key)                  │
└─────────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. CLI Format Parity

The storage preserves the exact structure of local CLI files:
- **metadata.json** → `sessions` container (one document per session)
- **transcript.jsonl** → `transcripts` container (one document per line)
- **events.jsonl** → `events` container (one document per line)

No data transformation or loss occurs during sync.

### 2. Multi-Device Origin Tracking

Every document includes:
- `user_id`: The user who owns the session
- `host_id`: The machine where the session originated

This enables:
- Querying sessions by origin machine
- Detecting conflicts from concurrent edits
- Audit trail for session activity

### 3. Incremental Sync

Sequence numbers enable efficient delta sync:
- Transcript lines have `sequence` (0, 1, 2, ...)
- Event lines have `sequence` (0, 1, 2, ...)
- Sync daemon queries current count, sends only new lines

### 4. Query-Optimized Partitioning

Container partition keys are chosen for efficient queries:

| Container | Partition Key | Rationale |
|-----------|---------------|-----------|
| `sessions` | `/user_id` | List all sessions for a user efficiently |
| `transcripts` | `/partition_key` | All messages for a session in one partition |
| `events` | `/partition_key` | All events for a session in one partition |

The composite partition key format is: `{user_id}|{project_slug}|{session_id}`

## Components

### CosmosFileStorage

The main storage class providing:

```python
class CosmosFileStorage:
    # Lifecycle
    async def initialize() -> None
    async def close() -> None
    async def verify_connection() -> bool

    # Session metadata (sessions container)
    async def upsert_session_metadata(user_id, host_id, metadata) -> None
    async def get_session_metadata(user_id, session_id) -> dict | None
    async def list_sessions(user_id, project_slug?, limit?) -> list[dict]
    async def delete_session(user_id, project_slug, session_id) -> bool

    # Transcript messages (transcripts container)
    async def sync_transcript_lines(user_id, host_id, project_slug, session_id, lines, start_sequence?) -> int
    async def get_transcript_lines(user_id, project_slug, session_id, after_sequence?) -> list[dict]
    async def get_transcript_count(user_id, project_slug, session_id) -> int

    # Events (events container)
    async def sync_event_lines(user_id, host_id, project_slug, session_id, lines, start_sequence?) -> int
    async def get_event_lines(user_id, project_slug, session_id, after_sequence?) -> list[dict]
    async def get_event_count(user_id, project_slug, session_id) -> int
```

### CosmosFileConfig

Configuration for Cosmos DB connection:

```python
@dataclass
class CosmosFileConfig:
    endpoint: str           # Cosmos DB account endpoint
    database_name: str      # Database name (default: "amplifier-db")
    auth_method: str        # "default_credential" or "key"
    key: str | None         # Account key (only for key auth)

    @classmethod
    def from_env(cls) -> CosmosFileConfig
```

### Identity Module

Utilities for consistent user/device identification:

```python
class IdentityContext:
    @classmethod
    def initialize() -> None
    
    @classmethod
    def get_user_id() -> str
    
    @classmethod
    def get_host_id() -> str
```

## Data Flow

### Sync Flow (Daemon → Cosmos)

```
1. Daemon discovers local sessions
   └── Scans ~/.amplifier/projects/*/sessions/*/

2. For each session, daemon calls server API:
   └── POST /api/v1/sessions/{id}/sync-status
       Returns: { event_count, transcript_count }

3. Daemon reads local files and calculates delta:
   └── new_events = local_events[event_count:]
   └── new_transcripts = local_transcripts[transcript_count:]

4. Daemon sends incremental updates:
   └── POST /api/v1/sessions/{id}/metadata
   └── POST /api/v1/sessions/{id}/events
   └── POST /api/v1/sessions/{id}/transcript

5. Server uses CosmosFileStorage to persist:
   └── storage.upsert_session_metadata(...)
   └── storage.sync_event_lines(...)
   └── storage.sync_transcript_lines(...)
```

### Query Flow (Client → Cosmos)

```
1. Client requests session list:
   └── GET /api/v1/sessions?user_id=X&project=Y

2. Server queries Cosmos:
   └── storage.list_sessions(user_id, project_slug)

3. Cosmos executes efficient partition query:
   └── Query scoped to user_id partition
   └── Optional project_slug filter

4. Results returned with full metadata
```

## Document Schemas

### Session Document

```json
{
    "id": "session-id",
    "user_id": "user-123",
    "host_id": "laptop-01",
    "session_id": "session-id",
    "project_slug": "my-project",
    "bundle": "foundation",
    "created": "2024-01-15T10:00:00Z",
    "updated": "2024-01-15T12:00:00Z",
    "turn_count": 5,
    "parent_id": null,
    "trace_id": "trace-xyz",
    "config": { ... },
    "_type": "session",
    "synced_at": "2024-01-15T12:00:00Z"
}
```

### Transcript Document

```json
{
    "id": "session-id_msg_0",
    "partition_key": "user-123|my-project|session-id",
    "user_id": "user-123",
    "host_id": "laptop-01",
    "project_slug": "my-project",
    "session_id": "session-id",
    "sequence": 0,
    "role": "user",
    "content": "Hello",
    "timestamp": "2024-01-15T10:00:00Z",
    "turn": 0,
    "_type": "transcript_message",
    "synced_at": "2024-01-15T12:00:00Z"
}
```

### Event Document

```json
{
    "id": "session-id_evt_0",
    "partition_key": "user-123|my-project|session-id",
    "user_id": "user-123",
    "host_id": "laptop-01",
    "project_slug": "my-project",
    "session_id": "session-id",
    "sequence": 0,
    "event": "session.start",
    "ts": "2024-01-15T10:00:00Z",
    "lvl": "info",
    "turn": 0,
    "data": { ... },
    "data_truncated": false,
    "data_size_bytes": 150,
    "_type": "event",
    "synced_at": "2024-01-15T12:00:00Z"
}
```

## Large Event Handling

Events larger than 400KB (Cosmos document size considerations) are handled specially:

1. **Detection**: `len(json.dumps(event).encode()) > 400KB`
2. **Truncation**: Only summary fields stored:
   - `event`, `ts`, `lvl`, `turn`
   - `data_truncated: true`
   - `data_size_bytes`: original size
3. **Full data**: Remains in local `events.jsonl` file

This approach:
- Prevents Cosmos document size limit issues
- Maintains queryable metadata for all events
- Preserves full data locally for session-analyst access

## Authentication

### DefaultAzureCredential (Recommended)

Uses Azure Identity SDK's credential chain:
1. Environment variables (AZURE_CLIENT_ID, etc.)
2. Managed Identity (Azure-hosted services)
3. Azure CLI (`az login`)
4. Visual Studio Code
5. Azure PowerShell

### Key-Based

Direct Cosmos DB account key for development/testing.

**Security Note**: Never commit keys to source control.

## Error Handling

The library defines specific exceptions:

| Exception | Cause |
|-----------|-------|
| `AuthenticationError` | Cosmos auth failed |
| `StorageConnectionError` | Network/connection issues |
| `StorageIOError` | Read/write operations failed |
| `SessionNotFoundError` | Session doesn't exist |

All exceptions inherit from `SessionStorageError` for unified handling.

## Testing

### Unit Tests

```bash
uv run pytest tests/test_identity.py -v
```

### Integration Tests (Requires Cosmos)

```bash
export AMPLIFIER_COSMOS_ENDPOINT="https://..."
uv run pytest tests/test_cosmos_file_storage.py -v
```

Integration tests:
- Create unique test data (UUID-based IDs)
- Clean up after each test
- Skip automatically if Cosmos not configured

## Future Considerations

### Potential Enhancements

1. **Batch Operations**: Use Cosmos transactional batch for atomic multi-document writes
2. **Change Feed**: React to Cosmos changes for real-time sync
3. **TTL Policies**: Auto-expire old session data
4. **Compression**: Compress large event payloads before storage

### Not In Scope

- Local file storage (handled by Amplifier CLI)
- Real-time WebSocket sync (separate service)
- Session replay/analysis (session-analyst tool)
