# Session Storage Architecture

This document describes the architecture of the Amplifier Session Storage system, designed to support both local file storage and Azure Cosmos DB with offline-first synchronization.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION LAYER                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ AmplifierTUI│  │ AmplifierAPI│  │   Recipes   │  │    Session-Analyst      │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘ │
└─────────┼────────────────┼────────────────┼─────────────────────┼───────────────┘
          │                │                │                     │
          ▼                ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SESSION STORAGE PROTOCOL                               │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                        SessionStorage (ABC)                                │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐   │ │
│  │  │ Session CRUD │ │   Events     │ │  Transcript  │ │    Queries      │   │ │
│  │  │ create/read/ │ │ append_event │ │ append_msg   │ │ query_events    │   │ │
│  │  │ update/list/ │ │ (append-only)│ │ compact      │ │ (projection!)   │   │ │
│  │  │ delete       │ │              │ │ rewind       │ │ get_aggregates  │   │ │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └─────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
          │                                          │
          ▼                                          ▼
┌─────────────────────────────┐             ┌─────────────────────────────────────────┐
│   LocalFileStorage          │             │        SyncedCosmosStorage              │
│ ┌─────────────────────────┐ │             │  ┌────────────────────────────────────┐ │
│ │ ~/.amplifier/           │ │             │  │           SyncEngine               │ │
│ │   projects/             │ │◄───────────►│  │  ┌─────────┐    ┌──────────────┐  │ │
│ │     {project}/          │ │  Sync       │  │  │ Change  │    │  Conflict    │  │ │
│ │       sessions/         │ │  Protocol   │  │  │ Tracker │    │  Resolver    │  │ │
│ │         {id}/           │ │             │  │  └─────────┘    └──────────────┘  │ │
│ │           metadata      │ │             │  └────────────────────────────────────┘ │
│ │           transcript    │ │             │         │                    │          │
│ │           events        │ │             │         ▼                    ▼          │
│ └─────────────────────────┘ │             │  ┌───────────────┐  ┌────────────────┐  │
└─────────────────────────────┘             │  │LocalFileCache │  │ CosmosDBClient │  │
                                            │  │ (offline ops) │  │ (cloud sync)   │  │
                                            │  └───────────────┘  └────────────────┘  │
                                            └─────────────────────────────────────────┘
                                                                           │
                                                                           ▼
                                            ┌─────────────────────────────────────────┐
                                            │            Azure Cosmos DB              │
                                            │  ┌──────────┐ ┌──────────┐ ┌─────────┐ │
                                            │  │ sessions │ │ messages │ │ events  │ │
                                            │  │/user_id  │ │/user_id- │ │/user_id-│ │
                                            │  │          │ │session_id│ │session  │ │
                                            │  └──────────┘ └──────────┘ └─────────┘ │
                                            │        ▲            │                   │
                                            │        │     Large events (>400KB)      │
                                            │        │            ▼                   │
                                            │        │     ┌──────────────┐           │
                                            │        └─────┤ event_chunks │           │
                                            │              │ (overflow)   │           │
                                            │              └──────────────┘           │
                                            └─────────────────────────────────────────┘
```

## Core Design Principles

### 1. User Isolation

**All queries MUST include user_id.** This is enforced at the protocol level:

```python
# Every method requires user_id
async def get_session(self, user_id: str, session_id: str) -> SessionMetadata | None
async def append_message(self, user_id: str, session_id: str, message: TranscriptMessage)
```

In Cosmos DB, user_id is part of the partition key, making cross-user queries **impossible by design**.

### 2. Event Projection Enforcement

Events can contain 100k+ tokens (full LLM responses). The protocol **prevents accidental retrieval**:

```python
# Safe: Returns EventSummary (small metadata only)
async def query_events(self, query: EventQuery) -> list[EventSummary]

# Explicit: Full data retrieval (separate method, use sparingly)
async def get_event_data(self, user_id: str, session_id: str, event_id: str) -> dict | None
```

The `EventProjection` TypedDict explicitly lists safe fields:
- `event_id`, `event_type`, `ts`, `session_id`, `turn`
- `model`, `usage`, `duration_ms`
- `has_tool_calls`, `has_error`, `error_type`, `tool_name`

**Never includes**: `data`, `content`, or other large payload fields.

### 3. Offline-First Operation

The `SyncedCosmosStorage` enables:
1. Writes go to local storage immediately
2. Changes tracked in sync queue
3. Background sync uploads when connected
4. Reads prefer local, fallback to cloud

### 4. Atomic Rewind Operations

Transcript and events must stay in sync. Rewind operations:
1. Create backup files (local) or snapshots (cloud)
2. Truncate both transcript and events atomically
3. Update session metadata (turn_count, message_count)

## Data Model

### Cosmos DB Containers

| Container | Partition Key | Document Types |
|-----------|---------------|----------------|
| `sessions` | `/user_id` | SessionMetadata |
| `transcript_messages` | `/user_id_session_id` | TranscriptMessage |
| `events` | `/user_id_session_id` | EventRecord |
| `event_chunks` | `/user_id_session_id` | EventChunk (for >400KB events) |
| `sync_state` | `/user_id_device_id` | SyncState, PendingChange |

### Session Metadata

```json
{
  "id": "{session_id}",
  "user_id": "user_abc123",
  "session_id": "c3843177-7ec7-4c7b-a9f0-24fab9291bf5",
  "project_slug": "amplifier-core",
  "parent_id": null,
  "created": "2025-01-15T10:30:00Z",
  "updated": "2025-01-15T11:45:00Z",
  "name": "Implementing auth system",
  "bundle": "foundation",
  "model": "claude-sonnet-4-20250514",
  "turn_count": 12,
  "message_count": 47,
  "event_count": 234
}
```

### Transcript Message

```json
{
  "id": "{session_id}_{sequence}",
  "user_id_session_id": "user_abc123_c3843177-...",
  "user_id": "user_abc123",
  "session_id": "c3843177-...",
  "sequence": 15,
  "role": "assistant",
  "content": "...",
  "tool_calls": [...],
  "timestamp": "2025-01-15T11:42:00Z",
  "turn": 6
}
```

### Event Record

```json
{
  "id": "{session_id}_{event_id}",
  "user_id_session_id": "user_abc123_c3843177-...",
  "user_id": "user_abc123",
  "session_id": "c3843177-...",
  "event_id": "evt_a1b2c3d4",
  "event_type": "llm:response",
  "ts": "2025-01-15T11:42:00.123Z",
  "turn": 6,
  
  "summary": {
    "model": "claude-sonnet-4-20250514",
    "usage": {"input_tokens": 12500, "output_tokens": 850},
    "duration_ms": 2340,
    "has_tool_calls": true
  },
  
  "data_size_bytes": 450000,
  "is_chunked": true,
  "chunk_count": 3
}
```

## Large Event Handling

Events exceeding 400KB are chunked:

```
Event Size        | Storage Strategy
------------------+------------------------------------------
< 400KB          | Single document with inline data
400KB - 2MB      | EventRecord (summary) + EventChunks
> 2MB            | Rejected (Cosmos DB limit)
```

Reassembly:
1. Query EventRecord to get chunk count
2. Fetch all chunks in parallel
3. Reassemble in order by chunk_index

## Sync Protocol

### Change Tracking

```python
@dataclass
class ChangeRecord:
    local_seq: int           # Monotonic sequence per device
    device_id: str           # Unique device identifier
    timestamp: datetime      # Wall-clock time
    change_type: ChangeType  # CREATE, UPDATE, APPEND, DELETE
    entity_type: EntityType  # SESSION, MESSAGE, EVENT
    session_id: str
    payload_ref: str         # Local file with actual data
```

### Sync States

```
                    ┌─────────────────┐
                    │    SYNCED       │
                    │  (no pending)   │
                    └────────┬────────┘
                             │
                    Local write occurs
                             │
                             ▼
                    ┌─────────────────┐
                    │    PENDING      │
                    │  (queued)       │
                    └────────┬────────┘
                             │
                    Network available
                             │
                             ▼
                    ┌─────────────────┐
                    │   UPLOADING     │
                    │  (in progress)  │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
         Success          Conflict         Error
            │                │                │
            ▼                ▼                ▼
    ┌───────────┐    ┌───────────────┐    ┌─────────────┐
    │  SYNCED   │    │  CONFLICTED   │    │   RETRY     │
    └───────────┘    │  (needs       │    │  (backoff)  │
                     │   resolution) │    └─────────────┘
                     └───────────────┘
```

### Conflict Resolution

| Conflict Type | Resolution Strategy |
|---------------|---------------------|
| Metadata update | Last-writer-wins with version vector |
| Concurrent appends | Merge by timestamp + device_id |
| Delete vs modify | Delete wins (can restore from history) |
| Rewind conflict | Requires user decision |

### Version Vectors

```python
@dataclass
class VersionVector:
    entries: dict[str, int]  # device_id -> sequence
    timestamp: datetime      # Wall clock for tiebreaker
    
    def happens_before(self, other: "VersionVector") -> bool:
        """Check if self causally precedes other."""
        ...
    
    def concurrent_with(self, other: "VersionVector") -> bool:
        """Check if two versions are concurrent (conflict)."""
        return not self.happens_before(other) and not other.happens_before(self)
```

## Security Model

### Authentication Flow

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                                       │
│  • Authenticate user (Entra ID / OAuth)                                   │
│  • Inject user_id into all storage calls                                  │
│  • NEVER trust user_id from client input                                  │
└────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    STORAGE LAYER                                           │
│  • user_id in partition key (Cosmos DB)                                   │
│  • user_id in file path (Local)                                           │
│  • ALL queries MUST include user_id                                       │
│  • Cross-user queries IMPOSSIBLE by design                                │
└────────────────────────────────────────────────────────────────────────────┘
```

### Partition Key Strategy

```python
PARTITION_KEYS = {
    "sessions": "/user_id",
    "transcript_messages": "/user_id_session_id",  # Composite
    "events": "/user_id_session_id",               # Composite
    "sync_state": "/user_id_device_id",            # Composite
}
```

Every query includes user_id, making cross-user access impossible at the database level.

## Module Structure

```
amplifier_session_storage/
├── __init__.py              # Main package exports
├── protocol.py              # SessionStorage ABC and types
├── exceptions.py            # Custom exceptions
│
├── local/
│   ├── storage.py           # LocalFileStorage implementation
│   └── file_ops.py          # Atomic file operations
│
├── cosmos/
│   ├── storage.py           # CosmosDBStorage implementation
│   ├── client.py            # Cosmos DB client wrapper
│   └── chunking.py          # Large event chunking
│
├── sync/
│   ├── engine.py            # SyncEngine orchestration
│   ├── tracker.py           # Change tracking
│   ├── conflict.py          # Conflict resolution
│   └── version.py           # Version vectors
│
└── synced/
    └── storage.py           # SyncedCosmosStorage (combines local + cosmos)
```

## Session Analyst Integration

The session-analyst agent requires special consideration due to event size:

### Safe Operations

```python
# ✅ List sessions (metadata only)
sessions = await storage.list_sessions(query)

# ✅ Get transcript (usually safe, <100KB)
transcript = await storage.get_transcript(user_id, session_id)

# ✅ Query events with projection (NEVER returns full data)
events = await storage.query_events(EventQuery(
    session_id=session_id,
    user_id=user_id,
    event_types=["llm:response", "tool:execute:post"],
))

# ✅ Get aggregates (computed server-side)
stats = await storage.get_event_aggregates(user_id, session_id)

# ✅ Rewind with backup
result = await storage.rewind_to_turn(user_id, session_id, turn=5)
```

### Dangerous Operations (Use Sparingly)

```python
# ⚠️ Full event data - can be 100k+ tokens
data = await storage.get_event_data(user_id, session_id, event_id)
```

## Migration Strategy

### Phase 1: Local-Only (Current Amplifier)
- Sessions stored locally at `~/.amplifier/projects/`
- No sync capability

### Phase 2: Hybrid (This Implementation)
- New sessions created locally AND synced to cloud
- Existing sessions migrated on-demand
- Full local operation continues to work

### Phase 3: Cloud-Primary (Future)
- Cloud is source of truth
- Local cache for offline operation
- Existing local sessions fully migrated

## Performance Considerations

### Query Patterns

| Operation | Expected Latency |
|-----------|------------------|
| Get session metadata | <100ms |
| Append message (local) | <50ms |
| Append message (synced) | <200ms |
| Query events (projected) | <200ms |
| Full event retrieval | 200ms - 2s (size dependent) |
| Sync catchup | <30s after connectivity |

### Cosmos DB RU Optimization

- Partition by user for isolation
- Composite keys for efficient range queries
- Aggregations computed server-side
- Chunked events reduce document size

## Testing Strategy

### Unit Tests
- Protocol compliance for each backend
- Atomic write verification
- Conflict resolution logic

### Integration Tests
- Multi-machine sync scenarios
- Offline operation and recovery
- Large event handling

### End-to-End Tests
- Session analyst workflows
- Rewind and recovery
- Cross-device session access
- Session sharing and discovery

## Session Sharing Architecture

The storage system supports session sharing for team collaboration and cross-team learning.

### Visibility Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Session Visibility                              │
├─────────────────────────────────────────────────────────────────────────┤
│  PRIVATE (default)     │ Only the session owner can access             │
│  TEAM                  │ Members of specified teams can view           │
│  ORGANIZATION          │ All organization members can view             │
│  PUBLIC                │ Any authenticated user can view               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Access Control Flow

```
check_access(user_id, session, required_permission):
    │
    ├─► Is user the owner?
    │       └─► YES → ALLOW (full access: read/write/delete)
    │
    ├─► Is visibility PRIVATE?
    │       └─► YES → DENY
    │
    ├─► Check visibility level:
    │       │
    │       ├─► PUBLIC:
    │       │   └─► Authenticated? → ALLOW read
    │       │
    │       ├─► ORGANIZATION:
    │       │   └─► Same org_id? → ALLOW read
    │       │
    │       └─► TEAM:
    │           └─► User in any session.team_ids? → ALLOW read
    │
    └─► DENY (insufficient permission)
```

### Additional Cosmos DB Containers

| Container | Partition Key | Purpose |
|-----------|---------------|---------|
| `shared_sessions` | `/org_id` | Index of shared sessions for discovery |
| `organizations` | `/org_id` | Organization definitions |
| `teams` | `/org_id` | Team definitions |
| `user_memberships` | `/user_id` | User org/team memberships |

### Shared Session Index

When a session's visibility changes from PRIVATE, it's indexed in `shared_sessions`:

```json
{
  "id": "{session_id}",
  "org_id": "org_123",
  "session_id": "sess_abc",
  "owner_user_id": "user_456",
  "owner_name": "Alice",
  "visibility": "team",
  "team_ids": ["team_a", "team_b"],
  "name": "Debug auth issue",
  "project_slug": "my-project",
  "tags": ["auth", "debugging"],
  "turn_count": 15,
  "created": "2025-01-30T10:00:00Z",
  "updated": "2025-01-30T14:30:00Z",
  "shared_at": "2025-01-30T12:00:00Z"
}
```

### Query Efficiency

| Query | Cross-Partition? | RU Estimate |
|-------|------------------|-------------|
| My sessions | No (user_id partition) | ~5 RU |
| All org shared sessions | No (org_id partition) | ~5-10 RU |
| My team sessions | No (org_id + filter) | ~10 RU |
| Recent team activity | No (org_id + time filter) | ~10 RU |

### Module Structure (Sharing)

```
amplifier_session_storage/
├── access/
│   ├── __init__.py
│   ├── controller.py      # AccessController
│   └── permissions.py     # Permission enum, AccessDecision
│
├── membership/
│   ├── __init__.py
│   └── store.py           # MembershipStore
│
└── (existing modules updated with sharing methods)
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Separate `shared_sessions` container | Efficient org-scoped queries without cross-partition reads |
| Owner always has full access | Security invariant, no exceptions |
| Read-only shared access by default | Explicit write grants required for safety |
| Denormalized owner_name in index | Avoids join to get display names |
| Tags for categorization | Enables filtering by topic/type |
