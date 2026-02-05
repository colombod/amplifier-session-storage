# Amplifier Session Storage - Data Schema Reference

> **Version**: 1.0.0  
> **Last Updated**: 2025-02-05  
> **Status**: Authoritative Source of Truth

This document defines the canonical schema for all data entities in amplifier-session-storage. All backend implementations (Cosmos DB, DuckDB, SQLite) MUST conform to this schema.

---

## Table of Contents

1. [Overview](#overview)
2. [Sessions](#sessions)
3. [Transcript Messages](#transcript-messages)
4. [Events](#events)
5. [Embeddings](#embeddings)
6. [Backend-Specific Notes](#backend-specific-notes)
7. [Schema Evolution](#schema-evolution)

---

## Overview

### Data Model

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER                                     │
│  (identified by user_id from Azure CLI or identity provider)    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐      ┌──────────────────┐     ┌────────────┐ │
│   │   SESSION   │──────│ TRANSCRIPT_MSG   │     │   EVENT    │ │
│   │  metadata   │ 1:N  │   messages       │     │  logs      │ │
│   └─────────────┘      └──────────────────┘     └────────────┘ │
│         │                      │                      │         │
│         │              ┌───────┴───────┐              │         │
│         │              │  EMBEDDING    │              │         │
│         │              │  vectors      │              │         │
│         │              └───────────────┘              │         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Concepts

- **user_id**: Primary isolation key. All data is partitioned by user.
- **session_id**: Unique identifier for a conversation session.
- **project_slug**: Logical grouping of sessions (maps to local project directories).
- **sequence**: 0-based order of messages in a transcript.
- **turn**: Conversation turn number (user message + assistant response = 1 turn).

---

## Sessions

### Schema Definition

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `session_id` | string | Yes | - | Unique session identifier (UUID format) |
| `user_id` | string | Yes | - | User identifier (from identity provider) |
| `host_id` | string | Yes | - | Host machine identifier |
| `project_slug` | string | Yes | `"default"` | Project grouping identifier |
| `created` | ISO 8601 datetime | Yes | - | Session creation timestamp |
| `updated` | ISO 8601 datetime | Yes | - | Last modification timestamp |
| `name` | string | No | `null` | User-defined session name |
| `description` | string | No | `null` | User-defined description |
| `bundle` | string | No | `null` | Bundle identifier used (e.g., `"bundle:foundation"`) |
| `model` | string | No | `null` | Primary LLM model used |
| `turn_count` | integer | Yes | `0` | Number of conversation turns |
| `message_count` | integer | Yes | `0` | Total messages in transcript |
| `event_count` | integer | Yes | `0` | Total events logged |
| `parent_id` | string | No | `null` | Parent session ID (for spawned/forked sessions) |
| `forked_from_turn` | integer | No | `null` | Turn number where fork occurred |
| `tags` | string[] | No | `[]` | User-defined tags |
| `visibility` | enum | Yes | `"private"` | Access level: `private`, `team`, `org`, `public` |
| `org_id` | string | No | `null` | Organization ID for sharing |
| `team_ids` | string[] | No | `[]` | Team IDs for sharing |
| `shared_at` | ISO 8601 datetime | No | `null` | When session was first shared |

### Example

```json
{
  "session_id": "abc123-def456-789012",
  "user_id": "user@example.com",
  "host_id": "laptop-001",
  "project_slug": "amplifier-core",
  "created": "2025-02-05T10:30:00.000Z",
  "updated": "2025-02-05T11:45:00.000Z",
  "name": "Debug authentication flow",
  "description": "Investigating JWT validation failures",
  "bundle": "bundle:foundation",
  "model": "claude-sonnet-4-20250514",
  "turn_count": 15,
  "message_count": 42,
  "event_count": 128,
  "parent_id": null,
  "forked_from_turn": null,
  "tags": ["debugging", "auth", "jwt"],
  "visibility": "private",
  "org_id": null,
  "team_ids": [],
  "shared_at": null
}
```

### Indexes

| Index | Fields | Purpose |
|-------|--------|---------|
| Primary | `user_id`, `session_id` | Unique session lookup |
| Project | `user_id`, `project_slug`, `created` | List sessions by project |
| Date | `user_id`, `created` | List sessions by date |
| Bundle | `user_id`, `bundle` | Filter by bundle |

---

## Transcript Messages

### Schema Definition

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `user_id` | string | Yes | - | User identifier |
| `session_id` | string | Yes | - | Parent session ID |
| `sequence` | integer | Yes | - | 0-based message order |
| `turn` | integer | No | `null` | Conversation turn number (can be null for system messages) |
| `role` | enum | Yes | - | Message role: `user`, `assistant`, `tool`, `system` |
| `content` | string/object | Yes | - | Message content (string or structured) |
| `timestamp` | ISO 8601 datetime | Yes | - | When message was created |
| `tool_calls` | object[] | No | `null` | Tool calls made (assistant messages only) |
| `tool_call_id` | string | No | `null` | Tool call this responds to (tool messages only) |
| `thinking` | string | No | `null` | Assistant thinking/reasoning (if captured) |
| `project_slug` | string | Yes | - | Project grouping (denormalized for queries) |

### Content Structure

**User messages:**
```json
{
  "role": "user",
  "content": "Help me debug this authentication issue"
}
```

**Assistant messages:**
```json
{
  "role": "assistant",
  "content": "I'll help you debug the authentication issue. Let me start by...",
  "thinking": "The user is experiencing auth failures. I should first check...",
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": "{\"file_path\": \"src/auth.py\"}"
      }
    }
  ]
}
```

**Tool messages:**
```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "File content here..."
}
```

**System messages:**
```json
{
  "role": "system",
  "content": "You are a helpful assistant..."
}
```

### Example (Full Message)

```json
{
  "user_id": "user@example.com",
  "session_id": "abc123-def456-789012",
  "sequence": 5,
  "turn": 3,
  "role": "assistant",
  "content": "I found the issue in your JWT validation logic...",
  "timestamp": "2025-02-05T10:35:42.123Z",
  "thinking": "Looking at the auth.py file, I can see that...",
  "tool_calls": null,
  "tool_call_id": null,
  "project_slug": "amplifier-core"
}
```

### Turn Numbering

- Turn starts at `1` with the first user message
- User message and its assistant response share the same turn number
- Tool messages share the turn of the assistant that called them
- System messages may have `turn: null`
- **Important**: Turn can be `null` for some messages - use `sequence` for reliable ordering

### Indexes

| Index | Fields | Purpose |
|-------|--------|---------|
| Primary | `user_id`, `session_id`, `sequence` | Unique message lookup |
| Turn | `user_id`, `session_id`, `turn` | Get messages by turn |
| Role | `user_id`, `session_id`, `role` | Filter by role |
| Full-text | `content` | Text search |

---

## Events

### Schema Definition

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `event_id` | string | Yes | - | Unique event identifier |
| `user_id` | string | Yes | - | User identifier |
| `session_id` | string | Yes | - | Parent session ID |
| `event_type` | string | Yes | - | Event type (e.g., `llm:request`, `tool:call`) |
| `ts` | ISO 8601 datetime | Yes | - | Event timestamp |
| `turn` | integer | No | `null` | Associated turn number |
| `level` | enum | Yes | `"INFO"` | Log level: `DEBUG`, `INFO`, `WARN`, `ERROR` |
| `data` | object | No | `null` | Event payload (can be very large) |
| `summary` | object | Yes | `{}` | Safe projection fields for queries |
| `is_chunked` | boolean | Yes | `false` | Whether data is stored in chunks |
| `chunk_count` | integer | No | `null` | Number of chunks if chunked |
| `data_size_bytes` | integer | Yes | `0` | Size of data payload |
| `project_slug` | string | Yes | - | Project grouping (denormalized) |

### Event Types

| Event Type | Description | Typical Data |
|------------|-------------|--------------|
| `llm:request` | LLM API request sent | model, messages, parameters |
| `llm:response` | LLM API response received | content, usage, duration |
| `tool:call` | Tool invocation | tool_name, arguments |
| `tool:result` | Tool result | output, duration |
| `session:start` | Session started | bundle, model |
| `session:end` | Session ended | turn_count, duration |
| `agent:spawn` | Sub-agent spawned | agent_id, instruction |
| `agent:complete` | Sub-agent completed | result, duration |
| `error` | Error occurred | error_type, message, stack |

### Summary Projection

The `summary` field contains safe, queryable fields extracted from `data`:

```json
{
  "model": "claude-sonnet-4-20250514",
  "duration_ms": 1234,
  "has_tool_calls": true,
  "has_error": false,
  "tool_names": ["read_file", "bash"],
  "usage": {
    "input_tokens": 1500,
    "output_tokens": 800
  }
}
```

**Fields NEVER in summary** (can be huge):
- `content`
- `data`
- `messages`
- `full_response`

### Chunking (Large Events)

Events with `data` > 400KB are chunked:

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | string | `{event_id}_chunk_{index}` |
| `event_id` | string | Parent event ID |
| `chunk_index` | integer | 0-based chunk number |
| `total_chunks` | integer | Total number of chunks |
| `chunk_data` | string | JSON-encoded chunk content |

### Example

```json
{
  "event_id": "evt_abc123def456",
  "user_id": "user@example.com",
  "session_id": "abc123-def456-789012",
  "event_type": "llm:response",
  "ts": "2025-02-05T10:35:42.123Z",
  "turn": 3,
  "level": "INFO",
  "data": null,
  "summary": {
    "model": "claude-sonnet-4-20250514",
    "duration_ms": 2341,
    "has_tool_calls": true,
    "usage": {
      "input_tokens": 1500,
      "output_tokens": 800
    }
  },
  "is_chunked": true,
  "chunk_count": 3,
  "data_size_bytes": 1250000,
  "project_slug": "amplifier-core"
}
```

### Indexes

| Index | Fields | Purpose |
|-------|--------|---------|
| Primary | `user_id`, `session_id`, `event_id` | Unique event lookup |
| Time | `user_id`, `session_id`, `ts` | Events by time |
| Type | `user_id`, `session_id`, `event_type` | Filter by type |

---

## Embeddings

### Schema Definition

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | User identifier |
| `session_id` | string | Yes | Parent session ID |
| `sequence` | integer | Yes | Message sequence number |
| `source` | enum | Yes | Content source: `user`, `assistant`, `thinking`, `tool` |
| `embedding` | float[] | Yes | Vector embedding (dimensions vary by model) |
| `model` | string | Yes | Embedding model used |
| `dimensions` | integer | Yes | Vector dimensions (e.g., 3072) |

### Embedding Sources

| Source | Description | Typical Use |
|--------|-------------|-------------|
| `user` | User message content | Find similar questions |
| `assistant` | Assistant response content | Find similar answers |
| `thinking` | Assistant thinking blocks | Find similar reasoning |
| `tool` | Tool output content | Find similar tool results |

### Example

```json
{
  "user_id": "user@example.com",
  "session_id": "abc123-def456-789012",
  "sequence": 5,
  "source": "assistant",
  "embedding": [0.123, -0.456, 0.789, ...],
  "model": "text-embedding-3-large",
  "dimensions": 3072
}
```

### Supported Embedding Models

| Model | Provider | Dimensions | Notes |
|-------|----------|------------|-------|
| `text-embedding-3-large` | OpenAI/Azure | 3072 | Recommended |
| `text-embedding-3-small` | OpenAI/Azure | 1536 | Faster, smaller |
| `text-embedding-ada-002` | OpenAI/Azure | 1536 | Legacy |

---

## Backend-Specific Notes

### Cosmos DB

**Containers:**
| Container | Partition Key | Purpose |
|-----------|---------------|---------|
| `sessions` | `/user_id` | Session metadata |
| `transcript_messages` | `/user_id_session_id` | Messages + embeddings |
| `events` | `/user_id_session_id` | Event metadata |
| `event_chunks` | `/user_id_session_id` | Large event chunks |

**Additional Fields:**
- `id`: Required by Cosmos (same as primary identifier)
- `_type`: Document type discriminator
- `user_id_session_id`: Composite partition key for co-location

**Vector Search:**
- Enabled via `AMPLIFIER_COSMOS_ENABLE_VECTOR=true`
- Uses flat vector index (no HNSW)
- Configured for 3072 dimensions

### DuckDB

**Tables:**
| Table | Primary Key |
|-------|-------------|
| `sessions` | `user_id, session_id` |
| `transcript_messages` | `user_id, session_id, sequence` |
| `events` | `user_id, session_id, event_id` |
| `embeddings` | `user_id, session_id, sequence, source` |

**Vector Search:**
- Native ARRAY type for embeddings
- Cosine similarity via custom function
- Full-text search via built-in FTS

### SQLite

**Tables:** Same as DuckDB

**Vector Search:**
- Embeddings stored as JSON arrays
- Vector operations via Python
- Full-text search via FTS5

---

## Schema Evolution

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-02-05 | Initial schema definition |

### Migration Guidelines

1. **Additive changes** (new optional fields): No migration needed
2. **Type changes**: Requires data migration script
3. **Required field additions**: Must provide default or migrate existing data
4. **Field removal**: Deprecate first, remove in next major version

### Compatibility Promise

- Minor version bumps: Backward compatible
- Major version bumps: May require migration
- All changes documented in this file

---

## Validation

### Required Validation Rules

1. **session_id**: Must be valid UUID format
2. **user_id**: Must be non-empty string
3. **timestamps**: Must be valid ISO 8601 format
4. **role**: Must be one of: `user`, `assistant`, `tool`, `system`
5. **visibility**: Must be one of: `private`, `team`, `org`, `public`
6. **sequence**: Must be >= 0
7. **turn**: Must be >= 1 (when not null)

### Data Integrity

- Session must exist before adding messages/events
- Message sequences must be contiguous (no gaps)
- Chunk counts must match actual chunks
- Embeddings must have correct dimensions for model

---

## API Reference

For programmatic access to this schema, see:
- `amplifier_session_storage/backends/base.py` - Abstract interface
- `amplifier_session_storage/backends/cosmos.py` - Cosmos implementation
- `amplifier_session_storage/backends/duckdb.py` - DuckDB implementation
- `amplifier_session_storage/backends/sqlite.py` - SQLite implementation

---

*This document is the authoritative source of truth for amplifier-session-storage data schema.*
