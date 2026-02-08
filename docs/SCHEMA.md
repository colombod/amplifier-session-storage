# Amplifier Session Storage - Data Schema Reference

> **Version**: 2.0.0  
> **Last Updated**: 2025-02-06  
> **Status**: Authoritative Source of Truth

This document defines the canonical schema for all data entities in amplifier-session-storage. All backend implementations (Cosmos DB, DuckDB, SQLite) MUST conform to this schema.

---

## Table of Contents

1. [Overview](#overview)
2. [Sessions](#sessions)
3. [Transcript Messages](#transcript-messages)
4. [Transcript Vectors](#transcript-vectors)
5. [Events](#events)
6. [Schema Meta](#schema-meta)
7. [Backend-Specific Notes](#backend-specific-notes)
8. [Schema Evolution](#schema-evolution)

---

## Overview

### Data Model

```
+------------------------------------------------------------------+
|                         USER                                     |
|  (identified by user_id from Azure CLI or identity provider)     |
+------------------------------------------------------------------+
|                                                                  |
|   +-------------+      +------------------+     +------------+   |
|   |   SESSION   |------|  TRANSCRIPT_MSG  |     |   EVENT    |   |
|   |  metadata   | 1:N  |   messages       |     |  logs      |   |
|   +-------------+      +------------------+     +------------+   |
|                                |                      |          |
|                         +------+------+               |          |
|                         |             |               |          |
|                   +-----+-----+ +----+----+          |          |
|                   | TRANSCRIPT| | EVENT   |          |          |
|                   |  VECTORS  | | CHUNKS  |          |          |
|                   +-----------+ +---------+          |          |
|                                                                  |
+------------------------------------------------------------------+
```

### Key Concepts

- **user_id**: Primary isolation key. All data is partitioned by user.
- **session_id**: Unique identifier for a conversation session.
- **project_slug**: Logical grouping of sessions (maps to local project directories).
- **sequence**: 0-based order of messages in a transcript.
- **turn**: Conversation turn number (user message + assistant response = 1 turn).
- **parent_id**: Links a transcript vector record back to its source transcript message.
- **content_type**: Discriminates vector records by source: `user_query`, `assistant_response`, `assistant_thinking`, `tool_output`.

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

The transcripts table stores conversation messages. It contains **no vector columns** -- all embeddings are stored in the separate transcript vectors table.

### Schema Definition

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | string | Yes | - | `{session_id}_msg_{sequence}` |
| `user_id` | string | Yes | - | User identifier |
| `host_id` | string | Yes | - | Host machine identifier |
| `project_slug` | string | Yes | - | Project grouping (denormalized for queries) |
| `session_id` | string | Yes | - | Parent session ID |
| `sequence` | integer | Yes | - | 0-based message order |
| `role` | enum | Yes | - | Message role: `user`, `assistant`, `tool`, `system` |
| `content` | JSON | Yes | - | Message content (string or structured JSON, see Content Structure below) |
| `turn` | integer | No | `null` | Conversation turn number (can be null for system messages) |
| `ts` | ISO 8601 datetime | Yes | - | When message was created (normalized from metadata.timestamp or top-level timestamp) |
| `metadata` | JSON object | No | `null` | Dynamic metadata dict containing timestamp and other module-added fields |
| `synced_at` | ISO 8601 datetime | Yes | - | When record was last synced |

> **Note:** The `metadata` field is intentionally dynamic and can contain various fields added by different Amplifier modules (e.g., `timestamp`, `source`, `confidence`, `redaction_applied`). The `ts` field is normalized at write time for efficient querying.

> **Note:** Fields such as `text_content`, `tool_calls`, `tool_call_id`, and `thinking` are logical fields stored within the JSON `content` column, not discrete DuckDB columns. See the Content Structure section below for how they appear inside `content`.

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

## Transcript Vectors

Vector embeddings are stored in a dedicated table, separate from transcript messages. Each record holds one vector for one content type (optionally one chunk of a longer text), linking back to its parent transcript message via `parent_id`.

### Schema Definition

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | string | Yes | - | `{parent_id}_{content_type}_{chunk_index}` |
| `parent_id` | string | Yes | - | References transcript message ID |
| `user_id` | string | Yes | - | User identifier |
| `session_id` | string | Yes | - | Parent session ID |
| `project_slug` | string | No | `null` | Project grouping |
| `content_type` | enum | Yes | - | `user_query`, `assistant_response`, `assistant_thinking`, `tool_output` |
| `chunk_index` | integer | Yes | `0` | 0-based chunk position |
| `total_chunks` | integer | Yes | `1` | Total chunks for this content type |
| `span_start` | integer | Yes | `0` | Character offset start in original text |
| `span_end` | integer | Yes | - | Character offset end in original text |
| `token_count` | integer | Yes | - | Tokens in this chunk (tiktoken cl100k_base) |
| `source_text` | string | Yes | - | The text that was embedded |
| `vector` | float[] | No (nullable during embedding generation) | `null` | Embedding vector (dimensions vary by model) |
| `embedding_model` | string | No | `null` | Embedding model used |
| `created_at` | ISO 8601 datetime | Yes | now | When record was created |

### Content Types

| Content Type | Source | Typical Use |
|-------------|--------|-------------|
| `user_query` | User message content | Find similar questions |
| `assistant_response` | Assistant text blocks | Find similar answers |
| `assistant_thinking` | Assistant thinking blocks | Find similar reasoning |
| `tool_output` | Tool output content (truncated to 10000 chars) | Find similar tool results |

### Chunking

Texts exceeding 8192 tokens are split into chunks:
- Target chunk size: 1024 tokens
- Overlap between chunks: 128 tokens
- Minimum chunk size: 64 tokens (smaller segments merged with previous)
- Content-aware splitting: markdown-aware for assistant, line-aware for tool, sentence-aware for user
- Token counting: tiktoken `cl100k_base` encoding

A single content extraction may produce multiple vector records when chunking activates. All chunks share the same `parent_id` and `content_type`, differentiated by `chunk_index`.

### Example

```json
{
  "id": "sess_abc_msg_5_assistant_thinking_0",
  "parent_id": "sess_abc_msg_5",
  "user_id": "user@example.com",
  "session_id": "sess_abc",
  "project_slug": "amplifier-core",
  "content_type": "assistant_thinking",
  "chunk_index": 0,
  "total_chunks": 2,
  "span_start": 0,
  "span_end": 4096,
  "token_count": 1024,
  "source_text": "The user wants to understand the vector architecture...",
  "vector": [0.123, -0.456, 0.789, "..."],
  "embedding_model": "text-embedding-3-large",
  "created_at": "2025-02-06T10:30:00.000Z"
}
```

### Supported Embedding Models

| Model | Provider | Dimensions | Notes |
|-------|----------|------------|-------|
| `text-embedding-3-large` | OpenAI/Azure | 3072 | Recommended |
| `text-embedding-3-small` | OpenAI/Azure | 1536 | Faster, smaller |
| `text-embedding-ada-002` | OpenAI/Azure | 1536 | Legacy |

### Indexes

| Index | Fields | Purpose |
|-------|--------|---------|
| Primary | `id` | Unique vector record lookup |
| HNSW | `vector` | Cosine similarity search (DuckDB only, single index) |
| Parent | `parent_id` | Join to transcript message |
| Session | `user_id`, `session_id` | Filter vectors by session |
| User | `user_id` | Filter vectors by user |

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

### Indexes

| Index | Fields | Purpose |
|-------|--------|---------|
| Primary | `user_id`, `session_id`, `event_id` | Unique event lookup |
| Time | `user_id`, `session_id`, `ts` | Events by time |
| Type | `user_id`, `session_id`, `event_type` | Filter by type |

---

## Schema Meta

A simple key-value table that tracks schema version and metadata for migration support.

### Schema Definition

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `key` | string | Yes | Metadata key (primary key) |
| `value` | string | Yes | Metadata value |

### Current Keys

| Key | Value | Description |
|-----|-------|-------------|
| `version` | `"2"` | Schema version. `1` = inline vectors on transcripts. `2` = externalized vectors in transcript_vectors. |

### DDL

**DuckDB**:
```sql
CREATE TABLE IF NOT EXISTS schema_meta (
    key VARCHAR PRIMARY KEY,
    value VARCHAR
)
```

**SQLite**:
```sql
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT
)
```

On backend initialization, the version is checked. If version < 2, auto-migration moves inline vectors from `transcripts` to `transcript_vectors` and sets version to 2. See `MULTI_VECTOR_IMPLEMENTATION.md` for migration details.

---

## Backend-Specific Notes

### Cosmos DB

**Containers:**
| Container | Partition Key | Purpose |
|-----------|---------------|---------|
| `sessions` | `/user_id` | Session metadata |
| `transcript_messages` | `/user_id_session_id` | Messages (type=transcript_message) + vectors (type=transcript_vector) |
| `events` | `/user_id_session_id` | Event metadata |
| `event_chunks` | `/user_id_session_id` | Large event chunks |

**Additional Fields:**
- `id`: Required by Cosmos (same as primary identifier)
- `type`: Document type discriminator (`transcript_message` or `transcript_vector`)
- `partition_key`: Composite partition key for co-location (`{user_id}_{session_id}`)

**Vector Search:**
- Single vector index on `/vector` path (quantizedFlat)
- Enabled via `AMPLIFIER_COSMOS_ENABLE_VECTOR=true`
- Configured for 3072 dimensions
- 1.8MB safety limit on transcript documents (defense-in-depth below Cosmos 2MB hard limit)

### DuckDB

**Tables:**
| Table | Primary Key |
|-------|-------------|
| `sessions` | `user_id, session_id` |
| `transcripts` | `id` (format: `{session_id}_msg_{sequence}`) |
| `transcript_vectors` | `id` (format: `{parent_id}_{content_type}_{chunk_index}`) |
| `events` | `user_id, session_id, event_id` |
| `schema_meta` | `key` |

**Vector Search:**
- Native `FLOAT[N]` array type for vectors in `transcript_vectors`
- Single HNSW index (`idx_vectors_hnsw`) with cosine metric
- `vectors_with_context` view (JOIN of transcript_vectors + transcripts)
- HNSW persistence enabled for disk-based databases
- HNSW limitation: WHERE filters cause sequential scan fallback (acceptable for single-user DBs)
- Full-text search via LIKE on transcripts.content and transcript_vectors.source_text

### SQLite

**Tables:** Same as DuckDB

**Vector Search:**
- Vectors stored as JSON arrays in `vector_json TEXT` column
- Vector operations via Python numpy (brute-force cosine similarity)
- Full-text search via LIKE on transcripts.content and transcript_vectors.source_text

---

## Schema Evolution

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-02-05 | Initial schema definition with inline vector columns on transcripts |
| 2.0.0 | 2025-02-06 | Externalized vectors to transcript_vectors table. Added schema_meta table. Added chunking support. Removed vector columns from transcripts. |

### Migration Guidelines

1. **Additive changes** (new optional fields): No migration needed
2. **Type changes**: Requires data migration script
3. **Required field additions**: Must provide default or migrate existing data
4. **Field removal**: Deprecate first, remove in next major version
5. **Schema version bump**: Update `schema_meta.version`, add auto-migration logic

### Auto-Migration (v1 -> v2)

On initialization, backends check `schema_meta.version`. If version < 2:
1. Create `transcript_vectors` table if not exists
2. Move inline vectors from `transcripts` to `transcript_vectors`
3. Drop old vector columns and indexes from `transcripts`
4. Set version to 2

Migration is idempotent and safe to run multiple times.

### Compatibility Promise

- Minor version bumps: Backward compatible
- Major version bumps: May require migration (auto-migration provided where possible)
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
8. **content_type**: Must be one of: `user_query`, `assistant_response`, `assistant_thinking`, `tool_output`
9. **chunk_index**: Must be >= 0 and < total_chunks
10. **token_count**: Must be > 0

### Data Integrity

- Session must exist before adding messages/events
- Message sequences must be contiguous (no gaps)
- Chunk counts must match actual chunks
- Vector dimensions must match embedding model
- `parent_id` in transcript_vectors must reference a valid transcript message

---

## API Reference

For programmatic access to this schema, see:
- `amplifier_session_storage/backends/base.py` - Abstract interface
- `amplifier_session_storage/backends/cosmos.py` - Cosmos implementation
- `amplifier_session_storage/backends/duckdb.py` - DuckDB implementation
- `amplifier_session_storage/backends/sqlite.py` - SQLite implementation
- `amplifier_session_storage/content_extraction.py` - Content extraction and token counting
- `amplifier_session_storage/chunking.py` - Semantic chunking pipeline

---

*This document is the authoritative source of truth for amplifier-session-storage data schema.*
