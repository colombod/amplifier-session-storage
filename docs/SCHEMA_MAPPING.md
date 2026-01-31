# Schema Mapping: Disk ↔ Cosmos DB

This document defines the authoritative mapping between on-disk session storage (compatible with amplifier-app-cli and session-analyst) and Cosmos DB storage.

## Design Principle

**Single source of truth**: Both storage backends MUST represent the same logical data model. Field names and semantics are identical across backends.

## Session Metadata

### On-Disk: `metadata.json`

```json
{
  "session_id": "abc123-def456",
  "created": "2025-01-31T12:00:00.000Z",
  "updated": "2025-01-31T12:30:00.000Z",
  "bundle": "bundle:foundation",
  "model": "claude-sonnet-4-20250514",
  "turn_count": 5,
  "message_count": 12,
  "event_count": 45,
  "name": "Debug auth issue",
  "description": "Investigating login failures",
  "parent_id": null,
  "project_slug": "default",
  "tags": ["debugging", "auth"]
}
```

### Cosmos DB: `sessions` Container

| Field | Type | Partition | Required | Notes |
|-------|------|-----------|----------|-------|
| `id` | string | - | Yes | Same as session_id (Cosmos requirement) |
| `session_id` | string | - | Yes | Primary identifier |
| `user_id` | string | PK | Yes | **Partition key** for user isolation |
| `project_slug` | string | - | Yes | Project grouping |
| `created` | ISO datetime | - | Yes | Creation timestamp |
| `updated` | ISO datetime | - | Yes | Last modification |
| `name` | string | - | No | User-defined name |
| `description` | string | - | No | User-defined description |
| `bundle` | string | - | No | Bundle identifier |
| `model` | string | - | No | LLM model used |
| `turn_count` | int | - | Yes | Number of conversation turns |
| `message_count` | int | - | Yes | Total messages in transcript |
| `event_count` | int | - | Yes | Total events logged |
| `parent_id` | string | - | No | Parent session (for spawned) |
| `forked_from_turn` | int | - | No | Fork point if forked |
| `visibility` | enum | - | Yes | private/team/org/public |
| `org_id` | string | - | No | Organization for sharing |
| `team_ids` | string[] | - | No | Teams for sharing |
| `shared_at` | ISO datetime | - | No | When first shared |
| `tags` | string[] | - | No | User-defined tags |
| `_type` | string | - | Yes | Always "session" |

### Field Mapping

| Disk Field | Cosmos Field | Notes |
|------------|--------------|-------|
| `session_id` | `session_id`, `id` | Cosmos needs `id` for document identity |
| `created` | `created` | Same format (ISO 8601) |
| `updated` | `updated` | Same format |
| `bundle` | `bundle` | Same |
| `model` | `model` | Same |
| `turn_count` | `turn_count` | Same |
| `message_count` | `message_count` | Same |
| `event_count` | `event_count` | Same |
| `name` | `name` | Same |
| `description` | `description` | Same |
| `parent_id` | `parent_id` | Same |
| `project_slug` | `project_slug` | **New on disk** (was implicit in path) |
| - | `user_id` | **Cosmos only** (from identity) |
| - | `visibility` | **Cosmos only** (for sharing) |
| - | `org_id` | **Cosmos only** (for sharing) |
| - | `team_ids` | **Cosmos only** (for sharing) |
| - | `shared_at` | **Cosmos only** (for sharing) |
| `tags` | `tags` | Same |

### Migration Notes

**Disk → Cosmos:**
- `user_id` derived from `IdentityContext.get_user_id()`
- `project_slug` derived from directory path (`~/.amplifier/projects/{slug}/sessions/`)
- `visibility` defaults to `private`

**Cosmos → Disk:**
- `user_id`, `visibility`, `org_id`, `team_ids`, `shared_at` are NOT written to disk
- These are cloud-only features for multi-tenant sharing

---

## Transcript Messages

### On-Disk: `transcript.jsonl`

Each line is a JSON object:

```json
{"role": "user", "content": "Hello", "timestamp": "2025-01-31T12:00:00.000Z"}
{"role": "assistant", "content": "Hi there!", "timestamp": "2025-01-31T12:00:01.000Z", "tool_calls": [...]}
```

### Cosmos DB: `transcript_messages` Container

| Field | Type | Partition | Required | Notes |
|-------|------|-----------|----------|-------|
| `id` | string | - | Yes | `{session_id}_msg_{sequence}` |
| `user_id_session_id` | string | PK | Yes | Partition key |
| `user_id` | string | - | Yes | For queries |
| `session_id` | string | - | Yes | For queries |
| `sequence` | int | - | Yes | Order in transcript |
| `role` | enum | - | Yes | user/assistant/tool/system |
| `content` | any | - | Yes | Message content |
| `timestamp` | ISO datetime | - | Yes | When message was created |
| `turn` | int | - | Yes | Conversation turn number |
| `tool_calls` | array | - | No | Tool calls (assistant only) |
| `tool_call_id` | string | - | No | Tool response ID (tool only) |
| `_type` | string | - | Yes | Always "transcript_message" |

### Field Mapping

| Disk Field | Cosmos Field | Notes |
|------------|--------------|-------|
| `role` | `role` | Same |
| `content` | `content` | Same |
| `timestamp` | `timestamp` | Same format |
| `tool_calls` | `tool_calls` | Same |
| - | `sequence` | **Derived** (line number in JSONL) |
| - | `turn` | **Derived** (from message grouping) |
| - | `tool_call_id` | Same if present |

---

## Events

### On-Disk: `events.jsonl`

Each line is a JSON object (can be VERY large - 100k+ tokens):

```json
{"ts": "2025-01-31T12:00:00.000Z", "event": "llm:request", "session_id": "abc123", "lvl": "INFO", "data": {...}}
```

### Cosmos DB: `events` Container

| Field | Type | Partition | Required | Notes |
|-------|------|-----------|----------|-------|
| `id` | string | - | Yes | Unique event ID |
| `event_id` | string | - | Yes | Same as id |
| `user_id_session_id` | string | PK | Yes | Partition key |
| `user_id` | string | - | Yes | For queries |
| `session_id` | string | - | Yes | For queries |
| `event_type` | string | - | Yes | Event type (e.g., "llm:request") |
| `ts` | ISO datetime | - | Yes | Timestamp |
| `turn` | int | - | No | Turn number if applicable |
| `is_chunked` | bool | - | Yes | Whether data is chunked |
| `chunk_count` | int | - | No | Number of chunks |
| `data_size_bytes` | int | - | Yes | Size of data payload |
| `summary` | object | - | Yes | Safe projection fields |
| `data` | object | - | No | Inline data (if not chunked) |
| `_type` | string | - | Yes | Always "event" |

### Chunking Strategy

Events larger than 400KB are chunked and stored in `event_chunks` container:

| Field | Type | Notes |
|-------|------|-------|
| `id` | string | `{event_id}_chunk_{index}` |
| `event_id` | string | Parent event |
| `chunk_index` | int | 0-indexed chunk number |
| `total_chunks` | int | Total number of chunks |
| `chunk_data` | string | JSON-encoded chunk |

### Summary Projection

The `summary` field contains ONLY safe fields that can be returned in queries:

```json
{
  "model": "claude-sonnet",
  "duration_ms": 1234,
  "has_tool_calls": true,
  "has_error": false,
  "usage": {
    "input_tokens": 1000,
    "output_tokens": 500
  }
}
```

**NEVER included in summary:**
- `content` (can be huge)
- `data` (can be huge)
- `messages` (can be huge)

---

## Container Structure (Cosmos DB)

| Container | Partition Key | Purpose |
|-----------|---------------|---------|
| `sessions` | `/user_id` | Session metadata |
| `transcript_messages` | `/user_id_session_id` | Conversation messages |
| `events` | `/user_id_session_id` | Event metadata + small events |
| `event_chunks` | `/user_id_session_id` | Large event chunks |
| `shared_sessions` | `/visibility` | Shared session index |
| `organizations` | `/org_id` | Organization data |
| `teams` | `/org_id` | Team data |
| `user_memberships` | `/user_id` | User membership data |

---

## User Isolation

**CRITICAL**: All queries MUST include user isolation.

### Cosmos Queries
```sql
-- CORRECT: User isolated
SELECT * FROM c WHERE c.user_id = @user_id AND c.session_id = @session_id

-- WRONG: No user isolation (security vulnerability!)
SELECT * FROM c WHERE c.session_id = @session_id
```

### Disk Storage
- Path includes implicit user: `~/.amplifier/projects/{project}/sessions/{session}/`
- Single-user by design (no user_id in metadata.json)

---

## Sync Considerations

### Local → Cloud Sync
1. Read `metadata.json` → Create `SessionMetadata`
2. Add `user_id` from `IdentityContext`
3. Add `project_slug` from path
4. Set `visibility` to `private` (default)
5. Upload to Cosmos

### Cloud → Local Sync
1. Read `SessionMetadata` from Cosmos
2. Write `metadata.json` (excluding cloud-only fields)
3. Download transcript messages → Write `transcript.jsonl`
4. Download events → Write `events.jsonl`

### Conflict Resolution
- `updated` timestamp used for last-writer-wins
- Transcript messages are append-only (no conflicts)
- Events are append-only (no conflicts)
- Rewind operations create REWIND blocks (not deletions)

---

## Testing Requirements

1. **Round-trip tests**: Write to disk → Read to Cosmos → Write to disk → Compare
2. **Field parity tests**: All common fields map correctly
3. **Chunking tests**: Large events chunk and reassemble correctly
4. **Isolation tests**: User A cannot access User B's data
5. **Projection tests**: Query results never contain full event data
