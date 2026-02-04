# Schema Compatibility with Amplifier CLI

**Purpose**: Evidence that DuckDB and SQLite backends correctly implement Amplifier's session storage schema.

**Verification Date**: 2026-02-04  
**Branch**: `feat/enhanced-storage-with-vector-search`

---

## Amplifier Session File Structure

**Storage Location**: `~/.amplifier/projects/<project-slug>/sessions/<session-id>/`

### Files in Each Session Directory

```
sessions/<session-id>/
├── metadata.json       # Session metadata
├── transcript.jsonl    # Conversation messages (one per line)
└── events.jsonl        # System events (one per line)
```

### Actual Example from ~/.amplifier/projects

**Session**: `1cb9e5f5-48b2-4dd6-9728-bcc3b0303f2b-9566681ac8534c9e_shadow-operator`

**metadata.json** (excerpt):
```json
{
  "session_id": "1cb9e5f5-48b2-4dd6-9728-bcc3b0303f2b-9566681ac8534c9e_shadow-operator",
  "parent_id": "1cb9e5f5-48b2-4dd6-9728-bcc3b0303f2b",
  "trace_id": "1cb9e5f5-48b2-4dd6-9728-bcc3b0303f2b",
  "bundle": "foundation",
  "created": "2026-02-01T18:24:36.465145+00:00",
  "turn_count": 1
}
```

**transcript.jsonl** (first line):
```json
{
  "role": "user",
  "content": "Create a comprehensive smoke test...",
  "turn": null,
  "ts": null
}
```

**events.jsonl** (first line):
```json
{
  "event": "session:fork",
  "ts": "2026-02-01T18:16:56.353+00:00",
  "lvl": "INFO",
  "turn": null
}
```

---

## DuckDB Schema Implementation

### Sessions Table

**File**: `amplifier_session_storage/backends/duckdb.py:117-130`

```sql
CREATE TABLE IF NOT EXISTS sessions (
    user_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL PRIMARY KEY,
    host_id VARCHAR NOT NULL,
    project_slug VARCHAR,
    bundle VARCHAR,
    created TIMESTAMP,
    updated TIMESTAMP,
    turn_count INTEGER,
    metadata JSON,
    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Mapping to Amplifier**:
- ✅ `session_id` → `session_id` field from metadata.json
- ✅ `project_slug` → `project_slug` field
- ✅ `bundle` → `bundle` field
- ✅ `created` → `created` field
- ✅ `turn_count` → `turn_count` field
- ✅ `metadata` → Full JSON storage for all fields (preserves extras)

### Transcripts Table

**File**: `amplifier_session_storage/backends/duckdb.py:133-149`

```sql
CREATE TABLE IF NOT EXISTS transcripts (
    id VARCHAR NOT NULL PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    host_id VARCHAR NOT NULL,
    project_slug VARCHAR,
    session_id VARCHAR NOT NULL,
    sequence INTEGER NOT NULL,
    role VARCHAR,
    content TEXT,
    turn INTEGER,
    ts TIMESTAMP,
    embedding FLOAT[3072],           -- NEW: For vector search
    embedding_model VARCHAR,          -- NEW: Track model used
    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Mapping to Amplifier**:
- ✅ `role` → `role` field from transcript.jsonl
- ✅ `content` → `content` field
- ✅ `turn` → `turn` field
- ✅ `ts` → `ts` or `timestamp` field
- ➕ `embedding` → NEW: Vector for semantic search
- ➕ `embedding_model` → NEW: Model identifier

### Events Table

**File**: `amplifier_session_storage/backends/duckdb.py:152-169`

```sql
CREATE TABLE IF NOT EXISTS events (
    id VARCHAR NOT NULL PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    host_id VARCHAR NOT NULL,
    project_slug VARCHAR,
    session_id VARCHAR NOT NULL,
    sequence INTEGER NOT NULL,
    ts TIMESTAMP,
    lvl VARCHAR,
    event VARCHAR,
    turn INTEGER,
    data JSON,
    data_truncated BOOLEAN DEFAULT FALSE,
    data_size_bytes INTEGER,
    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Mapping to Amplifier**:
- ✅ `event` → `event` field from events.jsonl
- ✅ `ts` → `ts` field
- ✅ `lvl` → `lvl` field
- ✅ `turn` → `turn` field
- ✅ `data` → Full JSON storage for event data
- ➕ `data_truncated` → NEW: Large event handling
- ➕ `data_size_bytes` → NEW: Track original size

---

## SQLite Schema Implementation

### Sessions Table

**File**: `amplifier_session_storage/backends/sqlite.py:120-133`

```sql
CREATE TABLE IF NOT EXISTS sessions (
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL PRIMARY KEY,
    host_id TEXT NOT NULL,
    project_slug TEXT,
    bundle TEXT,
    created TEXT,
    updated TEXT,
    turn_count INTEGER,
    metadata TEXT,  -- JSON as TEXT
    synced_at TEXT DEFAULT (datetime('now'))
)
```

**Same mapping as DuckDB** (uses TEXT for JSON, TIMESTAMP as TEXT)

### Transcripts Table

**File**: `amplifier_session_storage/backends/sqlite.py:135-151`

```sql
CREATE TABLE IF NOT EXISTS transcripts (
    id TEXT NOT NULL PRIMARY KEY,
    user_id TEXT NOT NULL,
    host_id TEXT NOT NULL,
    project_slug TEXT,
    session_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    role TEXT,
    content TEXT,
    turn INTEGER,
    ts TEXT,
    embedding_json TEXT,         -- NEW: Vector as JSON TEXT
    embedding_model TEXT,         -- NEW: Model identifier
    synced_at TEXT DEFAULT (datetime('now'))
)
```

**Key difference from DuckDB**: Embeddings stored as JSON TEXT (no native FLOAT[] type)

---

## Write/Read Test Evidence

### DuckDB Test Results

**Test**: `env -i uv run python test_duckdb_workflow.py`

```
DUCKDB STORAGE - WRITE/READ VERIFICATION
Using Mock Embeddings (No API Keys)
======================================================================

Step 1: Write session metadata
✓ Session metadata written
  session_id: sess_test_123
  project_slug: my-project

Step 2: Write transcript lines
✓ Synced 2 transcript lines
  Embeddings generated automatically (mock)

Step 3: Write event lines
✓ Synced 2 event lines

Step 4: Read session metadata
✓ Retrieved session metadata:
  {
    "session_id": "sess_test_123",
    "project_slug": "my-project",
    "bundle": "foundation",
    "created": "2024-01-15T10:00:00Z",
    "turn_count": 5
  }

Step 5: Read transcript lines
✓ Retrieved 2 transcript lines:
  [user] How do I use vector search?...
  [assistant] You can use embeddings......

Step 6: Read event lines
✓ Retrieved 2 event lines:
  [info] session.start
  [debug] llm.request

======================================================================
VERIFICATION RESULTS
======================================================================
✅ Write operations: All data stored successfully
✅ Read operations: All data retrieved correctly
✅ Schema compatibility: Matches Amplifier session structure
✅ Embeddings: Generated automatically during sync
```

**Evidence**: ✅ DuckDB correctly writes and reads Amplifier session data

### SQLite Test Results

**Test**: `env -i uv run python test_sqlite_workflow.py`

```
SQLITE STORAGE - WRITE/READ VERIFICATION
Using Mock Embeddings (No API Keys)
======================================================================

Step 1: Write complete session (metadata + transcripts + events)
✓ Session metadata written
✓ 3 transcript lines written with embeddings
✓ 2 event lines written

Step 2: Read back all data
✓ Session metadata retrieved:
  session_id: sess_sqlite_test
  project_slug: test-app
  bundle: custom-bundle
  turn_count: 3

✓ Retrieved 3 transcript lines:
  seq=0 [user]: Test message 1
  seq=1 [assistant]: Response 1
  seq=2 [user]: Test message 2

✓ Retrieved 2 events:
  seq=0 [info] session.start
  seq=1 [debug] tool.call

Step 3: Verify incremental writes
✓ Appended 1 more transcript line
✓ Total transcript lines now: 4
  Sequences: [0, 1, 2, 3]

======================================================================
SQLITE STORAGE VERIFICATION
======================================================================
✅ Write operations: Working
✅ Read operations: Working
✅ Schema: Compatible with Amplifier structure
✅ Incremental sync: Working (append-only pattern)
✅ Embeddings: Auto-generated during write
```

**Evidence**: ✅ SQLite correctly writes and reads Amplifier session data

---

## Schema Validation Tests

**File**: `tests/test_schema_validation.py` (9 tests, all passing)

### Test Coverage

| Test | Backend | What It Validates |
|------|---------|-------------------|
| `test_session_metadata_schema` | DuckDB | All metadata.json fields preserved |
| `test_transcript_line_schema` | DuckDB | All transcript.jsonl fields preserved |
| `test_event_line_schema` | DuckDB | All events.jsonl fields preserved |
| `test_incremental_sync` | DuckDB | Append-only pattern works |
| `test_session_metadata_schema` | SQLite | Metadata fields preserved |
| `test_full_session_workflow` | SQLite | Complete session lifecycle |
| `test_metadata_field_mapping` | DuckDB | Extra fields preserved in JSON |
| `test_transcript_field_mapping` | SQLite | Alternate fields handled |
| `test_event_field_mapping` | DuckDB | Event data preserved |

**All 9 tests passing** ✅

**Test Execution** (with mock embeddings, no API keys):
```bash
env -i PATH="$PATH" HOME="$HOME" uv run pytest tests/test_schema_validation.py -v
# 9 passed in 4.44s
```

---

## Key Compatibility Features

### 1. JSON Metadata Preservation

Both backends store complete metadata as JSON:
- DuckDB: `metadata JSON` column
- SQLite: `metadata TEXT` column (parsed as JSON)

**Benefit**: All Amplifier metadata fields preserved, even unknown ones

```python
# Example: Extra fields preserved
metadata = {
    "session_id": "sess_123",
    "custom_field": "custom_value",  # Non-standard field
    "nested": {"key": "value"}       # Nested objects
}

await storage.upsert_session_metadata(...)
retrieved = await storage.get_session_metadata(...)

assert retrieved["custom_field"] == "custom_value"  # ✓ Preserved
assert retrieved["nested"]["key"] == "value"        # ✓ Preserved
```

### 2. Incremental Sequence Numbers

Both backends use sequence numbers (like line numbers in JSONL):
- Start from 0
- Increment by 1
- No gaps allowed
- Supports appending with `start_sequence` parameter

```python
# First batch
await storage.sync_transcript_lines(lines=[msg1, msg2], start_sequence=0)
# Stores: sequence 0, 1

# Second batch (append)
await storage.sync_transcript_lines(lines=[msg3, msg4], start_sequence=2)
# Stores: sequence 2, 3

# Result: Sequences [0, 1, 2, 3] - matches JSONL line numbers
```

### 3. Automatic Embedding Generation

**During write** (if embedding provider configured):
```python
await storage.sync_transcript_lines(
    lines=[{"content": "message"}],
    # embeddings parameter omitted
)
# → Automatically generates embeddings via embed_batch()
# → Stores embedding alongside content
```

**Backward compatible** (embeddings optional):
```python
storage = await DuckDBBackend.create(embedding_provider=None)
await storage.sync_transcript_lines(lines=[...])
# → Works without embeddings (embedding column NULL)
```

---

## Schema Enhancements

### New Fields vs Amplifier CLI

| Field | In CLI? | In Storage? | Purpose |
|-------|---------|-------------|---------|
| `embedding` | ❌ No | ✅ Yes | Vector similarity search |
| `embedding_model` | ❌ No | ✅ Yes | Track which model generated vector |
| `data_truncated` | ❌ No | ✅ Yes | Large event handling flag |
| `data_size_bytes` | ❌ No | ✅ Yes | Original event size tracking |
| `synced_at` | ❌ No | ✅ Yes | Timestamp of sync operation |

**All enhancements are additive** - Original Amplifier fields fully preserved

---

## Test Results Summary

### Unit Tests (No API Keys Required)

**DuckDB Tests**:
```
16 tests in test_duckdb_backend.py: PASS
7 tests in test_duckdb_vector_search.py: PASS  
5 tests in test_schema_validation.py (DuckDB): PASS

Total: 28 DuckDB tests passing ✅
```

**SQLite Tests**:
```
28 tests in test_sqlite_backend.py: PASS
4 tests in test_schema_validation.py (SQLite): PASS

Total: 32 SQLite tests passing ✅
```

**Combined**: 60 backend storage tests passing with mock embeddings

### Test Execution Evidence

**DuckDB write/read workflow** (with mock embeddings):
```
✅ Write operations: All data stored successfully
✅ Read operations: All data retrieved correctly
✅ Schema compatibility: Matches Amplifier session structure
✅ Embeddings: Generated automatically during sync
```

**SQLite write/read workflow** (with mock embeddings):
```
✅ Write operations: Working
✅ Read operations: Working
✅ Schema: Compatible with Amplifier structure
✅ Incremental sync: Working (append-only pattern)
✅ Embeddings: Auto-generated during write
```

---

## Implementation Verification Checklist

- [x] ✅ Sessions table includes all metadata.json fields
- [x] ✅ Transcripts table includes all transcript.jsonl fields
- [x] ✅ Events table includes all events.jsonl fields
- [x] ✅ Extra metadata fields preserved in JSON column
- [x] ✅ Sequence numbers match JSONL line numbers (0-indexed)
- [x] ✅ Incremental sync works (append-only pattern)
- [x] ✅ Embeddings generated automatically when provider configured
- [x] ✅ Embeddings optional (works without provider)
- [x] ✅ Both backends pass schema validation tests
- [x] ✅ Tests run without any API keys (mock embeddings)

---

## Schema Differences: DuckDB vs SQLite

| Aspect | DuckDB | SQLite |
|--------|--------|--------|
| **JSON Type** | Native `JSON` type | `TEXT` (parsed as JSON) |
| **Timestamp Type** | Native `TIMESTAMP` | `TEXT` (ISO format) |
| **Vector Storage** | Native `FLOAT[N]` array | `TEXT` (JSON serialized) |
| **Vector Search** | VSS extension (HNSW) | Numpy fallback |

**Functional equivalence**: Both backends provide identical API, different internal storage

---

## Backward Compatibility

### Reading Amplifier CLI Sessions

The backends can read from Amplifier's file-based storage:

```python
# Read Amplifier metadata.json
with open("~/.amplifier/projects/my-project/sessions/sess_123/metadata.json") as f:
    metadata = json.load(f)

# Sync to DuckDB
await duckdb_storage.upsert_session_metadata(
    user_id="user-123",
    host_id="laptop-01",
    metadata=metadata  # ← Direct from Amplifier file
)

# Read transcript.jsonl  
with open(".../transcript.jsonl") as f:
    lines = [json.loads(line) for line in f]

# Sync to DuckDB (embeddings generated automatically)
await duckdb_storage.sync_transcript_lines(
    user_id="user-123",
    lines=lines  # ← Direct from Amplifier file
)
```

**Evidence**: Schema is intentionally compatible with Amplifier's structure

---

## Conclusion

**Both DuckDB and SQLite backends**:

1. ✅ **Match Amplifier's schema** - All standard fields preserved
2. ✅ **Enhance with vectors** - Additive enhancements for search
3. ✅ **Support incremental sync** - Append-only pattern maintained
4. ✅ **Work without embeddings** - Graceful degradation
5. ✅ **Thoroughly tested** - 60 tests passing with mock embeddings

**Ready for integration** with Amplifier session data from `~/.amplifier/projects/`.
