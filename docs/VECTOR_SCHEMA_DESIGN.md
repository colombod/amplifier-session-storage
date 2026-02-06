# Vector Schema Design - Externalized Multi-Vector Strategy

> **Version**: 2.0.0  
> **Last Updated**: 2025-02-06  
> **Status**: Implemented across all 3 backends (DuckDB, SQLite, Cosmos DB)

**Purpose**: Technical reference for how embeddings are stored externally from transcripts and what content they represent.

**Key Insight**: Different content types need different embeddings for effective search. Vectors are stored in a dedicated table, not inline on transcript rows.

---

## The Problem with Single Embedding Column

**The naive approach (NOT used)**:
```sql
CREATE TABLE transcripts (
    ...
    embedding FLOAT[3072],  -- Single vector for everything
    embedding_model VARCHAR
)
```

**Issues**:
1. Can't search "only in assistant responses" vs "only in thinking"
2. Can't weight different content types differently
3. Loses information about what the vector represents
4. Mixed signal (user question + assistant reasoning + tool output all look the same)

---

## Design Evolution

### Previously Considered: Inline Multi-Column (Option A)

The earlier v0.2.0 design placed 4 vector columns directly on the `transcripts` table:

```sql
-- v0.2.0 approach (NO LONGER USED)
CREATE TABLE transcripts (
    ...
    user_query_vector FLOAT[3072],
    assistant_response_vector FLOAT[3072],
    assistant_thinking_vector FLOAT[3072],
    tool_output_vector FLOAT[3072]
)
```

**Problems discovered**:
- No chunking support: texts exceeding embedding model token limits (8192) could not be split
- 4 HNSW indexes required (storage and rebuild overhead)
- Sparse table: most rows had 3 of 4 vector columns as NULL
- Tight coupling between conversation content and vector data

### Implemented Design: Externalized Vector Table (Option C)

Vectors are stored in a dedicated `transcript_vectors` table. Each row has a single `vector` column and a `content_type` discriminator. Multiple rows per transcript message are supported for chunked content.

---

## Implemented Design: Externalized Vectors

### Transcripts Table (Vector-Free)

```sql
CREATE TABLE transcripts (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    project_slug VARCHAR,
    sequence INTEGER NOT NULL,

    -- Original Amplifier fields
    role VARCHAR NOT NULL,
    content JSON NOT NULL,
    turn INTEGER,
    ts TIMESTAMP,

    -- Extracted text for display/search
    text_content TEXT,

    -- Sync metadata
    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

No embedding columns. No vector metadata. This table stores only conversation data.

### Transcript Vectors Table

```sql
CREATE TABLE transcript_vectors (
    id VARCHAR NOT NULL PRIMARY KEY,
    parent_id VARCHAR NOT NULL,         -- References transcripts.id
    user_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    project_slug VARCHAR,
    content_type VARCHAR NOT NULL,      -- 'user_query', 'assistant_response', etc.
    chunk_index INTEGER NOT NULL DEFAULT 0,
    total_chunks INTEGER NOT NULL DEFAULT 1,
    span_start INTEGER NOT NULL DEFAULT 0,
    span_end INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    source_text TEXT NOT NULL,          -- The text that was embedded
    vector FLOAT[3072],                -- Single vector column
    embedding_model VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

-- Single HNSW index (replaces 4 per-column indexes)
CREATE INDEX idx_vectors_hnsw
ON transcript_vectors USING HNSW (vector)
WITH (metric = 'cosine');

-- Supporting indexes for lookups and filtering
CREATE INDEX idx_vectors_parent ON transcript_vectors (parent_id);
CREATE INDEX idx_vectors_session ON transcript_vectors (user_id, session_id);
CREATE INDEX idx_vectors_user ON transcript_vectors (user_id);
```

### Why This Design?

**Enables precise search queries** via `content_type` filtering:

```sql
-- Search ONLY user questions
SELECT tv.*, t.content, t.role
FROM transcript_vectors tv
JOIN transcripts t ON tv.parent_id = t.id
WHERE tv.content_type = 'user_query'
  AND array_cosine_similarity(tv.vector, @query) > 0.8

-- Search ONLY assistant reasoning
SELECT tv.*, t.content, t.role
FROM transcript_vectors tv
JOIN transcripts t ON tv.parent_id = t.id
WHERE tv.content_type = 'assistant_thinking'
  AND array_cosine_similarity(tv.vector, @query) > 0.8

-- Search everything (all content types)
SELECT tv.*, t.content, t.role
FROM transcript_vectors tv
JOIN transcripts t ON tv.parent_id = t.id
ORDER BY array_cosine_similarity(tv.vector, @query) DESC
LIMIT 10
```

**Additional benefits**:
- Single HNSW index covers all content types
- Chunking is natural (multiple rows with different `chunk_index`)
- `source_text` enables full-text search on the embedded content
- Dense table (every row has exactly one non-NULL vector)
- Clean separation of concerns

### HNSW Index Limitation

DuckDB's HNSW index does not support pre-filtered queries. When a `WHERE user_id = ?` clause is present, DuckDB falls back to sequential scan instead of using the HNSW index. This is acceptable because DuckDB databases are single-user (one `.db` file per user), meaning the `WHERE` filter matches all rows in the database. The sequential scan effectively searches the same dataset the HNSW index would.

---

## Schema Versioning

A `schema_meta` table tracks the current schema version:

```sql
CREATE TABLE schema_meta (
    key VARCHAR PRIMARY KEY,
    value VARCHAR
)
-- key='version', value='2' for externalized vectors
```

On initialization, backends check the version and auto-migrate from version 1 (inline vectors) to version 2 (externalized vectors). See `MULTI_VECTOR_IMPLEMENTATION.md` for migration details.

---

## Field Population Logic

### User Messages

```python
if message["role"] == "user":
    # -> content_type = "user_query"
    # -> source_text = message["content"]
    # -> vector = embed(source_text)
    # All stored in transcript_vectors
```

**Result**: One `transcript_vectors` row with `content_type='user_query'`

### Assistant Messages (Complex)

```python
if message["role"] == "assistant":
    content_blocks = message["content"]  # Array

    # Extract thinking
    thinking_parts = [b["thinking"] for b in content_blocks if b["type"] == "thinking"]
    if thinking_parts:
        text = "\n\n".join(thinking_parts)
        chunks = chunk_text(text, "assistant_thinking")  # May produce multiple chunks
        # Each chunk -> one transcript_vectors row

    # Extract text responses
    text_parts = [b["text"] for b in content_blocks if b["type"] == "text"]
    if text_parts:
        text = "\n\n".join(text_parts)
        chunks = chunk_text(text, "assistant_response")
        # Each chunk -> one transcript_vectors row

    # Skip tool_call blocks - not embeddable
```

**Result**: 1-N `transcript_vectors` rows (1 per content type per chunk)

### Tool Messages

```python
if message["role"] == "tool":
    tool_content = message["content"]
    if isinstance(tool_content, str) and len(tool_content) <= 10000:
        # -> content_type = "tool_output"
        # -> source_text = tool_content
        # -> chunks if > 8192 tokens
    elif isinstance(tool_content, str):
        # Truncate to MAX_TOOL_OUTPUT_LENGTH (10000 chars) first
        tool_content = tool_content[:10000]
```

**Result**: One or more `transcript_vectors` rows with `content_type='tool_output'`

---

## Events Schema - Structured Search (No Embeddings)

### Events Table Schema

```sql
CREATE TABLE events (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    sequence INTEGER NOT NULL,

    -- Event fields
    event VARCHAR NOT NULL,
    ts TIMESTAMP NOT NULL,
    lvl VARCHAR,
    turn INTEGER,
    data JSON,

    -- Extracted fields for fast structured search
    tool_name VARCHAR,
    error_type VARCHAR,
    model_used VARCHAR,

    -- Size tracking
    data_truncated BOOLEAN,
    data_size_bytes INTEGER,
    synced_at TIMESTAMP
)

-- Indexes for structured search (NO vector indexes)
CREATE INDEX idx_event_type ON events(user_id, event, ts);
CREATE INDEX idx_tool_name ON events(user_id, tool_name, ts);
CREATE INDEX idx_level ON events(user_id, lvl, ts);
```

### Why NO Embeddings for Events?

1. **Events are telemetry** - Structured data optimized for filtering
2. **High volume** - Thousands of events per session
3. **Not conversational** - Event types are categorical, not semantic
4. **Storage cost** - Would multiply embedding storage requirements
5. **Search patterns** - Users search by event type, tool, level (structured)

---

## Complete Extraction Pipeline

```python
def extract_all_embeddable_content(message: dict) -> dict[str, str]:
    """
    Extract all embeddable content from a transcript message.

    Returns dict with keys:
        - user_query: str | None
        - assistant_response: str | None
        - assistant_thinking: str | None
        - tool_output: str | None

    Only the relevant key(s) for the message role will be populated.
    """
```

After extraction, each non-None text is:
1. Counted for tokens via tiktoken (`cl100k_base` encoding)
2. If > 8192 tokens: chunked into ~1024-token segments with 128-token overlap
3. Each chunk embedded and stored as a `transcript_vectors` row

---

## Storage Overhead Analysis

### Per Vector Record

| Component | Size |
|-----------|------|
| Vector (3072-d, float32) | ~12 KB |
| Source text | Variable (avg ~500 bytes for short, up to ~4 KB for chunks) |
| Metadata fields | ~200 bytes |

### Per Message (Typical)

| Message Type | Vector Records | Total Vector Storage |
|--------------|---------------|---------------------|
| **User** | 1 record | ~12.5 KB |
| **Assistant (simple)** | 1 record (response only) | ~12.5 KB |
| **Assistant (with thinking)** | 2 records | ~25 KB |
| **Assistant (long thinking, 16k tokens)** | 2 records (response) + ~2 records (chunked thinking) | ~50 KB |
| **Tool** | 1 record | ~12.5 KB |

**For 1000-message session** (typical mix):
- Vector storage: ~15-20 MB
- Transcript storage (no vectors): ~500 KB

**Mitigation**:
- Externalized table means transcripts stay small for fast content queries
- Use quantized indexes (Cosmos: 128 bytes per vector)
- Don't embed tool outputs > 10000 chars
- Use DuckDB for local (no cloud storage costs)

---

## Summary

### Content Types

| Content Type | Contains | From Role | From Field |
|-------------|----------|-----------|------------|
| `user_query` | User questions/input | user | `content` (string) |
| `assistant_response` | Assistant's visible responses | assistant | `content[].text` blocks |
| `assistant_thinking` | Assistant's reasoning | assistant | `content[].thinking` blocks |
| `tool_output` | Tool execution results | tool | `content` (string, truncated to 10000 chars) |

### Events Search Strategy

**No embeddings** - Use structured search:
- Filter by `event` type
- Filter by `tool_name` (extracted from data)
- Filter by `lvl` (ERROR, WARNING, etc.)
- Filter by date range

**Rationale**: Events are telemetry, not conversation - structured queries are more appropriate.

---

## Implementation Status

All items completed:

- Schema updated in all three backends (DuckDB, SQLite, Cosmos)
- Externalized `transcript_vectors` table/document type implemented
- Single HNSW index (DuckDB) / single quantizedFlat index (Cosmos)
- Chunking pipeline for texts > 8192 tokens
- Token counting via tiktoken
- Extraction functions implemented (`content_extraction.py`)
- Schema versioning via `schema_meta` table
- Auto-migration from inline vectors to externalized vectors
- Search queries use `content_type` filtering on transcript_vectors
- Full-text search covers both `transcripts.content` and `transcript_vectors.source_text`

**See also**:
- `EMBEDDING_STRATEGY.md` - What content gets embedded and chunking details
- `MULTI_VECTOR_IMPLEMENTATION.md` - Complete implementation details
- `HYBRID_SEARCH_GUIDE.md` - User guide for search features
