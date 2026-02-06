# Multi-Vector Embedding Implementation - Complete Guide

**Status**: Implemented across all 3 backends (externalized vector architecture)  
**Version**: 0.3.0 (breaking change from 0.2.0 - vectors externalized from transcripts)  
**Schema Version**: 2

---

## Overview

The session storage uses **4 separate content types for vector embeddings**, stored in a dedicated `transcript_vectors` table (DuckDB/SQLite) or as `transcript_vector` documents (Cosmos DB). This externalized architecture replaces the earlier inline 4-column design, enabling chunking support and a single vector index.

### Why Multi-Vector?

Amplifier assistant messages contain **multiple content types** in a single message:
- **Thinking blocks**: Internal reasoning (often 1000s of tokens)
- **Text blocks**: User-visible responses
- **Tool calls**: Function invocations

Embedding these together loses semantic specificity. With multi-vector:
- Search **only** in reasoning: "How did the AI approach this problem?"
- Search **only** in responses: "What did it tell me about X?"
- Search **only** in user queries: "When did I ask about Y?"

### Why Externalized?

The earlier inline approach stored 4 vector columns directly on the `transcripts` table. This had limitations:
- No chunking support (long texts exceeded embedding model token limits)
- 4 separate HNSW indexes (storage and maintenance overhead)
- Sparse columns (most rows had 3 of 4 vectors as NULL)
- Tight coupling between conversation data and vector data

The externalized design stores vectors in a separate table with:
- One vector column, one HNSW index
- Native chunking support (multiple vector records per content type)
- `source_text` stored alongside each vector for full-text search over embedded content
- Clean separation of concerns

---

## Schema Design

### Transcripts Table (Vector-Free)

The transcripts table contains no vector columns:

**DuckDB**:
```sql
CREATE TABLE transcripts (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    project_slug VARCHAR,
    sequence INTEGER NOT NULL,
    role VARCHAR NOT NULL,
    content JSON NOT NULL,
    turn INTEGER,
    ts TIMESTAMP,
    text_content TEXT,
    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### Transcript Vectors Table

**DuckDB**:
```sql
CREATE TABLE transcript_vectors (
    id VARCHAR NOT NULL PRIMARY KEY,
    parent_id VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    project_slug VARCHAR,
    content_type VARCHAR NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    total_chunks INTEGER NOT NULL DEFAULT 1,
    span_start INTEGER NOT NULL DEFAULT 0,
    span_end INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    source_text TEXT NOT NULL,
    vector FLOAT[3072],
    embedding_model VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

-- Single HNSW index
CREATE INDEX idx_vectors_hnsw
ON transcript_vectors USING HNSW (vector)
WITH (metric = 'cosine');

-- Supporting indexes
CREATE INDEX idx_vectors_parent ON transcript_vectors (parent_id);
CREATE INDEX idx_vectors_session ON transcript_vectors (user_id, session_id);
CREATE INDEX idx_vectors_user ON transcript_vectors (user_id);
```

**SQLite**:
```sql
CREATE TABLE transcript_vectors (
    id TEXT NOT NULL PRIMARY KEY,
    parent_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    project_slug TEXT,
    content_type TEXT NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    total_chunks INTEGER NOT NULL DEFAULT 1,
    span_start INTEGER NOT NULL DEFAULT 0,
    span_end INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    source_text TEXT NOT NULL,
    vector_json TEXT,
    embedding_model TEXT,
    created_at TEXT DEFAULT (datetime('now'))
)
```

**Cosmos DB** (document structure):
```json
{
  "id": "sess_abc_msg_5_assistant_thinking_0",
  "type": "transcript_vector",
  "partition_key": "user-123_sess_abc",
  "user_id": "user-123",
  "host_id": "laptop-01",
  "project_slug": "my-project",
  "session_id": "sess_abc",
  "parent_id": "sess_abc_msg_5",
  "content_type": "assistant_thinking",
  "chunk_index": 0,
  "total_chunks": 2,
  "span_start": 0,
  "span_end": 4096,
  "token_count": 1024,
  "source_text": "The user wants to understand...",
  "vector": [0.1, 0.2, "..., 3072 floats"],
  "embedding_model": "text-embedding-3-large"
}
```

### Schema Versioning

A `schema_meta` table tracks the schema version:

```sql
CREATE TABLE schema_meta (
    key VARCHAR PRIMARY KEY,
    value VARCHAR
)
-- Current: key='version', value='2'
```

On initialization, the backend checks the schema version and auto-migrates from version 1 (inline vectors) to version 2 (externalized vectors) if needed.

---

## Content Extraction Rules

### User Messages
**Structure**: Simple string
```json
{"role": "user", "content": "How do I use vector search?"}
```
**Extraction**: `content` -> `user_query` content type

### Assistant Messages
**Structure**: Array of content blocks
```json
{
  "role": "assistant",
  "content": [
    {"type": "thinking", "thinking": "The user wants to understand..."},
    {"type": "text", "text": "To use vector search..."},
    {"type": "tool_call", "id": "...", "name": "read_file", "input": {...}}
  ]
}
```

**Extraction**:
- All `thinking` blocks -> joined -> `assistant_thinking` content type
- All `text` blocks -> joined -> `assistant_response` content type
- Tool calls **NOT embedded** (no semantic value)

### Tool Messages
**Structure**: String output
```json
{"role": "tool", "content": "{file contents or tool result}"}
```
**Extraction**: `content` (truncated to 10000 chars via `MAX_TOOL_OUTPUT_LENGTH`) -> `tool_output` content type

### Chunking Pipeline

After extraction, texts exceeding 8192 tokens are chunked:

1. **Count tokens** via tiktoken (`cl100k_base` encoding)
2. **If <= 8192 tokens**: Single chunk, single vector record
3. **If > 8192 tokens**: Split into ~1024-token chunks with 128-token overlap
   - Content-aware splitting: markdown-aware for assistant content, line-aware for tool output, sentence-aware for user queries
   - Minimum chunk size of 64 tokens (runts merged with previous chunk)
4. **Each chunk** becomes a separate `transcript_vectors` record with its own `chunk_index` and `total_chunks`

---

## API Usage

### Writing Data (Automatic Embedding Generation)

```python
from amplifier_session_storage.backends import DuckDBBackend, DuckDBConfig
from amplifier_session_storage.embeddings import OpenAIEmbeddings

# Create backend with embedding provider
embeddings = OpenAIEmbeddings.from_env()
storage = await DuckDBBackend.create(
    config=DuckDBConfig(db_path="sessions.db"),
    embedding_provider=embeddings
)

# Sync transcript - embeddings generated automatically
lines = [
    {"role": "user", "content": "How does MMR work?"},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "MMR balances relevance and diversity..."},
            {"type": "text", "text": "MMR stands for Maximum Marginal Relevance..."}
        ]
    }
]

synced = await storage.sync_transcript_lines(
    user_id="user-123",
    host_id="laptop-01",
    project_slug="my-project",
    session_id="sess_abc",
    lines=lines
)

# Output: Generated embeddings: 1 user, 1 responses, 1 thinking, 0 tool
```

**What happens**:
1. Extracts 3 texts: user query, assistant thinking, assistant response
2. Checks token counts; chunks any text exceeding 8192 tokens
3. Generates embeddings in batch API calls
4. Stores transcript row (no vectors) in `transcripts`
5. Stores each vector + metadata as a record in `transcript_vectors`

### Searching - Target Specific Content Types

```python
# Generate query embedding
query_vec = await embeddings.embed_text("diversity algorithm")

# Search ONLY in assistant thinking (reasoning)
results = await storage.vector_search(
    user_id="user-123",
    query_vector=query_vec,
    vector_columns=["assistant_thinking"],  # Only search thinking blocks
    top_k=10
)

# Search ONLY in user queries
results = await storage.vector_search(
    user_id="user-123",
    query_vector=query_vec,
    vector_columns=["user_query"],  # Only search what user asked
    top_k=10
)

# Search across ALL content types (default)
results = await storage.vector_search(
    user_id="user-123",
    query_vector=query_vec,
    # vector_columns not specified = search all
    top_k=10
)
```

### Search Strategy: Over-Fetch + Dedup

Since vectors are externalized as separate rows (potentially multiple chunks per content type per message), search results must be deduplicated to return one result per parent transcript message.

**All backends** use the same approach:
1. Query `transcript_vectors` ordered by similarity
2. Over-fetch 3x the requested `top_k` to ensure enough unique parents
3. Deduplicate in Python using a `dict` keyed by `parent_id`, keeping the entry with the highest similarity score

**DuckDB** (using `vectors_with_context` view):
```sql
SELECT parent_id, session_id, project_slug, sequence,
       content, role, turn, ts, content_type,
       array_cosine_similarity(vector, <vec_literal>) AS similarity
FROM vectors_with_context
WHERE user_id = ? AND content_type IN (?)
ORDER BY similarity DESC
LIMIT ?  -- top_k * 3
```

```python
# Python-side dedup
seen: dict[str, SearchResult] = {}
for row in results_raw:
    parent_id = row[0]
    score = float(row[9])
    if parent_id not in seen or score > seen[parent_id].score:
        seen[parent_id] = SearchResult(...)
```

**Cosmos DB**:
```sql
SELECT TOP @fetchLimit c.parent_id, c.session_id, c.content_type,
       VectorDistance(c.vector, @queryVector) AS distance
FROM c
WHERE c.type = "transcript_vector"
  AND c.user_id = @userId
  AND ARRAY_CONTAINS(@contentTypes, c.content_type)
ORDER BY VectorDistance(c.vector, @queryVector)
```

---

## Performance Characteristics

### Embedding Generation (During Ingestion)

```
10 messages with mixed content:
- 10 user queries
- 10 assistant responses
- 10 assistant thinking blocks
- 3 tool outputs
= 33 texts to embed -> 1-3 API calls (batched!)
```

Chunked texts produce additional vector records but embedding calls are still batched efficiently.

### Search Performance

**DuckDB**:
- **Single HNSW index**: O(log n) similarity search across all content types
- **HNSW limitation**: `WHERE user_id = ?` causes fallback to sequential scan. Acceptable because DuckDB files are single-user (one database per user), so the filter matches all rows.
- **Dedup overhead**: Python-side dict lookup, negligible

**SQLite**:
- **Numpy fallback**: O(n) brute-force cosine similarity
- **Acceptable for**: <10k vector records

**Cosmos DB**:
- **Single quantizedFlat index** on `/vector` path
- **O(n) optimized** with quantization
- **RU cost**: Per-query, not per-content-type (single index vs. 4 previously)

---

## Migration from Inline Vectors

### Auto-Migration (v0.2.0 -> v0.3.0)

The library automatically migrates databases from schema version 1 (inline vectors) to version 2 (externalized vectors) on initialization:

1. Detects schema version via `schema_meta` table
2. If version < 2 and inline vector columns exist on `transcripts`:
   - Re-extracts `source_text` from stored content via `extract_all_embeddable_content()`
   - Copies vectors to `transcript_vectors` with `chunk_index=0, total_chunks=1`
   - Drops old columns: `user_query_vector`, `assistant_response_vector`, `assistant_thinking_vector`, `tool_output_vector`, `embedding_model`, `vector_metadata`
   - Drops old HNSW indexes: `idx_user_query_vector`, `idx_assistant_response_vector`, `idx_assistant_thinking_vector`, `idx_tool_output_vector`
   - Sets schema version to 2

Migration is idempotent: if `transcript_vectors` already has data, or old columns do not exist, the migration skips and sets version to 2.

### Manual Migration (If Needed)

**DuckDB/SQLite**:
```bash
# Delete old database and re-ingest (cleanest approach)
rm sessions.db
python your_ingestion_script.py
```

**Cosmos DB**:
The library detects the old 4-path vector embedding policy and replaces it with the single `/vector` path policy on container initialization.

---

## Backend-Specific Implementation

### DuckDB

**Transcript Storage**: No vector columns  
**Vector Storage**: `transcript_vectors` table with native `FLOAT[3072]`  
**View**: `vectors_with_context` (JOIN of transcript_vectors + transcripts) provides vector rows enriched with parent message content  
**Search**: VSS extension, single HNSW index, `array_cosine_similarity()`  
**HNSW Persistence**: For disk-based databases, `SET hnsw_enable_experimental_persistence = true` is configured  
**Performance**: O(log n) via HNSW (with sequential scan fallback when WHERE filters are present)

### SQLite

**Transcript Storage**: No vector columns  
**Vector Storage**: `transcript_vectors` table with `vector_json TEXT` (JSON serialized)  
**Search**: Numpy cosine similarity (brute-force)  
**Performance**: O(n) - acceptable for <10k vector records

### Cosmos DB

**Transcript Storage**: `transcript_message` documents (no vector fields)  
**Vector Storage**: `transcript_vector` documents in the same container  
**Index**: Single `quantizedFlat` vector index on `/vector` path  
**Search**: `VectorDistance(c.vector, @queryVector)` with `TOP @fetchLimit`  
**Size Validation**: Transcript documents validated against 1.8MB safety limit (200KB margin below Cosmos 2MB hard limit). Without inline vectors, transcripts are much smaller, but the check exists as defense-in-depth.

---

## Implementation Details

### Efficient Batch Processing

**The `_embed_non_none()` helper**:
```python
async def _embed_non_none(self, texts: list[str | None]) -> list[list[float] | None]:
    """Generate embeddings only for non-None texts."""
    # Input: ["text1", None, "text2", None, "text3"]
    # Extracts: [(0, "text1"), (2, "text2"), (4, "text3")]
    # Calls API: ["text1", "text2", "text3"] -> single batch call
    # Returns: [vec1, None, vec2, None, vec3]
```

**Benefits**:
- Single API call for all non-None texts
- Preserves ordering with None placeholders
- Minimal API cost

### Content Extraction

**The `extract_all_embeddable_content()` function**:
```python
def extract_all_embeddable_content(message: dict) -> dict:
    """Extract all embeddable content from a message.

    Returns dict with keys:
    - user_query: str | None
    - assistant_response: str | None
    - assistant_thinking: str | None
    - tool_output: str | None
    """
```

**Handles complex cases**:
- Multiple thinking blocks -> joined with "\n\n"
- Multiple text blocks -> joined with "\n\n"
- Large tool outputs -> truncated to 10000 chars (`MAX_TOOL_OUTPUT_LENGTH`)
- Missing/null content -> returns None

### Vector Record ID Format

Each vector record has a deterministic ID:
```
{parent_id}_{content_type}_{chunk_index}
```

Where `parent_id` = `{session_id}_msg_{sequence}`.

Example: `sess_abc_msg_5_assistant_thinking_0` is the first chunk of the thinking content for message sequence 5 in session `sess_abc`.

---

## Configuration

### Environment Variables

**Embedding Provider** (required for vector search):
```bash
# Option 1: OpenAI Direct
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_EMBEDDING_DIMENSIONS=3072
OPENAI_EMBEDDING_CACHE_SIZE=1000

# Option 2: Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com/openai/deployments/text-embedding-3-large
AZURE_OPENAI_API_KEY=xxx
# Or use RBAC:
AZURE_OPENAI_USE_RBAC=true  # Requires az login
```

**DuckDB**:
```bash
AMPLIFIER_DUCKDB_PATH=~/.amplifier/sessions.db
AMPLIFIER_DUCKDB_VECTOR_DIMENSIONS=3072
```

**SQLite**:
```bash
AMPLIFIER_SQLITE_PATH=~/.amplifier/sessions_sqlite.db
AMPLIFIER_SQLITE_VECTOR_DIMENSIONS=3072
```

**Cosmos DB**:
```bash
AMPLIFIER_COSMOS_ENDPOINT=https://xxx.documents.azure.com:443/
AMPLIFIER_COSMOS_DATABASE=amplifier-sessions
AMPLIFIER_COSMOS_AUTH_METHOD=default_credential
AMPLIFIER_COSMOS_ENABLE_VECTOR=true
```

---

## Real-World Examples

### Example 1: Finding How AI Reasoned

"Find when the AI decided to refactor the authentication module"

```python
query = "refactor authentication module decision"
query_vec = await embeddings.embed_text(query)

# Search thinking blocks - this is where decisions are made
results = await storage.vector_search(
    user_id="user-123",
    query_vector=query_vec,
    vector_columns=["assistant_thinking"],
    top_k=5
)

# Results show messages where AI *thought* about refactoring auth
# Chunked thinking blocks are deduplicated - one result per parent message
```

### Example 2: Finding User Questions

```python
query = "authentication setup"
query_vec = await embeddings.embed_text(query)

results = await storage.vector_search(
    user_id="user-123",
    query_vector=query_vec,
    vector_columns=["user_query"],
    top_k=10
)
# Returns only user messages asking about authentication
```

### Example 3: Cross-Content Search

```python
query = "database schema design"
query_vec = await embeddings.embed_text(query)

results = await storage.vector_search(
    user_id="user-123",
    query_vector=query_vec,
    # Default: searches all 4 content types in transcript_vectors
    top_k=10
)
# Returns user queries, AI responses, AI reasoning, and tool outputs
# All deduplicated by parent transcript message
```

---

## Future Enhancements

### Potential Improvements

1. **Weighted multi-vector search**:
   - Weight thinking blocks higher for "why" questions
   - Weight responses higher for "what" questions

2. **Vector column auto-selection**:
   - Query classification: "how did you decide?" -> search thinking
   - "what did you say?" -> search responses

3. **Cross-session semantic clustering**:
   - Group similar thinking patterns across sessions
   - Find recurring debugging strategies

4. **Embedding model migration**:
   - Re-embed with newer models
   - Compare search quality across models
   - `embedding_model` field on each vector record supports mixed models

---

## Summary

**Externalized multi-vector architecture enables**:
- Precise semantic search by content type (user, assistant, thinking, tool)
- Chunking support for long texts exceeding 8192 tokens
- Single vector index per backend (reduced from 4)
- Full-text search over `source_text` alongside semantic search
- Clean separation of conversation data and vector data
- Schema versioning with automatic migration from inline vectors

**All 3 backends** (DuckDB, SQLite, Cosmos) implement identical APIs using the externalized vector pattern.
