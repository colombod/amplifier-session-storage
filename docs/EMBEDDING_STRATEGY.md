# Embedding Strategy and Field Mapping

> **Version**: 2.0.0  
> **Last Updated**: 2025-02-06  
> **Status**: Implemented and Tested

This document describes what content gets embedded, how it is chunked and stored, and how the externalized vector architecture enables semantic search.

---

## Overview

The library uses **multi-vector embeddings** with an **externalized vector table**. Vectors are not stored inline on transcript rows. Instead, a dedicated `transcript_vectors` table (DuckDB/SQLite) or `transcript_vector` document type (Cosmos DB) holds all vector data separately from conversation content.

Four content types are extracted and embedded independently:

| Content Type | Source | Use Case |
|-------------|--------|----------|
| `user_query` | User message content | "When did I ask about X?" |
| `assistant_response` | Assistant text blocks | "What did the AI tell me about X?" |
| `assistant_thinking` | Assistant thinking blocks | "How did the AI reason about X?" |
| `tool_output` | Tool result content | "What files/outputs contained X?" |

Each content type is stored as one or more vector records in `transcript_vectors`, with a `parent_id` linking back to the source transcript row. Long texts are chunked, producing multiple vector records per content type.

---

## Amplifier Session Data Structure

### Directory Structure (On Disk)

```
~/.amplifier/projects/<project-slug>/sessions/<session-id>/
+-- metadata.json       # Session metadata (not embedded)
+-- transcript.jsonl    # Conversation messages (embedded)
+-- events.jsonl        # System events (not embedded)
```

### What Gets Embedded

| Source | Embedded | Notes |
|--------|----------|-------|
| `transcript.jsonl` | Yes | Multi-vector per message, stored in transcript_vectors |
| `metadata.json` | No | Structured metadata, use filters |
| `events.jsonl` | No | Telemetry data, use structured search |

---

## Content Extraction

Content extraction is handled by `amplifier_session_storage/content_extraction.py`.

### Token Counting

All token operations use tiktoken with the `cl100k_base` encoding (compatible with text-embedding-3-large and GPT-4 family models). A lazy singleton encoder avoids repeated initialization cost:

```python
import tiktoken

EMBED_TOKEN_LIMIT = 8192
_TIKTOKEN_ENCODING = "cl100k_base"

def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    encoder = _get_encoder()
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])
```

### User Messages

**Structure**:
```json
{"role": "user", "content": "How do I use vector search?"}
```

**Extraction**: Direct string -> `user_query` content type

### Assistant Messages

**Structure** (complex content array):
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
- Tool calls -> **NOT embedded** (metadata, not semantic content)

### Tool Messages

**Structure**:
```json
{"role": "tool", "content": "{file contents or tool result}"}
```

**Extraction**: Content (truncated to 10000 chars via `MAX_TOOL_OUTPUT_LENGTH`) -> `tool_output` content type

### Extraction Function

```python
from amplifier_session_storage.content_extraction import extract_all_embeddable_content

result = extract_all_embeddable_content(message)
# Returns:
# {
#     "user_query": "...",           # or None
#     "assistant_response": "...",   # or None
#     "assistant_thinking": "...",   # or None
#     "tool_output": "..."           # or None
# }
```

---

## Semantic Chunking

Texts exceeding 8192 tokens (`EMBED_TOKEN_LIMIT`) are split into smaller chunks for embedding. This is handled by `amplifier_session_storage/chunking.py`.

### Chunking Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `CHUNK_TARGET_TOKENS` | 1024 | Target tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | 128 | Overlap between adjacent chunks |
| `CHUNK_MIN_TOKENS` | 64 | Minimum chunk size (runts appended to previous) |
| `EMBED_TOKEN_LIMIT` | 8192 | Threshold above which chunking activates |

### Content-Aware Splitting

Different content types use different splitting strategies:

| Content Type | Strategy | Logic |
|-------------|----------|-------|
| `assistant_thinking` | Markdown-aware | Preserves code blocks as atomic segments, splits on paragraph boundaries (`\n\n`), then sentence-splits oversized paragraphs |
| `assistant_response` | Markdown-aware | Same as thinking |
| `tool_output` | Line-aware | Splits on newlines |
| `user_query` | Sentence-aware | Splits on `.!?` + whitespace |

### Chunk Data Structure

```python
@dataclass
class ChunkResult:
    text: str           # The chunk text
    span_start: int     # Character offset start in original text
    span_end: int       # Character offset end in original text
    chunk_index: int    # 0-based position
    total_chunks: int   # Total chunks produced
    token_count: int    # Tokens in this chunk
```

### Short Text Handling

Texts at or below the 8192-token limit produce a single `ChunkResult` spanning the full text. No splitting occurs.

### Chunk Merging

After splitting, small segments are merged up to the target token count. Each chunk except the first includes a 128-token overlap tail from the previous chunk. Trailing chunks smaller than 64 tokens are appended to the previous chunk rather than creating a runt.

---

## Embedding Generation

### Automatic During Ingestion

```python
# Embeddings generated automatically when provider configured
storage = await DuckDBBackend.create(embedding_provider=embeddings)

await storage.sync_transcript_lines(
    user_id="user-123",
    host_id="laptop-01",
    project_slug="my-project",
    session_id="sess_abc",
    lines=transcript_lines
)
# Output: Generated embeddings: 5 user, 5 responses, 3 thinking, 2 tool
```

### Batch Efficiency

The implementation batches all non-None texts into a single API call per content type:

```python
# 10 messages with mixed content:
# - 10 user queries
# - 10 assistant responses
# - 8 assistant thinking blocks
# - 3 tool outputs
# = 31 texts -> 1-2 API calls (batched!)
```

Chunked texts (e.g., a 16k-token thinking block split into 2 chunks) produce additional embedding calls, but these are still batched efficiently.

### Externalized Storage

Each embedding is stored as a separate record in `transcript_vectors`:

```python
# Vector record ID format:
# {session_id}_msg_{sequence}_{content_type}_{chunk_index}

# For a chunked assistant_thinking block (3 chunks):
# sess_abc_msg_5_assistant_thinking_0
# sess_abc_msg_5_assistant_thinking_1
# sess_abc_msg_5_assistant_thinking_2
```

---

## Search Capabilities

### Targeted Vector Search

```python
# Search ONLY in user queries
results = await storage.vector_search(
    user_id="user-123",
    query_vector=query_vec,
    vector_columns=["user_query"],
    top_k=10
)

# Search ONLY in assistant thinking (reasoning)
results = await storage.vector_search(
    user_id="user-123",
    query_vector=query_vec,
    vector_columns=["assistant_thinking"],
    top_k=10
)

# Search ALL content types (default)
results = await storage.vector_search(
    user_id="user-123",
    query_vector=query_vec,
    # vector_columns not specified = search all 4
    top_k=10
)
```

### Search Type Options

```python
from amplifier_session_storage.backends import TranscriptSearchOptions

options = TranscriptSearchOptions(
    query="vector database design",
    search_type="hybrid",          # full_text, semantic, hybrid
    search_in_user=True,           # Include user messages
    search_in_assistant=True,      # Include assistant responses
    search_in_thinking=True,       # Include thinking blocks
    mmr_lambda=0.7                 # Relevance vs diversity
)
```

The `search_in_*` flags control both semantic and full-text search. For semantic search, they filter by `content_type` in `transcript_vectors`. For full-text search, they also search `source_text` in `transcript_vectors` filtered by the same `content_type` values.

---

## Backend Implementation

### DuckDB

- **Transcript Storage**: `transcripts` table contains no vector columns
- **Vector Storage**: `transcript_vectors` table with native `FLOAT[3072]` vector column
- **Index**: Single HNSW index (`idx_vectors_hnsw`) on the `vector` column
- **Search**: `array_cosine_similarity()` on `vectors_with_context` view (JOIN of transcript_vectors + transcripts), over-fetch 3x, Python-side dedup by `parent_id`

```sql
-- Single HNSW index on the externalized vector column
CREATE INDEX idx_vectors_hnsw
ON transcript_vectors USING HNSW (vector)
WITH (metric = 'cosine');
```

**HNSW limitation**: DuckDB's HNSW index does not support pre-filtered queries. A `WHERE user_id = ?` clause causes fallback to sequential scan. This is acceptable because DuckDB databases are single-user (one `.db` file per user), so the filter is effectively a no-op against the full dataset.

### SQLite

- **Vector Storage**: `transcript_vectors` table with `vector_json TEXT` (JSON serialized)
- **Indexes**: None (brute-force numpy cosine similarity)
- **Search**: Loads vectors, computes cosine similarity via numpy, dedup by `parent_id`

### Cosmos DB

- **Vector Storage**: `transcript_vector` documents in the `transcript_messages` container
- **Index**: Single `quantizedFlat` vector index on `/vector` path
- **Search**: `VectorDistance(c.vector, @queryVector)`, Python-side dedup by `parent_id`

---

## Use Cases

### 1. Find User Questions

**Query**: "When did I ask about authentication?"

```python
results = await storage.vector_search(
    query_vector=embed("authentication setup"),
    vector_columns=["user_query"]
)
# Returns only user messages about authentication
```

### 2. Find AI Reasoning

**Query**: "How did the AI decide on the architecture?"

```python
results = await storage.vector_search(
    query_vector=embed("architecture decision"),
    vector_columns=["assistant_thinking"]
)
# Returns thinking blocks with architectural reasoning
```

### 3. Find AI Responses

**Query**: "What did the AI tell me about testing?"

```python
results = await storage.vector_search(
    query_vector=embed("testing strategies"),
    vector_columns=["assistant_response"]
)
# Returns user-visible responses about testing
```

### 4. Find Tool Outputs

**Query**: "Which files contained database schemas?"

```python
results = await storage.vector_search(
    query_vector=embed("database schema"),
    vector_columns=["tool_output"]
)
# Returns tool messages with schema-related content
```

### 5. Cross-Content Search

**Query**: "Everything about vector search"

```python
results = await storage.vector_search(
    query_vector=embed("vector search implementation")
    # No vector_columns = search all 4 types
)
# Returns user questions, AI responses, AI reasoning, and tool outputs
```

---

## Performance Characteristics

### Embedding Generation

| Scenario | API Calls | Notes |
|----------|-----------|-------|
| 10 messages, no cache | 1-2 | Batched efficiently |
| 10 messages, warm cache | 0-1 | Cache hits reduce calls |
| Re-sync existing | 0 | Embeddings already stored |

### Search Performance

| Backend | Index Type | Complexity | Notes |
|---------|------------|------------|-------|
| DuckDB | HNSW (single index) | O(log n) | Sequential scan with WHERE filter (acceptable, single-user DB) |
| SQLite | Numpy | O(n) | Brute-force over all vector records |
| Cosmos | quantizedFlat (single index) | O(n) optimized | ~50-100ms typical |

### Storage Overhead

| Component | Size |
|-----------|------|
| 1 vector (3072-d, float32) | ~12 KB |
| Metadata per vector record | ~200 bytes |
| Source text per record | Variable |

Typical per-message overhead:
- User message (1 content type, 1 chunk): ~12.5 KB in vector records
- Assistant with thinking (2 content types, 1 chunk each): ~25 KB in vector records
- Long thinking block (16k tokens, 2 chunks): ~37 KB in vector records

---

## Configuration

### Environment Variables

```bash
# Embedding Provider (Azure OpenAI)
AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com/openai/deployments/text-embedding-3-large
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-large
AZURE_OPENAI_EMBEDDING_DIMENSIONS=3072
AZURE_OPENAI_USE_RBAC=true  # Or use AZURE_OPENAI_API_KEY

# Cache
OPENAI_EMBEDDING_CACHE_SIZE=1000  # Default 1000 entries
```

### Embedding Model Recommendations

| Model | Dimensions | Cost | Quality |
|-------|------------|------|---------|
| `text-embedding-3-large` | 3072 | Higher | Best |
| `text-embedding-3-small` | 1536 | Lower | Good |
| `text-embedding-ada-002` | 1536 | Legacy | Acceptable |

---

## Summary

| What | Where | Content Type |
|------|-------|-------------|
| User questions | `role=user`, `content` | `user_query` |
| AI responses | `role=assistant`, `content[].text` | `assistant_response` |
| AI reasoning | `role=assistant`, `content[].thinking` | `assistant_thinking` |
| Tool outputs | `role=tool`, `content` | `tool_output` |
| Session metadata | `metadata.json` | Not embedded (use filters) |
| Events | `events.jsonl` | Not embedded (use structured search) |

All vectors stored externally in `transcript_vectors` table / `transcript_vector` documents, with chunking support for long texts and a single HNSW/quantizedFlat index per backend.

---

## Resilience (v0.3.0)

All embedding API calls are protected by three layers of resilience, implemented in `amplifier_session_storage.embeddings.resilience` and applied by `EmbeddingMixin`:

### Batch Splitting

Large embedding requests are split into groups of 16 texts (`EMBED_BATCH_SIZE`). Each sub-batch is independent -- a failed batch returns `None` for those positions without affecting other batches.

### Retry with Exponential Backoff

Transient failures are retried automatically:

| Setting | Value |
|---------|-------|
| Max retries | 5 |
| Backoff base | 1.0s |
| Backoff cap | 60.0s |
| Backoff multiplier | 2.0x |
| Retryable status codes | 429, 500, 502, 503, 504 |
| Retryable exceptions | ConnectionError, TimeoutError, OSError |

Respects `Retry-After` headers from the API response.

### Circuit Breaker

After 5 consecutive retryable failures, the circuit breaker opens and all embedding calls fail fast for 60 seconds. After the timeout, a single probe request is allowed (half-open state). If it succeeds, the circuit closes. If it fails, the circuit re-opens.

Only retryable failures (429, 5xx, network errors) trip the breaker. Non-retryable errors (authentication, configuration) propagate immediately without affecting the breaker state.

The circuit breaker is shared process-wide across all backends.

### Graceful Degradation

When embedding generation fails during `sync_transcript_lines`, transcript documents are always stored. The error is logged at ERROR level with:
- `EMBEDDING_FAILURE` prefix for grep/alerting
- Full identity chain: user, project, session, message count
- Clear explanation of what succeeded vs. what failed
- Full stack trace via `exc_info=True`

---

## Embedding Lifecycle Management (v0.3.0)

### backfill_embeddings()

Generates vectors for transcripts where `has_vectors` is `false`. Finds missing transcripts with a single query (`WHERE has_vectors = false`), then processes them in batches through the extract-chunk-embed pipeline. Safe to call repeatedly.

### rebuild_vectors()

Deletes ALL vectors for a session, resets `has_vectors` to `false`, then regenerates from scratch. Use for model upgrades, dimension changes, or corruption recovery.

Both methods return `EmbeddingOperationResult` with `transcripts_found`, `vectors_stored`, `vectors_failed`, and `errors` (capped at 50).
