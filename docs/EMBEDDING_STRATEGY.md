# Embedding Strategy and Field Mapping

> **Version**: 1.0.0  
> **Last Updated**: 2025-02-05  
> **Status**: Implemented and Tested

This document describes what content gets embedded and how for semantic search capabilities.

---

## Overview

The library uses **multi-vector embeddings** - 4 separate vectors per transcript message:

| Vector Column | Source | Use Case |
|---------------|--------|----------|
| `user_query_vector` | User message content | "When did I ask about X?" |
| `assistant_response_vector` | Assistant text blocks | "What did the AI tell me about X?" |
| `assistant_thinking_vector` | Assistant thinking blocks | "How did the AI reason about X?" |
| `tool_output_vector` | Tool result content | "What files/outputs contained X?" |

This enables **targeted semantic search** - search only in reasoning, only in responses, or across all content types.

---

## Amplifier Session Data Structure

### Directory Structure (On Disk)

```
~/.amplifier/projects/<project-slug>/sessions/<session-id>/
├── metadata.json       # Session metadata (not embedded)
├── transcript.jsonl    # Conversation messages (embedded)
└── events.jsonl        # System events (not embedded)
```

### What Gets Embedded

| Source | Embedded | Notes |
|--------|----------|-------|
| `transcript.jsonl` | ✅ Yes | Multi-vector per message |
| `metadata.json` | ❌ No | Structured metadata, use filters |
| `events.jsonl` | ❌ No | Telemetry data, use structured search |

---

## Content Extraction

Content extraction is handled by `amplifier_session_storage/content_extraction.py`.

### User Messages

**Structure**:
```json
{"role": "user", "content": "How do I use vector search?"}
```

**Extraction**: Direct string → `user_query_vector`

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
- All `thinking` blocks → joined → `assistant_thinking_vector`
- All `text` blocks → joined → `assistant_response_vector`
- Tool calls → **NOT embedded** (metadata, not semantic content)

### Tool Messages

**Structure**:
```json
{"role": "tool", "content": "{file contents or tool result}"}
```

**Extraction**: Content (truncated to 1000 chars) → `tool_output_vector`

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
    # embeddings parameter optional - auto-generated if not provided
)
# Output: Generated embeddings: 5 user, 5 responses, 3 thinking, 2 tool
```

### Batch Efficiency

The implementation batches all non-None texts into a single API call:

```python
# 10 messages with mixed content:
# - 10 user queries
# - 10 assistant responses
# - 8 assistant thinking blocks
# - 3 tool outputs
# = 31 texts → 1-2 API calls (batched!)
```

### Vector Metadata

Each message stores metadata about which vectors exist:

```json
{
  "has_user_query": true,
  "has_assistant_response": true,
  "has_assistant_thinking": true,
  "has_tool_output": false
}
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

---

## Backend Implementation

### DuckDB

- **Storage**: Native `FLOAT[3072]` arrays
- **Indexes**: 4 HNSW indexes for O(log n) search
- **Search**: `array_cosine_similarity()` with `GREATEST()` for multi-vector

```sql
-- 4 vector columns with HNSW indexes
CREATE INDEX idx_user_query_vector ON transcripts USING HNSW (user_query_vector);
CREATE INDEX idx_assistant_response_vector ON transcripts USING HNSW (assistant_response_vector);
CREATE INDEX idx_assistant_thinking_vector ON transcripts USING HNSW (assistant_thinking_vector);
CREATE INDEX idx_tool_output_vector ON transcripts USING HNSW (tool_output_vector);
```

### SQLite

- **Storage**: JSON TEXT serialization
- **Indexes**: None (brute-force numpy)
- **Search**: Cosine similarity via numpy, max across all vectors

### Cosmos DB

- **Storage**: JSON arrays in document fields
- **Indexes**: 4 quantizedFlat vector indexes
- **Search**: `VectorDistance()` with `LEAST()` for multi-vector

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

| Backend | Index Type | Complexity | 10k Messages |
|---------|------------|------------|--------------|
| DuckDB | HNSW | O(log n) | ~15 comparisons/column |
| SQLite | Numpy | O(n) | ~40k vector comparisons |
| Cosmos | quantizedFlat | O(n) optimized | ~50-100ms |

### Storage Overhead

| Component | Size per Message |
|-----------|------------------|
| Text content | ~500 bytes avg |
| 4 vectors (3072-d each) | ~48 KB (float32) |
| Vector metadata | ~100 bytes |

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

## Testing

### Test Coverage

| Area | Tests | Status |
|------|-------|--------|
| Content extraction | 18 tests | ✅ Passing |
| Multi-vector generation | 12 tests | ✅ Passing |
| Vector search | 15 tests | ✅ Passing |
| Hybrid search | 8 tests | ✅ Passing |

### Verified Against Real Data

Tested against real Amplifier sessions from `~/.amplifier/projects/`:
- ✅ User messages extracted correctly
- ✅ Thinking blocks extracted (found 33+ in real sessions)
- ✅ Response blocks extracted
- ✅ Tool outputs handled (found 38+ in real sessions)
- ✅ Complex multi-block messages parsed correctly

---

## Summary

| What | Where | Embedded As |
|------|-------|-------------|
| User questions | `role=user`, `content` | `user_query_vector` |
| AI responses | `role=assistant`, `content[].text` | `assistant_response_vector` |
| AI reasoning | `role=assistant`, `content[].thinking` | `assistant_thinking_vector` |
| Tool outputs | `role=tool`, `content` | `tool_output_vector` |
| Session metadata | `metadata.json` | Not embedded (use filters) |
| Events | `events.jsonl` | Not embedded (use structured search) |

**Multi-vector search enables precise semantic queries** - find not just "messages about X" but specifically "user questions about X" or "AI reasoning about X" or "tool outputs containing X".
