# Hybrid Search Guide

This guide explains how to use hybrid search capabilities in amplifier-session-storage.

## Overview

Hybrid search combines three powerful techniques:

1. **Full-Text Search** - Keyword matching in transcript content and vector source_text
2. **Semantic Search** - Embedding-based similarity via externalized transcript_vectors
3. **MMR Re-Ranking** - Maximum Marginal Relevance for diversity

```
User Query
    |
    v
[Embed Query] ---------> query_vector (computed ONCE)
    |                          |
    v                          v
[Full-Text Search]     [Semantic Search]
  (transcripts.content    (transcript_vectors.vector
   + transcript_vectors    cosine similarity)
   .source_text)
    |                          |
    v                          v
       Merge & Deduplicate
              |
              v
       [MMR Re-Ranking] --> Diverse + Relevant Results
```

The query is embedded exactly once. The resulting vector is reused for semantic search and MMR re-ranking, avoiding redundant API calls.

---

## Quick Start

```python
from amplifier_session_storage import CosmosBackend, AzureOpenAIEmbeddings
from amplifier_session_storage.backends import TranscriptSearchOptions

# Initialize with embeddings
embeddings = AzureOpenAIEmbeddings.from_env()
async with CosmosBackend.create(embedding_provider=embeddings) as storage:

    # Hybrid search with MMR
    results = await storage.search_transcripts(
        user_id="user-123",
        options=TranscriptSearchOptions(
            query="How do I implement vector search in Cosmos DB?",
            search_type="hybrid",
            mmr_lambda=0.7,  # Favor relevance
            search_in_user=True,
            search_in_assistant=True,
            search_in_thinking=True
        ),
        limit=10
    )

    for result in results:
        print(f"Session: {result.session_id}")
        print(f"Content: {result.content[:100]}...")
        print(f"Score: {result.score:.3f}")
        print(f"Source: {result.source}")
        print("---")
```

---

## Search Types

### Full-Text Search

Best for: Exact keyword matching, known terminology

```python
options = TranscriptSearchOptions(
    query="CosmosDB VectorDistance",
    search_type="full_text"
)
```

**How it works:**
- Searches `transcripts.content` using SQL `CONTAINS` (Cosmos) or `LIKE` (DuckDB/SQLite)
- Also searches `transcript_vectors.source_text`, filtered by `content_type`
- Results from both sources are merged and deduplicated by `(session_id, sequence)`
- Fast and precise for known keywords
- Case-insensitive matching
- No embedding API costs

The `search_in_user`, `search_in_assistant`, and `search_in_thinking` flags control which `content_type` values are searched in `transcript_vectors.source_text`. This allows full-text keyword search to be scoped to specific content types, just like semantic search.

### Semantic Search

Best for: Conceptual similarity, paraphrased queries

```python
options = TranscriptSearchOptions(
    query="How do I set up cloud storage for sessions?",
    search_type="semantic"
)
```

**How it works:**
- Converts query to embedding vector (single API call)
- Searches `transcript_vectors` by cosine similarity
- Filters by `content_type` based on `search_in_*` flags
- Over-fetches 3x and deduplicates by `parent_id`
- Understands meaning, not just keywords
- Query "database" matches "storage", "persistence", etc.

### Hybrid Search (Recommended)

Best for: Maximum recall and relevance

```python
options = TranscriptSearchOptions(
    query="embedding generation at ingestion time",
    search_type="hybrid",
    mmr_lambda=0.7  # Tune relevance vs diversity
)
```

**How it works:**
1. Embeds query **once** into a vector
2. Gets candidates from full-text search (transcripts + source_text)
3. Gets candidates from semantic search (reuses pre-computed vector)
4. Merges and deduplicates results
5. Applies MMR re-ranking for relevance + diversity
6. Returns top-k results

---

## MMR Lambda Parameter

The `mmr_lambda` parameter controls the relevance-diversity trade-off:

### Lambda = 1.0 (Pure Relevance)
```python
options = TranscriptSearchOptions(
    query="vector search",
    search_type="hybrid",
    mmr_lambda=1.0  # Only most relevant results
)
```

**When to use:**
- You want the absolute best matches
- Redundancy is acceptable
- Precision over diversity

**Example**: Finding exact implementation details

### Lambda = 0.7 (Relevance-Focused, Default)
```python
options = TranscriptSearchOptions(
    query="session storage architecture",
    search_type="hybrid",
    mmr_lambda=0.7  # Mostly relevant, some variety
)
```

**When to use:**
- General purpose search (recommended default)
- Want top matches with some variety
- Balance precision and exploration

**Example**: Understanding a topic with related context

### Lambda = 0.5 (Balanced)
```python
options = TranscriptSearchOptions(
    query="debugging strategies",
    search_type="hybrid",
    mmr_lambda=0.5  # Equal relevance and diversity
)
```

**When to use:**
- Exploring a topic broadly
- Want to see different perspectives
- Research and discovery

**Example**: Learning about a new area

### Lambda = 0.0 (Pure Diversity)
```python
options = TranscriptSearchOptions(
    query="design patterns",
    search_type="hybrid",
    mmr_lambda=0.0  # Maximum variety
)
```

**When to use:**
- Broad exploration
- Avoid redundant information
- Coverage over precision

**Example**: Finding diverse examples of a pattern

### MMR Matched Vector Retrieval

During MMR re-ranking, each candidate needs its embedding vector to compute diversity. The library fetches the matched vector from `transcript_vectors` using the `content_type` from the search result:

```python
# When a search result needs its vector for MMR:
ct = result.metadata.get("content_type")
embedding = await self._get_embedding(
    user_id, result.session_id, result.sequence, ct
)
```

When `content_type` is available, the fetch is precise (`WHERE parent_id = ? AND content_type = ?`). When not available, the first vector for that parent is returned.

---

## Searching Different Content Types

The `search_in_*` flags control both semantic and full-text search. They map to `content_type` values in `transcript_vectors`:

| Flag | Content Type | Searched |
|------|-------------|----------|
| `search_in_user=True` | `user_query` | User messages |
| `search_in_assistant=True` | `assistant_response` | Assistant text responses |
| `search_in_thinking=True` | `assistant_thinking` | Assistant reasoning blocks |

Tool output (`tool_output`) is included by default when any flag is true.

### Search User Messages Only

```python
options = TranscriptSearchOptions(
    query="how do I...",
    search_in_user=True,
    search_in_assistant=False,
    search_in_thinking=False
)
```

### Search Assistant Responses Only

```python
options = TranscriptSearchOptions(
    query="implementation details",
    search_in_user=False,
    search_in_assistant=True,
    search_in_thinking=False
)
```

### Search Thinking Blocks

```python
options = TranscriptSearchOptions(
    query="reasoning about architecture",
    search_in_user=False,
    search_in_assistant=False,
    search_in_thinking=True
)
```

### Search Everything (Default)

```python
options = TranscriptSearchOptions(
    query="cosmos db configuration",
    search_in_user=True,
    search_in_assistant=True,
    search_in_thinking=True
)
```

---

## Filtering Search Results

### By Project

```python
from amplifier_session_storage.backends import SearchFilters

options = TranscriptSearchOptions(
    query="authentication",
    filters=SearchFilters(
        project_slug="amplifier-core"
    )
)
```

### By Date Range

```python
options = TranscriptSearchOptions(
    query="bug fix",
    filters=SearchFilters(
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-02-01T00:00:00Z"
    )
)
```

### Combined Filters

```python
options = TranscriptSearchOptions(
    query="implementation",
    filters=SearchFilters(
        project_slug="amplifier",
        start_date="2024-01-01T00:00:00Z",
        bundle="foundation"
    )
)
```

---

## Event Search

### Search by Event Type

```python
from amplifier_session_storage.backends import EventSearchOptions

# Find all LLM requests
results = await storage.search_events(
    user_id="user-123",
    options=EventSearchOptions(
        event_type="llm.request"
    ),
    limit=100
)
```

### Search by Tool Name

```python
# Find all bash tool executions
results = await storage.search_events(
    user_id="user-123",
    options=EventSearchOptions(
        tool_name="bash"
    ),
    limit=50
)
```

### Search by Level

```python
# Find all errors
results = await storage.search_events(
    user_id="user-123",
    options=EventSearchOptions(
        level="error"
    ),
    limit=100
)
```

### Combined Event Search

```python
# Find all tool errors in a specific project
results = await storage.search_events(
    user_id="user-123",
    options=EventSearchOptions(
        event_type="tool.error",
        level="error",
        filters=SearchFilters(
            project_slug="my-project",
            start_date="2024-01-01T00:00:00Z"
        )
    )
)
```

---

## Analytics & Statistics

### Session Statistics

```python
from amplifier_session_storage.backends import SearchFilters

# Overall statistics
stats = await storage.get_session_statistics(
    user_id="user-123"
)

print(f"Total sessions: {stats['total_sessions']}")
print(f"Sessions by project: {stats['sessions_by_project']}")
print(f"Sessions by bundle: {stats['sessions_by_bundle']}")

# Filtered statistics
stats = await storage.get_session_statistics(
    user_id="user-123",
    filters=SearchFilters(
        start_date="2024-01-01T00:00:00Z",
        project_slug="amplifier"
    )
)
```

---

## Performance Optimization

### Single Query Embedding

Hybrid search embeds the query exactly once. The pre-computed vector is passed to `_semantic_search_with_vector()`, avoiding the double-embed that would occur if full-text and semantic paths each embedded independently:

```python
# Internal implementation:
async def _hybrid_search_transcripts(self, user_id, options, limit):
    # Compute query embedding ONCE
    query_vector = await self.embedding_provider.embed_text(options.query)
    query_np = np.array(query_vector)

    # Full-text results (no embedding needed)
    text_results = await self._full_text_search_transcripts(...)

    # Semantic results REUSE pre-computed vector
    semantic_results = await self._semantic_search_with_vector(
        user_id, options, candidate_limit, query_vector  # passed in
    )

    # Merge, deduplicate, MMR re-rank with query_np
```

### Embedding Cache

The embedding cache reduces API calls:

```python
# Configure cache size
embeddings = AzureOpenAIEmbeddings(
    endpoint=...,
    api_key=...,
    cache_size=2000  # Increase for more caching
)

# Monitor cache
stats = embeddings.get_cache_stats()
print(f"Cached embeddings: {stats['size']}/{stats['max_entries']}")
print(f"Utilization: {stats['utilization']:.1%}")
```

**Recommendations:**
- Default 1000 is good for most use cases
- Increase to 2000-5000 for heavy search workloads
- Decrease to 100-500 for memory-constrained environments

### Batch Operations

Always use batch operations for multiple embeddings:

```python
# Bad: Individual calls
embeddings_list = []
for text in texts:
    emb = await embeddings.embed_text(text)  # N API calls
    embeddings_list.append(emb)

# Good: Batch call
embeddings_list = await embeddings.embed_batch(texts)  # 1 API call
```

### Index Optimization

Ensure Cosmos DB containers have the single-path vector index:

```bash
# Check container indexing policy in Azure Portal
# Look for "vectorIndexes" section with:
# - path: "/vector"
# - type: "quantizedFlat" (or "diskANN" for >100k records)
# - dimensions: 3072
```

---

## Cost Considerations

### Embedding API Costs

Azure OpenAI charges per token for embedding generation:

| Model | Dimensions | Cost per 1M tokens |
|-------|------------|-------------------|
| text-embedding-3-small | 1536 | ~$0.02 |
| text-embedding-3-large | 3072 | ~$0.13 |

**Cost reduction strategies:**

1. **Use cache** - Default 1000-entry cache eliminates redundant calls
2. **Batch calls** - Use `embed_batch()` for efficiency
3. **Smaller model** - Consider text-embedding-3-small for dev/test
4. **Selective embedding** - Only embed user/assistant content, skip system messages

### Storage Costs

Vector embeddings increase storage size, but externalization keeps transcript queries fast:

| Component | Size per Vector Record |
|-----------|----------------------|
| Vector (3072-d, float32) | ~12 KB |
| Source text + metadata | ~500-4000 bytes |

**Cosmos DB optimization:**
- Single quantizedFlat index (128 bytes compressed per vector)
- Consider pruning old sessions
- Use DuckDB for local development (no cloud costs)

---

## Advanced Patterns

### Progressive Enhancement

Start simple, add features as needed:

```python
# Level 1: Basic sync (no embeddings)
storage = await CosmosBackend.create()  # No embedding provider
await storage.sync_transcript_lines(...)  # Works without embeddings

# Level 2: Add embeddings later
embeddings = AzureOpenAIEmbeddings.from_env()
storage = await CosmosBackend.create(embedding_provider=embeddings)
# Now ingestion generates embeddings automatically in transcript_vectors

# Level 3: Backfill old data
await storage.upsert_embeddings(...)  # Add embeddings to existing data
```

### Multi-Backend Strategy

```python
# Development: DuckDB (fast, local)
dev_storage = await DuckDBBackend.create(
    embedding_provider=embeddings
)

# Production: Cosmos DB (cloud, multi-device)
prod_storage = await CosmosBackend.create(
    embedding_provider=embeddings
)

# Testing: SQLite (lightweight, embedded)
test_storage = await SQLiteBackend.create()
```

---

## FAQ

**Q: Can I use the library without embeddings?**  
A: Yes. Full-text search works without an embedding provider. Semantic/hybrid search will gracefully degrade to full-text.

**Q: How do I know if vector search is working?**  
A: Call `await storage.supports_vector_search()` - returns `True` if both embedding provider and vector indexes are available.

**Q: What if my Cosmos DB account does not support vector search?**  
A: You can still use full-text search, or use DuckDB/SQLite backends which have built-in vector support.

**Q: Should I use DuckDB or SQLite for local development?**  
A: DuckDB is recommended - better performance, native vector support, analytical capabilities.

**Q: How do I monitor embedding API costs?**  
A: Check cache stats with `embeddings.get_cache_stats()` - high cache utilization = lower API costs.

**Q: Can I switch backends without data loss?**  
A: The storage abstraction uses the same interface, but each backend stores data independently. You will need to re-ingest data when switching.

**Q: How does chunking affect search results?**  
A: Chunked texts produce multiple vector records, but search results are deduplicated by parent transcript message. You always get one result per message, with the score from the best-matching chunk.
