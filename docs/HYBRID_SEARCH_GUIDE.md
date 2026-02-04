# Hybrid Search Guide

This guide explains how to use hybrid search capabilities in amplifier-session-storage.

## Overview

Hybrid search combines three powerful techniques:

1. **Full-Text Search** - Keyword matching in transcript content
2. **Semantic Search** - Embedding-based similarity (understands meaning)
3. **MMR Re-Ranking** - Maximum Marginal Relevance for diversity

```
User Query
    ↓
[Full-Text Search] → Candidates
    +
[Semantic Search]  → Candidates
    ↓
Merge & Deduplicate
    ↓
[MMR Re-Ranking] → Diverse + Relevant Results
```

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
- Uses SQL `CONTAINS` (Cosmos) or `LIKE` (DuckDB/SQLite)
- Fast and precise for known keywords
- Case-insensitive matching
- No embedding API costs

### Semantic Search

Best for: Conceptual similarity, paraphrased queries

```python
options = TranscriptSearchOptions(
    query="How do I set up cloud storage for sessions?",
    search_type="semantic"
)
```

**How it works:**
- Converts query to embedding vector
- Finds similar embeddings in database
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
1. Gets candidates from both full-text AND semantic search
2. Merges and deduplicates results
3. Applies MMR re-ranking for relevance + diversity
4. Returns top-k results

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

---

## Searching Different Content Types

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
# ❌ Bad: Individual calls
embeddings_list = []
for text in texts:
    emb = await embeddings.embed_text(text)  # N API calls
    embeddings_list.append(emb)

# ✅ Good: Batch call
embeddings_list = await embeddings.embed_batch(texts)  # 1 API call
```

### Index Optimization

Ensure Cosmos DB containers have proper indexes:

```bash
# Check container indexing policy in Azure Portal
# Look for "vectorIndexes" section with:
# - path: "/embedding"
# - type: "quantizedFlat" (or "diskANN" for >100k docs)
# - dimensions: 3072
```

---

## Cost Considerations

### Embedding API Costs

Azure OpenAI charges per token for embedding generation:

| Model | Dimensions | Cost per 1M tokens |
|-------|------------|--------------------|
| text-embedding-3-small | 1536 | ~$0.02 |
| text-embedding-3-large | 3072 | ~$0.13 |

**Cost reduction strategies:**

1. **Use cache** - Default 1000-entry cache eliminates redundant calls
2. **Batch calls** - Use `embed_batch()` for efficiency
3. **Smaller model** - Consider text-embedding-3-small for dev/test
4. **Selective embedding** - Only embed user/assistant content, skip system messages

### Storage Costs

Vector embeddings increase storage size:

| Component | Size per Message |
|-----------|------------------|
| Text content | ~500 bytes avg |
| Embedding (3072-d) | ~12 KB (float32) |

**Cosmos DB optimization:**
- Use `quantizedFlat` index (128 bytes compressed)
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
# Now ingestion generates embeddings automatically

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

## Migration Checklist

- [ ] Backup existing data (if needed)
- [ ] Configure Azure OpenAI environment variables
- [ ] Enable Cosmos DB vector search feature
- [ ] Upgrade library to v0.2.0
- [ ] Test with small dataset first
- [ ] Verify vector search works (`supports_vector_search()`)
- [ ] Monitor embedding cache hit rate
- [ ] Tune MMR lambda for your use case
- [ ] Clear old containers or backfill embeddings
- [ ] Update ingestion pipeline to generate embeddings
- [ ] Test search quality with real queries
- [ ] Monitor API costs and adjust cache size

---

## FAQ

**Q: Can I use v0.2.0 without embeddings?**  
A: Yes! Full-text search works without an embedding provider. Semantic/hybrid search will gracefully degrade to full-text.

**Q: How do I know if vector search is working?**  
A: Call `await storage.supports_vector_search()` - returns `True` if both embedding provider and vector indexes are available.

**Q: What if my Cosmos DB account doesn't support vector search?**  
A: You can still use full-text search, or use DuckDB/SQLite backends which have built-in vector support.

**Q: Should I use DuckDB or SQLite for local development?**  
A: DuckDB is recommended - better performance, native vector support, analytical capabilities.

**Q: How do I monitor embedding API costs?**  
A: Check cache stats with `embeddings.get_cache_stats()` - high cache utilization = lower API costs.

**Q: Can I switch backends without data loss?**  
A: The storage abstraction uses the same interface, but each backend stores data independently. You'll need to re-ingest data when switching.
