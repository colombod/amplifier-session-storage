# Upgrade Guide - v0.1.0 to v0.2.0

This guide explains how to upgrade from the basic file-sync storage (v0.1.0) to the enhanced hybrid search storage (v0.2.0).

## Breaking Changes

⚠️ **IMPORTANT**: Version 0.2.0 adds embedding support to the schema. Since we're in early development, the recommended approach is to **clear and rebuild** your Cosmos DB data.

### What's New in v0.2.0

1. **Storage Backend Abstraction** - Multiple backends (Cosmos, DuckDB, SQLite)
2. **Hybrid Search** - Full-text + semantic + MMR re-ranking
3. **Embedding Support** - Automatic embedding generation during ingestion
4. **Multiple Backends** - Choose Cosmos for cloud, DuckDB/SQLite for local
5. **Advanced Search** - Search by event type, tool name, date range, content
6. **Analytics** - Cross-session statistics and aggregations

### Schema Changes

#### Transcripts Container
New fields added:
- `embedding` - FLOAT[3072] vector for semantic search
- `embedding_model` - Model identifier (e.g., "text-embedding-3-large")

#### Events Container
No schema changes (backward compatible)

#### Sessions Container
No schema changes (backward compatible)

---

## Migration Options

### Option 1: Clean Start (Recommended for Development)

Since you're in early development, the cleanest approach is:

```bash
# 1. Backup existing data (optional)
# Export sessions using old version if needed

# 2. Delete Cosmos DB containers
# Via Azure Portal: Delete "sessions", "transcripts", "events" containers

# 3. Upgrade library
uv pip install --upgrade git+https://github.com/colombod/amplifier-session-storage@feat/enhanced-storage-with-vector-search

# 4. Configure embeddings
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_EMBEDDING_MODEL="text-embedding-3-large"

# 5. Run ingestion with new version (embeddings generated automatically)
# Your ingestion pipeline will create new containers with vector indexes
```

### Option 2: Backfill Embeddings (For Production Data)

If you have data you need to preserve:

```python
from amplifier_session_storage import CosmosBackend, AzureOpenAIEmbeddings
from amplifier_session_storage.backends import SearchFilters

# Initialize
embeddings = AzureOpenAIEmbeddings.from_env()
async with CosmosBackend.create(embedding_provider=embeddings) as storage:
    
    # Get all sessions
    sessions = await storage.search_sessions(
        user_id="your-user-id",
        filters=SearchFilters(),
        limit=1000
    )
    
    # For each session, backfill embeddings
    for session in sessions:
        session_id = session["session_id"]
        project_slug = session["project_slug"]
        
        # Get transcript lines
        transcripts = await storage.get_transcript_lines(
            user_id="your-user-id",
            project_slug=project_slug,
            session_id=session_id
        )
        
        # Generate embeddings
        texts = [t.get("content", "") for t in transcripts]
        vectors = await embeddings.embed_batch(texts)
        
        # Upsert embeddings
        embedding_docs = [
            {
                "sequence": i,
                "text": texts[i],
                "vector": vectors[i],
                "metadata": {"model": embeddings.model_name}
            }
            for i in range(len(texts))
        ]
        
        await storage.upsert_embeddings(
            user_id="your-user-id",
            project_slug=project_slug,
            session_id=session_id,
            embeddings=embedding_docs
        )
        
        print(f"Backfilled {len(embedding_docs)} embeddings for {session_id}")
```

---

## Cosmos DB Configuration Updates

### Enable Vector Search

Vector search in Cosmos DB requires:

1. **Account-level feature flag** - Contact Azure support or use preview portal
2. **Vector indexes on containers** - Automatically created by v0.2.0

### Environment Variables

New variables in v0.2.0:

```bash
# Vector search control (default: true)
export AMPLIFIER_COSMOS_ENABLE_VECTOR="true"

# Azure OpenAI embeddings (required for semantic search)
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_EMBEDDING_MODEL="text-embedding-3-large"
export AZURE_OPENAI_EMBEDDING_DIMENSIONS="3072"
export AZURE_OPENAI_EMBEDDING_CACHE_SIZE="1000"
```

---

## API Changes

### Storage Backend Selection

The library now provides a unified `StorageBackend` interface with multiple implementations:

```python
from amplifier_session_storage import CosmosBackend, DuckDBBackend, SQLiteBackend
from amplifier_session_storage import AzureOpenAIEmbeddings

# Create embedding provider (optional, enables semantic search)
embeddings = AzureOpenAIEmbeddings.from_env()

# Choose your backend:

# Cosmos DB (cloud, team-wide sync)
storage = await CosmosBackend.create(embedding_provider=embeddings)

# DuckDB (local, fast with HNSW vector index)
storage = await DuckDBBackend.create(
    config=DuckDBConfig(db_path="~/.amplifier/sessions.duckdb"),
    embedding_provider=embeddings
)

# SQLite (local, simplest)
storage = await SQLiteBackend.create(
    config=SQLiteConfig(db_path="~/.amplifier/sessions.sqlite"),
    embedding_provider=embeddings
)
```

#### Search Operations

```python
from amplifier_session_storage.backends import TranscriptSearchOptions

# Full-text search
results = await storage.search_transcripts(
    user_id="user-123",
    options=TranscriptSearchOptions(
        query="vector search",
        search_type="full_text"
    ),
    limit=10
)

# Semantic search
results = await storage.search_transcripts(
    user_id="user-123",
    options=TranscriptSearchOptions(
        query="how to implement embeddings",
        search_type="semantic"
    ),
    limit=10
)

# Hybrid search with MMR re-ranking
results = await storage.search_transcripts(
    user_id="user-123",
    options=TranscriptSearchOptions(
        query="cosmos db setup",
        search_type="hybrid",
        mmr_lambda=0.7  # Favor relevance over diversity
    ),
    limit=10
)
```

---

## Testing Your Upgrade

```python
# Verify vector search is enabled
async with CosmosBackend.create(embedding_provider=embeddings) as storage:
    supports_vectors = await storage.supports_vector_search()
    print(f"Vector search available: {supports_vectors}")
    
    # Test embedding cache
    cache_stats = embeddings.get_cache_stats()
    print(f"Cache: {cache_stats}")
    
    # Test hybrid search
    results = await storage.search_transcripts(
        user_id="your-user-id",
        options=TranscriptSearchOptions(
            query="test query",
            search_type="hybrid"
        ),
        limit=5
    )
    print(f"Found {len(results)} results")
```

---

## Troubleshooting

### "Vector search not available"

**Cause**: Cosmos DB account doesn't have vector search enabled

**Solution**: 
1. Check if `AMPLIFIER_COSMOS_ENABLE_VECTOR=true` is set
2. Verify Cosmos DB account has vector search feature enabled
3. Check Azure portal for vector search capability

### "Embedding provider required"

**Cause**: Trying semantic/hybrid search without embedding provider

**Solution**:
```python
# Always pass embedding provider for semantic search
embeddings = AzureOpenAIEmbeddings.from_env()
storage = await CosmosBackend.create(embedding_provider=embeddings)
```

### "Dimension mismatch"

**Cause**: Embeddings have different dimensions than configured

**Solution**:
- Ensure `AZURE_OPENAI_EMBEDDING_DIMENSIONS` matches your model
- text-embedding-3-small: 1536
- text-embedding-3-large: 3072

### Graceful Degradation

If embeddings are unavailable, the system automatically falls back:

```
hybrid search (requested)
  ↓ (no embedding provider)
full_text search (automatic fallback)
```

This ensures searches work even without embeddings configured.

---

## Performance Considerations

### Embedding Cache

The LRU cache reduces API calls for repeated queries:

```python
# Configure cache size
embeddings = AzureOpenAIEmbeddings(
    endpoint=...,
    api_key=...,
    cache_size=1000  # Default: 1000 queries
)

# Monitor cache effectiveness
stats = embeddings.get_cache_stats()
print(f"Cache hit rate: {stats['utilization']}")
```

### MMR Lambda Tuning

Adjust `mmr_lambda` based on your use case:

| Lambda | Use Case |
|--------|----------|
| 0.9-1.0 | When you want the most relevant results only |
| 0.7-0.8 | General purpose (recommended) |
| 0.5 | Balance relevance and diversity |
| 0.2-0.3 | When you want variety over exact matches |
| 0.0 | Maximum diversity (good for exploration) |

---

## Next Steps

After upgrading:

1. ✅ Verify vector search is working
2. ✅ Test hybrid search with sample queries
3. ✅ Monitor embedding cache performance
4. ✅ Configure MMR lambda for your use case
5. ✅ Set up monitoring for embedding API usage/costs

---

## Upgrading from v0.2.0 to v0.3.0

### What Changed

1. **`has_vectors` flag on transcript documents** -- A new boolean field `has_vectors` tracks whether vector embeddings exist for each transcript. Set to `false` on creation, flipped to `true` when vectors are stored.
   - DuckDB: `has_vectors BOOLEAN DEFAULT FALSE` column added via schema migration v3
   - SQLite: `has_vectors INTEGER DEFAULT 0` column added via schema migration v3
   - Cosmos DB: Schema-less; new documents get the field automatically. Queries use `NOT IS_DEFINED(c.has_vectors) OR c.has_vectors = false` for backward compatibility.

2. **Schema migration v3** -- DuckDB and SQLite automatically run `ALTER TABLE transcripts ADD COLUMN has_vectors` during `initialize()`. No manual action needed.

3. **Embedding resilience** -- All embedding API calls now have:
   - Retry with exponential backoff (up to 5 retries, 1s-60s, respects Retry-After headers)
   - Circuit breaker (5 consecutive retryable failures = 60s cooldown)
   - Batch splitting (groups of 16 texts, per-batch error isolation)
   - Graceful degradation: transcript sync succeeds even if embeddings fail

4. **New methods: `backfill_embeddings()` and `rebuild_vectors()`** -- Replaces the manual backfill pattern from v0.2.0:
   ```python
   # Old (v0.2.0) -- manual, fragile
   embeddings = await provider.embed_batch(texts)
   await storage.upsert_embeddings(user_id, project, session, embeddings)

   # New (v0.3.0) -- handles chunking, retry, progress
   result = await storage.backfill_embeddings(user_id, project, session)
   ```

5. **`get_session_sync_stats()` fix** -- The Cosmos DB implementation was using `GROUP BY` with multiple aggregates, which the Python SDK does not support. Fixed to use 6 separate single-aggregate queries.

6. **Structured error logging** -- Embedding failures are now logged with `EMBEDDING_FAILURE` prefix, full identity chain (user, project, session), and clear success/failure breakdown.

### Migration Steps

1. **Update dependency**: Bump `amplifier-session-storage` to `>=0.3.0`
2. **Restart services**: Schema migration v3 runs automatically on `initialize()`
3. **Backfill existing sessions** (optional): If you have sessions without vectors, use `backfill_embeddings()` instead of the manual approach from v0.2.0
4. **No breaking changes**: All existing APIs are backward compatible

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/colombod/amplifier-session-storage/issues
- Check logs for detailed error messages
- Use `supports_vector_search()` to diagnose capabilities
