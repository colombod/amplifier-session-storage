# Vector Search Implementation Guide

This document explains how vector search works across different backends.

## Overview

All three backends support vector similarity search, but with different implementations:

| Backend | Vector Support | Index Type | Notes |
|---------|---------------|------------|-------|
| **Cosmos DB** | ✅ Native | quantizedFlat / diskANN | Requires account-level feature enablement |
| **DuckDB** | ✅ VSS Extension | HNSW | Requires `INSTALL vss; LOAD vss` |
| **SQLite** | ⚠️ Numpy Fallback | N/A (brute-force) | sqlite-vss not widely available |

---

## Cosmos DB Vector Search

### Configuration

**Required**: Vector search must be enabled at account level

```bash
# Enable via Azure Portal
# Navigate to account → Features → "Vector Search in Azure Cosmos DB for NoSQL"

# Or via CLI (if available in your region)
az cosmosdb update \
  --name your-account \
  --resource-group your-rg \
  --capabilities EnableServerless EnableNoSQLVectorSearch
```

### How It Works

Cosmos DB uses the `VectorDistance` function:

```sql
SELECT c.id, c.content,
       VectorDistance(c.embedding, @query_vector) AS distance
FROM c
WHERE c.user_id = @user_id AND c.embedding != null
ORDER BY VectorDistance(c.embedding, @query_vector)
```

**Vector Index Configuration** (automatic via library):
```json
{
  "vectorIndexes": [{
    "path": "/embedding",
    "type": "quantizedFlat",
    "quantizationByteSize": 128,
    "vectorDataType": "float32",
    "dimensions": 3072
  }]
}
```

**Performance**:
- quantizedFlat: Good for <100k documents
- diskANN: Better for larger datasets (>100k docs)

---

## DuckDB Vector Search

### Installation

```bash
# VSS extension auto-installs on first use
# No separate installation needed
```

### How It Works

DuckDB uses the VSS extension with HNSW indexes:

```sql
-- Install and load extension
INSTALL vss;
LOAD vss;

-- Create HNSW index
CREATE INDEX idx_embedding ON transcripts USING HNSW (embedding)
WITH (metric = 'cosine');

-- Query with array_cosine_similarity
SELECT id, content,
       array_cosine_similarity(embedding, [0.1, 0.2, ...]::FLOAT[3072]) AS similarity
FROM transcripts
ORDER BY similarity DESC
LIMIT 10;
```

### Critical Implementation Detail

**DuckDB-VSS requires constant array expressions** for HNSW index usage.

❌ **This doesn't work** (parameter binding):
```python
query_vector = [0.1, 0.2, ...]
sql = "SELECT array_cosine_similarity(vec, ?::FLOAT[3072]) FROM table"
conn.execute(sql, [query_vector])  # Falls back to sequential scan!
```

✅ **This works** (string interpolation):
```python
def format_vector_literal(vec):
    vec_str = "[" + ", ".join(str(float(x)) for x in vec) + "]"
    return f"{vec_str}::FLOAT[{len(vec)}]"

vec_literal = format_vector_literal(query_vector)
sql = f"SELECT array_cosine_similarity(vec, {vec_literal}) FROM table"
conn.execute(sql)  # Uses HNSW index!
```

**Why**: The HNSW index optimizer needs the query vector at planning time, not execution time. Parameter binding creates `VALUE_PARAMETER` expressions; the optimizer only recognizes `VALUE_CONSTANT`.

**Security**: The vector values come from trusted embedding models, not user input. If you accept user-provided vectors, validate they're numeric:

```python
if not all(isinstance(x, (int, float)) for x in vector):
    raise ValueError("Vector must contain only numeric values")
```

### Performance

- **With HNSW index**: O(log n) similarity search
- **Without index**: O(n) sequential scan
- **Index creation**: Fast for <100k documents, slower for larger datasets

---

## SQLite Vector Search

### Current Implementation

SQLite uses **numpy-based brute-force search** as fallback:

```python
# Fetch all embeddings from database
rows = cursor.fetchall()

# Compute cosine similarities with numpy
query_np = np.array(query_vector)
for row in rows:
    embedding_np = np.array(json.loads(row['embedding_json']))
    similarity = np.dot(query_np, embedding_np) / (
        np.linalg.norm(query_np) * np.linalg.norm(embedding_np)
    )
    
# Sort by similarity, return top-k
```

**Performance**: O(n) - loads all embeddings into memory

### Why Not sqlite-vss?

The `sqlite-vss` extension is not widely available:
- Not in standard SQLite distributions
- Requires manual compilation
- Limited package availability

**For small datasets** (<10k documents), numpy fallback is acceptable.

**For larger datasets**: Use DuckDB or Cosmos DB instead.

### Future Enhancement

If `sqlite-vss` becomes more available, we can add optimized path:

```python
async def vector_search(self, ...):
    if self._vss_available:
        return await self._vss_vector_search(...)  # Fast path
    else:
        return await self._numpy_vector_search(...)  # Fallback
```

---

## Embedding Generation During Ingestion

All backends support automatic embedding generation:

```python
# Embeddings generated automatically during sync
await storage.sync_transcript_lines(
    user_id="user-123",
    host_id="laptop-01",
    project_slug="project",
    session_id="session",
    lines=transcript_lines,
    # embeddings parameter optional - generated if not provided
)
```

**How it works**:

1. Check if `embeddings` parameter provided
2. If not, check if `embedding_provider` configured
3. If yes, call `embedding_provider.embed_batch(texts)`
4. Store embeddings alongside transcript content

**Cache efficiency**:
```python
embeddings = AzureOpenAIEmbeddings(cache_size=1000)

# First batch - hits API
await storage.sync_transcript_lines(lines=batch1)  # API calls

# Second batch with duplicate content - hits cache
await storage.sync_transcript_lines(lines=batch2)  # Mostly cache hits
```

---

## Graceful Degradation

The system automatically handles missing capabilities:

### No Embedding Provider

```python
storage = await DuckDBBackend.create(embedding_provider=None)

# Semantic search requested
results = await storage.search_transcripts(
    options=TranscriptSearchOptions(
        query="...",
        search_type="semantic"  # Will fallback to full_text
    )
)
# Automatically uses full_text search instead
```

### No VSS Extension (DuckDB)

```python
# If VSS extension fails to load
await storage.supports_vector_search()  # Returns False

# Semantic search requested
results = await storage.search_transcripts(
    options=TranscriptSearchOptions(search_type="semantic")
)
# Falls back to full_text automatically
```

### No Vector Indexes (Cosmos DB)

```python
# If vector search not enabled on account
await storage.supports_vector_search()  # Returns False

# Hybrid search requested
results = await storage.search_transcripts(
    options=TranscriptSearchOptions(search_type="hybrid")
)
# Falls back to full_text
```

---

## Performance Comparison

**Setup**: 10,000 documents, 3072-dimensional embeddings, top-10 query

| Backend | Index Type | Query Time | Notes |
|---------|------------|------------|-------|
| **Cosmos DB** | quantizedFlat | ~50-100ms | Best for distributed access |
| **Cosmos DB** | diskANN | ~20-50ms | Best for >100k docs |
| **DuckDB** | HNSW | ~5-20ms | Best for local analytics |
| **SQLite** | Numpy fallback | ~200-500ms | Only for small datasets |

**Memory Usage**:

| Backend | Index Memory | Notes |
|---------|--------------|-------|
| Cosmos DB | Cloud-managed | ~128 bytes per vector (quantized) |
| DuckDB | In-process | ~200-400 bytes per vector (HNSW graph) |
| SQLite | None | Loads vectors on demand |

---

## Distance Metrics

All backends use **cosine similarity** for consistency:

```
Cosine Similarity = dot(A, B) / (norm(A) * norm(B))

Range: [-1, 1]
- 1.0 = Identical direction
- 0.0 = Orthogonal
- -1.0 = Opposite direction
```

**Why cosine?**
- Scale-invariant (only direction matters)
- Standard for text embeddings
- Supported by all backends

**Alternative metrics** (DuckDB only):
- L2 distance: `array_distance()`
- Inner product: `array_negative_inner_product()`

---

## Troubleshooting

### "Vector search not available"

**Check 1**: Embedding provider configured?
```python
storage = await DuckDBBackend.create(embedding_provider=embeddings)  # Required!
```

**Check 2**: Backend supports vectors?
```python
supports = await storage.supports_vector_search()
print(f"Vector search: {supports}")
```

**Check 3**: DuckDB VSS extension loaded?
```bash
# Test manually
uv run python -c "
import duckdb
conn = duckdb.connect(':memory:')
conn.execute('INSTALL vss')
conn.execute('LOAD vss')
print('VSS available')
"
```

### "HNSW index not used" (DuckDB)

**Cause**: Using parameter binding instead of string interpolation

**Fix**: Use `_format_vector_literal()` to embed vector in SQL:
```python
vec_literal = DuckDBBackend._format_vector_literal(query_vector)
sql = f"SELECT ... array_cosine_similarity(vec, {vec_literal}) ..."
```

**Verify**: Check EXPLAIN plan for `HNSW_INDEX_SCAN` node:
```python
explain = conn.execute(f"EXPLAIN {sql}").fetchall()
has_hnsw = any("HNSW" in str(row) for row in explain)
```

### Slow vector search (SQLite)

**Cause**: Using numpy fallback (O(n) complexity)

**Solutions**:
1. Use DuckDB instead (same local storage, faster search)
2. Limit dataset size (<10k documents)
3. Add filtering to reduce candidates before vector search

---

## Best Practices

### 1. Choose the Right Backend

```python
# Cloud sync, multi-device
backend = CosmosBackend  # Use Cosmos

# Local development, fast analytics
backend = DuckDBBackend  # Use DuckDB

# Embedded app, small dataset
backend = SQLiteBackend  # Use SQLite
```

### 2. Use Appropriate Index Types

**Cosmos DB**:
- <100k docs: `quantizedFlat`
- >100k docs: `diskANN`

**DuckDB**:
- Always use HNSW (auto-created by library)
- Tune `ef_search` for speed vs accuracy trade-off

### 3. Batch Embedding Generation

```python
# ✅ Good: Batch operation
texts = [line["content"] for line in lines]
embeddings = await provider.embed_batch(texts)
await storage.sync_transcript_lines(lines=lines, embeddings=embeddings)

# ❌ Bad: Individual calls
for line in lines:
    embedding = await provider.embed_text(line["content"])
    # N API calls instead of 1
```

### 4. Monitor Cache Performance

```python
stats = embedding_provider.get_cache_stats()
print(f"Cache hit rate: {stats['utilization']:.1%}")

# Low utilization? Increase cache size
provider = AzureOpenAIEmbeddings(cache_size=2000)
```

---

## Migration Path

### From Full-Text to Hybrid Search

**Step 1**: Add embedding provider
```python
# Before
storage = await DuckDBBackend.create()

# After  
embeddings = AzureOpenAIEmbeddings.from_env()
storage = await DuckDBBackend.create(embedding_provider=embeddings)
```

**Step 2**: Regenerate embeddings for existing data
```python
# Backfill embeddings
sessions = await storage.search_sessions(user_id, filters, limit=1000)
for session in sessions:
    # Get transcripts
    transcripts = await storage.get_transcript_lines(...)
    
    # Generate embeddings
    texts = [t["content"] for t in transcripts]
    vectors = await embeddings.embed_batch(texts)
    
    # Update
    await storage.upsert_embeddings(...)
```

**Step 3**: Switch to hybrid search
```python
# Now you can use hybrid search
results = await storage.search_transcripts(
    options=TranscriptSearchOptions(search_type="hybrid")
)
```

---

## References

- **Cosmos DB**: [Vector Search Documentation](https://learn.microsoft.com/azure/cosmos-db/nosql/vector-search)
- **DuckDB-VSS**: [VSS Extension Docs](https://duckdb.org/docs/stable/core_extensions/vss)
- **MMR Algorithm**: [Carbonell & Goldstein, 1998](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf)
- **Reference Implementation**: [AIGeekSquad/AIContext](https://github.com/AIGeekSquad/AIContext) (C#)
