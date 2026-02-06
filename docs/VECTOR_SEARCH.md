# Vector Search Implementation Guide

This document explains how vector search works across different backends with the externalized vector architecture.

## Overview

All three backends support vector similarity search using the `transcript_vectors` table (DuckDB/SQLite) or `transcript_vector` documents (Cosmos DB). Vectors are stored separately from transcript content, with a single index per backend.

| Backend | Vector Support | Index Type | Notes |
|---------|---------------|------------|-------|
| **Cosmos DB** | Native | quantizedFlat / diskANN | Single `/vector` path; requires account-level feature enablement |
| **DuckDB** | VSS Extension | HNSW | Single `idx_vectors_hnsw` index; requires `INSTALL vss; LOAD vss` |
| **SQLite** | Numpy Fallback | N/A (brute-force) | sqlite-vss not widely available |

---

## DuckDB Vector Search

### Installation

```bash
# VSS extension auto-installs on first use
# No separate installation needed
```

### How It Works

DuckDB uses the VSS extension with a single HNSW index on the `transcript_vectors` table. A `vectors_with_context` view JOINs vector records with their parent transcript rows, providing both the vector and the full message content in one query.

```sql
-- Install and load extension
INSTALL vss;
LOAD vss;

-- Single HNSW index on transcript_vectors
CREATE INDEX idx_vectors_hnsw
ON transcript_vectors USING HNSW (vector)
WITH (metric = 'cosine');

-- Search via vectors_with_context view
SELECT parent_id, session_id, project_slug, sequence,
       content, role, turn, ts, content_type,
       array_cosine_similarity(vector, [0.1, 0.2, ...]::FLOAT[3072]) AS similarity
FROM vectors_with_context
WHERE user_id = ?
  AND content_type IN ('user_query', 'assistant_response')
ORDER BY similarity DESC
LIMIT 30;  -- over-fetch 3x for dedup
```

### Deduplication

Since multiple vector records can exist per transcript message (different content types, chunked texts), results must be deduplicated. All backends use the same Python-side approach:

```python
# Over-fetch 3x the requested top_k
results_raw = conn.execute(query, params).fetchall()

# Deduplicate by parent_id, keeping best similarity
seen: dict[str, SearchResult] = {}
for row in results_raw:
    parent_id = row[0]
    score = float(row[9])
    if parent_id not in seen or score > seen[parent_id].score:
        seen[parent_id] = SearchResult(...)

# Return top_k after dedup
final = sorted(seen.values(), key=lambda r: r.score, reverse=True)[:top_k]
```

### Full-Text Search on source_text

Full-text search queries both the main `transcripts.content` field and the `transcript_vectors.source_text` field. The `source_text` search respects the same `content_type` filtering as semantic search:

```sql
-- Search source_text in transcript_vectors
SELECT tv.parent_id, tv.session_id, tv.content_type,
       t.content, t.role, t.turn, t.ts, t.sequence
FROM transcript_vectors tv
JOIN transcripts t ON tv.parent_id = t.id
WHERE tv.user_id = ?
  AND tv.content_type IN ('user_query', 'assistant_thinking')
  AND tv.source_text LIKE '%search_term%'
```

Results from both sources are merged and deduplicated by `(session_id, sequence)`.

### Critical Implementation Detail

**DuckDB-VSS requires constant array expressions** for HNSW index usage.

This does NOT work (parameter binding):
```python
query_vector = [0.1, 0.2, ...]
sql = "SELECT array_cosine_similarity(vec, ?::FLOAT[3072]) FROM table"
conn.execute(sql, [query_vector])  # Falls back to sequential scan!
```

This works (string interpolation):
```python
def format_vector_literal(vec):
    vec_str = "[" + ", ".join(str(float(x)) for x in vec) + "]"
    return f"{vec_str}::FLOAT[{len(vec)}]"

vec_literal = format_vector_literal(query_vector)
sql = f"SELECT array_cosine_similarity(vec, {vec_literal}) FROM table"
conn.execute(sql)  # Uses HNSW index!
```

**Why**: The HNSW index optimizer needs the query vector at planning time, not execution time. Parameter binding creates `VALUE_PARAMETER` expressions; the optimizer only recognizes `VALUE_CONSTANT`.

**Security**: The vector values come from trusted embedding models, not user input. If you accept user-provided vectors, validate they are numeric:

```python
if not all(isinstance(x, (int, float)) for x in vector):
    raise ValueError("Vector must contain only numeric values")
```

### HNSW Index Limitation with WHERE Filters

DuckDB's HNSW index does not support pre-filtered queries. When a `WHERE user_id = ?` clause is present, DuckDB falls back to sequential scan rather than using the HNSW index.

**Why this is acceptable**: DuckDB databases are single-user. Each user has their own `.db` file, so `WHERE user_id = ?` matches all rows in the database. The sequential scan searches the same dataset the HNSW index would. For the typical dataset size (thousands to tens of thousands of vector records per user), sequential scan performance is adequate.

**Verification**: Check EXPLAIN plan for `HNSW_INDEX_SCAN` node:
```python
explain = conn.execute(f"EXPLAIN {sql}").fetchall()
has_hnsw = any("HNSW" in str(row) for row in explain)
# Will be False when WHERE clause is present - this is expected
```

### Performance

- **With HNSW index (no WHERE filter)**: O(log n) similarity search
- **With WHERE filter**: O(n) sequential scan (acceptable for single-user DBs)
- **Index creation**: Fast for <100k vector records

---

## Cosmos DB Vector Search

### Configuration

**Required**: Vector search must be enabled at account level

```bash
# Enable via Azure Portal
# Navigate to account -> Features -> "Vector Search in Azure Cosmos DB for NoSQL"

# Or via CLI (if available in your region)
az cosmosdb update \
  --name your-account \
  --resource-group your-rg \
  --capabilities EnableServerless EnableNoSQLVectorSearch
```

### How It Works

Cosmos DB uses the `VectorDistance` function against the single `/vector` path on `transcript_vector` documents:

```sql
SELECT TOP @fetchLimit
    c.parent_id, c.session_id, c.content_type,
    c.source_text, c.chunk_index, c.total_chunks,
    VectorDistance(c.vector, @queryVector) AS distance
FROM c
WHERE c.type = "transcript_vector"
  AND c.user_id = @userId
  AND ARRAY_CONTAINS(@contentTypes, c.content_type)
ORDER BY VectorDistance(c.vector, @queryVector)
```

**Vector Index Configuration** (single index, automatic via library):
```json
{
  "vectorIndexes": [{
    "path": "/vector",
    "type": "quantizedFlat",
    "quantizationByteSize": 128,
    "vectorDataType": "float32",
    "dimensions": 3072
  }]
}
```

**Deduplication**: Same Python-side `dict` approach as DuckDB, keyed by `parent_id`.

**Full-text on source_text**: Uses `CONTAINS(c.source_text, @query)` with `ARRAY_CONTAINS(@contentTypes, c.content_type)` filtering.

**Note**: Cosmos returns distance (lower = better), which is converted to similarity (higher = better) in the Python layer.

### Performance

- quantizedFlat: Good for <100k vector records
- diskANN: Better for larger datasets (>100k records)

---

## SQLite Vector Search

### Current Implementation

SQLite uses **numpy-based brute-force search** as fallback:

```python
# Fetch vectors from transcript_vectors, filtered by content_type
rows = cursor.fetchall()

# Compute cosine similarities with numpy
query_np = np.array(query_vector)
results = []
for row in rows:
    vec = json.loads(row['vector_json'])
    vec_np = np.array(vec)
    similarity = np.dot(query_np, vec_np) / (
        np.linalg.norm(query_np) * np.linalg.norm(vec_np)
    )
    results.append((row, similarity))

# Sort by similarity, deduplicate by parent_id, return top-k
```

**Performance**: O(n) - loads all matching vectors into memory

### Why Not sqlite-vss?

The `sqlite-vss` extension is not widely available:
- Not in standard SQLite distributions
- Requires manual compilation
- Limited package availability

**For small datasets** (<10k vector records), numpy fallback is acceptable.

**For larger datasets**: Use DuckDB or Cosmos DB instead.

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
)
```

**How it works**:

1. Extract embeddable content from each message
2. Chunk texts exceeding 8192 tokens
3. Generate embeddings via batch API call
4. Store transcript row (no vectors) in `transcripts`
5. Store each vector + metadata in `transcript_vectors`

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

**Setup**: 10,000 transcript messages (producing ~15,000 vector records), 3072-dimensional embeddings, top-10 query

| Backend | Index Type | Query Time | Notes |
|---------|------------|------------|-------|
| **Cosmos DB** | quantizedFlat | ~50-100ms | Best for distributed access |
| **Cosmos DB** | diskANN | ~20-50ms | Best for >100k records |
| **DuckDB** | HNSW | ~5-20ms | Best for local analytics (no WHERE filter) |
| **DuckDB** | Sequential | ~20-50ms | With WHERE user_id filter (single-user DB) |
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
uv run python -c "
import duckdb
conn = duckdb.connect(':memory:')
conn.execute('INSTALL vss')
conn.execute('LOAD vss')
print('VSS available')
"
```

### "HNSW index not used" (DuckDB)

**Cause 1**: Using parameter binding instead of string interpolation

**Fix**: Use `_format_vector_literal()` to embed vector in SQL:
```python
vec_literal = DuckDBBackend._format_vector_literal(query_vector)
sql = f"SELECT ... array_cosine_similarity(vec, {vec_literal}) ..."
```

**Cause 2**: WHERE clause present (e.g., `WHERE user_id = ?`)

**Expected behavior**: DuckDB HNSW does not support pre-filtered queries. The sequential scan fallback is acceptable for single-user databases.

### Slow vector search (SQLite)

**Cause**: Using numpy fallback (O(n) complexity)

**Solutions**:
1. Use DuckDB instead (same local storage, faster search)
2. Limit dataset size (<10k vector records)
3. Add content_type filtering to reduce candidates before vector search

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

### 2. Use Content-Type Filtering

```python
# More precise: search only relevant content types
results = await storage.vector_search(
    user_id="user-123",
    query_vector=query_vec,
    vector_columns=["assistant_thinking"],  # Only reasoning
    top_k=10
)
# Fewer vector records scanned, faster results
```

### 3. Batch Embedding Generation

```python
# Good: Let sync handle batching automatically
await storage.sync_transcript_lines(lines=lines)
# Internally batches all non-None texts into minimal API calls

# Bad: Individual calls
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

## References

- **Cosmos DB**: [Vector Search Documentation](https://learn.microsoft.com/azure/cosmos-db/nosql/vector-search)
- **DuckDB-VSS**: [VSS Extension Docs](https://duckdb.org/docs/stable/core_extensions/vss)
- **MMR Algorithm**: [Carbonell & Goldstein, 1998](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf)
