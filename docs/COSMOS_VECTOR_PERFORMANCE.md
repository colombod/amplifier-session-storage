# Cosmos DB Vector Search - High Performance Configuration

**Purpose**: Configure Cosmos DB for optimal vector search performance with the externalized single-vector-path architecture.

**Target**: Production-grade semantic search over large session datasets

---

## Vector Index Types

Cosmos DB NoSQL supports two vector index types:

| Type | Performance | Accuracy | Best For |
|------|-------------|----------|----------|
| **quantizedFlat** | Fast | High | <100k vectors |
| **diskANN** | Very Fast | Tunable | >100k vectors |

### quantizedFlat Configuration

**Recommended for session storage** (typically <100k vector records):

```python
vector_index_policy = {
    "vectorIndexes": [
        {
            "path": "/vector",
            "type": "quantizedFlat",
            "quantizationByteSize": 128,
            "vectorDataType": "float32",
            "dimensions": 3072
        }
    ]
}
```

**Benefits**:
- **Single index** covers all content types (user_query, assistant_response, assistant_thinking, tool_output)
- **96% storage reduction** per vector: 12,288 bytes -> 128 bytes (quantized)
- **Faster queries**: Smaller index fits in memory
- **High accuracy**: Minimal quality loss with 128-byte quantization
- **Simpler management**: One index to maintain vs. four in the previous architecture

### diskANN Configuration

**For large deployments** (>100k vector records):

```python
vector_index_policy = {
    "vectorIndexes": [
        {
            "path": "/vector",
            "type": "diskANN",
            "quantizationByteSize": 128,
            "vectorDataType": "float32",
            "dimensions": 3072
        }
    ]
}
```

**Benefits**:
- **Scales to millions**: Disk-based index
- **Tunable accuracy**: Trade speed for precision
- **Lower memory**: Index stored on disk

---

## Re-Ranking with Oversampling

### The Strategy

Cosmos DB vector search returns approximate nearest neighbors. For high accuracy:

1. **Oversample** - Fetch more candidates than needed (e.g., fetch 50, return top 10)
2. **Deduplicate** - Multiple vector records may exist per parent message (chunks, content types); keep best score per parent
3. **Re-rank** - Use exact similarity computation on deduplicated results
4. **MMR** - Apply MMR for diversity after re-ranking

### Implementation

```python
async def search_with_reranking(
    storage: CosmosBackend,
    query_vector: list[float],
    top_k: int = 10,
    oversample_factor: int = 5
) -> list[SearchResult]:
    """
    High-accuracy search with oversampling and re-ranking.

    Args:
        query_vector: Query embedding
        top_k: Final number of results
        oversample_factor: Fetch this many times top_k for re-ranking
    """
    # Step 1: Oversample from Cosmos vector search
    candidates = await storage.vector_search(
        user_id="user-123",
        query_vector=query_vector,
        top_k=top_k * oversample_factor  # Fetch 50 for top-10 query
    )
    # Candidates are already deduplicated by parent_id (Python-side dict)

    # Step 2: Re-rank with exact cosine similarity
    import numpy as np
    query_np = np.array(query_vector)

    for candidate in candidates:
        vector = candidate.metadata.get("embedding_vector")
        if vector:
            vector_np = np.array(vector)
            exact_similarity = np.dot(query_np, vector_np) / (
                np.linalg.norm(query_np) * np.linalg.norm(vector_np)
            )
            candidate.score = exact_similarity

    # Sort by exact scores
    candidates.sort(key=lambda x: x.score, reverse=True)

    # Step 3: Apply MMR for diversity (optional)
    from amplifier_session_storage.search.mmr import compute_mmr

    vectors = [np.array(c.metadata["embedding_vector"]) for c in candidates]
    mmr_indices = compute_mmr(
        vectors=vectors,
        query=query_np,
        lambda_param=0.7,  # 70% relevance, 30% diversity
        top_k=top_k
    )

    return [candidates[i] for i in mmr_indices]
```

**Performance**:
- **First pass**: Fast approximate search on single `/vector` index
- **Re-ranking**: Exact computation on small candidate set (cheap)
- **Result**: High accuracy with minimal latency penalty

---

## Required Cosmos DB Configuration

### 1. Enable Vector Search

**At account level** (one-time setup):

```bash
# Via Azure Portal
# Navigate to account -> Features -> "Vector Search in Azure Cosmos DB for NoSQL"
# Enable the feature

# Or via CLI (if supported in your region)
az cosmosdb update \
  --name your-cosmos-account \
  --resource-group your-resource-group \
  --subscription your-subscription-id \
  --capabilities EnableServerless EnableNoSQLVectorSearch
```

### 2. Create Database and Containers

**Database**:
```bash
az cosmosdb sql database create \
  --account-name your-cosmos-account \
  --resource-group your-resource-group \
  --name your-database \
  --throughput 400  # Minimum for serverless
```

**Transcript messages container** (stores both messages and vectors):
```bash
az cosmosdb sql container create \
  --account-name your-cosmos-account \
  --database-name your-database \
  --name transcript_messages \
  --partition-key-path "/partition_key" \
  --idx @vector-index-policy.json
```

**vector-index-policy.json**:
```json
{
  "indexingMode": "consistent",
  "automatic": true,
  "includedPaths": [
    {"path": "/parent_id/?"},
    {"path": "/content_type/?"},
    {"path": "/user_id/?"},
    {"path": "/session_id/?"},
    {"path": "/type/?"}
  ],
  "excludedPaths": [
    {"path": "/content/*"},
    {"path": "/source_text/*"},
    {"path": "/vector/*"},
    {"path": "/\"_etag\"/?"},
    {"path": "/*"}
  ],
  "vectorIndexes": [
    {
      "path": "/vector",
      "type": "quantizedFlat",
      "quantizationByteSize": 128,
      "vectorDataType": "float32",
      "dimensions": 3072
    }
  ]
}
```

Key indexing decisions:
- **Included paths** cover the scalar fields used in WHERE clauses for filtering by parent, content type, user, and session
- **Excluded paths** prevent indexing of large text fields (`content`, `source_text`) and the vector itself (handled by the vector index)
- **Single vector index** on `/vector` replaces the previous 4-path approach

### 3. Grant RBAC Permissions

**Required role**: Cosmos DB Built-in Data Contributor

```bash
az cosmosdb sql role assignment create \
  --account-name your-cosmos-account \
  --resource-group your-resource-group \
  --scope "/" \
  --role-definition-id 00000000-0000-0000-0000-000000000002 \
  --principal-id $(az ad signed-in-user show --query id -o tsv)
```

RBAC propagation takes 5-15 minutes.

---

## Performance Tuning Options

### Quantization Byte Size

| Size | Storage | Accuracy | Speed |
|------|---------|----------|-------|
| **64** | Smallest | Lower | Fastest |
| **128** | Recommended | High | Fast |
| **256** | Larger | Higher | Medium |
| **512** | Large | Very High | Slower |

**Recommendation**: **128 bytes** - Best balance for text-embedding-3-large (3072 dims)

### Oversampling Strategy

```python
# For top-10 results
oversample_candidates = {
    "high_precision": 100,   # 10x oversampling
    "balanced": 50,          # 5x oversampling (recommended)
    "fast": 20               # 2x oversampling
}
```

**Trade-off**:
- Higher oversampling = better accuracy, higher RU cost
- Lower oversampling = faster, cheaper, slightly lower accuracy

**Note**: With the externalized vector architecture, oversampling is more important because multiple vector records may exist per parent message (from chunking and multiple content types). The 3x over-fetch used for deduplication is separate from the oversampling for accuracy.

**Recommendation**: **5x oversampling** (fetch 50, return 10)

### Re-Ranking Methods

**Option 1: Exact Cosine Similarity** (Recommended)
```python
# Compute exact similarity on oversampled candidates
exact_score = np.dot(query, vector) / (np.linalg.norm(query) * np.linalg.norm(vector))
```

**Option 2: MMR Diversity** (For varied results)
```python
# Apply MMR to balance relevance + diversity
mmr_indices = compute_mmr(vectors, query, lambda_param=0.7, top_k=10)
```

**Option 3: Hybrid** (Best accuracy + diversity)
```python
# 1. Oversample (50 candidates)
# 2. Deduplicate by parent_id (best score per message)
# 3. Re-rank with exact similarity
# 4. Apply MMR for diversity
# 5. Return top-10
```

---

## Storage Analysis

### Per Vector Record in Cosmos

| Component | Raw Size | Quantized Size |
|-----------|----------|---------------|
| Vector (3072-d, float32) | 12,288 bytes | 128 bytes (quantized index) |
| Source text | Variable (~500-4000 bytes) | Same (not indexed) |
| Metadata fields | ~300 bytes | Same |
| Total document | ~13-17 KB | ~13-17 KB (index: 128 bytes) |

### Comparison with Previous 4-Path Architecture

| Aspect | Old (4 inline vectors) | New (externalized) |
|--------|----------------------|-------------------|
| Vector indexes | 4 quantizedFlat | 1 quantizedFlat |
| Index storage per message | ~512 bytes (4 x 128) | ~128-256 bytes (1-2 records) |
| Document size (transcript) | ~50-60 KB (with 4 vectors) | ~500 bytes - 2 KB (no vectors) |
| Vector documents | N/A (inline) | Separate, ~13-17 KB each |
| RU per search | Proportional to 4 VectorDistance calls | 1 VectorDistance call |
| Chunking support | No | Yes (multiple records per content type) |

The new architecture trades more documents for simpler indexing and search queries. Total storage is comparable, but query costs are lower due to the single index path.

---

## Recommended Configuration for Session Analyst

### For Team-Wide Sessions (<100k vector records)

```python
cosmos_config = {
    "endpoint": "https://your-cosmos-account.documents.azure.com:443/",
    "database": "amplifier-sessions",
    "auth_method": "default_credential",
    "enable_vector_search": True,

    # Vector index configuration
    "vector_index_type": "quantizedFlat",
    "quantization_byte_size": 128,
    "vector_dimensions": 3072,

    # Search optimization
    "oversample_factor": 5,       # Fetch 5x candidates
    "rerank_method": "exact",     # Exact cosine after fetch
    "mmr_lambda": 0.7,            # 70% relevance, 30% diversity
}
```

**Expected performance**:
- **Query latency**: 50-100ms (network + search + dedup + re-rank)
- **Accuracy**: 95%+ (quantization + exact re-rank)
- **Cost**: ~5 RU per search (single index path)

### For Large Datasets (>100k vector records)

```python
cosmos_config = {
    # ... same as above ...
    "vector_index_type": "diskANN",  # Change to diskANN
    "quantization_byte_size": 128,
}
```

**Expected performance**:
- **Query latency**: 20-50ms (faster than quantizedFlat at scale)
- **Accuracy**: Tunable via oversampling
- **Cost**: ~3-5 RU per search

---

## Document Size Validation

Cosmos DB has a hard 2MB per-document limit. The library validates transcript documents against a 1.8MB safety threshold (200KB margin):

```python
cosmos_size_limit = 1_800_000  # 1.8MB safety margin

if doc_size > cosmos_size_limit:
    logger.warning(f"Document exceeds size limit ({doc_size:,} bytes)")
    doc["content"] = "[Content truncated - original size exceeded Cosmos limit]"
    doc["content_truncated"] = True
    doc["original_content_size"] = original_size
```

With vectors externalized, transcript documents are much smaller (no inline vector arrays). This check exists as defense-in-depth for unusually large message content.

Vector documents are inherently sized safely: each contains one 3072-d vector (~12 KB) plus source text (at most ~4 KB for a 1024-token chunk) plus metadata.

---

## Implementation in CosmosBackend

**Current implementation** (in amplifier-session-storage):

The `CosmosBackend` class implements:
- Single quantizedFlat vector index on `/vector` path (128-byte compression)
- `VectorDistance(c.vector, @queryVector)` queries filtered by `content_type`
- Python-side deduplication by `parent_id`
- Auto-migration from 4-path to single-path vector index policy
- Configurable via `CosmosConfig`

**What is configured**:
```python
# In backends/cosmos.py
vector_index = {
    "path": "/vector",
    "type": "quantizedFlat",
    "quantizationByteSize": 128,
    "vectorDataType": "float32",
    "dimensions": self.config.vector_dimensions
}
```

---

## Summary

**For session analyst bundle using Cosmos DB**:

```python
# Environment configuration
AMPLIFIER_COSMOS_ENDPOINT=https://your-cosmos-account.documents.azure.com:443/
AMPLIFIER_COSMOS_DATABASE=amplifier-sessions
AMPLIFIER_COSMOS_AUTH_METHOD=default_credential
AMPLIFIER_COSMOS_ENABLE_VECTOR=true

# Create storage
from amplifier_session_storage.backends.cosmos import CosmosBackend
storage = await CosmosBackend.create(CosmosConfig.from_env())

# High-accuracy search with oversampling
candidates = await storage.vector_search(
    user_id="team-user",
    query_vector=query_embedding,
    top_k=50  # 5x oversample for top-10
)
# Results are already deduplicated by parent_id

# Re-rank with MMR
from amplifier_session_storage.search.mmr import compute_mmr
final_results = compute_mmr(
    vectors=[c.embedding for c in candidates],
    query=query_embedding,
    lambda_param=0.7,
    top_k=10
)
```

**This configuration delivers**:
- 96% index storage reduction (quantization)
- Single vector index (reduced from 4)
- High accuracy (oversample + exact re-rank)
- Result diversity (MMR)
- Fast queries (<100ms typical)
- Chunking support for long texts
- Scalable to millions of vector records
