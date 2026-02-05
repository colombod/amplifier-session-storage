# Azure Cosmos DB Best Practices for Session Storage

> Research document covering query optimization, data setup, and vector exclusion patterns for the Cosmos DB backend.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Query Best Practices](#query-best-practices)
3. [Excluding Vectors from Query Results](#excluding-vectors-from-query-results)
4. [Data Setup and Schema Design](#data-setup-and-schema-design)
5. [Vector Indexing Configuration](#vector-indexing-configuration)
6. [Partition Key Strategy](#partition-key-strategy)
7. [Schema Evolution Patterns](#schema-evolution-patterns)
8. [Implementation Recommendations](#implementation-recommendations)

---

## Executive Summary

Azure Cosmos DB for NoSQL provides powerful capabilities for session storage with vector search, but requires careful attention to query patterns, projection, and indexing to optimize RU (Request Unit) costs and performance.

**Key Findings:**
- **Explicit property projection** significantly reduces RU cost compared to `SELECT *`
- **JSON projection syntax** provides fine-grained control over returned fields
- **Hierarchical partition keys** enable scalable user-scoped data
- **quantizedFlat vector indexes** balance cost and performance for moderate vector volumes
- **Schema versioning** enables safe evolution without downtime

---

## Query Best Practices

### Explicit Property Selection vs SELECT *

**The Problem:** `SELECT *` returns the full JSON document, including large vector arrays (up to 48KB per message with 4 embeddings). This:
- Increases RU consumption
- Increases network latency
- Wastes bandwidth on unused data

**The Solution:** Always use explicit property projection in production queries.

#### Basic Projection Patterns

```sql
-- BAD: Returns everything including vectors
SELECT * FROM c WHERE c.user_id = @userId

-- GOOD: Explicit property list
SELECT c.id, c.session_id, c.user_id, c.sequence, c.turn, 
       c.role, c.content, c.ts
FROM c 
WHERE c.user_id = @userId

-- GOOD: SELECT VALUE for single property
SELECT VALUE c.content FROM c WHERE c.id = @id
```

#### JSON Object Projection

Cosmos DB supports constructing new JSON objects in the SELECT clause:

```sql
-- Project a shaped object (excludes all vector columns)
SELECT {
    "id": c.id,
    "session_id": c.session_id,
    "sequence": c.sequence,
    "turn": c.turn,
    "role": c.role,
    "content": c.content,
    "ts": c.ts,
    "embedding_model": c.embedding_model
} AS message
FROM c
WHERE c.partition_key = @partitionKey
ORDER BY c.sequence
```

### Query Performance Optimization

1. **Always include partition key in WHERE clause**
   ```sql
   -- GOOD: Scoped to single partition
   WHERE c.partition_key = @pk AND c.session_id = @sessionId
   
   -- BAD: Cross-partition fan-out (expensive)
   WHERE c.session_id = @sessionId
   ```

2. **Use TOP N for bounded results**
   ```sql
   SELECT TOP 100 c.id, c.content, c.ts
   FROM c
   WHERE c.partition_key = @pk
   ORDER BY c.ts DESC
   ```

3. **Analyze query metrics**
   - Use `x-ms-request-charge` header to monitor RU consumption
   - Use `x-ms-documentdb-query-metrics` for detailed execution stats
   - Target: Most queries should be single-partition and <10 RU

---

## Excluding Vectors from Query Results

### Strategy 1: Explicit Projection (Recommended)

The most reliable and performant approach:

```sql
-- Define the exact columns to return, omitting vector fields
SELECT 
    c.id,
    c.session_id,
    c.user_id,
    c.sequence,
    c.turn,
    c.role,
    c.content,
    c.ts,
    c.embedding_model,
    c.vector_metadata
FROM c
WHERE c.partition_key = @pk
```

**Advantages:**
- Full control over returned data
- Cosmos DB only materializes requested properties
- Minimum RU cost
- Clear contract in code

**Disadvantages:**
- Must update queries when adding new non-vector columns
- Verbose for tables with many columns

### Strategy 2: Application-Layer Stripping

If you must use broader queries, strip vectors in application code:

```python
def _strip_vectors(doc: dict) -> dict:
    """Remove vector columns from document before returning."""
    vector_columns = [
        'user_query_vector',
        'assistant_response_vector', 
        'assistant_thinking_vector',
        'tool_output_vector'
    ]
    return {k: v for k, v in doc.items() if k not in vector_columns}
```

**Note:** This still consumes RUs for reading vectors—use only as a fallback.

### Strategy 3: Stored Procedures for Complex Operations

Stored procedures run server-side within a single partition and can filter results before returning:

```javascript
// Stored procedure to get session messages without vectors
function getSessionMessages(sessionId, limit) {
    var context = getContext();
    var container = context.getCollection();
    var response = context.getResponse();
    
    var query = {
        query: "SELECT c.id, c.session_id, c.sequence, c.turn, c.role, " +
               "c.content, c.ts, c.embedding_model, c.vector_metadata " +
               "FROM c WHERE c.session_id = @sessionId " +
               "ORDER BY c.sequence",
        parameters: [{ name: "@sessionId", value: sessionId }]
    };
    
    var accept = container.queryDocuments(
        container.getSelfLink(),
        query,
        { maxItemCount: limit },
        function(err, items) {
            if (err) throw err;
            response.setBody(items);
        }
    );
    
    if (!accept) throw new Error("Query not accepted");
}
```

**When to use stored procedures:**
- Complex multi-step operations within a partition
- When you need atomicity guarantees
- Batch processing with filtering

**Caution:** Stored procedures still pay RU for underlying reads. They reduce network transfer but not storage reads.

### Strategy 4: UDFs (Not Recommended for Projection)

User-Defined Functions compute scalar values per document and **add RU overhead** without leveraging indexes. They're not suitable for projection/filtering of large fields.

```javascript
// DON'T do this - inefficient for vector exclusion
function stripVectors(doc) {
    delete doc.user_query_vector;
    delete doc.assistant_response_vector;
    // ... UDFs add per-document overhead
    return doc;
}
```

**Recommendation:** Use native JSON projection instead of UDFs for field exclusion.

---

## Data Setup and Schema Design

### Container Design Pattern

For session/transcript/event data, use a **single container with type discriminator**:

```json
// Session document
{
    "id": "session_abc123",
    "type": "session",
    "partition_key": "user123|project-slug|session_abc123",
    "user_id": "user123",
    "session_id": "abc123",
    "project_slug": "project-slug",
    "created": "2026-01-15T10:00:00Z",
    "updated": "2026-01-15T12:00:00Z",
    "turn_count": 15,
    "schemaVersion": 2
}

// Transcript document  
{
    "id": "session_abc123_msg_0001",
    "type": "transcript",
    "partition_key": "user123|project-slug|session_abc123",
    "user_id": "user123",
    "session_id": "abc123",
    "sequence": 1,
    "role": "user",
    "content": "Hello, help me with...",
    "ts": "2026-01-15T10:01:00Z",
    "user_query_vector": [0.1, 0.2, ...],  // 3072 dimensions
    "embedding_model": "text-embedding-3-large",
    "schemaVersion": 2
}
```

**Benefits:**
- Single query fetches session + messages
- All related data shares partition key
- Simplified indexing policy
- Aligned with event-sourcing patterns

### Indexing Policy Configuration

```json
{
    "indexingMode": "consistent",
    "automatic": true,
    "includedPaths": [
        { "path": "/user_id/?" },
        { "path": "/session_id/?" },
        { "path": "/type/?" },
        { "path": "/ts/?" },
        { "path": "/sequence/?" },
        { "path": "/role/?" },
        { "path": "/turn/?" }
    ],
    "excludedPaths": [
        { "path": "/content/*" },           // Don't index full content text
        { "path": "/user_query_vector/*" }, // Handled by vector index
        { "path": "/assistant_response_vector/*" },
        { "path": "/assistant_thinking_vector/*" },
        { "path": "/tool_output_vector/*" },
        { "path": "/*" }                    // Exclude everything else by default
    ],
    "vectorIndexes": [
        { "path": "/user_query_vector", "type": "quantizedFlat" },
        { "path": "/assistant_response_vector", "type": "quantizedFlat" },
        { "path": "/assistant_thinking_vector", "type": "quantizedFlat" },
        { "path": "/tool_output_vector", "type": "quantizedFlat" }
    ]
}
```

**Key Points:**
- Explicitly include only queried scalar properties
- Exclude large text content from standard indexes (use full-text search separately)
- Exclude vector arrays from standard indexing (use vector indexes)
- Default exclude everything to prevent indexing new unknown fields

---

## Vector Indexing Configuration

### Vector Embedding Policy

```python
vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/user_query_vector",
            "dataType": "float32",
            "dimensions": 3072,  # text-embedding-3-large
            "distanceFunction": "cosine"
        },
        {
            "path": "/assistant_response_vector",
            "dataType": "float32", 
            "dimensions": 3072,
            "distanceFunction": "cosine"
        },
        {
            "path": "/assistant_thinking_vector",
            "dataType": "float32",
            "dimensions": 3072,
            "distanceFunction": "cosine"
        },
        {
            "path": "/tool_output_vector",
            "dataType": "float32",
            "dimensions": 3072,
            "distanceFunction": "cosine"
        }
    ]
}
```

### Choosing Index Type

| Index Type | Best For | Trade-offs |
|------------|----------|------------|
| `flat` | <1,000 vectors per partition | Exact search, higher RU |
| `quantizedFlat` | 1,000 - 50,000 vectors per partition | Compressed storage, good recall |
| `diskANN` | >50,000 vectors per partition | Approximate search, lowest latency at scale |

**For session storage:** `quantizedFlat` is typically optimal—sessions have moderate vector counts and benefit from compression.

### Vector Search Query Pattern

```sql
-- Vector search with TOP N (required for vector search)
SELECT TOP 50
    c.id,
    c.session_id,
    c.sequence,
    c.content,
    c.ts,
    VectorDistance(c.user_query_vector, @queryVector) AS similarity
FROM c
WHERE c.partition_key = @partitionKey
ORDER BY VectorDistance(c.user_query_vector, @queryVector)
```

**Critical:** Always include `TOP N` clause with vector search to bound the search space.

---

## Partition Key Strategy

### Hierarchical Partition Keys (Recommended)

For user-scoped session data, use hierarchical partition keys:

```python
# Three-level hierarchy
partition_key_definition = {
    "paths": ["/user_id", "/project_slug", "/session_id"],
    "kind": "MultiHash"
}
```

**Access Patterns:**

| Query Scope | Partition Key Prefix | Cost |
|-------------|---------------------|------|
| All user sessions | `/user_id` | Low (single user's partitions) |
| User's project sessions | `/user_id`, `/project_slug` | Lower (narrower scope) |
| Single session | `/user_id`, `/project_slug`, `/session_id` | Lowest (single partition) |

### Composite Partition Key (Alternative)

If hierarchical keys aren't available, use a composite string:

```python
partition_key = f"{user_id}|{project_slug}|{session_id}"
```

**Trade-off:** Must always provide full key for queries, but ensures data locality.

---

## Schema Evolution Patterns

### Schema Versioning

Add a `schemaVersion` field to all documents:

```json
{
    "id": "...",
    "schemaVersion": 2,
    // ... other fields
}
```

### Migration Strategy

1. **Add new fields with defaults** - Always backward compatible
2. **Read-time transformation** - Handle multiple versions in application
3. **Background migration** - Use Change Feed to backfill

```python
def normalize_document(doc: dict) -> dict:
    """Transform document to latest schema version."""
    version = doc.get('schemaVersion', 1)
    
    if version < 2:
        # Add fields introduced in v2
        doc['vector_metadata'] = doc.get('vector_metadata', {})
        doc['schemaVersion'] = 2
        
    return doc
```

### Change Feed for Migrations

```python
async def migrate_documents(container):
    """Background migration using Change Feed."""
    async for changes in container.query_items_change_feed():
        for doc in changes:
            if doc.get('schemaVersion', 1) < CURRENT_VERSION:
                migrated = normalize_document(doc)
                await container.upsert_item(migrated)
```

---

## Implementation Recommendations

### 1. Query Layer Abstraction

Create a query builder that enforces vector exclusion:

```python
class CosmosQueryBuilder:
    """Build queries with automatic vector exclusion."""
    
    VECTOR_COLUMNS = [
        'user_query_vector',
        'assistant_response_vector',
        'assistant_thinking_vector', 
        'tool_output_vector'
    ]
    
    TRANSCRIPT_COLUMNS = [
        'id', 'session_id', 'user_id', 'sequence', 'turn',
        'role', 'content', 'ts', 'embedding_model', 'vector_metadata'
    ]
    
    @classmethod
    def select_transcript(cls, include_vectors: bool = False) -> str:
        """Generate SELECT clause for transcripts."""
        columns = cls.TRANSCRIPT_COLUMNS.copy()
        if include_vectors:
            columns.extend(cls.VECTOR_COLUMNS)
        return ', '.join(f'c.{col}' for col in columns)
```

### 2. Connection Configuration

```python
cosmos_config = {
    "endpoint": os.environ["COSMOS_ENDPOINT"],
    "credential": DefaultAzureCredential(),
    "database": "session-storage",
    "container": "transcripts",
    "consistency_level": "Session",  # Good balance for sessions
    "max_retry_attempts": 3,
    "preferred_regions": ["West US 2"],  # Closest region
}
```

### 3. Monitoring RU Consumption

```python
async def execute_query_with_metrics(container, query, params):
    """Execute query and log RU consumption."""
    results = []
    total_ru = 0
    
    async for page in container.query_items(
        query=query,
        parameters=params,
        populate_query_metrics=True
    ).by_page():
        results.extend(page)
        total_ru += page.get_response_headers().get('x-ms-request-charge', 0)
    
    logger.debug(f"Query consumed {total_ru} RUs for {len(results)} results")
    return results
```

### 4. Best Practices Checklist

- [ ] Always use explicit property projection (no `SELECT *`)
- [ ] Include partition key in all queries
- [ ] Use `TOP N` with vector search
- [ ] Configure indexing policy to exclude vector paths from standard indexes
- [ ] Add `schemaVersion` to all documents
- [ ] Monitor RU consumption in production
- [ ] Use hierarchical partition keys for user-scoped data
- [ ] Keep item sizes small (<5KB excluding vectors where possible)
- [ ] Strip vectors in application layer as fallback defense

---

## References

- [SELECT - Azure Cosmos DB for NoSQL](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/query/select)
- [Vector Search in Azure Cosmos DB](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/vector-search)
- [Hierarchical Partition Keys](https://learn.microsoft.com/en-us/azure/cosmos-db/hierarchical-partition-keys)
- [Schema Versioning Design Pattern](https://devblogs.microsoft.com/cosmosdb/azure-cosmos-db-design-patterns-part-9-schema-versioning/)
- [Large Item Size Design Patterns](https://devblogs.microsoft.com/cosmosdb/4-design-patterns-to-deal-with-large-item-sizes/)
- [Query Performance Tips](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/performance-tips-query-sdk)
