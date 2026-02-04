# Multi-Vector Embedding Implementation - Complete Guide

**Status**: ✅ Implemented across all 3 backends  
**Version**: 0.2.0 (breaking change from 0.1.0)  
**Tests**: 159 passing, 19 skipped  

---

## Overview

The enhanced session storage now supports **4 separate vector embeddings per message** instead of a single embedding. This enables precise semantic search across different content types in Amplifier conversations.

### Why Multi-Vector?

Amplifier assistant messages contain **multiple content types** in a single message:
- **Thinking blocks**: Internal reasoning (often 1000s of tokens)
- **Text blocks**: User-visible responses
- **Tool calls**: Function invocations

Embedding these together loses semantic specificity. With multi-vector:
- Search **only** in reasoning: "How did the AI approach this problem?"
- Search **only** in responses: "What did it tell me about X?"
- Search **only** in user queries: "When did I ask about Y?"

---

## Schema Design

### Transcript Table - 4 Vector Columns

**DuckDB**:
```sql
CREATE TABLE transcripts (
    ...
    content JSON,  -- Original content preserved as JSON
    
    -- Four separate vector columns
    user_query_vector FLOAT[3072],           -- From user messages
    assistant_response_vector FLOAT[3072],   -- From assistant text blocks
    assistant_thinking_vector FLOAT[3072],   -- From assistant thinking blocks
    tool_output_vector FLOAT[3072],          -- From tool result messages
    
    embedding_model VARCHAR,
    vector_metadata JSON  -- Tracks which vectors are populated
)

-- Four HNSW indexes for O(log n) search
CREATE INDEX idx_user_query_vector ON transcripts USING HNSW (user_query_vector);
CREATE INDEX idx_assistant_response_vector ON transcripts USING HNSW (assistant_response_vector);
CREATE INDEX idx_assistant_thinking_vector ON transcripts USING HNSW (assistant_thinking_vector);
CREATE INDEX idx_tool_output_vector ON transcripts USING HNSW (tool_output_vector);
```

**SQLite**:
```sql
CREATE TABLE transcripts (
    ...
    content_json TEXT,  -- JSON serialized
    
    -- Four vector columns as JSON TEXT
    user_query_vector_json TEXT,
    assistant_response_vector_json TEXT,
    assistant_thinking_vector_json TEXT,
    tool_output_vector_json TEXT,
    
    embedding_model TEXT,
    vector_metadata TEXT
)
```

**Cosmos DB**:
```json
{
  "id": "session_msg_0",
  "content": { ... },  // Preserved as-is
  
  // Four vector arrays
  "user_query_vector": [0.1, 0.2, ..., 3072 floats],
  "assistant_response_vector": [0.3, 0.4, ...],
  "assistant_thinking_vector": [0.5, 0.6, ...],
  "tool_output_vector": [0.7, 0.8, ...],
  
  "vector_metadata": {
    "has_user_query": true,
    "has_assistant_response": true,
    "has_assistant_thinking": true,
    "has_tool_output": false
  },
  "embedding_model": "text-embedding-3-large"
}
```

---

## Content Extraction Rules

### User Messages
**Structure**: Simple string
```json
{"role": "user", "content": "How do I use vector search?"}
```
**Extraction**: `content` → `user_query_vector`

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
- All `thinking` blocks → joined → `assistant_thinking_vector`
- All `text` blocks → joined → `assistant_response_vector`
- Tool calls **NOT embedded** (no semantic value)

### Tool Messages
**Structure**: String output
```json
{"role": "tool", "content": "{file contents or tool result}"}
```
**Extraction**: `content` (truncated to 1000 chars) → `tool_output_vector`

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
2. Generates 3 embeddings in a single API call (batch)
3. Stores each in appropriate vector column
4. `tool_output_vector` remains NULL (no tool message)

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

### Search Strategy: GREATEST/LEAST

**DuckDB/SQLite** (uses GREATEST for max similarity):
```sql
SELECT ...,
       GREATEST(
           CASE WHEN user_query_vector IS NOT NULL 
                THEN array_cosine_similarity(user_query_vector, query) 
                ELSE 0.0 END,
           CASE WHEN assistant_response_vector IS NOT NULL 
                THEN array_cosine_similarity(assistant_response_vector, query) 
                ELSE 0.0 END,
           ...
       ) AS similarity
ORDER BY similarity DESC
```

**Cosmos DB** (uses LEAST for min distance):
```sql
SELECT ...,
       LEAST(
           (IS_DEFINED(c.user_query_vector) 
               ? VectorDistance(c.user_query_vector, @query_vector) 
               : 1.0),
           (IS_DEFINED(c.assistant_response_vector) 
               ? VectorDistance(c.assistant_response_vector, @query_vector) 
               : 1.0),
           ...
       ) AS best_distance
ORDER BY best_distance
```

---

## Performance Characteristics

### Embedding Generation (During Ingestion)

**Single-vector (old)**:
```
10 messages → 10 API calls (if no cache)
```

**Multi-vector (new)**:
```
10 messages with mixed content:
- 10 user queries
- 10 assistant responses  
- 10 assistant thinking blocks
- 3 tool outputs
= 33 texts to embed → 1-3 API calls (batched!)
```

**Efficiency gain**: ~70% fewer API calls due to intelligent batching

### Search Performance

**DuckDB**:
- **With HNSW indexes**: O(log n) per vector column
- **4 columns searched**: ~4 × O(log n) = still O(log n)
- **10k messages**: ~13-15 comparisons per column

**SQLite**:
- **Numpy fallback**: O(n) but searches 4 vectors
- **10k messages**: Loads all 40k vectors, computes max similarity
- **Memory**: ~480 MB for 10k messages (4 × 3072 × 4 bytes × 10k)

**Cosmos DB**:
- **quantizedFlat indexes**: O(n) but optimized with quantization
- **4 indexes**: Queries can target specific vector or all
- **Cost**: RU cost × 4 if searching all vectors

---

## Migration Guide

### Breaking Changes from v0.1.0

**Old schema**:
```sql
content TEXT,
embedding FLOAT[3072]
```

**New schema**:
```sql
content JSON,
user_query_vector FLOAT[3072],
assistant_response_vector FLOAT[3072],
assistant_thinking_vector FLOAT[3072],
tool_output_vector FLOAT[3072]
```

### Migration Steps

1. **No backward compatibility** - Schema is incompatible
2. **Clear old data** or create new database/container
3. **Re-ingest sessions** with new embeddings

**DuckDB/SQLite**:
```bash
# Delete old database
rm sessions.db

# Create new with updated schema (automatic on first run)
python your_ingestion_script.py
```

**Cosmos DB**:
```bash
# Delete old containers
az cosmosdb sql container delete --name transcripts ...

# Re-run initialization (creates containers with new vector indexes)
# The library will auto-create 4 vector indexes
```

---

## Real-World Examples

### Example 1: Finding How AI Reasoned

```python
# User wants to understand how the AI approached a problem
query = "How did you decide to use DuckDB?"
query_vec = await embeddings.embed_text(query)

# Search ONLY in thinking blocks
results = await storage.vector_search(
    user_id="user-123",
    query_vector=query_vec,
    vector_columns=["assistant_thinking"],  # Only reasoning
    top_k=5
)

# Results will show messages where AI *thought* about DuckDB choice
# Even if it didn't mention DuckDB in the response to user
```

### Example 2: Finding User Questions

```python
# Find when user asked about a specific topic
query = "authentication setup"
query_vec = await embeddings.embed_text(query)

results = await storage.vector_search(
    user_id="user-123",
    query_vector=query_vec,
    vector_columns=["user_query"],  # Only user questions
    top_k=10
)

# Returns only user messages asking about authentication
```

### Example 3: Finding Tool Outputs

```python
# Find sessions where specific file content was examined
query = "package.json dependencies"
query_vec = await embeddings.embed_text(query)

results = await storage.vector_search(
    user_id="user-123",
    query_vector=query_vec,
    vector_columns=["tool_output"],  # Only tool results
    top_k=10
)

# Returns tool messages containing package.json content
```

---

## Implementation Details

### Efficient Batch Processing

**The `_embed_non_none()` helper**:
```python
async def _embed_non_none(self, texts: list[str | None]) -> list[list[float] | None]:
    """Generate embeddings only for non-None texts."""
    # Input: ["text1", None, "text2", None, "text3"]
    # Extracts: [(0, "text1"), (2, "text2"), (4, "text3")]
    # Calls API: ["text1", "text2", "text3"] → single batch call
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
- Multiple thinking blocks → joined with "\n\n"
- Multiple text blocks → joined with "\n\n"
- Large tool outputs → truncated to 1000 chars
- Missing/null content → returns None

### Vector Metadata

**Purpose**: Track which vectors exist in each message
```json
{
  "has_user_query": true,
  "has_assistant_response": true,
  "has_assistant_thinking": true,
  "has_tool_output": false
}
```

**Uses**:
- Analytics: Count messages with thinking blocks
- Query optimization: Skip vector columns that are always NULL
- Debugging: Verify embedding generation

---

## Testing Evidence

### Test Coverage

**Content Extraction**: 18 tests ✅
- User message extraction
- Assistant message parsing (thinking + text blocks)
- Tool message truncation
- Real Amplifier message structures
- Edge cases (empty content, missing fields)

**DuckDB Backend**: 23 tests ✅
- Multi-vector generation during sync
- HNSW index usage verification
- Vector search with column filtering
- Hybrid search with MMR

**SQLite Backend**: 28 tests ✅
- Multi-vector generation
- Numpy fallback for vector search
- All CRUD operations
- Content JSON storage

**Schema Validation**: 9 tests ✅
- Amplifier metadata.json compatibility
- transcript.jsonl field mapping
- events.jsonl structure
- Incremental sync patterns

### Real Data Validation

**Tested against**: `~/.amplifier/projects/*/sessions/*/`

**Verified**:
- ✅ Session metadata schema matches
- ✅ Transcript lines schema compatible
- ✅ Event lines structure preserved
- ✅ Complex assistant messages parsed correctly
- ✅ Thinking blocks extracted (found 33 in real session)
- ✅ Tool outputs handled (found 38 in real session)

---

## Backend-Specific Implementation

### DuckDB

**Storage**: Native `FLOAT[3072]` arrays  
**Search**: VSS extension with HNSW indexes  
**Performance**: O(log n) per vector column  

**Vector Search Query**:
```sql
SELECT ...,
       GREATEST(
           CASE WHEN user_query_vector IS NOT NULL 
                THEN array_cosine_similarity(user_query_vector, query) 
                ELSE 0.0 END,
           CASE WHEN assistant_response_vector IS NOT NULL 
                THEN array_cosine_similarity(assistant_response_vector, query) 
                ELSE 0.0 END,
           CASE WHEN assistant_thinking_vector IS NOT NULL 
                THEN array_cosine_similarity(assistant_thinking_vector, query) 
                ELSE 0.0 END,
           CASE WHEN tool_output_vector IS NOT NULL 
                THEN array_cosine_similarity(tool_output_vector, query) 
                ELSE 0.0 END
       ) AS similarity
FROM transcripts
WHERE (user_query_vector IS NOT NULL OR assistant_response_vector IS NOT NULL 
       OR assistant_thinking_vector IS NOT NULL OR tool_output_vector IS NOT NULL)
ORDER BY similarity DESC
```

**Key Technique**: String interpolation for query vector (required for HNSW)
```python
vec_literal = DuckDBBackend._format_vector_literal(query_vector)
# Returns: "[0.1, 0.2, 0.3, ...]::FLOAT[3072]"
```

### SQLite

**Storage**: JSON TEXT serialization  
**Search**: Numpy cosine similarity (brute-force)  
**Performance**: O(n) - acceptable for <10k messages  

**Vector Search Logic**:
```python
# Load all 4 vector types
for row in rows:
    similarities = []
    
    if row['user_query_vector_json']:
        vec = json.loads(row['user_query_vector_json'])
        similarities.append(cosine_similarity(query, vec))
    
    if row['assistant_response_vector_json']:
        vec = json.loads(row['assistant_response_vector_json'])
        similarities.append(cosine_similarity(query, vec))
    
    # ... same for thinking and tool_output
    
    # Use maximum similarity (best match across all vectors)
    max_similarity = max(similarities) if similarities else 0.0
```

### Cosmos DB

**Storage**: JSON arrays in document fields  
**Search**: VectorDistance function with quantizedFlat indexes  
**Performance**: O(n) but distributed and optimized  

**Vector Indexes** (4 separate indexes):
```json
{
  "vectorIndexes": [
    {
      "path": "/user_query_vector",
      "type": "quantizedFlat",
      "dimensions": 3072,
      "quantizationByteSize": 128
    },
    {
      "path": "/assistant_response_vector",
      "type": "quantizedFlat",
      "dimensions": 3072,
      "quantizationByteSize": 128
    },
    // ... same for thinking and tool_output
  ]
}
```

**Vector Search Query**:
```sql
SELECT ...,
       LEAST(
           (IS_DEFINED(c.user_query_vector) 
               ? VectorDistance(c.user_query_vector, @query_vector) 
               : 1.0),
           (IS_DEFINED(c.assistant_response_vector) 
               ? VectorDistance(c.assistant_response_vector, @query_vector) 
               : 1.0),
           (IS_DEFINED(c.assistant_thinking_vector) 
               ? VectorDistance(c.assistant_thinking_vector, @query_vector) 
               : 1.0),
           (IS_DEFINED(c.tool_output_vector) 
               ? VectorDistance(c.tool_output_vector, @query_vector) 
               : 1.0)
       ) AS best_distance
FROM c
WHERE (IS_DEFINED(c.user_query_vector) OR IS_DEFINED(c.assistant_response_vector)
       OR IS_DEFINED(c.assistant_thinking_vector) OR IS_DEFINED(c.tool_output_vector))
ORDER BY best_distance
```

**Note**: Uses distance (lower = better), converted to similarity (higher = better) in results

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

## Search Examples

### Example 1: Debug Session by Reasoning

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

# Results show:
# - Message seq=15: "The authentication code has multiple concerns..."
# - Message seq=23: "Refactoring auth into separate modules would..."
```

### Example 2: Find What AI Told User

"What did the AI say about performance optimization?"

```python
query = "performance optimization techniques"
query_vec = await embeddings.embed_text(query)

# Search response blocks - what user saw
results = await storage.vector_search(
    user_id="user-123",
    query_vector=query_vec,
    vector_columns=["assistant_response"],
    top_k=5
)

# Results show user-visible responses about performance
```

### Example 3: Cross-Content Search

"Find anything related to database schema"

```python
query = "database schema design"
query_vec = await embeddings.embed_text(query)

# Search ALL content types
results = await storage.vector_search(
    user_id="user-123",
    query_vector=query_vec,
    # Default: searches all 4 vectors
    top_k=10
)

# Returns:
# - User queries about database
# - AI responses about schema
# - AI reasoning about design decisions
# - Tool outputs showing database files
```

---

## Verification Commands

### Verify Schema

**DuckDB**:
```bash
uv run python -c "
import asyncio
from amplifier_session_storage.backends import DuckDBBackend, DuckDBConfig

async def main():
    storage = await DuckDBBackend.create(DuckDBConfig(db_path=':memory:'))
    # Check column exists
    result = storage.conn.execute('''
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'transcripts' 
        AND column_name LIKE '%vector%'
    ''').fetchall()
    print('Vector columns:', [r[0] for r in result])

asyncio.run(main())
"
```

**Expected output**:
```
Vector columns: ['user_query_vector', 'assistant_response_vector', 
                 'assistant_thinking_vector', 'tool_output_vector']
```

### Verify Embedding Generation

```bash
uv run python -c "
import asyncio
from amplifier_session_storage.backends import DuckDBBackend, DuckDBConfig
from amplifier_session_storage.embeddings import OpenAIEmbeddings

async def main():
    embeddings = OpenAIEmbeddings.from_env()
    storage = await DuckDBBackend.create(
        DuckDBConfig(db_path=':memory:'),
        embedding_provider=embeddings
    )
    
    # Real assistant message with thinking
    lines = [{
        'role': 'assistant',
        'content': [
            {'type': 'thinking', 'thinking': 'I need to analyze the schema design...'},
            {'type': 'text', 'text': 'Here is my recommendation...'}
        ]
    }]
    
    await storage.sync_transcript_lines(
        user_id='test', host_id='test', project_slug='test',
        session_id='test', lines=lines
    )
    
    # Check vectors stored
    result = storage.conn.execute('''
        SELECT vector_metadata 
        FROM transcripts 
        WHERE session_id = 'test'
    ''').fetchone()
    
    print('Vector metadata:', result[0])

asyncio.run(main())
"
```

**Expected output**:
```
Generated embeddings: 0 user, 1 responses, 1 thinking, 0 tool
Vector metadata: {"has_user_query": false, "has_assistant_response": true, 
                  "has_assistant_thinking": true, "has_tool_output": false}
```

---

## Future Enhancements

### Potential Improvements

1. **Weighted multi-vector search**: 
   - Weight thinking blocks higher for "why" questions
   - Weight responses higher for "what" questions

2. **Vector column auto-selection**:
   - Query classification: "how did you decide?" → search thinking
   - "what did you say?" → search responses

3. **Cross-session semantic clustering**:
   - Group similar thinking patterns across sessions
   - Find recurring debugging strategies

4. **Embedding model migration**:
   - Re-embed with newer models (e.g., ada-003 when released)
   - Compare search quality across models

---

## Commits

Implementation completed across 3 commits:

1. **DuckDB**: Part of `98df6f1` - Initial multi-vector with HNSW
2. **SQLite**: `3cd5ee7` - Multi-vector with numpy fallback
3. **Cosmos DB**: `38bb17b` - Multi-vector with VectorDistance

**Total changes**:
- 3 backend files modified (~1200 lines changed)
- 4 helper methods added
- 18 content extraction tests
- 159 total tests passing

---

## Summary

**Multi-vector embeddings transform session search from**:
- "Find messages about X" (generic)

**To**:
- "Find when user asked about X" (precise)
- "Find how AI reasoned about X" (precise)
- "Find what AI told user about X" (precise)
- "Find which files contained X" (precise)

**All 3 backends** (DuckDB, SQLite, Cosmos) now have this capability with identical APIs, enabling rich semantic search across Amplifier conversation history.
