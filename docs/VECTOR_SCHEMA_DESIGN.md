# Vector Schema Design - Multi-Embedding Strategy

**Purpose**: Design document for how embeddings are stored and what content they represent.

**Key Insight**: Different content types need different embeddings for effective search.

---

## The Problem with Single Embedding Column

**Current naive approach**:
```sql
CREATE TABLE transcripts (
    ...
    embedding FLOAT[3072],  -- Single vector for everything
    embedding_model VARCHAR
)
```

**Issues**:
1. ❌ Can't search "only in assistant responses" vs "only in thinking"
2. ❌ Can't weight different content types differently
3. ❌ Loses information about what the vector represents
4. ❌ Mixed signal (user question + assistant reasoning + tool output all look the same)

---

## Recommended Design: Multi-Vector Schema

### Transcripts Table Schema

```sql
CREATE TABLE transcripts (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    sequence INTEGER NOT NULL,
    
    -- Original Amplifier fields
    role VARCHAR NOT NULL,  -- "user", "assistant", "tool"
    content JSON,           -- Full original content (preserved as-is)
    turn INTEGER,
    ts TIMESTAMP,
    
    -- Extracted text for display/search
    text_content TEXT,      -- Combined readable text
    
    -- Multiple embedding vectors (each for specific content type)
    user_query_embedding FLOAT[3072],      -- User questions/input
    assistant_response_embedding FLOAT[3072],  -- Assistant text responses
    assistant_thinking_embedding FLOAT[3072],  -- Assistant reasoning
    tool_output_embedding FLOAT[3072],     -- Tool execution results
    
    -- Metadata about embeddings
    embedding_model VARCHAR,
    embedding_metadata JSON,  -- {has_user_query, has_response, has_thinking, has_tool_output}
    
    synced_at TIMESTAMP
)
```

### Why Multiple Embedding Columns?

**Enables precise search queries**:

```sql
-- Search ONLY user questions
SELECT * FROM transcripts
WHERE user_query_embedding IS NOT NULL
  AND array_cosine_similarity(user_query_embedding, @query) > 0.8

-- Search ONLY assistant reasoning
SELECT * FROM transcripts
WHERE assistant_thinking_embedding IS NOT NULL
  AND array_cosine_similarity(assistant_thinking_embedding, @query) > 0.8

-- Search responses but NOT thinking
SELECT * FROM transcripts
WHERE assistant_response_embedding IS NOT NULL
  AND array_cosine_similarity(assistant_response_embedding, @query) > 0.8

-- Search everything (hybrid across all vectors)
SELECT * FROM transcripts
WHERE (
    array_cosine_similarity(user_query_embedding, @query) > 0.7 OR
    array_cosine_similarity(assistant_response_embedding, @query) > 0.7 OR
    array_cosine_similarity(assistant_thinking_embedding, @query) > 0.7 OR
    array_cosine_similarity(tool_output_embedding, @query) > 0.7
)
```

---

## Field Population Logic

### User Messages

```python
if message["role"] == "user":
    user_query_embedding = embed(message["content"])
    # All other embeddings: NULL
```

**Result**:
```
sequence=0, role=user, user_query_embedding=[0.1, 0.2, ...], others=NULL
```

### Assistant Messages (Complex)

```python
if message["role"] == "assistant":
    content_blocks = message["content"]  # Array
    
    # Extract thinking
    thinking_parts = [b["thinking"] for b in content_blocks if b["type"] == "thinking"]
    if thinking_parts:
        assistant_thinking_embedding = embed("\n".join(thinking_parts))
    
    # Extract text responses
    text_parts = [b["text"] for b in content_blocks if b["type"] == "text"]
    if text_parts:
        assistant_response_embedding = embed("\n".join(text_parts))
    
    # Skip tool_call blocks - not embeddable
```

**Result**:
```
sequence=1, role=assistant,
    assistant_thinking_embedding=[0.3, 0.4, ...],
    assistant_response_embedding=[0.5, 0.6, ...],
    others=NULL
```

### Tool Messages

```python
if message["role"] == "tool":
    # Tool output (may be large!)
    tool_content = message["content"]
    if isinstance(tool_content, str) and len(tool_content) < 10000:
        tool_output_embedding = embed(tool_content)
    # All other embeddings: NULL
```

**Result**:
```
sequence=2, role=tool, tool_output_embedding=[0.7, 0.8, ...], others=NULL
```

---

## Events Schema - Structured Search (No Embeddings)

### Events Table Schema

```sql
CREATE TABLE events (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    sequence INTEGER NOT NULL,
    
    -- Event fields
    event VARCHAR NOT NULL,     -- "llm.request", "tool.call", etc.
    ts TIMESTAMP NOT NULL,
    lvl VARCHAR,                -- "INFO", "DEBUG", "ERROR"
    turn INTEGER,
    
    -- Event data (may be large)
    data JSON,
    data_truncated BOOLEAN,
    data_size_bytes INTEGER,
    
    -- Extracted fields for structured search
    tool_name VARCHAR,          -- Extracted from data.tool
    error_type VARCHAR,         -- Extracted from data.error_type
    model_used VARCHAR,         -- Extracted from data.model
    
    synced_at TIMESTAMP
)

-- Indexes for structured search
CREATE INDEX idx_events_type ON events(user_id, event, ts);
CREATE INDEX idx_events_tool ON events(user_id, tool_name, ts);
CREATE INDEX idx_events_level ON events(user_id, lvl, ts);
```

### Why NO Embeddings for Events?

**Reasons**:
1. **Events are telemetry** - Structured data optimized for filtering
2. **High volume** - Thousands of events per session
3. **Not conversational** - Event types are categorical, not semantic
4. **Storage cost** - Would 10x embedding storage requirements
5. **Search patterns** - Users search by event type, tool, level (structured)

**Search queries for events** (all structured, no embeddings needed):

```python
# Find all bash tool executions
await storage.search_events(
    options=EventSearchOptions(
        event_type="tool.call",
        tool_name="bash"
    )
)

# Find all errors
await storage.search_events(
    options=EventSearchOptions(
        level="ERROR"
    )
)

# Find LLM requests for specific model
await storage.search_events(
    options=EventSearchOptions(
        event_type="llm.request",
        filters=SearchFilters(
            start_date="2024-01-01",
            end_date="2024-02-01"
        )
    )
)
```

**No semantic search needed** - these are precise, structured queries.

**Future consideration**: If we want to search error messages semantically:
```python
# Extract error message text from data field
error_message = event["data"].get("error", {}).get("message", "")
# Could embed this for "find similar errors"
# But this is rare and can be added later
```

---

## Embedding Metadata Strategy

### Option A: Separate Columns (Recommended)

**Schema**:
```sql
user_query_embedding FLOAT[3072],
assistant_response_embedding FLOAT[3072],
assistant_thinking_embedding FLOAT[3072],
tool_output_embedding FLOAT[3072],
embedding_model VARCHAR
```

**Pros**:
- ✅ Clear what each vector represents
- ✅ Easy to search specific content types
- ✅ No parsing needed to understand vector source
- ✅ Standard SQL NULL handling

**Cons**:
- Sparse table (most rows have 1-2 NULLs)
- Slightly more storage overhead

### Option B: Single Column + Metadata

**Schema**:
```sql
embedding FLOAT[3072],
embedding_source VARCHAR,  -- "user_query", "assistant_response", "assistant_thinking", "tool_output"
embedding_model VARCHAR
```

**Pros**:
- Single vector column
- Dense table

**Cons**:
- ❌ Can't search multiple content types in single query efficiently
- ❌ Need to parse embedding_source to filter
- ❌ More complex query logic

### Option C: Separate Rows per Content Type

**Schema** (each content type = separate row):
```sql
sequence INTEGER,          -- Logical message sequence
sub_sequence INTEGER,      -- 0=thinking, 1=response, 2=tool_output
content_type VARCHAR,      -- "user_query", "assistant_thinking", "assistant_response"
text_content TEXT,
embedding FLOAT[3072],
```

**Pros**:
- Clean separation
- Easy to search by content type
- Standard row-based approach

**Cons**:
- Breaks 1:1 mapping with transcript.jsonl lines
- More rows (3-4x)

**Recommendation**: **Option A** - Multiple embedding columns

---

## Updated Schema Design

### Transcripts Table (Final)

```sql
CREATE TABLE transcripts (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    project_slug VARCHAR,
    sequence INTEGER NOT NULL,
    
    -- Original Amplifier fields
    role VARCHAR NOT NULL,
    content JSON NOT NULL,  -- Original content (preserved as-is)
    turn INTEGER,
    ts TIMESTAMP,
    
    -- VECTOR COLUMNS (multiple embeddings per message)
    user_query_vector FLOAT[3072],           -- For role=user content
    assistant_response_vector FLOAT[3072],   -- For role=assistant text blocks
    assistant_thinking_vector FLOAT[3072],   -- For role=assistant thinking blocks
    tool_output_vector FLOAT[3072],          -- For role=tool content
    
    -- Vector metadata
    embedding_model VARCHAR,
    vector_metadata JSON,  -- {user_query: bool, assistant_response: bool, assistant_thinking: bool, tool_output: bool}
    
    -- Sync metadata
    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_session (user_id, session_id, sequence),
    INDEX idx_user_query_vector USING HNSW (user_query_vector),
    INDEX idx_assistant_response_vector USING HNSW (assistant_response_vector),
    INDEX idx_assistant_thinking_vector USING HNSW (assistant_thinking_vector),
    INDEX idx_tool_output_vector USING HNSW (tool_output_vector)
)
```

### Events Table (No Embeddings)

```sql
CREATE TABLE events (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    sequence INTEGER NOT NULL,
    
    -- Event fields
    event VARCHAR NOT NULL,
    ts TIMESTAMP NOT NULL,
    lvl VARCHAR,
    turn INTEGER,
    data JSON,
    
    -- Extracted fields for fast structured search
    tool_name VARCHAR,      -- FROM data.tool
    error_type VARCHAR,     -- FROM data.error_type  
    model_used VARCHAR,     -- FROM data.model
    
    -- Size tracking
    data_truncated BOOLEAN,
    data_size_bytes INTEGER,
    synced_at TIMESTAMP,
    
    -- Indexes for structured search (NO vector indexes)
    INDEX idx_event_type (user_id, event, ts),
    INDEX idx_tool_name (user_id, tool_name, ts),
    INDEX idx_level (user_id, lvl, ts)
)
```

**Note**: Events use **structured search only** - no embedding columns needed.

---

## Extraction Functions

### For User Messages

```python
def extract_user_content(message: dict) -> dict[str, str]:
    """Extract content from user message."""
    return {
        "user_query": message.get("content", "")
    }
```

**Produces**: Single `user_query` text for embedding

### For Assistant Messages

```python
def extract_assistant_content(message: dict) -> dict[str, str]:
    """Extract content from assistant message with content array."""
    content = message.get("content", [])
    
    result = {
        "assistant_response": "",
        "assistant_thinking": ""
    }
    
    if isinstance(content, str):
        result["assistant_response"] = content
        return result
    
    if isinstance(content, list):
        thinking_parts = []
        text_parts = []
        
        for block in content:
            if block.get("type") == "thinking":
                thinking_parts.append(block.get("thinking", ""))
            elif block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            # Skip tool_call blocks
        
        result["assistant_thinking"] = "\n\n".join(thinking_parts)
        result["assistant_response"] = "\n\n".join(text_parts)
    
    return result
```

**Produces**: Two separate texts - `assistant_response` and `assistant_thinking`

### For Tool Messages

```python
def extract_tool_content(message: dict) -> dict[str, str]:
    """Extract content from tool message."""
    content = message.get("content", "")
    
    # Limit very large tool outputs
    if isinstance(content, str):
        if len(content) > 10000:
            content = content[:10000] + "... (truncated)"
    else:
        content = str(content)[:10000]
    
    return {
        "tool_output": content
    }
```

**Produces**: Single `tool_output` text (truncated if huge)

---

## Complete Extraction Pipeline

```python
def extract_all_embeddable_content(message: dict) -> dict[str, str]:
    """
    Extract all embeddable content from a transcript message.
    
    Returns dict with keys:
        - user_query: str | None
        - assistant_response: str | None
        - assistant_thinking: str | None
        - tool_output: str | None
    
    Only the relevant key(s) for the message role will be populated.
    """
    role = message.get("role")
    
    if role == "user":
        return {
            "user_query": message.get("content", ""),
            "assistant_response": None,
            "assistant_thinking": None,
            "tool_output": None
        }
    
    elif role == "assistant":
        extracted = extract_assistant_content(message)
        return {
            "user_query": None,
            "assistant_response": extracted.get("assistant_response") or None,
            "assistant_thinking": extracted.get("assistant_thinking") or None,
            "tool_output": None
        }
    
    elif role == "tool":
        extracted = extract_tool_content(message)
        return {
            "user_query": None,
            "assistant_response": None,
            "assistant_thinking": None,
            "tool_output": extracted.get("tool_output") or None
        }
    
    else:
        # Unknown role - no embeddings
        return {
            "user_query": None,
            "assistant_response": None,
            "assistant_thinking": None,
            "tool_output": None
        }
```

---

## Embedding Generation During Sync

### Updated sync_transcript_lines Logic

```python
async def sync_transcript_lines(
    self,
    user_id: str,
    host_id: str,
    project_slug: str,
    session_id: str,
    lines: list[dict],
    start_sequence: int = 0,
    embeddings: dict[str, list[list[float]]] | None = None,  # Changed!
) -> int:
    """
    Sync transcript lines with multi-vector embeddings.
    
    Args:
        embeddings: Optional pre-computed embeddings as dict:
            {
                "user_query": [[0.1, ...], [0.2, ...], ...],
                "assistant_response": [[0.3, ...], ...],
                "assistant_thinking": [[0.4, ...], ...],
                "tool_output": [[0.5, ...], ...]
            }
            Each list matches length of lines (use None for non-applicable)
    """
    if not lines:
        return 0
    
    # Generate embeddings if not provided
    if embeddings is None and self.embedding_provider:
        embeddings = await self._generate_embeddings_for_lines(lines)
    
    # Store each line with its relevant embeddings
    for i, line in enumerate(lines):
        sequence = start_sequence + i
        role = line.get("role")
        
        # Determine which embeddings to store
        user_query_vec = embeddings.get("user_query", [None]*len(lines))[i]
        assistant_response_vec = embeddings.get("assistant_response", [None]*len(lines))[i]
        assistant_thinking_vec = embeddings.get("assistant_thinking", [None]*len(lines))[i]
        tool_output_vec = embeddings.get("tool_output", [None]*len(lines))[i]
        
        # Insert/update with all vectors
        await self._store_transcript(
            sequence=sequence,
            line=line,
            user_query_vector=user_query_vec,
            assistant_response_vector=assistant_response_vec,
            assistant_thinking_vector=assistant_thinking_vec,
            tool_output_vector=tool_output_vec
        )
```

### Batch Embedding Generation

```python
async def _generate_embeddings_for_lines(
    self, lines: list[dict]
) -> dict[str, list[list[float] | None]]:
    """
    Generate all embeddings for a batch of transcript lines.
    
    Returns dict with one list per embedding type, where each list
    has one entry per input line (None if not applicable).
    """
    # Extract all embeddable content
    user_queries = []
    assistant_responses = []
    assistant_thinkings = []
    tool_outputs = []
    
    for line in lines:
        extracted = extract_all_embeddable_content(line)
        user_queries.append(extracted["user_query"])
        assistant_responses.append(extracted["assistant_response"])
        assistant_thinkings.append(extracted["assistant_thinking"])
        tool_outputs.append(extracted["tool_output"])
    
    # Generate embeddings for each type (only non-None values)
    user_query_embeddings = await self._embed_non_none(user_queries)
    assistant_response_embeddings = await self._embed_non_none(assistant_responses)
    assistant_thinking_embeddings = await self._embed_non_none(assistant_thinkings)
    tool_output_embeddings = await self._embed_non_none(tool_outputs)
    
    return {
        "user_query": user_query_embeddings,
        "assistant_response": assistant_response_embeddings,
        "assistant_thinking": assistant_thinking_embeddings,
        "tool_output": tool_output_embeddings,
    }

async def _embed_non_none(self, texts: list[str | None]) -> list[list[float] | None]:
    """Generate embeddings only for non-None texts."""
    # Collect non-None texts with their indices
    texts_to_embed = [(i, text) for i, text in enumerate(texts) if text]
    
    if not texts_to_embed:
        return [None] * len(texts)
    
    # Generate embeddings (single batch call!)
    just_texts = [text for _, text in texts_to_embed]
    embeddings = await self.embedding_provider.embed_batch(just_texts)
    
    # Map back to original positions
    result = [None] * len(texts)
    for (original_idx, _), embedding in zip(texts_to_embed, embeddings):
        result[original_idx] = embedding
    
    return result
```

**Efficiency**: Still uses batch operations! Single API call per content type.

---

## Search Query Examples

### Search Only User Questions

```python
results = await storage.search_transcripts(
    user_id="user-123",
    options=TranscriptSearchOptions(
        query="how do I implement vector search?",
        search_type="semantic",
        search_in_user=True,           # Only user messages
        search_in_assistant=False,
        search_in_thinking=False
    )
)

# SQL generated:
# SELECT * WHERE role = 'user' 
#   AND user_query_vector IS NOT NULL
#   AND array_cosine_similarity(user_query_vector, @query) > threshold
```

### Search Only Assistant Reasoning

```python
results = await storage.search_transcripts(
    user_id="user-123",
    options=TranscriptSearchOptions(
        query="why did it choose DuckDB",
        search_type="semantic",
        search_in_user=False,
        search_in_assistant=False,
        search_in_thinking=True  # Only thinking blocks!
    )
)

# SQL generated:
# SELECT * WHERE role = 'assistant'
#   AND assistant_thinking_vector IS NOT NULL
#   AND array_cosine_similarity(assistant_thinking_vector, @query) > threshold
```

### Hybrid Search Across Everything

```python
results = await storage.search_transcripts(
    user_id="user-123",
    options=TranscriptSearchOptions(
        query="vector database comparison",
        search_type="hybrid",
        search_in_user=True,
        search_in_assistant=True,
        search_in_thinking=True
    )
)

# Searches ALL vector columns:
# user_query_vector, assistant_response_vector, assistant_thinking_vector, tool_output_vector
# Merges results, applies MMR re-ranking
```

---

## Storage Overhead Analysis

### Per Message Storage

| Message Type | Original Data | Vectors | Total Storage |
|--------------|---------------|---------|---------------|
| **User** | ~500 bytes | 1 vector (12 KB) | ~12.5 KB |
| **Assistant (simple)** | ~500 bytes | 1 vector (12 KB) | ~12.5 KB |
| **Assistant (with thinking)** | ~2 KB | 2 vectors (24 KB) | ~26 KB |
| **Tool** | ~5 KB | 1 vector (12 KB) | ~17 KB |

**Average**: ~15-20 KB per message (vs ~500 bytes without embeddings)

**For 1000-message session**:
- Without embeddings: ~500 KB
- With embeddings: ~15-20 MB

**Cost consideration**: Embeddings are 30-40x the storage size!

**Mitigation**:
- Use quantized indexes (Cosmos: 128 bytes vs 12 KB)
- Don't embed tool outputs >10KB
- Use DuckDB for local (no cloud storage costs)

---

## Summary

### Vector Columns Defined

| Column Name | Contains | From Role | From Field |
|-------------|----------|-----------|------------|
| `user_query_vector` | User questions/input | user | `content` (string) |
| `assistant_response_vector` | Assistant's visible responses | assistant | `content[].text` blocks |
| `assistant_thinking_vector` | Assistant's reasoning | assistant | `content[].thinking` blocks |
| `tool_output_vector` | Tool execution results | tool | `content` (string, truncated) |

### Events Search Strategy

**No embeddings** - Use structured search:
- Filter by `event` type
- Filter by `tool_name` (extracted from data)
- Filter by `lvl` (ERROR, WARNING, etc.)
- Filter by date range

**Rationale**: Events are telemetry, not conversation - structured queries are more appropriate.

---

## Next Steps

1. **Update schema** in all three backends (DuckDB, SQLite, Cosmos)
2. **Implement extraction functions** (user, assistant, tool)
3. **Update embedding generation** to produce multiple vectors
4. **Update search queries** to use appropriate vector columns
5. **Add tests** with real assistant message structure
6. **Update documentation** with new schema

**This is a significant change but necessary for correct semantic search over real Amplifier sessions!**
