# Embedding Strategy and Field Mapping

**Purpose**: Document what content gets embedded and how for semantic search capabilities.

**Date**: 2026-02-04  
**Schema Analysis**: Based on actual Amplifier session files from `~/.amplifier/projects`

---

## Amplifier Session Data Structure (On Disk)

### Directory Structure

```
~/.amplifier/projects/<project-slug>/sessions/<session-id>/
├── metadata.json       # Session metadata (not embedded)
├── transcript.jsonl    # Conversation messages (embedded)
└── events.jsonl        # System events (not embedded in v0.2.0)
```

---

## transcript.jsonl Schema Analysis

### Actual Field Structure (from real sessions)

**All messages have**:
```json
{
  "role": "user" | "assistant" | "tool",
  "content": <varies by role>,
  "turn": <integer or null>,
  "timestamp": "<ISO datetime>"
}
```

### Role-Specific Content Structure

#### User Messages

**Structure**:
```json
{
  "role": "user",
  "content": "User's text input as a string",
  "turn": null,
  "timestamp": null
}
```

**What to embed**: The entire `content` string
- This captures user intent, questions, requests
- Critical for finding "what did I ask about X?"

**Example**:
```json
{
  "role": "user",
  "content": "Create a comprehensive smoke test for the lol-coaching bundle",
  "turn": null
}
```

#### Assistant Messages

**Structure** (complex!):
```json
{
  "role": "assistant",
  "content": [
    {
      "type": "thinking",
      "thinking": "<extended thinking text>",
      "signature": "<cryptographic signature>"
    },
    {
      "type": "text",
      "text": "<response text shown to user>"
    },
    {
      "type": "tool_call",
      "id": "toolu_xxx",
      "name": "tool_name",
      "input": { ... }
    }
  ],
  "turn": null,
  "timestamp": null
}
```

**What to embed**:

1. **Text blocks** (`type: "text"`):
   - Extract `content[].text` fields
   - This is what the user sees
   - Critical for "what did the assistant say about X?"

2. **Thinking blocks** (`type: "thinking"`):
   - Extract `content[].thinking` fields
   - Contains reasoning, analysis, internal thoughts
   - Critical for "how did it reason about X?"
   - Often contains valuable context not in final response

3. **Tool calls** (`type: "tool_call"`):
   - Do NOT embed the tool call structure itself
   - The tool's response (in tool messages) may be embedded separately

**Extraction logic**:
```python
def extract_assistant_content(message: dict) -> str:
    """Extract embeddable content from assistant message."""
    content = message.get("content", "")
    
    if isinstance(content, str):
        return content  # Simple string content
    
    if isinstance(content, list):
        # Complex content array - extract text and thinking
        parts = []
        for block in content:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "thinking":
                parts.append(block.get("thinking", ""))
        return "\n\n".join(parts)
    
    return ""
```

#### Tool Messages

**Structure**:
```json
{
  "role": "tool",
  "content": "<tool output as string or structured data>",
  "turn": null
}
```

**What to embed**:
- The `content` field (tool output)
- Useful for "what files did I look at?" or "what was the bash output?"
- May be large (file contents, etc.) - consider length limits

---

## events.jsonl Schema Analysis

### Actual Field Structure

```json
{
  "event": "session:fork" | "llm.request" | "tool.call" | etc.,
  "ts": "<ISO datetime>",
  "lvl": "INFO" | "DEBUG" | "WARNING" | "ERROR",
  "turn": <integer or null>,
  "data": { ... event-specific data ... },
  "schema": "0.1.0",
  "session_id": "<session-id>"
}
```

### Event Types Observed

From real sessions:
- `session:fork` - Session forked (sub-session created)
- `session:start` - Session started
- `llm.request` - LLM API call
- `llm.response` - LLM response received
- `tool.call` - Tool execution
- `tool.result` - Tool result returned

### Embedding Strategy for Events

**Current implementation (v0.2.0)**: Events are NOT embedded

**Rationale**:
- Events are for telemetry/debugging, not conversation search
- Event data can be extremely large (100k+ tokens)
- Embedding events would drastically increase storage costs

**Search strategy**:
- Use structured queries (by event type, tool name, level)
- Use full-text search on event fields if needed
- Do NOT use semantic search on events

**Future consideration**: May embed event summaries (not full data) for:
- Error messages (semantic search for similar errors)
- Tool descriptions (find similar tool usage patterns)

---

## Embedding Field Mapping

### What Gets Embedded (Current Implementation)

| Source | Role | Field | Extraction | Why Embed |
|--------|------|-------|------------|-----------|
| **transcript.jsonl** | user | `content` | Direct string | User questions and requests |
| **transcript.jsonl** | assistant | `content[].text` | Extract from array | Assistant responses to user |
| **transcript.jsonl** | assistant | `content[].thinking` | Extract from array | Reasoning and analysis |
| **transcript.jsonl** | tool | `content` | Direct string | Tool outputs and results |

### What Does NOT Get Embedded

| Source | Why Not |
|--------|---------|
| **metadata.json** | Structured metadata, not conversational content |
| **events.jsonl** | Telemetry data, use structured search instead |
| Tool call structures | Just metadata, not searchable content |
| Signatures | Cryptographic data, not semantic content |

---

## Implementation in Storage Backends

### Current Content Extraction (INCOMPLETE!)

**File**: `backends/duckdb.py:408-411`

```python
# Current implementation (SIMPLE)
if embeddings is None and self.embedding_provider:
    texts = [line.get("content", "") for line in lines]
    # ↑ This only gets the content field as-is!
    embeddings = await self.embedding_provider.embed_batch(texts)
```

**Problem**: For assistant messages, `content` is an ARRAY, not a string!

```python
# What we're getting:
line.get("content", "")
# Returns: [{"type": "thinking", ...}, {"type": "text", ...}]
# Should return: "thinking text\n\nresponse text"
```

### Correct Content Extraction (NEEDED!)

```python
def extract_embeddable_content(line: dict) -> str:
    """
    Extract content for embedding from a transcript line.
    
    Handles:
    - User messages: Simple string content
    - Assistant messages: Complex content arrays (text + thinking)
    - Tool messages: String content
    """
    content = line.get("content", "")
    
    # Simple case: content is already a string
    if isinstance(content, str):
        return content
    
    # Complex case: content is an array (assistant messages)
    if isinstance(content, list):
        parts = []
        for block in content:
            block_type = block.get("type")
            
            if block_type == "text":
                # Text shown to user
                text = block.get("text", "")
                if text:
                    parts.append(text)
            
            elif block_type == "thinking":
                # Internal reasoning (extended thinking)
                thinking = block.get("thinking", "")
                if thinking:
                    parts.append(thinking)
            
            # Skip tool_call blocks - these are metadata, not searchable content
        
        return "\n\n".join(parts)
    
    # Fallback: convert to string
    return str(content)
```

### Updated Backend Implementation Needed

**All three backends need this update**:

```python
# In sync_transcript_lines():
if embeddings is None and self.embedding_provider:
    # CORRECT: Extract embeddable content
    texts = [extract_embeddable_content(line) for line in lines]
    embeddings = await self.embedding_provider.embed_batch(texts)
```

---

## Embedding Use Cases

### 1. Search User Questions

**Query**: "How do I implement vector search?"

**Searches**:
- User messages with `content` matching semantically
- Finds: "What's the best way to do semantic search?"
- Finds: "Can you explain embeddings?"

**Field**: `role=user`, `content` (string)

### 2. Search Assistant Responses

**Query**: "Explanations of MMR algorithm"

**Searches**:
- Assistant messages with `content[].text` blocks
- Finds responses that explained MMR
- Finds code examples and documentation

**Field**: `role=assistant`, `content[].text` (extracted from array)

### 3. Search Assistant Reasoning

**Query**: "Why did it choose DuckDB over SQLite?"

**Searches**:
- Assistant messages with `content[].thinking` blocks
- Finds internal reasoning about trade-offs
- Finds decision-making processes

**Field**: `role=assistant`, `content[].thinking` (extracted from array)

**This is unique and powerful** - you can search the AI's internal reasoning, not just what it showed you!

### 4. Search Tool Outputs

**Query**: "What files did I look at in the amplifier-core repo?"

**Searches**:
- Tool messages with file content or command output
- Finds: `read_file` tool results
- Finds: `bash` command outputs

**Field**: `role=tool`, `content` (string or structured)

---

## Search Configuration Options

### TranscriptSearchOptions

```python
@dataclass
class TranscriptSearchOptions:
    query: str
    search_in_user: bool = True       # Search user input
    search_in_assistant: bool = True  # Search assistant responses
    search_in_thinking: bool = True   # Search thinking blocks
    search_type: str = "hybrid"       # full_text, semantic, hybrid
```

**Implementation**:
- `search_in_user`: Filter `WHERE role = 'user'`
- `search_in_assistant`: Filter `WHERE role = 'assistant'`
- `search_in_thinking`: Requires extracting thinking content during embedding

**Current issue**: We don't have a way to separately search thinking vs. text responses!

**Solution**: Store separate embeddings for thinking vs. text, or include metadata to distinguish them.

---

## Proposed Enhancement: Multi-Vector Storage

### Option A: Combined Embedding (Current Approach)

**What we do**:
```python
# For assistant message
embedding = embed(thinking_text + "\n\n" + response_text)
# Single vector represents entire message
```

**Pros**:
- Simple implementation
- Single vector per message
- Efficient storage

**Cons**:
- Can't search thinking separately from responses
- Mixed signal (reasoning + output combined)

### Option B: Separate Embeddings (Better for Search)

**What we could do**:
```python
# For assistant message with thinking
thinking_embedding = embed(thinking_text)
response_embedding = embed(response_text)

# Store both with metadata
{
  "sequence": 5,
  "role": "assistant",
  "thinking_vector": [0.1, 0.2, ...],
  "response_vector": [0.3, 0.4, ...],
  "metadata": {"has_thinking": true, "has_response": true}
}
```

**Pros**:
- Can search thinking separately
- Better relevance (search matches specific content type)
- Supports "search only in reasoning" queries

**Cons**:
- 2x storage per assistant message
- More complex implementation
- 2x embedding API calls

**Recommendation**: Start with Option A (current), add Option B if needed

---

## Implementation Status

### What's Implemented (v0.2.0)

✅ Embedding infrastructure in place  
✅ All backends call `embed_batch()`  
✅ Cache working  
⚠️ Content extraction is SIMPLE (just `.get("content", "")`

### What Needs Updating

❌ **Content extraction for assistant messages**
- Currently: Gets array instead of text
- Needs: Extract text and thinking blocks
- Impact: Embeddings for assistant messages may be broken

❌ **Thinking block separation**
- Currently: No way to search thinking separately
- Needs: Metadata or separate embeddings
- Impact: Can't use `search_in_thinking` filter effectively

---

## Recommended Implementation Plan

### Phase 1: Fix Content Extraction (Critical)

**Priority**: HIGH - Required for embeddings to work correctly

**Changes needed**:
```python
# Add to backends/base.py or utils
def extract_embeddable_content(message: dict) -> str:
    """Extract text content for embedding from transcript message."""
    content = message.get("content", "")
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        texts = []
        for block in content:
            if block.get("type") == "text" and "text" in block:
                texts.append(block["text"])
            elif block.get("type") == "thinking" and "thinking" in block:
                texts.append(block["thinking"])
        return "\n\n".join(texts)
    
    return str(content)  # Fallback

# Update all backends:
texts = [extract_embeddable_content(line) for line in lines]
```

**Testing**:
- Add test with real assistant message structure
- Verify text and thinking both extracted
- Verify embeddings generated for combined content

### Phase 2: Add Thinking Metadata (Optional)

**Priority**: MEDIUM - Nice to have for advanced search

**Schema addition**:
```python
# Add to transcript documents
{
  "sequence": 5,
  "role": "assistant",
  "content": [...],
  "embedding": [0.1, 0.2, ...],
  "embedding_model": "text-embedding-3-large",
  "content_metadata": {
    "has_thinking": true,
    "has_text_response": true,
    "thinking_char_count": 1250,
    "response_char_count": 450
  }
}
```

**Search implementation**:
```python
# Filter by content type
if options.search_in_thinking and not options.search_in_assistant:
    WHERE content_metadata.has_thinking = true
```

### Phase 3: Separate Thinking Embeddings (Future)

**Priority**: LOW - Only if advanced analytics needed

**Would require**:
- Dual vector storage per assistant message
- Separate vector search queries
- More complex search merging

**Use case**: "Find all reasoning about architecture decisions" (thinking only)

---

## Current vs Correct Extraction

### What We're Doing Now (BROKEN for assistant messages)

```python
# Current code
texts = [line.get("content", "") for line in lines]

# For user message
texts[0] = "How do I use vector search?"  # ✓ Correct (string)

# For assistant message  
texts[1] = [{"type": "thinking", ...}, {"type": "text", ...}]  # ✗ WRONG (array!)
```

**Result**: Assistant message embeddings fail or produce incorrect vectors!

### What We Should Be Doing

```python
# Correct code
texts = [extract_embeddable_content(line) for line in lines]

# For user message
texts[0] = "How do I use vector search?"  # ✓ Correct (string)

# For assistant message
texts[1] = "The user wants me to...\n\nYou can use embeddings for semantic search..."
#          ↑ thinking block           ↑ text response
# ✓ Correct (combined string)
```

**Result**: All messages produce valid embeddings!

---

## Search Behavior Examples

### Example 1: Finding User Questions

**User query**: "sessions where I asked about testing"

**SQL**:
```sql
SELECT * FROM transcripts
WHERE role = 'user'
  AND (
    content LIKE '%testing%'  -- Full-text match
    OR array_cosine_similarity(embedding, query_vector) > 0.8  -- Semantic match
  )
```

**Finds**:
- "How do I write tests for this?"
- "Can you help me with pytest?"
- "What's the best testing approach?"

### Example 2: Finding Assistant Reasoning

**User query**: "how did it decide between options"

**SQL** (if we track thinking separately):
```sql
SELECT * FROM transcripts  
WHERE role = 'assistant'
  AND content_metadata.has_thinking = true
  AND array_cosine_similarity(thinking_embedding, query_vector) > 0.7
```

**Finds**:
- Thinking blocks with decision-making analysis
- Trade-off comparisons
- Architectural reasoning

### Example 3: Hybrid Search Across All Content

**User query**: "vector database comparisons"

**Process**:
1. Full-text search: `content LIKE '%vector database%'`
2. Semantic search: `array_cosine_similarity(embedding, query_vector)`
3. Merge results (user + assistant + tool)
4. MMR re-rank for diversity

**Finds**:
- User questions about vector DBs
- Assistant explanations
- Tool outputs showing benchmark results
- Mixed content types, ranked by relevance

---

## Current Implementation Gap

### Critical Issue

**The current implementation does NOT handle assistant content arrays correctly!**

**Evidence**:
```python
# backends/duckdb.py:408-411
texts = [line.get("content", "") for line in lines]
# ↑ This breaks for assistant messages!
```

**Impact**:
- Assistant message embeddings may fail
- Thinking blocks not included in search
- Search quality degraded for half the conversation

### Fix Required

**Add content extraction utility and use in all three backends**:

1. Create `extract_embeddable_content()` function
2. Update `DuckDBBackend.sync_transcript_lines()`
3. Update `SQLiteBackend.sync_transcript_lines()`
4. Update `CosmosBackend.sync_transcript_lines()`
5. Add tests with real assistant message structure

**Priority**: HIGH - Should be fixed before v0.2.0 release

---

## Summary Table

| Message Type | Field | Current Status | What to Embed |
|--------------|-------|----------------|---------------|
| **User** | `content` (string) | ✅ Working | Direct string |
| **Assistant** | `content[].text` | ❌ Broken | Extract text blocks |
| **Assistant** | `content[].thinking` | ❌ Not extracted | Extract thinking blocks |
| **Tool** | `content` (string/struct) | ⚠️ May work | String or stringify |
| **Events** | N/A | ✅ Correct | Don't embed (use structured search) |

**Current implementation**: 1 out of 4 message types handled correctly

**Required fix**: Content extraction function for complex content arrays

---

## Recommended Next Steps

1. **Implement `extract_embeddable_content()` utility**
2. **Update all three backends to use it**
3. **Add tests with real assistant message structure**
4. **Verify embeddings generated correctly for all message types**
5. **Consider adding content_metadata for advanced filtering**

**This is critical for the enhanced session storage to work correctly with real Amplifier sessions!**
