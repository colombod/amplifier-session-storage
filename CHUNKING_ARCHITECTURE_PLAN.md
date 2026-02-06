# Session Storage Chunking Architecture Plan

> Addresses: [Issue #4](https://github.com/microsoft/amplifier-session-sync/issues/4) (embedding token limit) and [Issue #5](https://github.com/microsoft/amplifier-session-sync/issues/5) (CosmosDB document size limit)

**Date**: 2026-02-06
**Status**: PLAN - Not yet implemented
**Author**: Architecture planning session

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Current Architecture Analysis](#2-current-architecture-analysis)
3. [Root Cause Analysis](#3-root-cause-analysis)
4. [Design Constraints](#4-design-constraints)
5. [Research Summary](#5-research-summary)
6. [Architecture Options Evaluated](#6-architecture-options-evaluated)
7. [Recommended Architecture](#7-recommended-architecture)
8. [Detailed Design](#8-detailed-design)
9. [Implementation Phases](#9-implementation-phases)
10. [Migration Strategy](#10-migration-strategy)
11. [Vector Count & Storage Impact Analysis](#11-vector-count--storage-impact-analysis)
12. [Open Questions & Risks](#12-open-questions--risks)
13. [References](#13-references)

---

## 1. Problem Statement

### Issue #4: Embedding Token Limit

The Azure OpenAI embedding model (`text-embedding-3-large`) has an **8,192 token limit**. Sub-agent
sessions routinely produce content with 10,000-75,000 tokens in a single message. The storage
library sends this text directly to the embedding API with **no token limit enforcement**, causing
the API to reject the request. This results in **~90% of transcript rejection warnings** during sync.

Evidence from production (2026-02-06):

| Session Agent               | Messages | Tokens Requested |
|-----------------------------|----------|------------------|
| `foundation:modular-builder`| 18       | 74,710           |
| `foundation:modular-builder`| 8        | 68,488           |
| `foundation:explorer`       | 20       | 65,231           |
| `foundation:bug-hunter`     | 1        | 57,420           |
| `foundation:session-analyst`| 20       | 44,409           |

Even a session with **1 message** exceeded the limit by 7x.

### Issue #5: CosmosDB Document Size Limit

CosmosDB has a **2MB document size limit**. Transcript documents with large content plus 4 embedded
vectors (each 3072 floats encoded as JSON arrays) can exceed this limit. The transcript storage path
has **no document size validation** (unlike events which have a 400KB check). This causes
`RequestEntityTooLarge` errors and accounts for ~10% of transcript rejection warnings.

### Combined Impact

Together, these issues mean that **all sub-agent sessions with substantial content fail to sync
transcripts**. This is the majority of the valuable analytical content (builder implementations,
explorer surveys, bug-hunter investigations, architect reviews).

---

## 2. Current Architecture Analysis

### Content Flow Pipeline

```
Session Event (events.jsonl)
    |
    v
content_extraction.py:extract_all_embeddable_content()
    |  Routes by message role:
    |  - user    -> user_query (no truncation)
    |  - assistant -> assistant_response + assistant_thinking (no truncation)
    |  - tool    -> tool_output (truncated at 10,000 chars)
    v
duckdb.py:_generate_multi_vector_embeddings()
    |  4 sequential _embed_non_none() calls
    |  Each batches all non-None texts into single API call
    v
azure_openai.py:embed_batch()
    |  NO token counting
    |  NO text truncation
    |  NO batching by token count
    |  Single client.embed() call with all texts
    v
Azure OpenAI API
    |  8,192 token limit per input
    |  REJECTS entire batch if ANY text exceeds limit
    v
ERROR: "max context length is 8192 tokens, requested N tokens"
```

### Storage Schema (DuckDB)

The `transcripts` table stores one row per message with 4 inline vector columns:

```sql
CREATE TABLE transcripts (
    id VARCHAR PRIMARY KEY,           -- "{session_id}_msg_{sequence}"
    user_id VARCHAR NOT NULL,
    host_id VARCHAR NOT NULL,
    project_slug VARCHAR,
    session_id VARCHAR NOT NULL,
    sequence INTEGER NOT NULL,
    role VARCHAR,
    content JSON,                     -- Full message content (string or array)
    turn INTEGER,
    ts TIMESTAMP,
    user_query_vector FLOAT[3072],         -- Vector 1
    assistant_response_vector FLOAT[3072], -- Vector 2
    assistant_thinking_vector FLOAT[3072], -- Vector 3
    tool_output_vector FLOAT[3072],        -- Vector 4
    embedding_model VARCHAR,
    vector_metadata JSON,
    synced_at TIMESTAMP
);
```

4 HNSW indexes (cosine metric), one per vector column.

### Vector Population Pattern

| Message Role | user_query | assistant_response | assistant_thinking | tool_output |
|-------------|------------|-------------------|-------------------|-------------|
| `user`      | Populated  | NULL              | NULL              | NULL        |
| `assistant` | NULL       | If text blocks    | If thinking blocks| NULL        |
| `tool`      | NULL       | NULL              | NULL              | Populated   |

Most messages populate only 1-2 of the 4 vectors. The schema is sparse by design.

### Search Pipeline

```sql
-- vector_search() builds:
SELECT ...,
    GREATEST(
        CASE WHEN user_query_vector IS NOT NULL
             THEN array_cosine_similarity(user_query_vector, query_vec) ELSE 0.0 END,
        CASE WHEN assistant_response_vector IS NOT NULL
             THEN array_cosine_similarity(assistant_response_vector, query_vec) ELSE 0.0 END,
        CASE WHEN assistant_thinking_vector IS NOT NULL
             THEN array_cosine_similarity(assistant_thinking_vector, query_vec) ELSE 0.0 END,
        CASE WHEN tool_output_vector IS NOT NULL
             THEN array_cosine_similarity(tool_output_vector, query_vec) ELSE 0.0 END
    ) as similarity
FROM transcripts
WHERE /* at least one vector IS NOT NULL */
ORDER BY similarity DESC
LIMIT N;
```

Hybrid search adds full-text `LIKE` matching, deduplication, and MMR re-ranking.

### Key Code Locations

| Component                | File                          | Lines     |
|--------------------------|-------------------------------|-----------|
| Content extraction       | `content_extraction.py`       | 20-174    |
| Multi-vector embedding   | `backends/duckdb.py`          | 243-283   |
| Embed without token check| `backends/duckdb.py`          | 209-241   |
| Schema creation          | `backends/duckdb.py`          | 314-413   |
| Transcript sync          | `backends/duckdb.py`          | 627-732   |
| Vector search            | `backends/duckdb.py`          | 1349-1453 |
| Hybrid search + MMR      | `backends/duckdb.py`          | 901-958   |
| Azure embed_batch        | `embeddings/azure_openai.py`  | 231-298   |
| Cosmos transcript sync   | `backends/cosmos.py`          | 782-871   |
| Cosmos event size check  | `backends/cosmos.py`          | 1235-1278 |

---

## 3. Root Cause Analysis

### Why texts exceed the embedding token limit

1. **No token counting anywhere in the pipeline.** Content extraction does not measure token count.
   The embedding provider does not check input length. There are no guards between extraction and
   the API call.

2. **Sub-agent sessions produce massive content.** A `modular-builder` agent writing code can
   produce thinking blocks of 50k+ tokens in a single message. An `explorer` surveying 20 files
   produces similarly large content. These are not edge cases; they are the normal output of
   specialized agents.

3. **Content types are extracted but not size-managed.** The `_extract_assistant_content` function
   joins all thinking blocks with `"\n\n"` into a single string. If an assistant message has 10
   thinking blocks of 5k tokens each, the combined thinking text is 50k tokens. Similarly, multiple
   text blocks are joined into one response string.

4. **The only truncation is for tool outputs at 10,000 characters** (not tokens). User queries and
   assistant content have no limits.

### Why CosmosDB documents exceed 2MB

1. **No size validation on transcript upserts.** Events have a 400KB threshold check
   (`cosmos.py:1235`), but transcripts go directly to `upsert_item()`.

2. **Vector data adds significant size.** 4 vectors x 3072 floats x ~8 bytes per float in JSON =
   ~100KB of vector data per document. Combined with large content, this pushes past 2MB.

3. **Content is stored untruncated.** The full message content (including tool results, code output,
   etc.) is stored in the `content` JSON field.

### The aggravating factor (user's hypothesis confirmed)

While thinking and response texts are NOT combined into a single embedding (they go to separate
vector columns), the issue is that **each individual content type can be enormous on its own**. A
single thinking block of 57,000 tokens is 7x the embedding limit. The problem is not combination;
it is the sheer size of individual content streams.

Additionally, the batch API call fails on the **entire batch** if any single text exceeds the limit.
So one oversized thinking block in a batch of 20 messages causes all 20 to fail.

---

## 4. Design Constraints

### Must Preserve

- **Original content stored intact.** Chunking is for embedding only. The `content` JSON field in
  transcripts must retain the full, unmodified message content for reconstruction and display.
- **Backward compatibility.** Existing data with inline vectors must continue to work. Searches must
  return results from both old (inline) and new (chunked) data.
- **Multi-backend support.** Solution must work across DuckDB, SQLite, and Cosmos DB backends.
- **Search quality.** Chunked content should produce search results at least as good as inline
  vectors for short content, and dramatically better than nothing for long content (which currently
  has no vectors at all).

### Must Address

- **Embedding token limit (8,192 tokens).** All text sent to the embedding API must be within
  limits.
- **CosmosDB document size (2MB).** Transcript documents must stay under the limit.
- **Search maps back to messages.** Users search for messages, not chunks. Chunk matches must
  resolve to parent messages.

### Should Consider

- **Vector count per entry.** More chunks = more vectors = more storage + slower index operations.
  Need a strategy for managing vector count growth.
- **Embedding API cost.** More chunks = more embedding API calls. Chunking strategy should not
  create excessive tiny chunks.
- **Chunking quality.** Chunks should be semantically coherent, not arbitrary byte/token splits.

---

## 5. Research Summary

### AIContext (C#) - Semantic Chunking via Embedding Breakpoints

**Repository**: https://github.com/AIGeekSquad/AIContext

**Approach**: 7-stage pipeline that splits text into sentences, creates overlapping context groups
(buffer windows), embeds each group, measures cosine distance between adjacent groups, and splits
at points where distance exceeds a percentile threshold.

**Key insights for our design**:
- Overlapping buffer windows (BufferSize parameter) capture local context without hierarchy
- Percentile-based breakpoint detection self-adapts to content variability
- Token limits enforced at two stages: segment level and group level
- Fallback cascade: semantic breakpoints -> single chunk -> sentence-per-chunk
- **Limitation**: Flat single-level approach. No multi-resolution retrieval.
- **Limitation**: Requires embedding each sentence group first to detect breakpoints (expensive
  for the purpose of deciding where to split).

**Tuning defaults**: MaxTokensPerChunk=512, BreakpointPercentileThreshold=0.75, BufferSize=1

### LlamaIndex - Hierarchical Chunking with Auto-Merging Retrieval

**Repository**: https://github.com/run-llama/llama_index

#### HierarchicalNodeParser

Creates a tree of chunks at multiple granularity levels by recursively applying SentenceSplitter:

```
Document (full text)
  +-- Level 0: SentenceSplitter(chunk_size=2048) -> [Node_A, Node_B, ...]
       +-- Level 1: SentenceSplitter(chunk_size=512)  -> [Node_A1, Node_A2, ...]
            +-- Level 2: SentenceSplitter(chunk_size=128) -> [Node_A1a, Node_A1b, ...]
```

**Only leaf nodes get embedded and indexed.** Parent nodes are stored in a docstore and retrieved
by ID during the merge phase. This keeps the vector index manageable.

#### AutoMergingRetriever

At query time:
1. Retrieve leaf nodes by embedding similarity (standard vector search)
2. For each retrieved leaf, look up its parent
3. If >50% of a parent's children are retrieved (`simple_ratio_thresh=0.5`), replace all those
   children with the parent node
4. Parent's score = average of merged children's scores
5. Iterate: merged parents can merge into grandparents if hierarchy has more levels

**Key insight**: The vector store is FLAT (no hierarchy). Hierarchy lives in node relationship
metadata and is exploited only at retrieval time.

#### SemanticDoubleMergingSplitter

Two-pass approach:
1. **Initial chunking**: Split into sentences, create chunks by appending sentences while semantic
   similarity > `appending_threshold` (0.8). Start new chunk when similarity < `initial_threshold`
   (0.6).
2. **Merging pass**: Iterate through initial chunks, merge adjacent chunks if similarity >
   `merging_threshold` (0.8) and combined size < `max_chunk_size`.

Uses spaCy for semantic similarity between chunks. Produces more coherent chunks than single-pass
splitting because the second pass can rejoin chunks that were artificially separated by local
dissimilarity.

#### SentenceSplitter (Cascading)

Multi-pass boundary-aware splitting:
1. Triple newline (paragraph) boundaries
2. NLTK sentence tokenizer
3. Punctuation boundaries (regex fallback)
4. Word-level splitting
5. Character-level splitting (last resort)

Merging uses `chunk_overlap` tokens from end of previous chunk prepended to next chunk.

### Key Takeaways for Our Design

| Approach | Strengths | Weaknesses | Fit for Us |
|----------|-----------|------------|------------|
| Simple truncation | Trivial to implement | Loses semantic content | Quick fix only |
| Token-aware sentence splitting | Predictable, fast | May split semantic units | Good baseline |
| Semantic breakpoint detection | Respects topic shifts | Expensive (embed to split) | Phase 2 enhancement |
| Hierarchical + auto-merge | Best retrieval quality | Most complex, most storage | Phase 3 goal |
| Double merging | Better chunk coherence | Needs similarity model | Phase 2 enhancement |

---

## 6. Architecture Options Evaluated

### Option A: Simple Truncation

Truncate text to 8,192 tokens before embedding. Store full content, embed only the head.

```
Pros: Trivial implementation, no schema change, no new dependencies
Cons: Loses tail content for search (a 75k token thinking block only embeds first 11%)
      Search quality degrades severely for long content
      Does not solve CosmosDB size issue
Verdict: INSUFFICIENT as sole solution. Acceptable as Phase 0 safety net.
```

### Option B: Chunk and Average Embeddings

Split text into chunks, embed each, average all chunk embeddings into one vector per type.

```
Pros: No schema change (still 4 vectors per message)
      Captures semantic content from all chunks
Cons: Averaged embeddings lose specificity ("muddled" signal)
      A chunk about authentication and a chunk about logging become one indistinct vector
      Research shows this degrades retrieval precision significantly
Verdict: REJECTED. Averaging destroys the precision that multi-vector search depends on.
```

### Option C: Chunked Vectors with Span Metadata (Recommended)

Split long texts into semantic chunks. Store each chunk as a separate vector record with span
metadata linking back to the parent message. Search across both inline vectors (short content) and
chunk vectors (long content).

```
Pros: Best search quality - each chunk is a precise semantic unit
      Span metadata enables precise retrieval (which part of the text matched)
      Naturally solves CosmosDB size (chunks are separate small documents)
      Backward compatible (short content stays inline)
      Extensible to hierarchical retrieval
Cons: Schema change (new table/collection)
      More complex search queries (JOIN/UNION with inline vectors)
      More vectors to index and search
Verdict: RECOMMENDED. Best balance of quality, complexity, and extensibility.
```

### Option D: Full Hierarchical Multi-Resolution (LlamaIndex Style)

Create multi-level hierarchy (4096/1024/256 tokens). Embed only leaves. Store parents in docstore.
Auto-merge at retrieval time.

```
Pros: Optimal retrieval quality at any granularity
      Auto-merging provides coherent context
Cons: Most complex implementation
      Highest storage overhead (3x chunk records)
      Requires docstore-like retrieval pattern (fetch by ID)
      May be over-engineered for session transcripts
Verdict: DEFERRED to Phase 3. Build toward it, but don't start here.
```

---

## 7. Recommended Architecture

### Overview: Chunked Vectors with Span Metadata + Phased Enhancement

```
Phase 0: Safety Net (truncation guard)
    |  Immediate fix: truncate before API call, log warning
    |
Phase 1: Token-Aware Sentence Chunking
    |  Split long texts at sentence/paragraph boundaries
    |  Store chunks in new transcript_chunks table
    |  Search across both inline and chunk vectors
    |
Phase 2: Semantic Chunking Enhancement
    |  Replace naive splitting with semantic breakpoint detection
    |  Double-merging for chunk coherence
    |  Markdown-aware splitting for structured content
    |
Phase 3: Hierarchical Retrieval with Auto-Merging
    |  Multi-level chunk hierarchy
    |  Auto-merge at retrieval time
    |  Optimal search quality
```

### Architecture Diagram

```
Content Extraction (existing)
    |
    v
+---------------------------------------------------+
| Chunking Pipeline (NEW)                           |
|                                                   |
|  1. Token counting (tiktoken)                     |
|  2. If tokens <= EMBED_TOKEN_LIMIT:               |
|       -> Pass through (inline vector path)        |
|  3. If tokens > EMBED_TOKEN_LIMIT:                |
|       -> Sentence splitting                       |
|       -> Token-aware merging with overlap          |
|       -> Span metadata generation                 |
|       -> Multiple chunk records                   |
+---------------------------------------------------+
    |                           |
    v                           v
Inline Path                 Chunk Path
(short content)             (long content)
    |                           |
    v                           v
transcripts table           transcript_chunks table
(4 vector columns)          (1 vector column + span metadata)
    |                           |
    +------ Search Union -------+
    |
    v
Unified Search Results (message-level)
```

### The Two-Path Strategy

**Fast Path (inline)**: For content within the embedding token limit (the common case for user
queries, short responses, moderate thinking blocks), the existing 4-vector-per-row architecture
is preserved unchanged. No schema change, no performance regression.

**Chunk Path (overflow)**: For content exceeding the token limit, the text is split into semantic
chunks. Each chunk gets its own vector record in a dedicated `transcript_chunks` table with span
metadata linking back to the parent message.

**Search unifies both paths**: Queries search inline vectors AND chunk vectors, deduplicate by
parent message, and return results at the message level.

---

## 8. Detailed Design

### 8.1 Chunking Pipeline

#### New Module: `chunking.py`

```
amplifier_session_storage/
    chunking.py           # NEW - chunking pipeline
    content_extraction.py # MODIFIED - adds token counting
    backends/
        duckdb.py         # MODIFIED - chunk storage + search
        sqlite.py         # MODIFIED - chunk storage + search
        cosmos.py         # MODIFIED - chunk storage + search + size validation
```

#### Token Counting

Add `tiktoken` dependency for accurate token counting matching the embedding model's tokenizer.

```python
import tiktoken

# text-embedding-3-large uses cl100k_base encoding
_encoder = tiktoken.get_encoding("cl100k_base")

EMBED_TOKEN_LIMIT = 8192
CHUNK_TARGET_TOKENS = 2048      # Target chunk size (well within limit)
CHUNK_OVERLAP_TOKENS = 128      # Overlap between adjacent chunks
CHUNK_MIN_TOKENS = 64           # Don't create tiny chunks
```

**Why 2048 target, not 8192?** Leaving headroom for:
- Token counting approximation errors
- Future model changes with lower limits
- Better search precision (smaller chunks = more specific matches)
- Alignment with LlamaIndex's recommended parent-level chunk size

#### Splitting Strategy

The chunking pipeline uses a cascading split strategy inspired by both AIContext and LlamaIndex:

```python
class ChunkingStrategy:
    """Split text into semantically coherent chunks within token limits."""

    def chunk_text(self, text: str, content_type: str) -> list[ChunkResult]:
        """
        Returns list of ChunkResult, each with:
        - text: the chunk text
        - span_start: character offset in original text
        - span_end: character offset in original text
        - chunk_index: 0-based position
        - token_count: tokens in this chunk
        """
        token_count = count_tokens(text)

        # Fast path: text fits in one chunk
        if token_count <= EMBED_TOKEN_LIMIT:
            return [ChunkResult(text=text, span_start=0, span_end=len(text),
                               chunk_index=0, token_count=token_count)]

        # Split into semantic units based on content type
        if content_type in ("assistant_thinking", "assistant_response"):
            segments = self._split_markdown_aware(text)
        else:
            segments = self._split_sentences(text)

        # Merge segments into chunks respecting token limits
        chunks = self._merge_segments_with_overlap(segments)

        return chunks
```

**Phase 1 - Sentence/Paragraph Splitting**:

```python
def _split_markdown_aware(self, text: str) -> list[Segment]:
    """Split respecting markdown structure."""
    segments = []

    # Priority 1: Split at markdown headers (## Header)
    # Priority 2: Split at triple newlines (paragraph breaks)
    # Priority 3: Split at double newlines
    # Priority 4: Split at sentence boundaries (., !, ?)
    # Priority 5: Split at word boundaries (last resort)

    # Keep code blocks intact (``` ... ```)
    # Keep list items as atomic units
    # Track character offsets for span metadata

    return segments  # list of Segment(text, start, end, is_boundary)
```

**Phase 1 - Token-Aware Merging with Overlap**:

```python
def _merge_segments_with_overlap(self, segments: list[Segment]) -> list[ChunkResult]:
    """Merge small segments into chunks up to CHUNK_TARGET_TOKENS."""
    chunks = []
    current_segments = []
    current_tokens = 0

    for segment in segments:
        seg_tokens = count_tokens(segment.text)

        # Oversized single segment: force-split at word boundaries
        if seg_tokens > CHUNK_TARGET_TOKENS:
            # Flush current
            if current_segments:
                chunks.append(self._build_chunk(current_segments, len(chunks)))
                current_segments = []
                current_tokens = 0
            # Force-split the oversized segment
            sub_chunks = self._force_split(segment, CHUNK_TARGET_TOKENS)
            chunks.extend(sub_chunks)
            continue

        # Would exceed target: flush and start new chunk
        if current_tokens + seg_tokens > CHUNK_TARGET_TOKENS:
            chunks.append(self._build_chunk(current_segments, len(chunks)))

            # Overlap: carry last N tokens of overlap from previous chunk
            overlap_segments = self._get_overlap_tail(current_segments, CHUNK_OVERLAP_TOKENS)
            current_segments = overlap_segments + [segment]
            current_tokens = sum(count_tokens(s.text) for s in current_segments)
        else:
            current_segments.append(segment)
            current_tokens += seg_tokens

    # Flush remaining
    if current_segments:
        # Don't create tiny final chunks - merge with previous if possible
        if current_tokens < CHUNK_MIN_TOKENS and chunks:
            chunks[-1] = self._extend_chunk(chunks[-1], current_segments)
        else:
            chunks.append(self._build_chunk(current_segments, len(chunks)))

    return chunks
```

**Phase 2 Enhancement - Semantic Breakpoint Detection** (added later):

```python
def _split_with_semantic_breakpoints(self, text: str) -> list[Segment]:
    """Phase 2: Use embedding similarity to find natural breakpoints."""

    # 1. Split into sentences
    sentences = self._split_sentences(text)

    # 2. Create overlapping groups (buffer_size=1)
    groups = self._create_buffered_groups(sentences, buffer_size=1)

    # 3. Embed each group (lightweight model or cached)
    group_embeddings = self._embed_groups(groups)

    # 4. Compute cosine distances between adjacent groups
    distances = [1 - cosine_sim(group_embeddings[i], group_embeddings[i+1])
                 for i in range(len(group_embeddings) - 1)]

    # 5. Find breakpoints at percentile threshold
    threshold = np.percentile(distances, 75)  # configurable
    breakpoints = [i for i, d in enumerate(distances) if d > threshold]

    # 6. Group sentences between breakpoints into segments
    return self._build_segments_from_breakpoints(sentences, breakpoints)
```

**Phase 2 Enhancement - Double Merging** (added later):

```python
def _double_merge_chunks(self, chunks: list[ChunkResult]) -> list[ChunkResult]:
    """Phase 2: Second pass to merge semantically similar adjacent chunks."""
    merged = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        # Look ahead up to merging_range chunks
        while i + 1 < len(chunks):
            next_chunk = chunks[i + 1]
            combined_tokens = current.token_count + next_chunk.token_count
            if combined_tokens > CHUNK_TARGET_TOKENS:
                break
            similarity = self._compute_chunk_similarity(current, next_chunk)
            if similarity > MERGING_THRESHOLD:  # 0.8 default
                current = self._merge_two_chunks(current, next_chunk)
                i += 1
            else:
                break
        merged.append(current)
        i += 1
    return merged
```

#### ChunkResult Data Structure

```python
@dataclass
class ChunkResult:
    text: str             # The chunk text (for embedding)
    span_start: int       # Character offset start in original text
    span_end: int         # Character offset end in original text
    chunk_index: int      # 0-based position in chunk sequence
    token_count: int      # Token count of this chunk
    total_chunks: int     # Total chunks for this content (set after all chunks created)
    overlap_start: int    # Where overlap begins (may differ from span_start)
    overlap_end: int      # Where overlap ends (may differ from span_end)
```

### 8.2 Schema Changes

#### New Table: `transcript_chunks` (DuckDB)

```sql
CREATE TABLE IF NOT EXISTS transcript_chunks (
    -- Identity
    id VARCHAR NOT NULL PRIMARY KEY,         -- "{parent_id}_chunk_{type}_{index}"
    parent_id VARCHAR NOT NULL,              -- FK to transcripts.id
    user_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,

    -- Chunk metadata
    content_type VARCHAR NOT NULL,           -- "user_query"|"assistant_response"|
                                             -- "assistant_thinking"|"tool_output"
    chunk_index INTEGER NOT NULL,            -- 0-based position in sequence
    total_chunks INTEGER NOT NULL,           -- Total chunks for this content type
    span_start INTEGER NOT NULL,             -- Character offset start in original text
    span_end INTEGER NOT NULL,               -- Character offset end in original text
    token_count INTEGER NOT NULL,            -- Tokens in this chunk

    -- Content and vector
    chunk_text TEXT NOT NULL,                -- The chunk text (for search display)
    vector FLOAT[3072],                      -- Embedding vector

    -- Metadata
    embedding_model VARCHAR,
    chunking_strategy VARCHAR,               -- "sentence_split_v1"|"semantic_v1"|etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_chunks_parent
    ON transcript_chunks (parent_id);

CREATE INDEX IF NOT EXISTS idx_chunks_session
    ON transcript_chunks (user_id, session_id);

CREATE INDEX IF NOT EXISTS idx_chunks_type
    ON transcript_chunks (content_type);

-- Vector index (single HNSW for all chunk vectors)
CREATE INDEX IF NOT EXISTS idx_chunks_vector
    ON transcript_chunks USING HNSW (vector) WITH (metric = 'cosine');
```

**Design decisions**:

1. **Single vector column** instead of 4: Chunks are already typed by `content_type`. One vector
   column with a type discriminator is more storage-efficient and simpler to query than 4 sparse
   columns.

2. **Single HNSW index**: All chunk vectors share one index, filtered by `content_type` at query
   time. This is more efficient than 4 separate indexes for the chunks table.

3. **`chunk_text` stored**: Enables search result display without joining back to the parent
   message. The parent message still has the full content.

4. **`parent_id` as foreign key**: Maps chunks back to their parent message for result
   aggregation.

#### Transcripts Table: No Change to Schema

The existing `transcripts` table schema is unchanged. For messages where content fits within the
token limit, inline vectors continue to be stored. For messages with chunked content, the inline
vector columns are set to NULL for the chunked content type, and the chunks are in
`transcript_chunks`.

A new field in `vector_metadata` tracks chunking:

```json
{
    "has_user_query_vector": true,
    "has_assistant_response_vector": false,
    "has_assistant_thinking_vector": false,
    "has_tool_output_vector": false,
    "chunked_types": ["assistant_thinking", "assistant_response"],
    "chunk_counts": {"assistant_thinking": 12, "assistant_response": 5}
}
```

#### Cosmos DB Collection: New Document Type

```json
{
    "id": "{session_id}_msg_{seq}_chunk_{type}_{idx}",
    "type": "transcript_chunk",
    "partitionKey": "{user_id}",
    "parent_id": "{session_id}_msg_{seq}",
    "session_id": "...",
    "content_type": "assistant_thinking",
    "chunk_index": 3,
    "total_chunks": 12,
    "span_start": 4096,
    "span_end": 6144,
    "token_count": 487,
    "chunk_text": "...",
    "vector": [0.123, 0.456, ...],
    "embedding_model": "text-embedding-3-large",
    "chunking_strategy": "sentence_split_v1"
}
```

Each chunk document is small (~30KB with vector data), well within Cosmos limits.

### 8.3 Modified Ingestion Pipeline

#### Updated Flow

```python
async def sync_transcript_lines(self, lines, ...):
    # 1. Extract content (existing)
    all_content = [extract_all_embeddable_content(line) for line in lines]

    # 2. NEW: Classify content by size
    inline_texts = {}   # content_type -> list[str|None] (within token limit)
    chunk_specs = {}    # (line_idx, content_type) -> list[ChunkResult]

    chunker = ChunkingStrategy()
    for i, content in enumerate(all_content):
        for content_type, text in content.items():
            if text is None:
                inline_texts.setdefault(content_type, []).append(None)
                continue

            token_count = count_tokens(text)
            if token_count <= EMBED_TOKEN_LIMIT:
                # Fast path: inline vector
                inline_texts.setdefault(content_type, []).append(text)
            else:
                # Chunk path: will go to transcript_chunks table
                inline_texts.setdefault(content_type, []).append(None)  # NULL inline
                chunks = chunker.chunk_text(text, content_type)
                chunk_specs[(i, content_type)] = chunks
                logger.info(f"Chunked {content_type} for line {i}: "
                           f"{token_count} tokens -> {len(chunks)} chunks")

    # 3. Embed inline texts (existing path, now only short texts)
    inline_embeddings = await self._generate_multi_vector_embeddings_from_texts(inline_texts)

    # 4. NEW: Embed chunks
    chunk_texts = []
    chunk_keys = []
    for (line_idx, content_type), chunks in chunk_specs.items():
        for chunk in chunks:
            chunk_texts.append(chunk.text)
            chunk_keys.append((line_idx, content_type, chunk))

    if chunk_texts:
        chunk_embeddings = await self._embed_non_none(chunk_texts)
    else:
        chunk_embeddings = []

    # 5. Store inline vectors in transcripts table (existing, with NULL for chunked types)
    # 6. NEW: Store chunk vectors in transcript_chunks table
    await self._store_chunks(chunk_keys, chunk_embeddings, lines, ...)
```

#### Batch Embedding Safety

The `_embed_non_none` method needs a token-limit safety check even for the inline path (as a
defense-in-depth measure):

```python
async def _embed_non_none(self, texts: list[str | None]) -> list[list[float] | None]:
    # ... existing filtering logic ...

    # NEW: Safety truncation for any text that somehow exceeds the limit
    safe_texts = []
    for text in just_texts:
        tokens = count_tokens(text)
        if tokens > EMBED_TOKEN_LIMIT:
            logger.warning(f"Text exceeds embed limit ({tokens} > {EMBED_TOKEN_LIMIT}), "
                          f"truncating. This should not happen if chunking is working.")
            text = truncate_to_tokens(text, EMBED_TOKEN_LIMIT)
        safe_texts.append(text)

    embeddings = await self.embedding_provider.embed_batch(safe_texts)
    # ...
```

### 8.4 Modified Search Pipeline

#### Unified Search Across Inline and Chunk Vectors

```python
async def _semantic_search_transcripts(self, options) -> list[SearchResult]:
    query_vector = await self.embedding_provider.embed_text(options.query)

    # Map search flags to content types
    target_types = self._resolve_search_targets(options)

    # Search BOTH inline vectors and chunk vectors
    inline_results = await self._search_inline_vectors(query_vector, target_types, options)
    chunk_results = await self._search_chunk_vectors(query_vector, target_types, options)

    # Merge: chunk results map to parent messages
    # If same message appears in both (inline for one type, chunks for another),
    # take the best score
    merged = self._merge_inline_and_chunk_results(inline_results, chunk_results)

    return merged[:options.limit]
```

#### Chunk Vector Search

```sql
-- Search chunk vectors
SELECT
    tc.parent_id,
    tc.session_id,
    tc.content_type,
    tc.chunk_index,
    tc.total_chunks,
    tc.span_start,
    tc.span_end,
    tc.chunk_text,
    array_cosine_similarity(tc.vector, {query_vec}::FLOAT[3072]) as similarity
FROM transcript_chunks tc
WHERE tc.vector IS NOT NULL
  AND tc.user_id = ?
  AND tc.content_type IN (?)          -- filtered by search flags
  AND similarity > 0.0
ORDER BY similarity DESC
LIMIT ? * 2;                          -- over-fetch for dedup
```

#### Result Aggregation

When multiple chunks from the same parent message match:

```python
def _merge_inline_and_chunk_results(self, inline, chunks):
    """Aggregate chunk matches to message level."""
    by_message = {}

    # Add inline results directly
    for result in inline:
        key = (result.session_id, result.sequence)
        by_message[key] = result

    # Add chunk results, keeping best score per message
    for chunk_result in chunks:
        key = (chunk_result.session_id, chunk_result.parent_sequence)
        existing = by_message.get(key)

        if existing is None or chunk_result.score > existing.score:
            # Enrich result with chunk span information
            result = SearchResult(
                session_id=chunk_result.session_id,
                sequence=chunk_result.parent_sequence,
                content=chunk_result.parent_content,  # full message
                score=chunk_result.score,
                source="semantic_chunk",
                chunk_info={
                    "content_type": chunk_result.content_type,
                    "chunk_index": chunk_result.chunk_index,
                    "total_chunks": chunk_result.total_chunks,
                    "span_start": chunk_result.span_start,
                    "span_end": chunk_result.span_end,
                    "matched_text": chunk_result.chunk_text,
                }
            )
            by_message[key] = result

    # Sort by score descending
    return sorted(by_message.values(), key=lambda r: r.score, reverse=True)
```

#### MMR Adaptation

For hybrid search with MMR re-ranking, the `_get_embedding` method needs to handle chunk vectors:

```python
async def _get_embedding(self, result) -> np.ndarray | None:
    """Get the best embedding vector for MMR re-ranking."""
    if result.source == "semantic_chunk" and result.chunk_info:
        # Fetch the specific chunk vector that produced this match
        chunk_id = f"{result.parent_id}_chunk_{result.chunk_info['content_type']}" \
                   f"_{result.chunk_info['chunk_index']}"
        return await self._fetch_chunk_vector(chunk_id)
    else:
        # Existing inline vector retrieval
        return await self._fetch_inline_vector(result)
```

### 8.5 CosmosDB Document Size Handling

#### Transcript Size Validation

Add size validation to `cosmos.py:sync_transcript_lines()`, mirroring the existing event size check:

```python
# In sync_transcript_lines, after building the document:
doc_json = json.dumps(doc)
doc_size = len(doc_json.encode("utf-8"))

COSMOS_TRANSCRIPT_SIZE_LIMIT = 1_800_000  # 1.8MB (safety margin below 2MB)

if doc_size > COSMOS_TRANSCRIPT_SIZE_LIMIT:
    logger.warning(
        f"Transcript document {doc_id} exceeds size limit "
        f"({doc_size:,} bytes > {COSMOS_TRANSCRIPT_SIZE_LIMIT:,}). "
        f"Storing without inline vectors."
    )
    # Remove inline vectors to reduce size
    for vec_col in ["user_query_vector", "assistant_response_vector",
                    "assistant_thinking_vector", "tool_output_vector"]:
        doc.pop(vec_col, None)

    # If still too large, truncate content
    doc_json = json.dumps(doc)
    doc_size = len(doc_json.encode("utf-8"))
    if doc_size > COSMOS_TRANSCRIPT_SIZE_LIMIT:
        doc["content_truncated"] = True
        doc["original_content_size"] = doc_size
        # Store only metadata, not full content
        doc["content"] = "[Content too large for Cosmos - stored in chunks only]"
```

#### Vector Externalization

For Cosmos, the chunking approach naturally externalizes vectors to separate documents. For messages
that need chunking, the parent transcript document stores NO vectors (they're all in chunk
documents). This keeps parent documents small.

### 8.6 Content-Type-Specific Chunking Strategies

Different content types benefit from different splitting approaches:

#### Assistant Thinking Blocks

Characteristics: Long-form reasoning, distinct phases, topic shifts, code interspersed.

**Strategy**: Markdown-aware splitting with emphasis on paragraph boundaries.
- Split at `\n\n` boundaries (reasoning paragraphs)
- Keep code blocks intact (don't split ``` regions)
- Respect numbered lists as atomic units
- Higher overlap (192 tokens) to maintain reasoning continuity

#### Assistant Responses

Characteristics: Structured markdown, headers, code blocks, lists.

**Strategy**: Structure-aware splitting.
- Split at markdown header boundaries (## sections)
- Keep code blocks intact
- Each header section becomes a potential chunk boundary
- Standard overlap (128 tokens)

#### User Queries

Characteristics: Usually short (under token limit). Rarely need chunking.

**Strategy**: Sentence-based splitting (fallback only).
- Almost never triggered
- Simple sentence split if somehow needed

#### Tool Output

Characteristics: Already truncated at 10,000 chars. Can be structured (JSON, code, logs).

**Strategy**: Line-based splitting.
- Split at newline boundaries
- Respect JSON structure if detected
- Lower overlap (64 tokens) since tool output is reference material

---

## 9. Implementation Phases

### Phase 0: Safety Net (Immediate - 1-2 days)

**Goal**: Stop the bleeding. Prevent API errors without any schema changes.

**Changes**:
1. Add `tiktoken` dependency
2. Add `truncate_to_tokens()` utility in `content_extraction.py`
3. In `_embed_non_none()`: truncate any text exceeding 8,192 tokens before API call
4. Log a warning when truncation occurs (to track frequency)
5. In `cosmos.py:sync_transcript_lines()`: add document size check before `upsert_item()`
6. For oversized Cosmos documents: strip vector fields, then truncate content if still too large

**Testing**: Run sync against the known-failing sessions from Issue #4 evidence table.

**What this does NOT solve**: Search quality for truncated content. A 75k token thinking block
will only have its first 8k tokens searchable. But at least the data syncs.

### Phase 1: Token-Aware Sentence Chunking (1-2 weeks)

**Goal**: Proper chunking with span vectors. Full search quality for long content.

**Changes**:
1. Implement `chunking.py` with `ChunkingStrategy` class
2. Implement sentence/paragraph splitting with markdown awareness
3. Implement token-aware merging with overlap
4. Add `transcript_chunks` table to all backends (DuckDB, SQLite, Cosmos)
5. Modify `sync_transcript_lines()` to use two-path strategy (inline vs chunks)
6. Modify search pipeline to query both inline and chunk vectors
7. Update MMR re-ranking to handle chunk vectors
8. Update `vector_metadata` to track chunking information
9. Add configuration for chunk sizes (`CHUNK_TARGET_TOKENS`, `CHUNK_OVERLAP_TOKENS`)

**Dependencies**: `tiktoken` (for token counting)

**Testing**:
- Unit tests for chunking: various content sizes, markdown structures, code blocks
- Integration tests: end-to-end sync and search with chunked content
- Regression tests: short content still uses fast inline path
- Performance tests: search latency with chunk vectors vs inline only

### Phase 2: Semantic Chunking Enhancement (1-2 weeks)

**Goal**: Better chunk boundaries that respect semantic coherence.

**Changes**:
1. Add semantic breakpoint detection (embedding-based distance calculation)
2. Implement double-merging pass for chunk coherence
3. Add `chunking_strategy` field to chunk records (versioning)
4. Configuration for breakpoint threshold, merging threshold
5. Evaluate: use lightweight embedding model (e.g., `text-embedding-3-small`) for breakpoint
   detection to reduce cost, and `text-embedding-3-large` for final chunk embeddings

**Decision point**: Whether the semantic chunking improvement justifies the added embedding cost.
Measure search quality (precision@k, recall@k) before and after on a test set.

**Dependencies**: None beyond Phase 1. Uses existing embedding provider for breakpoint detection.

### Phase 3: Hierarchical Retrieval with Auto-Merging (2-3 weeks)

**Goal**: Multi-resolution retrieval. Return the right granularity for each query.

**Changes**:
1. Implement hierarchical chunk creation (3 levels: 4096/1024/256 tokens)
2. Store parent chunks as non-embedded records (content only, no vector)
3. Add `parent_chunk_id` to chunk schema (for hierarchy navigation)
4. Implement auto-merging retriever logic in search pipeline
5. Add `chunk_level` to chunk records (0=leaf, 1=mid, 2=top)

**Design**: Only leaf chunks (level 0) get embedded and indexed. Parent chunks (levels 1, 2) are
stored with `vector=NULL` and retrieved by ID when auto-merging triggers. This follows LlamaIndex's
proven pattern: flat vector index + hierarchical metadata.

**Auto-merging logic**:
```
Query -> retrieve top-K leaf chunks
For each leaf chunk:
    Look up parent (level 1) chunk
    Count how many of parent's children are in results
    If > 50% of children present: replace children with parent
Repeat for level 2 if applicable
Return merged results
```

**Dependencies**: Phase 1 schema must support `parent_chunk_id` and `chunk_level` fields (add them
in Phase 1 as nullable, populate in Phase 3).

---

## 10. Migration Strategy

### Forward Compatibility

Phase 1 schema includes fields needed by Phase 3 (as nullable):

```sql
-- Added in Phase 1, used in Phase 3
parent_chunk_id VARCHAR,     -- NULL until Phase 3
chunk_level INTEGER DEFAULT 0, -- 0 (leaf) until Phase 3
```

### Existing Data

Existing transcript records with inline vectors continue to work. No migration of existing data is
required. The search pipeline queries both inline and chunk vectors.

**Optional re-indexing**: After Phase 1 ships, a background job can re-process existing sessions
that failed to sync (Issue #4 evidence). This would:
1. Read the original transcript content (stored intact in `content` field)
2. Run the chunking pipeline
3. Store chunk vectors
4. The messages that previously had NULL vectors now become searchable

### Schema Versioning

Add a `schema_version` field to the database metadata:

```python
SCHEMA_VERSIONS = {
    1: "Original 4-vector inline schema",
    2: "Added transcript_chunks table (Phase 1)",
    3: "Added hierarchical chunk fields (Phase 3)",
}
```

Backend `_initialize()` checks current version and applies migrations.

---

## 11. Vector Count & Storage Impact Analysis

### Current State

| Metric | Value |
|--------|-------|
| Vectors per message | 1-4 (sparse, usually 1-2) |
| Vector dimensions | 3072 (text-embedding-3-large) |
| Bytes per vector (DuckDB) | 3072 x 4 bytes = 12,288 bytes (~12KB) |
| Bytes per vector (Cosmos JSON) | 3072 x ~8 bytes = ~25KB |
| HNSW indexes | 4 (one per vector column) |

### With Chunking (Phase 1)

For a 75,000 token thinking block with 2048-token chunks:

```
75,000 / 2,048 = ~37 chunks (with overlap: ~40 chunks)
```

| Metric | Short Message | Long Message (75k tokens) |
|--------|---------------|---------------------------|
| Inline vectors | 1-2 | 0 (for chunked type) |
| Chunk vectors | 0 | ~40 |
| Total vectors | 1-2 | ~40 |
| Chunk storage (DuckDB) | 0 | 40 x 12KB = ~480KB |
| Chunk storage (Cosmos) | 0 | 40 x 30KB = ~1.2MB (across 40 documents) |

### Aggregate Impact Estimate

Based on the Issue #4 evidence (34 failing sessions out of 3,490 total = ~1% of sessions):

```
3,490 sessions x ~20 messages average = ~70,000 messages
1% with long content = ~700 messages needing chunking
Average 20 chunks per long message = ~14,000 new chunk vectors
14,000 chunks x 12KB = ~168MB additional DuckDB storage
14,000 chunks x 30KB = ~420MB additional Cosmos storage (across documents)
```

This is modest. The HNSW index growth from 14,000 additional vectors (on top of ~70,000 inline
vectors) is ~20%, well within DuckDB's capabilities.

### HNSW Index Implications

**DuckDB**: The new `idx_chunks_vector` is a single HNSW index for all chunk vectors. With the
`content_type` filter applied at query time, the search is efficient. HNSW search is
O(log N x M) where N is total vectors and M is the number of connections per node.

**Cosmos**: Uses quantizedFlat vector index. Chunk documents are small, separate documents.
Cosmos handles the indexing automatically.

### Cost Implications

Additional embedding API calls for chunking:

```
14,000 chunks x avg 2,048 tokens = ~28.7M tokens
text-embedding-3-large pricing: ~$0.13/1M tokens
Additional cost: ~$3.73 (one-time for re-indexing existing data)
Ongoing: ~$0.11/day (assuming 10 new failing sessions/day)
```

Negligible cost impact.

---

## 12. Open Questions & Risks

### Open Questions

1. **Chunk target size: 2048 vs 512 vs 1024?**
   - 2048: Fewer chunks, more context per chunk, still well within 8192 limit
   - 512: More chunks, more precise matching, aligns with LlamaIndex defaults
   - 1024: Middle ground
   - **Recommendation**: Start with 1024 for Phase 1. Evaluate search quality. The
     `CHUNK_TARGET_TOKENS` should be configurable.

2. **Should we use a lightweight model for semantic breakpoint detection in Phase 2?**
   - `text-embedding-3-small` (1536 dims) is faster and cheaper for breakpoint detection
   - Final chunk embeddings still use `text-embedding-3-large` (3072 dims)
   - **Recommendation**: Yes, use the small model for breakpoint detection only.

3. **Should the Phase 0 truncation be kept permanently as a safety net?**
   - Even after chunking is implemented, there could be edge cases
   - **Recommendation**: Yes, keep truncation as defense-in-depth. Log when it triggers.

4. **How should `_get_embedding()` work for MMR when a message has both inline and chunk vectors?**
   - Current: returns first non-null inline vector
   - With chunks: should return the best-matching chunk vector
   - **Recommendation**: Track which vector/chunk produced the match in SearchResult metadata.
     Use that specific vector for MMR re-ranking.

5. **Re-indexing strategy for existing failed sessions?**
   - The sync daemon could re-process failed sessions after Phase 1 ships
   - Need to identify which sessions failed and retry them
   - **Recommendation**: Add a "needs_reindex" flag or separate re-index job.

6. **Should we parallelize the 4 `_embed_non_none` calls in
   `_generate_multi_vector_embeddings`?**
   - Currently sequential (`await` one after another)
   - Could use `asyncio.gather()` for ~4x throughput
   - **Recommendation**: Yes, fix this as part of Phase 1 (low effort, high impact).

### Risks

1. **Search latency increase**: Querying two tables (inline + chunks) adds latency.
   - Mitigation: The UNION query can be optimized. Profile before and after.
   - If needed, create a materialized view combining both.

2. **HNSW index rebuild time**: Adding many chunk vectors requires index rebuild.
   - Mitigation: DuckDB HNSW indexes are built incrementally on INSERT.
   - Monitor index build time during bulk re-indexing.

3. **Semantic chunking quality**: Poor chunk boundaries could create "orphan" chunks that don't
   match any query well.
   - Mitigation: Overlap between chunks ensures no content is lost. Phase 2 semantic
     breakpoints improve boundary quality.

4. **tiktoken version drift**: The tokenizer must match what the Azure API uses internally.
   `text-embedding-3-large` uses `cl100k_base`.
   - Mitigation: Pin tiktoken version. Add a startup check that token counts are reasonable.

5. **Cosmos throughput**: 40 chunk documents per message means 40 upsert operations.
   - Mitigation: Use batch operations if Cosmos SDK supports it. Or use bulk executor.

---

## 13. References

### External Research

- **AIContext** (https://github.com/AIGeekSquad/AIContext) - Semantic chunking via embedding-based
  breakpoint detection. 7-stage pipeline. Overlapping context windows. C# implementation.

- **LlamaIndex** (https://github.com/run-llama/llama_index):
  - `HierarchicalNodeParser` - Multi-level chunk hierarchy (2048/512/128)
  - `AutoMergingRetriever` - Retrieval-time chunk recombination (ratio threshold 0.5)
  - `SemanticSplitterNodeParser` - Embedding-based adaptive breakpoints
  - `SemanticDoubleMergingSplitterNodeParser` - Two-pass merging with three thresholds
  - `SentenceSplitter` - Cascading boundary-aware splitting

### Internal Code References

| File | Lines | Description |
|------|-------|-------------|
| `content_extraction.py` | 20-174 | Content extraction by role |
| `backends/duckdb.py` | 209-283 | Embedding generation (no token limits) |
| `backends/duckdb.py` | 314-413 | Schema + HNSW indexes |
| `backends/duckdb.py` | 627-732 | Transcript sync pipeline |
| `backends/duckdb.py` | 1349-1453 | Vector search with GREATEST() |
| `backends/duckdb.py` | 901-958 | Hybrid search + MMR |
| `embeddings/azure_openai.py` | 231-298 | embed_batch (no token guard) |
| `backends/cosmos.py` | 782-871 | Cosmos transcript sync (no size check) |
| `backends/cosmos.py` | 1235-1278 | Cosmos event size validation (model for transcripts) |
| `search/mmr.py` | 60-144 | MMR algorithm (uses avg not max) |

### GitHub Issues

- [#4: Transcript sync fails: embedding text exceeds model token limit (8192)](https://github.com/microsoft/amplifier-session-sync/issues/4)
- [#5: Transcript documents exceed CosmosDB 2MB limit (RequestEntityTooLarge)](https://github.com/microsoft/amplifier-session-sync/issues/5)

---

## Appendix A: Phase 1 Task Breakdown

```
Phase 0: Safety Net
  [ ] Add tiktoken dependency to pyproject.toml
  [ ] Implement count_tokens() and truncate_to_tokens() utilities
  [ ] Add truncation guard in _embed_non_none() (all backends)
  [ ] Add document size check in cosmos.py:sync_transcript_lines()
  [ ] Add warning logging for truncation events
  [ ] Test against known-failing sessions

Phase 1: Token-Aware Sentence Chunking
  [ ] Implement chunking.py module
      [ ] ChunkResult dataclass
      [ ] ChunkingStrategy class
      [ ] Markdown-aware sentence splitting
      [ ] Token-aware segment merging with overlap
      [ ] Force-split for oversized segments
      [ ] Content-type-specific strategies
  [ ] Schema changes
      [ ] transcript_chunks table creation in DuckDB backend
      [ ] transcript_chunks table creation in SQLite backend
      [ ] transcript_chunk document type in Cosmos backend
      [ ] HNSW index on chunks vector column
      [ ] Schema version tracking
  [ ] Modified ingestion
      [ ] Two-path classification (inline vs chunk) in sync_transcript_lines
      [ ] Chunk embedding batch processing
      [ ] Chunk storage in transcript_chunks table
      [ ] Updated vector_metadata with chunking info
      [ ] Parallelize 4 _embed_non_none calls with asyncio.gather
  [ ] Modified search
      [ ] Chunk vector search query
      [ ] Result aggregation (chunk -> message level)
      [ ] Unified search across inline + chunk vectors
      [ ] MMR adaptation for chunk vectors
      [ ] SearchResult enrichment with chunk_info
  [ ] Testing
      [ ] Unit tests: chunking various content types and sizes
      [ ] Unit tests: markdown-aware splitting edge cases
      [ ] Integration tests: sync + search round-trip with chunks
      [ ] Regression tests: short content uses inline path
      [ ] Performance benchmarks: search latency with chunks
  [ ] Documentation
      [ ] Update EMBEDDING_STRATEGY.md
      [ ] Update MULTI_VECTOR_IMPLEMENTATION.md
      [ ] Update SCHEMA_MAPPING.md
```

---

## Appendix B: Configuration Reference

```python
# Embedding limits
EMBED_TOKEN_LIMIT = 8192          # Azure OpenAI text-embedding-3-large limit
EMBED_TOKEN_ENCODING = "cl100k_base"  # Tokenizer encoding

# Chunking parameters (Phase 1)
CHUNK_TARGET_TOKENS = 1024        # Target tokens per chunk
CHUNK_OVERLAP_TOKENS = 128        # Overlap between adjacent chunks
CHUNK_MIN_TOKENS = 64             # Minimum chunk size (avoid tiny chunks)

# Semantic chunking parameters (Phase 2)
BREAKPOINT_PERCENTILE = 0.75      # Percentile threshold for semantic breakpoints
BUFFER_SIZE = 1                   # Sentences of context for breakpoint detection
MERGING_THRESHOLD = 0.8           # Similarity threshold for double-merge pass

# Hierarchical parameters (Phase 3)
HIERARCHY_LEVELS = [4096, 1024, 256]  # Token sizes per level
AUTO_MERGE_RATIO = 0.5            # Fraction of children needed to trigger merge

# CosmosDB safety
COSMOS_TRANSCRIPT_SIZE_LIMIT = 1_800_000  # 1.8MB safety margin
COSMOS_CHUNK_BATCH_SIZE = 25      # Chunks per batch upsert

# CosmosDB safety
COSMOS_TRANSCRIPT_SIZE_LIMIT = 1_800_000  # 1.8MB safety margin
COSMOS_CHUNK_BATCH_SIZE = 25      # Chunks per batch upsert

# Tool output (existing)
MAX_TOOL_OUTPUT_LENGTH = 10_000   # Characters (not tokens)
```

---

## Appendix C: Architecture Review Findings

> This section captures the results of a critical architecture review conducted after the
> initial plan was written. The findings below represent binding corrections to the design.

### Review Verdict

The problem analysis, root cause analysis, and Phase 0 safety net are sound. The two-path
strategy (inline for short, chunks for long) is correct. However, the plan over-engineers
later phases and has correctness bugs in the search pipeline design.

**Revised complexity assessment:**
- Phases 0-1: Appropriate complexity (4/10)
- Phases 2-3: Over-engineered (9/10) - need significant scoping down

### Critical Corrections

#### C.1 HNSW + content_type Filtering Bug (MUST FIX)

The plan proposes filtering chunk search results by `content_type` after HNSW index retrieval.
**DuckDB HNSW indexes do not support pre-filtered search.** The index returns top-K globally,
then filters apply. This means:

- Searching for `content_type = 'assistant_thinking'` with `LIMIT 20` could return 20 results
  that are all `assistant_response` chunks, and the filter eliminates all of them.
- Over-fetching (e.g., `LIMIT 200`) is wasteful and unreliable.

**Resolution:** Drop `content_type` filtering from chunk vector search entirely. Let cosine
similarity naturally surface the most relevant chunks regardless of type. This also simplifies
the search query. The `content_type` is still stored as metadata for display purposes.

#### C.2 Over-fetch LIMIT Bug (MUST FIX)

The plan uses `LIMIT ? * 2` for chunk search to allow for deduplication. This is insufficient.
A single long message with 40 chunks could dominate the top-80 results, displacing valid
matches from other messages.

**Resolution:** Use SQL-level deduplication instead of Python-side over-fetching:

```sql
-- Get best chunk per parent message, then rank
WITH ranked_chunks AS (
    SELECT
        parent_id,
        chunk_text,
        content_type,
        chunk_index,
        span_start,
        span_end,
        array_cosine_similarity(vector, {query_vec}::FLOAT[3072]) as similarity,
        ROW_NUMBER() OVER (PARTITION BY parent_id ORDER BY
            array_cosine_similarity(vector, {query_vec}::FLOAT[3072]) DESC
        ) as rn
    FROM transcript_chunks
    WHERE vector IS NOT NULL AND user_id = ?
)
SELECT * FROM ranked_chunks
WHERE rn = 1
ORDER BY similarity DESC
LIMIT ?;
```

This guarantees one result per parent message (the best-matching chunk) regardless of how
many chunks that message has.

#### C.3 Evaluate Summary Embedding Alternative (MUST EVALUATE)

Before committing to the chunk table approach, the plan should have evaluated a simpler
alternative: **truncation + summary embedding**.

- Use the LLM to produce a ~500-token summary of the full content
- Embed the summary as the inline vector (no schema change)
- Store the summary in `vector_metadata` for display

**When summary embeddings might be BETTER than chunks:**
- "Find the session where we discussed authentication" (gestalt matching)
- "What sessions involved database migration work?" (topic-level search)

**When chunks are BETTER than summary embeddings:**
- "Find the exact part where the agent decided to use Redis" (precise location)
- "Show me the thinking where the architect evaluated caching approaches" (span-level)

**Decision required:** What is the dominant search use case? If it's primarily session-level
topic matching, summary embeddings are simpler and potentially more effective. If it's
span-level precision retrieval, chunks are necessary.

**Recommendation:** Implement chunks (the plan's approach) because:
1. The team-session-analyst already uses multi-vector search for span-level precision
2. Summary generation adds LLM latency and cost to every sync operation
3. Chunks are more general-purpose - they handle both use cases (best chunk score for
   ranking, chunk_info for span-level results)
4. Summary embedding could be added LATER as an additional vector on the parent message
   if gestalt matching proves valuable

But this evaluation should be documented explicitly so the decision is principled.

#### C.4 Remove Phase 3 Entirely (BINDING)

The auto-merging retriever (LlamaIndex pattern) solves a problem session transcripts don't
have. In document retrieval, you need multi-resolution because the right "answer unit" varies
(paragraph vs section vs chapter). In session search, the answer unit is always the
**message** - chunk embeddings help FIND the right message, and the response returns the full
message content via the parent join.

**Phase 3 is removed from the plan.** If future evidence proves otherwise, design it then.
Consequently:
- Remove `parent_chunk_id` and `chunk_level` forward-compat columns from Phase 1 schema
- Remove hierarchical parameters from configuration
- Remove Phase 3 from implementation phases

#### C.5 Demote Phase 2 to "Measure and Decide" (BINDING)

Semantic breakpoint detection requires embedding every sentence group just to decide where to
split. For session transcripts specifically, the content is already structurally organized:
- AI thinking blocks have clear paragraph breaks
- Responses have markdown headers and sections
- Code blocks have natural boundaries

The markdown-aware structural splitting in Phase 1 already respects these boundaries.

**Phase 2 becomes a checkpoint, not a planned phase:**
- After Phase 1 ships, measure search quality (precision@k, recall@k) on a test set
- If structural chunking produces adequate results (expected), stop
- If not, THEN investigate semantic chunking vs other approaches based on evidence

### Important Corrections to Schema

#### C.6 Remove Premature Columns

Remove from `transcript_chunks` table:
- `chunking_strategy VARCHAR` - premature versioning, add when actually needed
- `parent_chunk_id VARCHAR` - Phase 3 artifact, removed
- `chunk_level INTEGER` - Phase 3 artifact, removed

Remove from `vector_metadata` tracking:
- `chunked_types` and `chunk_counts` - the UNION search approach doesn't need them

**Simplified schema:**

```sql
CREATE TABLE IF NOT EXISTS transcript_chunks (
    id VARCHAR NOT NULL PRIMARY KEY,
    parent_id VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    content_type VARCHAR NOT NULL,
    chunk_index INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    vector FLOAT[3072],
    embedding_model VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Important Corrections to Process

#### C.7 Add Phase 0.5: Extract Shared Embedding Logic

The explorer confirmed that `_embed_non_none()` and `_generate_multi_vector_embeddings()` are
identically implemented in both DuckDB and Cosmos backends. Adding chunking logic to both
independently is a divergence bug waiting to happen.

**Insert Phase 0.5 between Phase 0 and Phase 1:**
- Extract shared embedding/chunking logic into a mixin or utility module
- Both backends delegate to the shared implementation
- This reduces Phase 1 blast radius from "modify two 2000-line files" to "modify one module"

#### C.8 Design Partial Failure Handling

If a message produces 40 chunks and the embedding API fails on chunks 39-40:

**Resolution:** All-or-nothing with truncated fallback.
- If any chunk embedding fails, discard all chunks for this message
- Fall back to truncating the original text to 8,192 tokens and storing as inline vector
- Log a warning with the failure details
- The message is still searchable (via truncated inline vector), just with reduced quality

#### C.9 Design Chunk Cleanup on Re-sync

If the sync daemon re-processes a session, chunk boundaries may differ between runs.

**Resolution:** Delete-before-insert within a transaction.
```python
# In _store_chunks():
# 1. Delete existing chunks for this parent
DELETE FROM transcript_chunks WHERE parent_id = ?
# 2. Insert new chunks
INSERT INTO transcript_chunks ...
# 3. Commit
```

This ensures idempotency and prevents orphaned chunks from previous runs.

#### C.10 Simplify Sentence Splitting Dependency

Phase 1 uses sentence splitting but doesn't specify the implementation.

**Resolution:** Use regex-based splitting for Phase 1 (zero new dependencies beyond tiktoken).

```python
import re

def split_sentences(text: str) -> list[str]:
    """Simple sentence splitting - sufficient for structurally organized AI output."""
    # Split on sentence-ending punctuation followed by whitespace
    # Handles common abbreviations (Mr., Dr., etc.) via negative lookbehind
    pattern = r'(?<!\b(?:Mr|Mrs|Ms|Dr|Prof|Jr|Sr|vs|etc|approx))\.\s+'
    return [s.strip() for s in re.split(pattern, text) if s.strip()]
```

NLTK and spaCy are overkill for this content type. AI-generated content is well-structured.

#### C.11 MMR Simplification for Phase 1

The MMR adaptation for chunk vectors is the most complex part of the search changes and the
plan's treatment is insufficient.

**Resolution for Phase 1:** Skip MMR re-ranking for chunk-sourced results. Apply MMR only
to inline results. This avoids the per-chunk DB lookup during MMR and the semantic mismatch
when comparing inline vectors against chunk vectors.

Add chunk-aware MMR as a Phase 1b enhancement after the basic pipeline is verified.

### Revised Phase Plan (BINDING)

```
Phase 0: Safety Net                                    [1-2 days]
    - Add tiktoken dependency
    - Truncation guard in _embed_non_none() (all backends)
    - Cosmos document size check in sync_transcript_lines()
    - Warning logging for truncation events
    - Test against known-failing sessions

Phase 0.5: Refactor Shared Embedding Logic             [1 day]
    - Extract _embed_non_none / _generate_multi_vector_embeddings
      from DuckDB and Cosmos into shared utility
    - Parallelize 4 _embed_non_none calls with asyncio.gather
    - Both backends delegate to shared implementation
    - Zero behavior change, just deduplication

Phase 1a: Basic Chunking (DuckDB only)                [3-5 days]
    - chunking.py module with ChunkingStrategy
    - Regex sentence splitting + paragraph boundaries
    - Token-aware merging with overlap
    - transcript_chunks table + HNSW index
    - Two-path ingestion (inline vs chunk)
    - Basic chunk search (SQL dedup, no MMR, no hybrid)
    - Unit + integration tests

Phase 1b: Search Integration                           [3-5 days]
    - UNION search across inline + chunk vectors
    - SQL-level deduplication (ROW_NUMBER PARTITION BY parent_id)
    - Hybrid search integration
    - SearchResult enrichment with chunk_info
    - Full-text search for chunks (LIKE on chunk_text)
    - Search integration tests

Phase 1c: Multi-backend + Polish                       [3-5 days]
    - Cosmos chunk documents (separate document type)
    - SQLite chunk table (sequential scan, no HNSW)
    - Chunk cleanup on re-sync (delete-before-insert)
    - Partial failure handling (all-or-nothing + truncated fallback)
    - MMR adaptation for chunk vectors (or explicit deferral)
    - Re-indexing job for previously failed sessions
    - Documentation updates

CHECKPOINT: Measure search quality.                    [1 day]
    - Build test query set with known-answer sessions
    - Measure precision@k and recall@k
    - Compare: inline-only vs inline+chunks vs chunks-only
    - Decision: Is structural chunking sufficient?
        - If YES -> DONE. Monitor and maintain.
        - If NO -> Investigate semantic chunking based on evidence.
```

**Total estimated effort: 2-3 weeks** (down from 5-8 weeks in the original plan)

### Discarded Elements

| Element | Reason |
|---------|--------|
| Phase 3 (Hierarchical + Auto-Merge) | Solves wrong problem; message is already the retrieval unit |
| Phase 2 (Semantic Chunking) | Demoted to evidence-based checkpoint |
| `parent_chunk_id` column | Phase 3 artifact |
| `chunk_level` column | Phase 3 artifact |
| `chunking_strategy` column | Premature versioning |
| `vector_metadata.chunked_types` | Unnecessary with UNION search |
| NLTK/spaCy dependency | Regex sufficient for AI-generated content |
| Content-type filtering in HNSW search | DuckDB HNSW doesn't support pre-filtering |
| `LIMIT * 2` over-fetch | Replaced with SQL-level dedup |
