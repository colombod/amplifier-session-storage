# Vector Externalization Architecture

> **Supersedes**: Sections 7, 8, 9 of `CHUNKING_ARCHITECTURE_PLAN.md`
> **Builds on**: Sections 1-5 (problem analysis, root causes, research) remain valid
> **Date**: 2026-02-06
> **Status**: PLAN - Not yet implemented

---

## Core Idea

Move ALL vectors out of the `transcripts` table into a dedicated `transcript_vectors`
table. Every vector record knows which parent message it belongs to, which content type
it represents, and which text span it covers. Short texts produce 1 vector entry. Long
texts are semantically chunked and produce N vector entries. The parent transcript stores
only content - no vectors, no size pressure.

This replaces the "two-path" design (inline vectors for short content + chunk table for
long content) with a single unified path for all vectors.

---

## Why This Is Better Than Two-Path

| Concern | Two-Path (original plan) | Externalized Vectors (this plan) |
|---------|--------------------------|----------------------------------|
| Search surface | UNION across 2 tables | Single table, single query |
| HNSW indexes | 4 on transcripts + 1 on chunks = 5 | 1 on transcript_vectors |
| MMR re-ranking | Mixed inline/chunk vectors, complex | Homogeneous vector set, trivial |
| Cosmos 2MB | Still need size checks on parent | Parent has no vectors, never hits 2MB |
| Schema complexity | Two schemas, two write paths | One vector schema, one write path |
| Re-indexing | Must handle both inline and chunks | Drop vectors, re-embed, insert |
| Backward compat | Must query old inline + new chunks | Clean migration, one query after |
| Code duplication | Inline path + chunk path | Single path for all content |

The existing `transcript_vectors` VIEW in DuckDB already separates vectors conceptually.
This architecture makes that separation physical.

---

## Table of Contents

1. [Schema Design](#1-schema-design)
2. [Chunking Pipeline](#2-chunking-pipeline)
3. [Ingestion Pipeline](#3-ingestion-pipeline)
4. [Search Pipeline](#4-search-pipeline)
5. [MMR Re-ranking](#5-mmr-re-ranking)
6. [Backend-Specific Details](#6-backend-specific-details)
7. [Migration Strategy](#7-migration-strategy)
8. [Implementation Phases](#8-implementation-phases)
9. [Task Breakdown](#9-task-breakdown)
10. [Open Questions](#10-open-questions)

---

## 1. Schema Design

### 1.1 Transcripts Table (Vector-Free)

```sql
CREATE TABLE IF NOT EXISTS transcripts (
    id VARCHAR NOT NULL PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    host_id VARCHAR NOT NULL,
    project_slug VARCHAR,
    session_id VARCHAR NOT NULL,
    sequence INTEGER NOT NULL,
    role VARCHAR,
    content JSON,
    turn INTEGER,
    ts TIMESTAMP,
    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_transcripts_session
    ON transcripts (user_id, project_slug, session_id, sequence);
```

**What changed**: Removed all 4 `FLOAT[3072]` vector columns, `embedding_model`,
`vector_metadata`. These fields move to the vectors table. The transcripts table is now
purely content storage.

**What this means for Cosmos**: A transcript document without 4 x 3072 float arrays drops
from ~100KB+ of vector data to just the content. Even the largest transcript messages
(which triggered the 2MB limit) are well within bounds.

### 1.2 Transcript Vectors Table (All Vectors)

```sql
CREATE TABLE IF NOT EXISTS transcript_vectors (
    -- Identity
    id VARCHAR NOT NULL PRIMARY KEY,
    parent_id VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,

    -- What this vector represents
    content_type VARCHAR NOT NULL,

    -- Span and chunking
    chunk_index INTEGER NOT NULL DEFAULT 0,
    total_chunks INTEGER NOT NULL DEFAULT 1,
    span_start INTEGER NOT NULL DEFAULT 0,
    span_end INTEGER NOT NULL,
    token_count INTEGER NOT NULL,

    -- The embedded text and its vector
    source_text TEXT NOT NULL,
    vector FLOAT[3072],

    -- Metadata
    embedding_model VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**ID format**: `{parent_id}_{content_type}_{chunk_index}`

Examples:
- `sess123_msg_5_user_query_0` - user query, single vector (not chunked)
- `sess123_msg_7_assistant_thinking_0` - first chunk of a thinking block
- `sess123_msg_7_assistant_thinking_14` - 15th chunk of same thinking block

**`content_type` values**: `user_query`, `assistant_response`, `assistant_thinking`, `tool_output`

**`source_text`**: The actual text that was embedded. For non-chunked content this is the
full text. For chunks this is the chunk text. Stored so search results can show the matched
text without joining back to the parent for display.

**`span_start` / `span_end`**: Character offsets into the original content text. Enables
highlighting the matched region in the full message.

### 1.3 Indexes

```sql
-- Single HNSW vector index (replaces 4 separate indexes)
CREATE INDEX IF NOT EXISTS idx_vectors_hnsw
    ON transcript_vectors USING HNSW (vector)
    WITH (metric = 'cosine');

-- Lookup indexes
CREATE INDEX IF NOT EXISTS idx_vectors_parent
    ON transcript_vectors (parent_id);

CREATE INDEX IF NOT EXISTS idx_vectors_session
    ON transcript_vectors (user_id, session_id);

CREATE INDEX IF NOT EXISTS idx_vectors_user
    ON transcript_vectors (user_id);
```

**Why one HNSW index instead of four**: The original 4-index design stored vectors in 4
separate columns, requiring 4 separate HNSW indexes. With all vectors in one column
discriminated by `content_type`, one HNSW index covers everything. This is more
storage-efficient and produces a single consistent nearest-neighbor ranking.

**Why no content_type index for HNSW filtering**: DuckDB HNSW does not support
pre-filtered search. The index returns global top-K, then filters apply. Instead of
fighting this, we let cosine similarity rank naturally and filter (if needed) in the
result aggregation layer.

### 1.4 Views (DuckDB)

```sql
-- For reading transcripts without vectors (most common operation)
-- Now trivial since transcripts table has no vectors
CREATE OR REPLACE VIEW transcript_messages AS
SELECT * FROM transcripts;

-- For bulk vector operations (re-indexing, analytics)
CREATE OR REPLACE VIEW vectors_with_context AS
SELECT
    v.*,
    t.role,
    t.content,
    t.turn,
    t.ts
FROM transcript_vectors v
JOIN transcripts t ON t.id = v.parent_id;
```

---

## 2. Chunking Pipeline

### 2.1 Module: `chunking.py`

```python
"""Semantic chunking for texts exceeding embedding token limits.

Short texts (within token limit) produce a single ChunkResult spanning the full text.
Long texts are split at structural boundaries and produce multiple ChunkResults.
All texts go through this pipeline - the output is always a list of ChunkResults.
"""

import re
import tiktoken

EMBED_TOKEN_LIMIT = 8192
CHUNK_TARGET_TOKENS = 1024
CHUNK_OVERLAP_TOKENS = 128
CHUNK_MIN_TOKENS = 64

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = _encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _encoder.decode(tokens[:max_tokens])


@dataclass
class ChunkResult:
    text: str
    span_start: int
    span_end: int
    chunk_index: int
    total_chunks: int      # set after all chunks created
    token_count: int


def chunk_text(text: str, content_type: str) -> list[ChunkResult]:
    """Chunk text for embedding. Always returns at least one ChunkResult."""
    token_count = count_tokens(text)

    # Short text: single chunk, no splitting needed
    if token_count <= EMBED_TOKEN_LIMIT:
        return [ChunkResult(
            text=text,
            span_start=0,
            span_end=len(text),
            chunk_index=0,
            total_chunks=1,
            token_count=token_count,
        )]

    # Long text: split into chunks
    segments = _split_into_segments(text, content_type)
    chunks = _merge_segments(segments)

    # Set total_chunks on all results
    for chunk in chunks:
        chunk.total_chunks = len(chunks)

    return chunks
```

### 2.2 Splitting Strategy

Structural splitting with content-type awareness. No NLP dependencies - AI-generated
content is already well-structured.

```python
def _split_into_segments(text: str, content_type: str) -> list[Segment]:
    """Split text into atomic segments at structural boundaries."""

    if content_type in ("assistant_thinking", "assistant_response"):
        return _split_markdown_aware(text)
    elif content_type == "tool_output":
        return _split_line_aware(text)
    else:
        return _split_sentence_aware(text)


def _split_markdown_aware(text: str) -> list[Segment]:
    """Split respecting markdown structure.

    Priority order:
    1. Markdown headers (## ...) - strongest boundary
    2. Code block boundaries (```) - keep code blocks intact
    3. Double newlines (paragraph breaks)
    4. Sentence boundaries (. ! ?)
    5. Single newlines (line breaks)
    """
    segments = []
    # Phase 1: Protect code blocks (replace with placeholders)
    # Phase 2: Split at headers and paragraph breaks
    # Phase 3: For oversized segments, split at sentence boundaries
    # Phase 4: Restore code blocks into their segments
    # Track character offsets throughout
    ...
    return segments


def _split_sentence_aware(text: str) -> list[Segment]:
    """Simple sentence-aware splitting for user queries."""
    pattern = r'(?<=[.!?])\s+'
    parts = re.split(pattern, text)
    ...
    return segments


def _split_line_aware(text: str) -> list[Segment]:
    """Line-based splitting for tool output."""
    lines = text.split('\n')
    ...
    return segments
```

### 2.3 Segment Merging with Overlap

```python
def _merge_segments(segments: list[Segment]) -> list[ChunkResult]:
    """Merge small segments into chunks up to CHUNK_TARGET_TOKENS.

    Uses overlapping windows: the last CHUNK_OVERLAP_TOKENS of chunk N
    are prepended to chunk N+1 for continuity.
    """
    chunks = []
    current_segments = []
    current_tokens = 0

    for segment in segments:
        seg_tokens = count_tokens(segment.text)

        # Oversized single segment: force-split at word boundaries
        if seg_tokens > CHUNK_TARGET_TOKENS:
            if current_segments:
                chunks.append(_build_chunk(current_segments, len(chunks)))
                current_segments = []
                current_tokens = 0
            chunks.extend(_force_split_segment(segment, len(chunks)))
            continue

        # Would exceed target: flush current chunk, start new with overlap
        if current_tokens + seg_tokens > CHUNK_TARGET_TOKENS and current_segments:
            chunks.append(_build_chunk(current_segments, len(chunks)))
            overlap = _get_overlap_tail(current_segments, CHUNK_OVERLAP_TOKENS)
            current_segments = overlap + [segment]
            current_tokens = sum(count_tokens(s.text) for s in current_segments)
            continue

        current_segments.append(segment)
        current_tokens += seg_tokens

    # Flush remaining
    if current_segments:
        if current_tokens < CHUNK_MIN_TOKENS and chunks:
            # Tiny tail: merge with previous chunk
            chunks[-1] = _extend_chunk(chunks[-1], current_segments)
        else:
            chunks.append(_build_chunk(current_segments, len(chunks)))

    return chunks
```

### 2.4 Safety Truncation (Defense-in-Depth)

Even with chunking, add a guard in the embedding provider:

```python
async def _embed_non_none(self, texts: list[str | None]) -> list[list[float] | None]:
    # ... existing None-filtering ...
    safe_texts = []
    for text in just_texts:
        tokens = count_tokens(text)
        if tokens > EMBED_TOKEN_LIMIT:
            logger.warning(
                f"Text exceeds embed limit ({tokens} > {EMBED_TOKEN_LIMIT}), "
                f"truncating. This indicates a chunking bug."
            )
            text = truncate_to_tokens(text, EMBED_TOKEN_LIMIT)
        safe_texts.append(text)

    embeddings = await self.embedding_provider.embed_batch(safe_texts)
    # ...
```

---

## 3. Ingestion Pipeline

### 3.1 Unified Flow

Every message goes through the same path. No branching between "inline" and "chunk".

```
Message
  |
  v
content_extraction.py:extract_all_embeddable_content()
  |  Returns: {"user_query": str|None, "assistant_response": str|None,
  |            "assistant_thinking": str|None, "tool_output": str|None}
  v
chunking.py:chunk_text() -- for each non-None content type
  |  Short text -> [1 ChunkResult]
  |  Long text  -> [N ChunkResults]
  v
embedding_provider.embed_batch() -- batch all chunk texts together
  |  All chunks across all content types in one batch
  |  Safety truncation as defense-in-depth
  v
Store: transcript row -> transcripts table (no vectors)
       vector rows    -> transcript_vectors table (1..N per content type)
```

### 3.2 Modified sync_transcript_lines

```python
async def sync_transcript_lines(self, user_id, host_id, project_slug,
                                 session_id, lines, start_sequence=0,
                                 embeddings=None):
    """Sync transcript lines with externalized vector storage."""

    # 1. Store transcript rows (content only, no vectors)
    stored = await self._store_transcript_rows(
        user_id, host_id, project_slug, session_id, lines, start_sequence
    )

    # 2. Generate vectors (if embedding provider configured)
    if self.embedding_provider and embeddings is None:
        await self._generate_and_store_vectors(
            user_id, session_id, lines, start_sequence
        )
    elif embeddings is not None:
        # Pre-computed embeddings (from sync daemon)
        await self._store_precomputed_vectors(
            user_id, session_id, lines, start_sequence, embeddings
        )

    return stored


async def _generate_and_store_vectors(self, user_id, session_id,
                                       lines, start_sequence):
    """Extract content, chunk, embed, store vectors."""

    all_vector_records = []
    all_texts_to_embed = []
    text_to_record_map = []  # maps embed index -> vector record

    for i, line in enumerate(lines):
        sequence = start_sequence + i
        parent_id = f"{session_id}_msg_{sequence}"
        content = extract_all_embeddable_content(line)

        for content_type, text in content.items():
            if text is None:
                continue

            chunks = chunk_text(text, content_type)

            for chunk in chunks:
                vector_id = f"{parent_id}_{content_type}_{chunk.chunk_index}"
                record = {
                    "id": vector_id,
                    "parent_id": parent_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "content_type": content_type,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "span_start": chunk.span_start,
                    "span_end": chunk.span_end,
                    "token_count": chunk.token_count,
                    "source_text": chunk.text,
                    "embedding_model": self.embedding_provider.model,
                }

                all_vector_records.append(record)
                all_texts_to_embed.append(chunk.text)
                text_to_record_map.append(len(all_vector_records) - 1)

    if not all_texts_to_embed:
        return

    # Single batch embed call for ALL chunks across ALL messages
    embeddings = await self._embed_non_none(all_texts_to_embed)

    # Attach vectors to records
    for idx, embedding in enumerate(embeddings):
        record_idx = text_to_record_map[idx]
        all_vector_records[record_idx]["vector"] = embedding

    # Bulk insert into transcript_vectors table
    await self._store_vector_records(all_vector_records)

    logger.info(
        f"Stored {len(all_vector_records)} vectors for "
        f"{len(lines)} messages in session {session_id}"
    )
```

### 3.3 Idempotent Re-sync

When re-syncing a session (e.g., after a chunking algorithm update), delete existing
vectors before inserting new ones:

```python
async def _store_vector_records(self, records):
    """Store vector records, replacing any existing vectors for the same parents."""

    # Group by parent_id
    parents = {r["parent_id"] for r in records}

    def _sync():
        with self.conn.cursor() as cur:
            # Delete existing vectors for these parents
            for parent_id in parents:
                cur.execute(
                    "DELETE FROM transcript_vectors WHERE parent_id = ?",
                    [parent_id]
                )

            # Bulk insert new vectors
            for record in records:
                vec_literal = self._format_vector_literal(record["vector"])
                cur.execute(f"""
                    INSERT INTO transcript_vectors (
                        id, parent_id, user_id, session_id,
                        content_type, chunk_index, total_chunks,
                        span_start, span_end, token_count,
                        source_text, vector, embedding_model
                    ) VALUES (
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?,
                        ?, {vec_literal}, ?
                    )
                """, [
                    record["id"], record["parent_id"],
                    record["user_id"], record["session_id"],
                    record["content_type"], record["chunk_index"],
                    record["total_chunks"],
                    record["span_start"], record["span_end"],
                    record["token_count"],
                    record["source_text"], record["embedding_model"],
                ])

    await asyncio.to_thread(_sync)
```

### 3.4 Parallelized Embedding

The original code calls `_embed_non_none` sequentially for each of the 4 content types.
With externalized vectors, we batch ALL texts (across all content types and all messages)
into a single `embed_batch` call. This is simpler AND faster.

For very large batches (e.g., 200+ chunks from a session with many long messages), the
embedding provider should internally batch to respect API limits:

```python
async def embed_batch(self, texts: list[str], batch_size: int = 100):
    """Embed texts, internally batching to respect API limits."""
    if len(texts) <= batch_size:
        return await self._embed_single_batch(texts)

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = await self._embed_single_batch(batch)
        results.extend(batch_results)
    return results
```

---

## 4. Search Pipeline

### 4.1 The Key Simplification

The current search builds a `GREATEST()` across 4 inline vector columns with 4 `CASE WHEN`
expressions. With externalized vectors, search becomes a single cosine similarity on one
column, with SQL-level deduplication to get the best match per parent message.

### 4.2 Semantic Search

```python
async def _semantic_search_transcripts(self, options) -> list[SearchResult]:
    query_vector = await self.embedding_provider.embed_text(options.query)
    vec_literal = self._format_vector_literal(query_vector)

    # Build optional content_type filter
    # Note: this filters AFTER HNSW retrieval, so we over-fetch
    type_filter = self._build_type_filter(options)

    sql = f"""
    WITH scored AS (
        SELECT
            v.parent_id,
            v.content_type,
            v.chunk_index,
            v.total_chunks,
            v.span_start,
            v.span_end,
            v.source_text,
            v.token_count,
            v.session_id,
            array_cosine_similarity(v.vector, {vec_literal}) as similarity
        FROM transcript_vectors v
        WHERE v.vector IS NOT NULL
          AND v.user_id = ?
          {type_filter}
    ),
    best_per_message AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY parent_id
                ORDER BY similarity DESC
            ) as rn
        FROM scored
        WHERE similarity > 0.0
    )
    SELECT
        b.parent_id,
        b.content_type,
        b.chunk_index,
        b.total_chunks,
        b.span_start,
        b.span_end,
        b.source_text,
        b.token_count,
        b.session_id,
        b.similarity,
        t.role,
        t.content,
        t.sequence,
        t.turn,
        t.ts
    FROM best_per_message b
    JOIN transcripts t ON t.id = b.parent_id
    WHERE b.rn = 1
    ORDER BY b.similarity DESC
    LIMIT ?
    """

    # ... execute and build SearchResult objects
```

**What `_build_type_filter` does**: Maps the existing `search_in_user`, `search_in_assistant`,
`search_in_thinking`, `search_in_tool` flags to a SQL `AND v.content_type IN (...)` clause.
If all flags are true (default), no filter is applied.

**Over-fetch consideration**: Since content_type filtering happens after HNSW retrieval,
we may need to request more than `LIMIT` from the CTE. In practice this rarely matters
because cosine similarity naturally ranks relevant content types higher. If a user searches
for "authentication flow" and the assistant_thinking chunk about auth is more similar than
a user_query about auth, the thinking chunk wins regardless.

If content_type filtering becomes important, increase the inner LIMIT:

```sql
-- Inner query fetches 5x, outer deduplicates and trims
LIMIT ? * 5  -- in the scored CTE
```

### 4.3 Full-Text Search

Full-text search can now search BOTH the parent content (for role-level matching) AND
chunk source_text (for span-level matching):

```python
async def _full_text_search_transcripts(self, options) -> list[SearchResult]:
    """Full-text search across transcripts and vector source texts."""

    sql = """
    SELECT DISTINCT
        t.id as parent_id,
        t.session_id,
        t.sequence,
        t.role,
        t.content,
        t.turn,
        t.ts,
        1.0 as similarity,
        v.source_text as matched_text,
        v.content_type,
        v.span_start,
        v.span_end
    FROM transcripts t
    LEFT JOIN transcript_vectors v ON v.parent_id = t.id
    WHERE t.user_id = ?
      AND (
          t.content::TEXT LIKE ?
          OR v.source_text LIKE ?
      )
    ORDER BY t.ts DESC
    LIMIT ?
    """
```

This is an improvement over the current full-text search which only searches the `content`
JSON column. Now it also matches against individual chunk texts, giving more precise results
and providing span information for where the match occurred.

### 4.4 Hybrid Search

```python
async def _hybrid_search_transcripts(self, options) -> list[SearchResult]:
    # 1. Get candidates from both search types
    candidate_limit = options.limit * 3
    semantic_results = await self._semantic_search_transcripts(
        options._replace(limit=candidate_limit)
    )
    fulltext_results = await self._full_text_search_transcripts(
        options._replace(limit=candidate_limit)
    )

    # 2. Merge and deduplicate by parent_id
    combined = self._merge_results(semantic_results, fulltext_results)

    if len(combined) <= options.limit:
        return combined

    # 3. MMR re-ranking (see section 5)
    return await self._mmr_rerank(combined, options)
```

### 4.5 SearchResult Enrichment

Search results now carry span information naturally:

```python
@dataclass
class SearchResult:
    session_id: str
    sequence: int
    role: str
    content: Any           # Full parent message content
    score: float
    source: str            # "semantic" | "full_text" | "hybrid"
    metadata: dict

    # NEW: Span information from the matched vector
    matched_text: str | None = None     # The chunk/text that matched
    content_type: str | None = None     # Which content type matched
    span_start: int | None = None       # Character offset in original
    span_end: int | None = None         # Character offset in original
    chunk_index: int | None = None      # Which chunk (0 if not chunked)
    total_chunks: int | None = None     # How many chunks total
```

The tool layer (`tool.py`) already only accesses `content`, `score`, `source`, and
`metadata`. The new span fields are additive - no breaking change for consumers.

---

## 5. MMR Re-ranking

### 5.1 The Simplification

With all vectors in one table, MMR becomes straightforward. No more mixed inline/chunk
confusion, no more `_get_embedding` returning "first non-null from 4 columns."

```python
async def _mmr_rerank(self, candidates, options) -> list[SearchResult]:
    """MMR re-ranking using vectors from the transcript_vectors table."""

    query_vector = await self.embedding_provider.embed_text(options.query)
    query_np = np.array(query_vector)

    # Fetch the specific vector that produced each candidate's match
    vectors = []
    valid_candidates = []

    for candidate in candidates:
        vec = await self._fetch_matched_vector(candidate)
        if vec is not None:
            vectors.append(np.array(vec))
            valid_candidates.append(candidate)

    if not vectors:
        return candidates[:options.limit]

    mmr_results = compute_mmr(
        vectors=vectors,
        query=query_np,
        lambda_param=options.mmr_lambda,
        top_k=options.limit,
    )

    return [valid_candidates[idx] for idx, _ in mmr_results]


async def _fetch_matched_vector(self, result: SearchResult) -> list[float] | None:
    """Fetch the vector that produced this search result."""

    # We know exactly which vector matched: parent_id + content_type + chunk_index
    vector_id = (
        f"{result.session_id}_msg_{result.sequence}"
        f"_{result.content_type}_{result.chunk_index or 0}"
    )

    row = self.conn.execute(
        "SELECT vector FROM transcript_vectors WHERE id = ?",
        [vector_id]
    ).fetchone()

    return list(row[0]) if row else None
```

**Why this is better than the current approach**: The current `_get_embedding` fetches all
4 inline vectors and returns the first non-null one. This means a result found via
`assistant_thinking_vector` might have its MMR computed using the `user_query_vector` - a
semantic mismatch. With externalized vectors, we fetch the exact vector that produced the
match.

### 5.2 Multi-Chunk MMR Awareness (Future Enhancement)

When multiple chunks from the same message match, MMR should treat them as related. A
simple approach: after MMR selection, if the top result is a chunk from a message with
multiple matching chunks, include the count in the result metadata:

```python
# After MMR selection, enrich with multi-chunk info
for result in mmr_results:
    matching_chunks = sum(
        1 for c in all_candidates
        if c.session_id == result.session_id
        and c.sequence == result.sequence
    )
    result.metadata["matching_chunks"] = matching_chunks
    result.metadata["total_chunks"] = result.total_chunks
```

This tells the consumer "this message had 7 out of 40 chunks match your query" - a signal
of broad relevance.

---

## 6. Backend-Specific Details

### 6.1 DuckDB

**Schema**: As described in section 1. Native `FLOAT[3072]` arrays with HNSW index.

**Vector literal format**: Must use `[1.0, 2.0]::FLOAT[N]` string literals (not parameter
binding) for HNSW index usage. Existing `_format_vector_literal()` method handles this.

**Bulk insert**: Use `INSERT INTO ... VALUES` in a loop within a single transaction.
DuckDB's HNSW index updates incrementally on INSERT.

**Views**: Replace existing `transcript_messages` and `transcript_vectors` views.
`transcript_messages` becomes trivial (just `SELECT * FROM transcripts`).
`transcript_vectors` is now a real table.

### 6.2 SQLite

**Schema**: Vector column stored as `vector_json TEXT` (JSON-serialized float array).

```sql
CREATE TABLE IF NOT EXISTS transcript_vectors (
    -- same columns as DuckDB, except:
    vector_json TEXT,       -- JSON array instead of FLOAT[3072]
    -- no HNSW index
);
```

**Search**: Brute-force numpy cosine similarity (existing pattern). Load all vectors,
compute similarity in Python, sort, return top-K. The numpy search is already implemented
in `_numpy_vector_search` - adapt it to query the new table.

**No HNSW**: SQLite doesn't support HNSW. The `sqlite-vss` virtual table could be used
but currently falls back to numpy anyway.

### 6.3 Cosmos DB

**Document type**: Vectors become separate documents with `type: "transcript_vector"`:

```json
{
    "id": "sess123_msg_5_assistant_thinking_3",
    "type": "transcript_vector",
    "partitionKey": "user_abc",
    "parentId": "sess123_msg_5",
    "sessionId": "sess123",
    "contentType": "assistant_thinking",
    "chunkIndex": 3,
    "totalChunks": 12,
    "spanStart": 3072,
    "spanEnd": 4096,
    "tokenCount": 487,
    "sourceText": "Let me analyze the authentication...",
    "vector": [0.123, 0.456, ...],
    "embeddingModel": "text-embedding-3-large"
}
```

**Same container**: Vectors share the container with transcripts and events, discriminated
by the `type` field. Same partition key (`user_id`) enables efficient cross-type queries.

**Vector indexing policy**: Update to index a single vector path instead of four:

```json
{
    "vectorIndexes": [
        {"path": "/vector", "type": "quantizedFlat"}
    ],
    "vectorEmbeddings": [
        {
            "path": "/vector",
            "dataType": "float32",
            "dimensions": 3072,
            "distanceFunction": "cosine"
        }
    ]
}
```

**Search**: Cosmos vector search queries the single vector field:

```sql
SELECT TOP @top_k
    c.id, c.parentId, c.sessionId, c.contentType,
    c.chunkIndex, c.totalChunks, c.spanStart, c.spanEnd,
    c.sourceText, c.tokenCount,
    VectorDistance(c.vector, @queryVector) AS distance
FROM c
WHERE c.type = 'transcript_vector'
  AND c.partitionKey = @userId
ORDER BY VectorDistance(c.vector, @queryVector)
```

Then JOIN with transcript documents client-side to get full message content.

**Document size**: Each vector document is ~30KB (vector data + metadata). Well within
Cosmos limits. Parent transcript documents no longer contain vectors, solving Issue #5.

**Cosmos container migration**: The existing `_ensure_container()` migration pattern handles
this. Detect old vector indexing policy, replace with new single-path policy.

---

## 7. Migration Strategy

### 7.1 Schema Migration (DuckDB)

```python
async def _migrate_to_externalized_vectors(self):
    """Migrate from inline vectors to externalized vector table."""

    # 1. Create new table
    self._create_transcript_vectors_table()

    # 2. Migrate existing inline vectors to new table
    # Only for rows that have non-NULL vectors
    self.conn.execute("""
        INSERT INTO transcript_vectors (
            id, parent_id, user_id, session_id,
            content_type, chunk_index, total_chunks,
            span_start, span_end, token_count,
            source_text, vector, embedding_model
        )
        SELECT
            id || '_user_query_0',
            id, user_id, session_id,
            'user_query', 0, 1,
            0, LENGTH(content::TEXT), 0,
            content::TEXT, user_query_vector,
            embedding_model
        FROM transcripts
        WHERE user_query_vector IS NOT NULL

        UNION ALL

        SELECT
            id || '_assistant_response_0',
            id, user_id, session_id,
            'assistant_response', 0, 1,
            0, LENGTH(content::TEXT), 0,
            content::TEXT, assistant_response_vector,
            embedding_model
        FROM transcripts
        WHERE assistant_response_vector IS NOT NULL

        UNION ALL

        SELECT
            id || '_assistant_thinking_0',
            id, user_id, session_id,
            'assistant_thinking', 0, 1,
            0, LENGTH(content::TEXT), 0,
            content::TEXT, assistant_thinking_vector,
            embedding_model
        FROM transcripts
        WHERE assistant_thinking_vector IS NOT NULL

        UNION ALL

        SELECT
            id || '_tool_output_0',
            id, user_id, session_id,
            'tool_output', 0, 1,
            0, LENGTH(content::TEXT), 0,
            content::TEXT, tool_output_vector,
            embedding_model
        FROM transcripts
        WHERE tool_output_vector IS NOT NULL
    """)

    # 3. Drop vector columns from transcripts
    # DuckDB supports ALTER TABLE DROP COLUMN
    for col in ['user_query_vector', 'assistant_response_vector',
                'assistant_thinking_vector', 'tool_output_vector',
                'embedding_model', 'vector_metadata']:
        self.conn.execute(f"ALTER TABLE transcripts DROP COLUMN IF EXISTS {col}")

    # 4. Drop old HNSW indexes (they reference dropped columns)
    for idx in ['idx_user_query_vector', 'idx_assistant_response_vector',
                'idx_assistant_thinking_vector', 'idx_tool_output_vector']:
        self.conn.execute(f"DROP INDEX IF EXISTS {idx}")

    # 5. Update views
    self._create_views()

    # 6. Record migration
    self._set_schema_version(2)
```

**Note on source_text during migration**: Migrated vectors use `content::TEXT` as
`source_text` because the original text that was embedded isn't stored separately. This
is imprecise for assistant messages where content is a JSON array of blocks. Phase 1b
can improve this by re-extracting content types from the JSON, but for search purposes
the full content text works as `source_text`.

### 7.2 Schema Versioning

```python
SCHEMA_VERSION = 2  # Current target

def _get_schema_version(self) -> int:
    """Check current schema version."""
    try:
        result = self.conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'"
        ).fetchone()
        return int(result[0]) if result else 1
    except Exception:
        return 1  # Original schema, no version tracking

def _set_schema_version(self, version: int):
    self.conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_meta (
            key VARCHAR PRIMARY KEY,
            value VARCHAR
        )
    """)
    self.conn.execute("""
        INSERT INTO schema_meta (key, value) VALUES ('version', ?)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
    """, [str(version)])
```

### 7.3 Backward Compatibility

During the migration window:
- **New code reads new schema**: Vectors from `transcript_vectors` table
- **Old data migrated in bulk**: The migration SQL copies existing inline vectors
- **Old sessions that failed to embed** (Issue #4): Will be re-processed by the re-index
  job after migration, now with chunking support

### 7.4 Re-indexing Previously Failed Sessions

After migration, a re-index job processes sessions that previously failed due to token
limits:

```python
async def reindex_session(self, user_id, session_id):
    """Re-extract, chunk, embed, and store vectors for a session."""

    # 1. Get all transcript messages for this session
    messages = await self.get_transcript_lines(user_id, session_id)

    if not messages:
        return 0

    # 2. Delete existing vectors for this session
    self.conn.execute(
        "DELETE FROM transcript_vectors WHERE session_id = ? AND user_id = ?",
        [session_id, user_id]
    )

    # 3. Re-process through the new pipeline (with chunking)
    await self._generate_and_store_vectors(
        user_id, session_id, messages, start_sequence=0
    )

    return len(messages)
```

---

## 8. Implementation Phases

### Phase 0: Safety Net [1-2 days]

Stop the bleeding. No schema changes.

```
[ ] Add tiktoken dependency to pyproject.toml
[ ] Implement count_tokens() and truncate_to_tokens() in content_extraction.py
[ ] Add truncation guard in _embed_non_none() across all backends
[ ] Add document size check in cosmos.py:sync_transcript_lines()
    - Strip vectors from doc if > 1.8MB
    - Truncate content if still too large
[ ] Warning logging for all truncation events
[ ] Test against known-failing sessions from Issue #4 evidence
```

### Phase 0.5: Extract Shared Embedding Logic [1 day]

Reduce blast radius before the schema change.

```
[ ] Create embeddings/pipeline.py (or similar shared module)
[ ] Extract _embed_non_none from DuckDB and Cosmos backends
[ ] Extract _generate_multi_vector_embeddings from both backends
[ ] Both backends delegate to shared implementation
[ ] Zero behavior change - just deduplication
[ ] Run existing tests to verify no regression
```

### Phase 1a: Vector Externalization + Chunking (DuckDB) [3-5 days]

The core schema change and chunking pipeline.

```
[ ] Implement chunking.py module
    [ ] ChunkResult dataclass
    [ ] chunk_text() function
    [ ] Markdown-aware splitting for assistant content
    [ ] Token-aware segment merging with overlap
    [ ] Force-split for oversized segments
    [ ] Unit tests for chunking (various sizes, markdown, code blocks)

[ ] Schema changes (DuckDB)
    [ ] transcript_vectors table creation
    [ ] HNSW index on vector column
    [ ] Lookup indexes (parent_id, session, user)
    [ ] schema_meta table for versioning

[ ] Modified ingestion (DuckDB)
    [ ] _generate_and_store_vectors() - unified chunking + embedding
    [ ] _store_transcript_rows() - content only, no vectors
    [ ] _store_vector_records() - bulk vector insert
    [ ] Delete-before-insert for idempotent re-sync
    [ ] Partial failure handling (all-or-nothing with truncated fallback)

[ ] Migration
    [ ] _migrate_to_externalized_vectors() SQL
    [ ] Schema version check in _initialize()
    [ ] Auto-migrate on startup

[ ] Basic search (DuckDB)
    [ ] Semantic search against transcript_vectors (CTE with ROW_NUMBER)
    [ ] JOIN to transcripts for content retrieval
    [ ] SearchResult with span enrichment
    [ ] Verify HNSW index is used (EXPLAIN query)
```

### Phase 1b: Full Search Integration [3-5 days]

Complete the search pipeline across all modes.

```
[ ] Full-text search across transcripts + vector source_text
[ ] Hybrid search (semantic + full-text merge)
[ ] MMR re-ranking using matched vector (not "first non-null")
[ ] Content-type filtering (search_in_user/assistant/thinking/tool flags)
[ ] SearchResult span fields (matched_text, content_type, span_start/end)
[ ] Update _get_embedding() to use transcript_vectors table
[ ] Integration tests: search round-trips with chunked and non-chunked content
[ ] Performance benchmarks: search latency vs old inline approach
```

### Phase 1c: Multi-Backend + Polish [3-5 days]

Extend to all backends, handle edge cases.

```
[ ] Cosmos backend
    [ ] Vector document type (type: "transcript_vector")
    [ ] Updated vector indexing policy (single /vector path)
    [ ] Container migration for existing data
    [ ] Vector search against new document type
    [ ] Cosmos-specific search (VectorDistance on single field)

[ ] SQLite backend
    [ ] transcript_vectors table (vector_json TEXT column)
    [ ] numpy brute-force search against new table
    [ ] Migration from inline vectors

[ ] Re-indexing
    [ ] reindex_session() method
    [ ] CLI or daemon integration for batch re-indexing
    [ ] Identify previously failed sessions for re-processing

[ ] Cleanup
    [ ] Remove VECTOR_COLUMNS constant (no longer needed)
    [ ] Remove _format_greatest_similarity helper
    [ ] Remove old view definitions
    [ ] Fix broken upsert_embeddings method (or remove if unused)
    [ ] Update documentation (EMBEDDING_STRATEGY.md, MULTI_VECTOR_IMPLEMENTATION.md,
        SCHEMA_MAPPING.md)
    [ ] Update test suite for new schema
```

### CHECKPOINT: Measure Search Quality [1 day]

```
[ ] Build test query set with known-answer sessions
[ ] Measure precision@10 and recall@10
[ ] Compare: old inline (from backup) vs new externalized
[ ] Evaluate chunked vs non-chunked result quality
[ ] Decision: Is structural chunking sufficient?
    - If YES -> Done. Monitor and maintain.
    - If NO -> Investigate semantic chunking based on evidence.
```

---

## 9. Task Breakdown

### Methods to Modify (per backend)

| Method | DuckDB | SQLite | Cosmos | Change |
|--------|--------|--------|--------|--------|
| `_initialize` | schema + migration | schema + migration | container policy | Add vectors table, migrate |
| `sync_transcript_lines` | Split into rows + vectors | Same | Same | Content to transcripts, vectors to vectors table |
| `vector_search` | Rewrite query | Rewrite numpy search | Rewrite Cosmos query | Query vectors table, JOIN for content |
| `_semantic_search_transcripts` | Use new vector_search | Same | Same | Simplified - single table |
| `_full_text_search_transcripts` | Add JOIN to vectors | Same | Same | Search source_text too |
| `_hybrid_search_transcripts` | Update merge logic | Same | Same | MMR uses matched vector |
| `_get_embedding` | Query vectors table | Same | Implement (missing) | Single-record fetch by vector ID |
| `get_transcript_lines` | No change | No change | No change | Already reads content only |
| `upsert_embeddings` | Fix or deprecate | Fix or deprecate | Fix or deprecate | Currently broken |

### Shared Code Changes

| Component | Change |
|-----------|--------|
| `content_extraction.py` | Add `count_tokens()`, `truncate_to_tokens()` |
| `chunking.py` | NEW - entire module |
| `embeddings/pipeline.py` | NEW - extracted shared embedding logic |
| `search/models.py` or `base.py` | Add span fields to `SearchResult` |
| `backends/base.py` | Update `sync_transcript_lines` signature, add `reindex_session` |

### Constants to Remove

```python
# DuckDB - duckdb.py:43
VECTOR_COLUMNS = frozenset({...})  # DELETE - no longer needed

# SQLite - sqlite.py:42
VECTOR_COLUMNS = frozenset({...})  # DELETE

# Cosmos - cosmos.py:57
VECTOR_FIELDS = {...}              # DELETE
```

### Test Files to Update

All vector-related test files need updating for the new schema:
- `test_duckdb_vector_search.py`
- `test_schema_validation.py`
- `test_mmr.py` (no change - pure algorithm)
- `test_content_extraction.py` (add chunking tests)
- NEW: `test_chunking.py`
- NEW: `test_vector_migration.py`

---

## 10. Open Questions

### 10.1 Source Text Accuracy During Migration

When migrating existing inline vectors to the new table, we don't have the original
extracted text (it was embedded but not stored). Options:

**A. Use content::TEXT**: Imprecise for assistant messages (includes JSON structure).
**B. Re-extract using content_extraction.py**: Accurate but requires processing each
message through the extraction pipeline during migration.
**C. Store NULL source_text for migrated rows**: Mark them as "legacy" and fill in
during re-indexing.

**Recommendation**: Option B for DuckDB/SQLite (can be done in Python during migration).
Option C for Cosmos (avoid re-processing all documents during container migration; fill
in during background re-index).

### 10.2 Embedding Provider Batch Size Limits

Azure OpenAI's embedding API may have per-request limits on total tokens or number of
inputs. With chunking producing many more texts per sync, we need to respect these limits.

**Recommendation**: Add internal batching in `embed_batch()` with configurable batch size
(default 100 texts per API call). The existing LRU cache helps reduce actual API calls.

### 10.3 HNSW Index Build Time

Building the HNSW index on the new table after bulk migration could take time for large
datasets. DuckDB builds HNSW incrementally on INSERT, but a bulk migration INSERT of
100k+ vectors may be slow.

**Recommendation**: Create the HNSW index AFTER the bulk migration INSERT, not before.
This lets DuckDB build the index in one pass rather than incrementally for each row.

### 10.4 Should We Keep embedding_model on Both Tables?

Currently `embedding_model` is on the transcripts table. With externalization, it belongs
on the vectors table (where the actual embeddings live). But keeping it on transcripts too
helps track "this message has been embedded with model X" without querying vectors.

**Recommendation**: Move to vectors table only. If needed for display, JOIN or cache.

### 10.5 Semantic Chunking (Deferred)

The user's preference is semantic chunking. Phase 1 uses structural chunking (markdown-aware
paragraph/sentence splitting) as a pragmatic starting point. Semantic chunking (embedding-based
breakpoint detection, double merging) can be added by replacing the `_split_into_segments`
implementation in `chunking.py` without any schema or pipeline changes. The externalized
vector architecture supports both approaches equally well - it's a chunking algorithm swap,
not an architecture change.

If structural chunking produces inadequate search results (measured at CHECKPOINT), the
semantic enhancement path is:

1. Split text into sentences
2. Embed each sentence (using lightweight model like text-embedding-3-small)
3. Compute cosine distances between adjacent sentence embeddings
4. Split at percentile-threshold breakpoints (AIContext approach)
5. Optional: double-merge pass to rejoin prematurely split chunks (LlamaIndex approach)
6. Feed resulting chunks through the existing pipeline (same storage, same search)

This is a pure `chunking.py` change. No schema, ingestion, or search changes needed.

---

## Summary

The externalized vector architecture replaces the original "two-path" design with a
cleaner normalized model:

```
transcripts table  -- content only, no vectors, no size pressure
       |
       | 1:N
       v
transcript_vectors table  -- ALL vectors (short=1, chunked=N), single HNSW index
```

**One write path**: Extract -> chunk -> embed -> store vectors separately
**One search surface**: Query transcript_vectors, JOIN for content
**One HNSW index**: Replaces 4 sparse column indexes
**One MMR approach**: Fetch the exact matched vector, no "first non-null" guessing
**Zero Cosmos 2MB risk**: Parent documents carry no vector data

Total estimated effort: **2-3 weeks** across all phases.

---

## Appendix: Architecture Review Findings

> Critical review conducted after the design was written. Findings below are binding
> corrections.

### R.1 HNSW + user_id Filtering (HIGH)

DuckDB HNSW returns global top-K across ALL rows, then `WHERE user_id = ?` filters.
In a multi-tenant DB, User A's search could miss their own relevant vectors if User B's
vectors dominate the HNSW top-K.

**Resolution**: This system uses one DuckDB file per user (the `AMPLIFIER_DUCKDB_PATH`
is user-scoped). Confirm this assumption. If shared multi-user DBs exist, either:
- Partition into per-user DB files (preferred)
- Over-fetch from HNSW (e.g., `LIMIT 500`), filter to user, then trim to requested limit
- Accept brute-force scan for filtered queries (fine under ~100K vectors)

**Action**: Add a verification step in Phase 1a to EXPLAIN the search query and confirm
HNSW index is used. Document the single-user-per-DB assumption.

### R.2 Add project_slug to transcript_vectors (HIGH)

Search commonly filters by project. Without `project_slug` on the vectors table, every
project-scoped search needs a JOIN to `transcripts` inside the CTE, defeating vector-first
scanning.

**Resolution**: Add `project_slug VARCHAR` to the `transcript_vectors` schema. Denormalize
it from the parent transcript. Update the schema in Section 1.2:

```sql
    -- Add after session_id:
    project_slug VARCHAR,
```

Update all write paths to populate it. Update search queries to filter directly.

### R.3 Migration source_text Must Use Re-extraction (HIGH)

The migration SQL uses `content::TEXT` as `source_text`. For assistant messages, `content`
is a JSON array like `[{"type":"thinking","thinking":"..."},{"type":"text","text":"..."}]`.
The `content::TEXT` produces the JSON serialization, not the extracted text that was
embedded. This is incorrect - not just imprecise.

**Resolution**: Migration MUST re-extract content using `content_extraction.py` rather
than using `content::TEXT`. The migration becomes a Python loop, not pure SQL:

```python
for row in transcripts_with_vectors:
    content = json.loads(row["content"])
    message = {"role": row["role"], "content": content}
    extracted = extract_all_embeddable_content(message)
    for content_type, text in extracted.items():
        if text and row[f"{content_type}_vector"] is not None:
            # Insert into transcript_vectors with correct source_text
            ...
```

This is slower than bulk SQL but produces correct `source_text` values.

### R.4 Full-Text Search JOIN Produces Non-deterministic Spans (MEDIUM)

The LEFT JOIN in section 4.3 with DISTINCT can produce arbitrary span information when
a message has multiple vector rows matching the LIKE clause.

**Resolution**: Split into two queries and merge:
1. Content search: `SELECT ... FROM transcripts WHERE content::TEXT LIKE ?`
2. Chunk search: `SELECT ... FROM transcript_vectors WHERE source_text LIKE ?`
3. Merge client-side, dedup by parent_id, prefer chunk match (has span info)

### R.5 MMR Should Batch-Fetch Vectors (LOW)

`_fetch_matched_vector` does N individual SELECTs for N candidates.

**Resolution**: Batch fetch with `WHERE id IN (?, ?, ...)`:

```python
async def _fetch_matched_vectors(self, results: list[SearchResult]):
    vector_ids = [self._build_vector_id(r) for r in results]
    rows = self.conn.execute(
        f"SELECT id, vector FROM transcript_vectors WHERE id IN ({placeholders})",
        vector_ids
    ).fetchall()
    return {row[0]: list(row[1]) for row in rows}
```

### R.6 Migration Needs Transaction + Idempotency (MEDIUM)

The migration does 8+ DDL/DML statements with no transaction. If it fails partway, the
next startup retry may fail on duplicate IDs.

**Resolution**: Add idempotency guard:
```python
if self._table_exists("transcript_vectors"):
    count = self.conn.execute("SELECT COUNT(*) FROM transcript_vectors").fetchone()[0]
    if count > 0:
        logger.info("Migration already completed, skipping")
        return
```

And wrap the data migration in a transaction.

### R.7 token_count Should Not Default to 0 for Migrated Rows (LOW)

Setting `token_count = 0` for migrated vectors is incorrect. Use `NULL` to indicate
"unknown" or compute an estimate from text length.

### R.8 Pre-computed Embeddings Path Unspecified (MEDIUM)

The sync daemon can pass pre-computed embeddings via the `embeddings` parameter to
`sync_transcript_lines`. The new design mentions `_store_precomputed_vectors` but doesn't
specify how pre-computed embeddings (which are flat 4-key dicts) map to the new chunked
vector records.

**Resolution**: Pre-computed embeddings represent non-chunked content (1 vector per type).
Store them as single-chunk records (`chunk_index=0, total_chunks=1`). The daemon doesn't
need to know about chunking - it just provides embeddings for content it could embed. If
content exceeded the token limit, the daemon wouldn't have an embedding for it, and the
backend falls through to the self-embedding path with chunking.

### R.9 Merge Phase 1a and 1b (MEDIUM)

Phase 1a creates the new schema but Phase 1b implements full search. You can't test 1a
without search working. The old search won't work against the new schema.

**Resolution**: Merge 1a and 1b into a single "Phase 1: Vector Externalization" phase.
The search must work before you can validate the migration. Target 5-8 days for the
combined phase.

### R.10 source_text Storage Cost (LOW)

Storing `source_text` on every chunk means a 50K-token thinking block chunked into ~50
chunks stores the full text ~50x (with overlap). This could 10-50x text storage for
large sessions.

**Resolution**: Accept the cost. The alternative (JOIN for every search result display)
is worse for latency. The storage increase is bounded by the number of chunked messages
(~1% of total). Monitor and revisit if storage becomes a concern.

### Revised Phase Plan (incorporating review)

```
Phase 0:   Safety net (truncation + Cosmos size check)           [1-2 days]
Phase 0.5: Extract shared embedding logic from backends          [1 day]
Phase 1:   Vector externalization + chunking + search (DuckDB)   [5-8 days]
           - Schema + migration (with re-extraction, not SQL cast)
           - Chunking pipeline
           - Full search pipeline (semantic, full-text, hybrid, MMR)
           - HNSW verification (EXPLAIN)
           - Integration tests
Phase 2:   Multi-backend (Cosmos, SQLite) + re-indexing          [3-5 days]
CHECKPOINT: Measure search quality                               [1 day]
```
