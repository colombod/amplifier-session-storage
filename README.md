# Amplifier Session Storage

Version 0.3.0 | Python >= 3.11 | MIT License

Session storage library for Amplifier with hybrid search, multiple backends, and embedding resilience.

**Core capabilities:**

- Three storage backends: Azure Cosmos DB, DuckDB, SQLite
- Hybrid search combining full-text, semantic, and MMR re-ranking
- Automatic embedding generation during ingestion with retry, circuit breaker, and batch splitting
- Externalized vector storage with chunking and overlap
- Embedding lifecycle management: backfill gaps, rebuild from scratch
- Pluggable embedding providers (Azure OpenAI, OpenAI direct)
- LRU cache for hot query embeddings
- Graceful degradation when embeddings are unavailable

---

## Table of Contents

1. [Architecture](#architecture)
2. [Protocol Split (ABC Hierarchy)](#protocol-split)
3. [Data Model](#data-model)
4. [Backend Comparison](#backend-comparison)
5. [Embedding Pipeline](#embedding-pipeline)
6. [Search Capabilities](#search-capabilities)
7. [Embedding Management](#embedding-management)
8. [Configuration](#configuration)
9. [Installation](#installation)
10. [Quick Start](#quick-start)
11. [Schema Migrations](#schema-migrations)
12. [Error Handling](#error-handling)
13. [MMR Algorithm](#mmr-algorithm)
14. [Development](#development)
15. [Changelog](#changelog)

---

## Architecture

### Storage Backend Abstraction

All backends implement the same `StorageBackend` interface, which is composed from three focused ABCs:

```
StorageBackend = StorageReader + StorageWriter + StorageAdmin
```

Each backend (`CosmosBackend`, `DuckDBBackend`, `SQLiteBackend`) inherits from both `StorageBackend` and `EmbeddingMixin`, which provides the shared embedding generation pipeline with resilience.

```
                    _StorageLifecycle
                   /        |        \
          StorageReader  StorageWriter  StorageAdmin
                   \        |        /
                    StorageBackend
                         |
                    EmbeddingMixin
                   /      |       \
          CosmosBackend  DuckDBBackend  SQLiteBackend
```

### Cosmos DB: Single-Container Architecture

Cosmos DB uses a single container (`session_data`) with a type discriminator field to store all document types. This follows Cosmos DB best practices for partition-aligned queries:

- **Partition key**: `partition_key` = `{user_id}_{project_slug}_{session_id}`
- **Type discriminator**: `type` field = `"session"`, `"transcript"`, `"event"`, or `"transcript_vector"`
- All documents for a session share the same partition key, enabling single-partition queries
- Column projections (`SESSION_PROJECTION`, `TRANSCRIPT_PROJECTION`, `EVENT_PROJECTION`) exclude vector fields to reduce RU cost

### DuckDB and SQLite: Separate Tables

DuckDB and SQLite use four tables:

| Table | Purpose |
|-------|---------|
| `sessions` | Session metadata |
| `transcripts` | Transcript messages (with `has_vectors` flag) |
| `events` | Event log entries |
| `transcript_vectors` | Externalized vector chunks |

A `schema_version` table tracks migration state for automatic upgrades.

### Externalized Vector Storage

Vector embeddings are stored separately from transcript documents. Each transcript message can produce multiple vector chunks (via text chunking with overlap). This design:

- Keeps transcript reads cheap (no vector data loaded)
- Supports multiple chunks per message (long content is split)
- Enables independent vector lifecycle (backfill, rebuild)
- Tracks vector status via `has_vectors` flag on transcript documents

---

## Protocol Split

The backend protocol is split into focused ABCs for granular dependency control. Consumers depend on the narrowest interface they need.

### StorageReader

Read, query, and search operations. Used by search tools, analytics dashboards, and read-only consumers.

```python
class StorageReader(_StorageLifecycle):
    async def get_session_metadata(user_id, session_id) -> dict | None
    async def get_transcript_lines(user_id, project_slug, session_id, after_sequence) -> list[dict]
    async def get_turn_context(user_id, session_id, turn, before, after) -> TurnContext
    async def get_message_context(session_id, sequence, user_id, before, after) -> MessageContext
    async def get_event_lines(user_id, project_slug, session_id, after_sequence) -> list[dict]
    async def search_sessions(user_id, filters, limit) -> list[dict]
    async def search_transcripts(user_id, options, limit) -> list[SearchResult]
    async def search_events(user_id, session_id, ...) -> list[SearchResult]
    async def supports_vector_search() -> bool
    async def vector_search(user_id, query_vector, filters, top_k) -> list[SearchResult]
    async def list_users(filters) -> list[str]
    async def list_projects(user_id, filters) -> list[str]
    async def list_sessions(user_id, project_slug, limit, offset) -> list[dict]
    async def get_active_sessions(user_id, project_slug, start_date, ...) -> list[dict]
    async def get_session_statistics(user_id, filters) -> dict
```

### StorageWriter

Write and sync operations. Used by the sync daemon and session upload server.

```python
class StorageWriter(_StorageLifecycle):
    async def upsert_session_metadata(user_id, host_id, metadata) -> None
    async def sync_transcript_lines(user_id, host_id, project_slug, session_id, lines, ...) -> int
    async def sync_event_lines(user_id, host_id, project_slug, session_id, lines, ...) -> int
    async def upsert_embeddings(user_id, project_slug, session_id, embeddings) -> int
    async def get_session_sync_stats(user_id, project_slug, session_id) -> SessionSyncStats
```

### StorageAdmin

Administrative operations. Used by management tools for lifecycle and maintenance.

```python
class StorageAdmin(_StorageLifecycle):
    async def delete_session(user_id, project_slug, session_id) -> bool
    async def backfill_embeddings(user_id, project_slug, session_id, ...) -> EmbeddingOperationResult
    async def rebuild_vectors(user_id, project_slug, session_id, ...) -> EmbeddingOperationResult
```

### StorageBackend

The composed type combining all three. Concrete backends implement this class.

```python
class StorageBackend(StorageReader, StorageWriter, StorageAdmin):
    pass
```

**Note on `user_id`**: empty string (`""`) means search across all users (team-wide). Non-empty string filters to that specific user.

---

## Data Model

### Session Metadata

```json
{
    "user_id": "dicolomb",
    "session_id": "c1ae0a48-ac3c-4d4f-bff4-10ad260cf86b",
    "host_id": "laptop-01",
    "project_slug": "amplifier",
    "bundle": "foundation",
    "created": "2026-01-15T10:00:00Z",
    "updated": "2026-01-15T11:30:00Z",
    "turn_count": 15,
    "metadata": { ... }
}
```

### Transcript Documents

```json
{
    "id": "sess-abc_msg_0",
    "type": "transcript",
    "user_id": "dicolomb",
    "session_id": "sess-abc",
    "sequence": 0,
    "role": "user",
    "content": "How do I implement vector search?",
    "turn": 0,
    "ts": "2026-01-15T10:00:00Z",
    "has_vectors": true
}
```

The `has_vectors` flag (added in v0.3.0) tracks whether vector embeddings exist for this transcript:

- Set to `false` when `sync_transcript_lines` stores the transcript
- Flipped to `true` when `_store_vector_records` successfully stores vectors
- Reset to `false` by `rebuild_vectors` before regeneration
- Queried by `backfill_embeddings` to find transcripts missing vectors

For Cosmos DB, backward compatibility with pre-0.3.0 documents uses `NOT IS_DEFINED(c.has_vectors) OR c.has_vectors = false`.

### Event Documents

```json
{
    "id": "sess-abc_evt_0",
    "type": "event",
    "user_id": "dicolomb",
    "session_id": "sess-abc",
    "sequence": 0,
    "event": "llm.request",
    "ts": "2026-01-15T10:00:01Z",
    "lvl": "info",
    "data": { ... },
    "data_truncated": false,
    "data_size_bytes": 1024
}
```

### Vector Documents (Externalized)

Each transcript message can produce multiple vector chunks:

```json
{
    "id": "sess-abc_msg_0_user_query_0",
    "type": "transcript_vector",
    "parent_id": "sess-abc_msg_0",
    "user_id": "dicolomb",
    "session_id": "sess-abc",
    "project_slug": "amplifier",
    "content_type": "user_query",
    "chunk_index": 0,
    "total_chunks": 1,
    "span_start": 0,
    "span_end": 42,
    "token_count": 12,
    "source_text": "How do I implement vector search?",
    "embedding_model": "text-embedding-3-large",
    "vector": [0.012, -0.034, ...]
}
```

Four content types are extracted per message:

| Content Type | Source | When Present |
|-------------|--------|-------------|
| `user_query` | User message content | `role == "user"` |
| `assistant_response` | Text blocks in assistant content | `role == "assistant"`, `type == "text"` |
| `assistant_thinking` | Thinking blocks in assistant content | `role == "assistant"`, `type == "thinking"` |
| `tool_output` | Tool result content (truncated to 10K chars) | `role == "tool"` |

---

## Backend Comparison

| Feature | Cosmos DB | DuckDB | SQLite |
|---------|----------|--------|--------|
| **Deployment** | Cloud (Azure) | Local file or in-memory | Local file or in-memory |
| **Multi-device sync** | Yes | No | No |
| **Vector search** | Native (quantizedFlat index) | VSS extension (HNSW) | sqlite-vss extension |
| **Async I/O** | Native async | `asyncio.to_thread()` wrapper | `aiosqlite` |
| **Authentication** | RBAC (DefaultAzureCredential) or key | None (local file) | None (local file) |
| **Schema migration** | Schema-less (backward compat) | Automatic (versioned) | Automatic (versioned) |
| **Best for** | Production, multi-device, team use | Development, analytics, CI | Embedded apps, testing |

### When to Use Each

- **Cosmos DB**: Production deployments with multi-device session sync, team-wide search, cloud-native infrastructure. Requires Azure subscription.
- **DuckDB**: Local development, CI testing with persistent state, single-machine analytics workloads. Fast columnar queries.
- **SQLite**: Embedded applications, lightweight testing, environments where DuckDB is unavailable. Single-file database.

---

## Embedding Pipeline

### Overview

The embedding pipeline transforms transcript messages into searchable vector embeddings:

```
Transcript Message
    |
    v
Content Extraction (4 content types)
    |
    v
Text Chunking (sentence/markdown/line-aware, with overlap)
    |
    v
Embedding Generation (batches of 16, retry + circuit breaker)
    |
    v
Vector Storage (externalized, per-chunk documents)
    |
    v
has_vectors = true (on transcript document)
```

### Content Extraction

Module: `amplifier_session_storage.content_extraction`

Handles complex Amplifier transcript message structures. Extracts separate text for each of the four vector content types from the message `role` and `content` fields. Assistant messages with complex content arrays (thinking blocks, text blocks, tool_call blocks) are decomposed into their constituent parts. Tool outputs are truncated to 10,000 characters.

Token counting uses `tiktoken` with the `cl100k_base` encoding (same as text-embedding-3-large). The embedding token limit is 8,192 tokens.

### Text Chunking

Module: `amplifier_session_storage.chunking`

Short texts (within the 8,192 token limit) produce a single chunk. Long texts are split at structural boundaries with overlap between chunks for continuity:

| Content Type | Splitting Strategy |
|-------------|-------------------|
| `assistant_response`, `assistant_thinking` | Markdown-aware: preserves fenced code blocks as atomic segments, splits on paragraph boundaries, then sentence boundaries |
| `tool_output` | Line-aware: splits on newline boundaries |
| `user_query` (and fallback) | Sentence-aware: splits after sentence-ending punctuation |

Chunking parameters:

| Parameter | Value |
|-----------|-------|
| Target tokens per chunk | 1,024 |
| Overlap between chunks | 128 tokens |
| Minimum chunk size | 64 tokens (runts merged into previous chunk) |

### Embedding Generation with Resilience

Module: `amplifier_session_storage.embeddings.resilience`

The `EmbeddingMixin` provides the shared embedding logic used by all three backends. It wraps every embedding API call with:

**Batch splitting** (`EMBED_BATCH_SIZE = 16`): Large batches are split into groups of 16 texts. Each sub-batch is independent -- a failed batch does not kill the rest.

**Retry with exponential backoff** (`RetryConfig`):

| Setting | Value |
|---------|-------|
| Max retries | 5 |
| Backoff base | 1.0s |
| Backoff cap | 60.0s |
| Backoff multiplier | 2.0x |
| Retryable status codes | 429, 500, 502, 503, 504 |
| Retryable exceptions | `ConnectionError`, `TimeoutError`, `OSError` |

Respects `Retry-After` headers from the API (capped at `backoff_max`).

**Circuit breaker** (`CircuitBreaker`):

| Setting | Value |
|---------|-------|
| Failure threshold | 5 consecutive retryable failures |
| Reset timeout | 60s |

States: `CLOSED` (normal) -> `OPEN` (fail fast) -> `HALF_OPEN` (probe). Only retryable failures (429, 5xx, network) trip the breaker. Non-retryable errors (auth, config) propagate immediately without affecting the breaker. The circuit breaker is shared across all backends in the process.

**Graceful degradation**: If embedding generation fails entirely, `sync_transcript_lines` still stores the transcript documents. The error is logged with a structured `EMBEDDING_FAILURE` prefix including the full identity chain (user, project, session, message count), and a clear explanation of what succeeded vs. what failed. The `exc_info=True` flag preserves the full stack trace.

### Embedding Cache

Module: `amplifier_session_storage.embeddings.cache`

`EmbeddingCache` provides an LRU cache for hot query embeddings. Keys are SHA-256 hashes of `{model_name}:{text}`. Configurable max entries (default 1,000). Automatic eviction of least recently used entries.

### Embedding Providers

Two providers are included, both implementing the `EmbeddingProvider` ABC:

**`AzureOpenAIEmbeddings`** (extra: `azure-openai`): Uses `azure-ai-inference` SDK. Supports RBAC authentication via `DefaultAzureCredential` or API key.

**`OpenAIEmbeddings`** (extra: `openai`): Uses `openai` SDK directly. API key authentication.

Both providers expose `embed_text()` for single texts and `embed_batch()` for batch operations, plus `dimensions` and `model_name` properties.

The `EmbeddingProvider` ABC:

```python
class EmbeddingProvider(ABC):
    @property
    def dimensions(self) -> int: ...
    @property
    def model_name(self) -> str: ...
    async def embed_text(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    async def close(self) -> None: ...
```

---

## Search Capabilities

### Search Types

| Type | Mechanism | Best For |
|------|-----------|----------|
| `full_text` | `CONTAINS()` in Cosmos, `LIKE` in DuckDB/SQLite | Exact terms, known keywords |
| `semantic` | Vector cosine similarity against query embedding | Conceptual similarity, paraphrased queries |
| `hybrid` | Both full-text and semantic, merged and re-ranked with MMR | Maximum recall and relevance (recommended) |

### Transcript Search

```python
results = await storage.search_transcripts(
    user_id="dicolomb",
    options=TranscriptSearchOptions(
        query="vector search implementation",
        search_type="hybrid",
        mmr_lambda=0.7,
        search_in_user=True,
        search_in_assistant=True,
        search_in_thinking=True,
        search_in_tool=False,
        filters=SearchFilters(
            project_slug="amplifier",
            start_date="2026-01-01T00:00:00Z",
            end_date="2026-02-01T00:00:00Z",
            bundle="foundation",
        ),
    ),
    limit=20,
)
```

### Event Search

```python
results = await storage.search_events(
    user_id="dicolomb",
    event_type="tool:pre",
    event_category="tool",
    tool_name="bash",
    level="info",
    start_date="2026-01-01T00:00:00Z",
    limit=50,
)
```

### Context Expansion

Search results can be expanded with surrounding context:

```python
# By turn number
context = await storage.get_turn_context(
    user_id="dicolomb",
    session_id="sess-abc",
    turn=15,
    before=3,
    after=1,
)
# Returns TurnContext with previous turns, current turn, and following turns

# By sequence number (when turns are null)
context = await storage.get_message_context(
    session_id="sess-abc",
    sequence=42,
    before=5,
    after=5,
)
# Returns MessageContext with surrounding messages
```

### Search Filters

```python
@dataclass
class SearchFilters:
    project_slug: str | None = None
    session_id: str | None = None
    start_date: str | None = None      # ISO-8601
    end_date: str | None = None        # ISO-8601
    bundle: str | None = None
    min_turn_count: int | None = None
    max_turn_count: int | None = None
    tags: list[str] = field(default_factory=list)
```

### Search Results

```python
@dataclass
class SearchResult:
    session_id: str
    project_slug: str
    sequence: int
    content: str
    metadata: dict[str, Any]
    score: float              # Higher = more relevant
    source: str               # "full_text", "semantic", or "hybrid"
```

### Graceful Degradation

When the embedding provider is not configured or vector indexes are unavailable:

- `hybrid` search falls back to `full_text`
- `semantic` search falls back to `full_text`
- `full_text` search always works

---

## Embedding Management

Two operations for managing the vector lifecycle, both on `StorageAdmin`:

### backfill_embeddings()

Generates embeddings for transcripts where `has_vectors` is `false`. Safe to call repeatedly -- skips transcripts that already have vectors.

```python
result = await storage.backfill_embeddings(
    user_id="dicolomb",
    project_slug="amplifier",
    session_id="sess-abc",
    batch_size=100,
    on_progress=lambda processed, total: print(f"{processed}/{total}"),
)
# result.transcripts_found = 42  (transcripts missing vectors)
# result.vectors_stored = 156    (chunks successfully embedded)
# result.vectors_failed = 0      (embedding failures)
# result.errors = []             (human-readable, capped at 50)
```

Use cases:
- Embedding provider was not configured during initial sync
- Embedding service was down during sync (graceful degradation stored transcripts without vectors)
- New sessions ingested before embedding provider was added

### rebuild_vectors()

Deletes ALL vectors for a session, resets `has_vectors` to `false`, then regenerates everything from transcript data.

```python
result = await storage.rebuild_vectors(
    user_id="dicolomb",
    project_slug="amplifier",
    session_id="sess-abc",
    batch_size=100,
    on_progress=lambda processed, total: print(f"{processed}/{total}"),
)
```

Use cases:
- Embedding model upgrade (switching from `text-embedding-3-small` to `text-embedding-3-large`)
- Dimension change
- Corruption recovery
- Chunking strategy change

### EmbeddingOperationResult

Both operations return:

```python
@dataclass
class EmbeddingOperationResult:
    transcripts_found: int    # Total transcripts in scope
    vectors_stored: int       # Successfully created vector documents
    vectors_failed: int       # Embedding failures (can be retried)
    errors: list[str]         # Human-readable messages (capped at 50)
```

When `vectors_failed > 0`, the caller can decide whether to retry with `backfill_embeddings()`.

---

## Configuration

### Environment Variables

#### Cosmos DB

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AMPLIFIER_COSMOS_ENDPOINT` | Yes | -- | Cosmos DB endpoint URL |
| `AMPLIFIER_COSMOS_DATABASE` | No | `amplifier-db` | Database name |
| `AMPLIFIER_COSMOS_AUTH_METHOD` | No | `default_credential` | `default_credential` (RBAC) or `key` |
| `AMPLIFIER_COSMOS_KEY` | If key auth | -- | Cosmos DB access key |
| `AMPLIFIER_COSMOS_ENABLE_VECTOR` | No | `true` | Enable vector search indexes |
| `AMPLIFIER_COSMOS_VECTOR_DIMENSIONS` | No | `3072` | Vector dimensions |

#### DuckDB

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AMPLIFIER_DUCKDB_PATH` | No | `:memory:` | Database file path (or `:memory:`) |
| `AMPLIFIER_DUCKDB_VECTOR_DIMENSIONS` | No | `3072` | Vector dimensions |

#### SQLite

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AMPLIFIER_SQLITE_PATH` | No | `:memory:` | Database file path (or `:memory:`) |
| `AMPLIFIER_SQLITE_VECTOR_DIMENSIONS` | No | `3072` | Vector dimensions |

#### Azure OpenAI Embeddings

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Yes | -- | Azure OpenAI endpoint |
| `AZURE_OPENAI_API_KEY` | If not RBAC | -- | API key |
| `AZURE_OPENAI_USE_RBAC` | No | `false` | Use `DefaultAzureCredential` |
| `AZURE_OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-large` | Model name |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | No | Model name | Deployment name |
| `AZURE_OPENAI_EMBEDDING_DIMENSIONS` | No | `3072` | Vector dimensions |
| `AZURE_OPENAI_EMBEDDING_CACHE_SIZE` | No | `1000` | LRU cache max entries |

### Config Classes

Each backend has a `@dataclass` config with a `from_env()` classmethod:

```python
@dataclass
class CosmosConfig:
    endpoint: str
    database_name: str
    auth_method: str = "default_credential"
    key: str | None = None
    enable_vector_search: bool = True
    vector_dimensions: int = 3072

@dataclass
class DuckDBConfig:
    db_path: str | Path = ":memory:"
    vector_dimensions: int = 3072

@dataclass
class SQLiteConfig:
    db_path: str | Path = ":memory:"
    vector_dimensions: int = 3072
```

---

## Installation

```bash
# Core only (minimal dependencies)
pip install git+https://github.com/colombod/amplifier-session-storage

# With specific backend
pip install "git+https://github.com/colombod/amplifier-session-storage[cosmos]"
pip install "git+https://github.com/colombod/amplifier-session-storage[duckdb]"
pip install "git+https://github.com/colombod/amplifier-session-storage[sqlite]"

# With embedding provider
pip install "git+https://github.com/colombod/amplifier-session-storage[azure-openai]"
pip install "git+https://github.com/colombod/amplifier-session-storage[openai]"

# Real-time sync support
pip install "git+https://github.com/colombod/amplifier-session-storage[sync]"

# Everything
pip install "git+https://github.com/colombod/amplifier-session-storage[all]"
```

### Optional Dependency Groups

| Extra | Packages |
|-------|----------|
| `cosmos` | `azure-cosmos >=4.5.0,<4.10.0`, `azure-identity >=1.15.0`, `aiohttp >=3.11.0` |
| `duckdb` | `duckdb >=0.10.0` |
| `sqlite` | `aiosqlite >=0.19.0` |
| `azure-openai` | `azure-ai-inference >=1.0.0b9`, `azure-core >=1.30.0` |
| `openai` | `openai >=1.0.0` |
| `sync` | `aiohttp >=3.11.0` |
| `all` | All of the above |

Core dependencies (always installed): `aiofiles`, `pydantic`, `pyyaml`, `numpy`, `tiktoken`.

### Development Setup

```bash
git clone https://github.com/colombod/amplifier-session-storage
cd amplifier-session-storage
uv sync --all-extras
uv run pytest tests/ -v
```

---

## Quick Start

### DuckDB (Simplest -- No Cloud Services)

```python
from amplifier_session_storage import DuckDBBackend, DuckDBConfig

async with DuckDBBackend.create(
    config=DuckDBConfig(db_path="./sessions.duckdb"),
) as storage:

    # Sync session metadata
    await storage.upsert_session_metadata(
        user_id="dicolomb",
        host_id="laptop-01",
        metadata={
            "session_id": "sess-001",
            "project_slug": "my-project",
            "bundle": "foundation",
            "created": "2026-01-15T10:00:00Z",
            "turn_count": 5,
        },
    )

    # Sync transcript lines
    await storage.sync_transcript_lines(
        user_id="dicolomb",
        host_id="laptop-01",
        project_slug="my-project",
        session_id="sess-001",
        lines=[
            {"role": "user", "content": "How do I use vector search?", "turn": 0},
            {"role": "assistant", "content": "You can use embeddings...", "turn": 0},
        ],
        start_sequence=0,
    )

    # Full-text search (no embeddings needed)
    results = await storage.search_transcripts(
        user_id="dicolomb",
        options=TranscriptSearchOptions(
            query="vector search",
            search_type="full_text",
        ),
    )
```

### Cosmos DB with RBAC and Embeddings

```python
from amplifier_session_storage import (
    CosmosBackend,
    AzureOpenAIEmbeddings,
    TranscriptSearchOptions,
)

# Initialize embedding provider (from environment variables)
embeddings = AzureOpenAIEmbeddings.from_env()

# Create backend (config from AMPLIFIER_COSMOS_* env vars)
async with CosmosBackend.create(embedding_provider=embeddings) as storage:

    # Sync with automatic embedding generation
    await storage.sync_transcript_lines(
        user_id="dicolomb",
        host_id="laptop-01",
        project_slug="amplifier",
        session_id="sess-abc",
        lines=transcript_lines,
        start_sequence=0,
    )

    # Hybrid search with MMR re-ranking
    results = await storage.search_transcripts(
        user_id="dicolomb",
        options=TranscriptSearchOptions(
            query="cosmos db configuration",
            search_type="hybrid",
            mmr_lambda=0.7,
            search_in_user=True,
            search_in_assistant=True,
        ),
        limit=20,
    )

    for result in results:
        print(f"[{result.score:.3f}] {result.session_id}: {result.content[:80]}...")
```

### Backfill Embeddings for Existing Data

```python
# Attach embedding provider to backend that was created without one
async with DuckDBBackend.create(
    config=DuckDBConfig(db_path="./sessions.duckdb"),
    embedding_provider=AzureOpenAIEmbeddings.from_env(),
) as storage:

    # Find and fill gaps
    result = await storage.backfill_embeddings(
        user_id="dicolomb",
        project_slug="my-project",
        session_id="sess-001",
        on_progress=lambda done, total: print(f"Progress: {done}/{total}"),
    )

    print(f"Found {result.transcripts_found} transcripts without vectors")
    print(f"Created {result.vectors_stored} vector chunks")
    if result.vectors_failed > 0:
        print(f"Failed: {result.vectors_failed} (can retry)")
```

---

## Schema Migrations

### DuckDB and SQLite

Both backends use automatic schema versioning. Migrations run during `initialize()`:

| Version | Migration | Description |
|---------|-----------|-------------|
| v1 | Initial | Base tables: `sessions`, `transcripts`, `events` with inline vector columns |
| v2 | `_migrate_to_externalized_vectors()` | Creates `transcript_vectors` table, migrates inline vectors, drops old columns |
| v3 | `_migrate_add_has_vectors()` | Adds `has_vectors` column (BOOLEAN/INTEGER) to `transcripts` with default `false`/`0` |

Migrations are idempotent -- safe to run multiple times. A `schema_version` table tracks the current version.

### Cosmos DB

Cosmos DB is schema-less. New fields (`has_vectors`) are added to documents as they are written. Queries use `NOT IS_DEFINED(c.has_vectors) OR c.has_vectors = false` for backward compatibility with documents created before v0.3.0.

---

## Error Handling

### Exception Hierarchy

All exceptions inherit from `SessionStorageError`, which carries a `message` and `details` dict:

```
SessionStorageError
    SessionNotFoundError
    SessionValidationError
    SessionExistsError
    EventNotFoundError
    StorageIOError
    ChunkingError
    SyncError
    ConflictError
    StorageConnectionError
    AuthenticationError
    ValidationError
    RewindError
    EventTooLargeError
    PermissionDeniedError
```

### Embedding Failure Handling

When embedding generation fails during `sync_transcript_lines`:

1. Transcript documents are always stored (Step 1 completes before Step 2)
2. The error is logged at ERROR level with structured format:

```
EMBEDDING_FAILURE: Vector generation failed for
  user=dicolomb project=amplifier session=sess-abc (5 messages).
  5 transcripts were stored successfully but will lack vector
  embeddings -- semantic/hybrid search will not cover these messages.
  error_type=HttpResponseError error=429 Too Many Requests
```

3. The full stack trace is included via `exc_info=True`
4. The `EMBEDDING_FAILURE` prefix enables grep/alerting
5. Missing vectors can be recovered later with `backfill_embeddings()`

### Circuit Breaker

`CircuitOpenError` is raised when the circuit breaker is open and embedding requests are rejected. This is a transient condition -- the circuit will probe again after `reset_timeout` (60s). Monitor via:

```python
from amplifier_session_storage.embeddings.mixin import get_circuit_breaker_stats

stats = get_circuit_breaker_stats()
# {"state": "closed", "failure_count": 0, "total_trips": 0}
```

---

## MMR Algorithm

Maximum Marginal Relevance (MMR) re-ranks search results to balance relevance and diversity.

**Formula**: `MMR = lambda * Sim(Di, Q) - (1 - lambda) * max(Sim(Di, Dj))`

Where `Di` is a candidate, `Q` is the query, and `Dj` are already-selected results.

**Lambda parameter**:

| Value | Behavior |
|-------|----------|
| 1.0 | Pure relevance (most similar to query) |
| 0.7 | Relevance-focused with some diversity (default) |
| 0.5 | Balanced |
| 0.3 | Diversity-focused |
| 0.0 | Pure diversity (maximum variety) |

Implementation in `amplifier_session_storage.search.mmr`, ported from the C# reference at [AIGeekSquad/AIContext](https://github.com/AIGeekSquad/AIContext). Original paper: Carbonell and Goldstein (1998).

---

## Development

```bash
# Install all dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=amplifier_session_storage

# Linting and formatting
uv run ruff check .
uv run ruff format .

# Type checking
uv run pyright

# Integration tests (requires Cosmos DB credentials)
AMPLIFIER_COSMOS_ENDPOINT="https://..." \
AMPLIFIER_COSMOS_DATABASE="amplifier-test-db" \
uv run pytest tests/test_cosmos_backend_integration.py -v
```

### Test Organization

| File | Scope | Requirements |
|------|-------|-------------|
| `tests/test_duckdb_backend.py` | DuckDB unit tests (in-memory) | `duckdb` |
| `tests/test_sqlite_backend.py` | SQLite unit tests (in-memory) | `aiosqlite` |
| `tests/test_protocol_split.py` | ABC protocol compliance | None |
| `tests/test_cosmos_backend_integration.py` | Cosmos DB integration | Real Cosmos DB instance |

---

## Changelog

### 0.3.0 (2026-02-10)

**Fixes:**
- Fixed Cosmos DB `get_session_sync_stats()` using unsupported `GROUP BY` with multiple aggregates. Replaced with 6 separate `SELECT VALUE` single-aggregate queries.

**Features:**
- Embedding resilience: retry with exponential backoff (429, 5xx), circuit breaker (5 failures = 60s cooldown), batch splitting (groups of 16), per-batch error isolation.
- `backfill_embeddings()` -- generate vectors for transcripts missing them.
- `rebuild_vectors()` -- delete and regenerate all vectors for a session.
- `has_vectors` flag on transcript documents for efficient missing-vector discovery.
- `_prepare_vector_records()` shared mixin for extract-chunk-embed pipeline.
- `EmbeddingOperationResult` dataclass for operation reporting.
- Schema migration v3 for DuckDB and SQLite (adds `has_vectors` column).

**Improvements:**
- Structured `EMBEDDING_FAILURE` error logging with identity chain and clear success/failure breakdown.
- Replaced unsafe `assert` with `ValueError` in both embedding providers.

### 0.2.0

- Storage backend abstraction (Cosmos DB, DuckDB, SQLite)
- Hybrid search (full-text + semantic + MMR re-ranking)
- Automatic embedding generation during ingestion
- Externalized vector storage with text chunking
- LRU cache for query embeddings
- Content extraction for 4 content types
- MMR re-ranking algorithm (ported from C# reference)
- Event search by type, tool, level, category
- Cross-session analytics and statistics
- Schema migration v2 (externalized vectors)

### 0.1.0

- Initial release with basic session storage

---

## License

MIT

## Credits

- MMR algorithm ported from [AIGeekSquad/AIContext](https://github.com/AIGeekSquad/AIContext)
- Original MMR paper: Carbonell and Goldstein (1998)
- Microsoft MADE:Explorations Team
