# Amplifier Session Storage

**Enhanced session storage for Amplifier with hybrid search and multiple backends**

## Overview

A powerful session storage library that provides:

- **Multiple Storage Backends**: Cosmos DB (cloud), DuckDB (local analytics), SQLite (embedded)
- **Hybrid Search**: Combines full-text + semantic search with MMR re-ranking
- **Automatic Embeddings**: Generate embeddings during ingestion for semantic search
- **Smart Caching**: LRU cache for hot query embeddings (minimize API costs)
- **Graceful Degradation**: Falls back to full-text when embeddings unavailable

## Installation

```bash
# Core only (minimal dependencies)
uv pip install git+https://github.com/colombod/amplifier-session-storage

# With Cosmos DB support
uv pip install "git+https://github.com/colombod/amplifier-session-storage[cosmos]"

# With DuckDB support
uv pip install "git+https://github.com/colombod/amplifier-session-storage[duckdb]"

# With SQLite support
uv pip install "git+https://github.com/colombod/amplifier-session-storage[sqlite]"

# With Azure OpenAI embeddings
uv pip install "git+https://github.com/colombod/amplifier-session-storage[azure-openai]"

# Full installation (all backends and embeddings)
uv pip install "git+https://github.com/colombod/amplifier-session-storage[all]"

# Development setup
git clone https://github.com/colombod/amplifier-session-storage
cd amplifier-session-storage
uv sync --all-extras
uv run pytest tests/ -v
```

## Quick Start - Hybrid Search

```python
from amplifier_session_storage import (
    CosmosBackend,
    AzureOpenAIEmbeddings,
    TranscriptSearchOptions
)

# Initialize backend with embeddings
embeddings = AzureOpenAIEmbeddings.from_env()
async with CosmosBackend.create(embedding_provider=embeddings) as storage:
    
    # Ingest with automatic embedding generation
    await storage.sync_transcript_lines(
        user_id="user-123",
        host_id="laptop-01",
        project_slug="my-project",
        session_id="sess-abc",
        lines=[
            {"role": "user", "content": "How do I implement vector search?", "turn": 0},
            {"role": "assistant", "content": "You can use embeddings...", "turn": 0},
        ],
        start_sequence=0,
        # Embeddings generated automatically during sync!
    )
    
    # Hybrid search with MMR re-ranking
    results = await storage.search_transcripts(
        user_id="user-123",
        options=TranscriptSearchOptions(
            query="vector search implementation",
            search_type="hybrid",  # "full_text", "semantic", or "hybrid"
            mmr_lambda=0.7,  # Balance relevance (1.0) vs diversity (0.0)
            search_in_user=True,
            search_in_assistant=True,
            search_in_thinking=True
        ),
        limit=10
    )
    
    for result in results:
        print(f"Session: {result.session_id}")
        print(f"Content: {result.content[:100]}...")
        print(f"Score: {result.score:.3f}")
```

## Key Features

### 1. Multiple Storage Backends

Choose the right backend for your use case:

| Backend | Best For | Vector Search |
|---------|----------|---------------|
| **Cosmos DB** | Cloud sync, multi-device, production | ✅ Native support |
| **DuckDB** | Local analytics, development, fast queries | ✅ VSS extension |
| **SQLite** | Embedded apps, lightweight, testing | ✅ sqlite-vss |

```python
# Cosmos DB - Cloud storage
from amplifier_session_storage import CosmosBackend
storage = await CosmosBackend.create(embedding_provider=embeddings)

# DuckDB - Local analytics
from amplifier_session_storage import DuckDBBackend
storage = await DuckDBBackend.create(embedding_provider=embeddings)

# SQLite - Embedded
from amplifier_session_storage import SQLiteBackend
storage = await SQLiteBackend.create(embedding_provider=embeddings)
```

### 2. Hybrid Search (Full-Text + Semantic + MMR)

Three search modes:

```python
# Full-text: Keyword matching
search_type="full_text"

# Semantic: Meaning-based (uses embeddings)
search_type="semantic"

# Hybrid: Combines both with MMR re-ranking (recommended)
search_type="hybrid"
```

**Why hybrid?**
- Full-text finds exact keyword matches
- Semantic finds conceptually similar content
- MMR re-ranks for relevance + diversity (avoids redundant results)

### 3. Automatic Embedding Generation

Embeddings are generated automatically during ingestion:

```python
# Just sync lines - embeddings generated automatically!
await storage.sync_transcript_lines(
    user_id="user-123",
    host_id="laptop-01",
    project_slug="project",
    session_id="session",
    lines=transcript_lines,
    # No need to manually generate embeddings
)
```

### 4. Smart Embedding Cache

LRU cache minimizes API calls and costs:

```python
embeddings = AzureOpenAIEmbeddings(
    endpoint=...,
    api_key=...,
    model="text-embedding-3-large",
    cache_size=1000  # Cache last 1000 query embeddings
)

# Check cache performance
stats = embeddings.get_cache_stats()
print(f"Cache utilization: {stats['utilization']:.1%}")
```

**Benefits:**
- Repeated queries hit cache (no API call)
- Configurable size based on workload
- Automatic LRU eviction

### 5. Advanced Search Capabilities

#### Search by Event Type

```python
from amplifier_session_storage.backends import EventSearchOptions

# Find all LLM requests
results = await storage.search_events(
    user_id="user-123",
    options=EventSearchOptions(
        event_type="llm.request"
    ),
    limit=100
)
```

#### Search by Tool Name

```python
# Find all bash tool executions
results = await storage.search_events(
    user_id="user-123",
    options=EventSearchOptions(
        tool_name="bash"
    )
)
```

#### Filter by Date Range

```python
from amplifier_session_storage.backends import SearchFilters

results = await storage.search_transcripts(
    user_id="user-123",
    options=TranscriptSearchOptions(
        query="implementation",
        filters=SearchFilters(
            project_slug="amplifier",
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-02-01T00:00:00Z"
        )
    )
)
```

### 6. Cross-Session Analytics

```python
from amplifier_session_storage.backends import SearchFilters

# Get statistics across all sessions
stats = await storage.get_session_statistics(
    user_id="user-123"
)

print(f"Total sessions: {stats['total_sessions']}")
print(f"By project: {stats['sessions_by_project']}")
print(f"By bundle: {stats['sessions_by_bundle']}")

# Filtered statistics
stats = await storage.get_session_statistics(
    user_id="user-123",
    filters=SearchFilters(
        start_date="2024-01-01T00:00:00Z",
        project_slug="amplifier"
    )
)
```

---

## Configuration

### Environment Variables

#### Cosmos DB

```bash
export AMPLIFIER_COSMOS_ENDPOINT="https://your-account.documents.azure.com:443/"
export AMPLIFIER_COSMOS_DATABASE="amplifier-db"
export AMPLIFIER_COSMOS_AUTH_METHOD="default_credential"  # or "key"
export AMPLIFIER_COSMOS_ENABLE_VECTOR="true"  # Enable vector search
```

#### Azure OpenAI Embeddings

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_EMBEDDING_MODEL="text-embedding-3-large"
export AZURE_OPENAI_EMBEDDING_DIMENSIONS="3072"
export AZURE_OPENAI_EMBEDDING_CACHE_SIZE="1000"
```

#### DuckDB

```bash
export AMPLIFIER_DUCKDB_PATH="./amplifier_sessions.duckdb"  # or ":memory:"
export AMPLIFIER_DUCKDB_VECTOR_DIMENSIONS="3072"
```

#### SQLite

```bash
export AMPLIFIER_SQLITE_PATH="./amplifier_sessions.db"  # or ":memory:"
export AMPLIFIER_SQLITE_VECTOR_DIMENSIONS="3072"
```

---

## Architecture

### Storage Abstraction

All backends implement the same `StorageBackend` interface:

```python
from amplifier_session_storage.backends import StorageBackend

# All backends support:
# - Session metadata operations
# - Transcript sync and search
# - Event sync and search  
# - Vector embeddings (if configured)
# - Analytics and aggregations
```

### Embedding Provider Abstraction

Pluggable embedding generation:

```python
from amplifier_session_storage.embeddings import EmbeddingProvider

# Current implementation:
from amplifier_session_storage import AzureOpenAIEmbeddings

# Easy to add more providers:
# - OpenAI Direct
# - Local models (sentence-transformers)
# - Custom implementations
```

### Search Flow

```
User Query
    ↓
┌─────────────────────────────────────┐
│ Full-Text Search (Keywords)         │
│ + Semantic Search (Embeddings)      │
└─────────────────────────────────────┘
    ↓
Merge & Deduplicate
    ↓
┌─────────────────────────────────────┐
│ MMR Re-Ranking                      │
│ (Balance relevance + diversity)     │
└─────────────────────────────────────┘
    ↓
Top-K Results
```

---

## MMR Algorithm

Maximum Marginal Relevance (MMR) re-ranks results to balance:
- **Relevance**: Similarity to query
- **Diversity**: Dissimilarity to already selected results

**Formula**: `MMR = λ × Sim(Di, Q) - (1-λ) × max(Sim(Di, Dj))`

**Lambda parameter:**
- `1.0` - Pure relevance (most similar results)
- `0.7` - Relevance-focused (recommended default)
- `0.5` - Balanced
- `0.3` - Diversity-focused
- `0.0` - Pure diversity (maximum variety)

**Ported from**: [AIGeekSquad/AIContext](https://github.com/AIGeekSquad/AIContext) (C# reference implementation)

---

## Usage Examples

### Example 1: Basic Ingestion with Embeddings

```python
from amplifier_session_storage import CosmosBackend, AzureOpenAIEmbeddings

# Setup
embeddings = AzureOpenAIEmbeddings.from_env()
async with CosmosBackend.create(embedding_provider=embeddings) as storage:
    
    # Sync session metadata
    await storage.upsert_session_metadata(
        user_id="user-123",
        host_id="laptop-01",
        metadata={
            "session_id": "sess-abc",
            "project_slug": "amplifier",
            "bundle": "foundation",
            "created": "2024-01-15T10:00:00Z",
            "turn_count": 10,
        }
    )
    
    # Sync transcripts with automatic embedding generation
    await storage.sync_transcript_lines(
        user_id="user-123",
        host_id="laptop-01",
        project_slug="amplifier",
        session_id="sess-abc",
        lines=[
            {"role": "user", "content": "How do I use vector search?", "turn": 0},
            {"role": "assistant", "content": "Vector search uses embeddings...", "turn": 0},
        ],
        start_sequence=0
        # Embeddings generated and stored automatically!
    )
```

### Example 2: Hybrid Search with Filters

```python
from amplifier_session_storage.backends import TranscriptSearchOptions, SearchFilters

# Search with multiple filters
results = await storage.search_transcripts(
    user_id="user-123",
    options=TranscriptSearchOptions(
        query="cosmos db configuration",
        search_type="hybrid",
        mmr_lambda=0.7,
        search_in_user=True,
        search_in_assistant=True,
        filters=SearchFilters(
            project_slug="amplifier",
            start_date="2024-01-01T00:00:00Z",
            bundle="foundation"
        )
    ),
    limit=20
)

for result in results:
    print(f"[{result.score:.3f}] {result.session_id}")
    print(f"  {result.content[:80]}...")
```

### Example 3: Event Analytics

```python
from amplifier_session_storage.backends import EventSearchOptions

# Find all tool errors
errors = await storage.search_events(
    user_id="user-123",
    options=EventSearchOptions(
        event_type="tool.error",
        level="error",
        filters=SearchFilters(start_date="2024-01-01T00:00:00Z")
    ),
    limit=50
)

print(f"Found {len(errors)} tool errors")

# Get usage statistics
stats = await storage.get_session_statistics(
    user_id="user-123",
    filters=SearchFilters(project_slug="amplifier")
)

print(f"Sessions: {stats['total_sessions']}")
print(f"By project: {stats['sessions_by_project']}")
```

### Example 4: Local Development with DuckDB

```python
from amplifier_session_storage import DuckDBBackend, AzureOpenAIEmbeddings

# Local database for fast development
embeddings = AzureOpenAIEmbeddings.from_env()
async with DuckDBBackend.create(
    config=DuckDBConfig(db_path="./dev.duckdb"),
    embedding_provider=embeddings
) as storage:
    
    # Same API as Cosmos DB!
    await storage.sync_transcript_lines(...)
    results = await storage.search_transcripts(...)
```

---

## Authentication

### Azure AD (Recommended)

```bash
# Login with Azure CLI
az login

# Set Cosmos endpoint
export AMPLIFIER_COSMOS_ENDPOINT="https://your-account.documents.azure.com:443/"
export AMPLIFIER_COSMOS_AUTH_METHOD="default_credential"
```

**Required RBAC role**: `Cosmos DB Built-in Data Contributor`

```bash
az cosmosdb sql role assignment create \
    --account-name your-cosmos-account \
    --resource-group your-resource-group \
    --role-definition-name "Cosmos DB Built-in Data Contributor" \
    --principal-id $(az ad signed-in-user show --query id -o tsv) \
    --scope "/"
```

### Key-Based (Development)

```bash
export AMPLIFIER_COSMOS_ENDPOINT="https://your-account.documents.azure.com:443/"
export AMPLIFIER_COSMOS_AUTH_METHOD="key"
export AMPLIFIER_COSMOS_KEY="your-cosmos-key"
```

---

## Search Capabilities

### Search Types

| Type | How It Works | Best For |
|------|--------------|----------|
| **full_text** | Keyword matching (CONTAINS/LIKE) | Exact terms, known keywords |
| **semantic** | Embedding similarity | Conceptual similarity, paraphrased queries |
| **hybrid** | Both + MMR re-ranking | Maximum recall and relevance |

### Search Scope

Control which message types to search:

```python
options = TranscriptSearchOptions(
    query="...",
    search_in_user=True,      # Search user input
    search_in_assistant=True,  # Search assistant responses
    search_in_thinking=True    # Search thinking blocks
)
```

### Filters

```python
from amplifier_session_storage.backends import SearchFilters

filters = SearchFilters(
    project_slug="amplifier",       # Filter by project
    session_id="sess-abc",          # Specific session
    start_date="2024-01-01T00:00:00Z",  # Date range start
    end_date="2024-02-01T00:00:00Z",    # Date range end
    bundle="foundation",            # Filter by bundle
    min_turn_count=5,              # Minimum conversation length
    max_turn_count=100,            # Maximum conversation length
)
```

---

## Data Model

### Container/Table Structure

All backends use the same logical structure:

#### Sessions
```json
{
    "user_id": "user-123",
    "session_id": "sess-abc",
    "host_id": "laptop-01",
    "project_slug": "my-project",
    "bundle": "foundation",
    "created": "2024-01-15T10:00:00Z",
    "turn_count": 10,
    "metadata": {...}
}
```

#### Transcripts
```json
{
    "id": "sess-abc_msg_0",
    "user_id": "user-123",
    "session_id": "sess-abc",
    "sequence": 0,
    "role": "user",
    "content": "Hello",
    "turn": 0,
    "ts": "2024-01-15T10:00:00Z",
    "embedding": [0.1, 0.2, ...],  // 3072-d vector
    "embedding_model": "text-embedding-3-large"
}
```

#### Events
```json
{
    "id": "sess-abc_evt_0",
    "user_id": "user-123",
    "session_id": "sess-abc",
    "sequence": 0,
    "event": "llm.request",
    "ts": "2024-01-15T10:00:00Z",
    "lvl": "info",
    "data": {...},
    "data_truncated": false,
    "data_size_bytes": 1024
}
```

---

## Embedding Models

### Supported Models

| Model | Dimensions | Use Case | Cost |
|-------|------------|----------|------|
| text-embedding-3-small | 1536 | Development, testing | Low |
| text-embedding-3-large | 3072 | Production (recommended) | Medium |
| text-embedding-ada-002 | 1536 | Legacy support | Low |

### Configuration

```python
# Automatic dimension detection
embeddings = AzureOpenAIEmbeddings(
    endpoint=...,
    api_key=...,
    model="text-embedding-3-large"  # Dimensions auto-detected (3072)
)

# Explicit dimensions
embeddings = AzureOpenAIEmbeddings(
    endpoint=...,
    api_key=...,
    model="custom-model",
    dimensions=1536  # Specify if unknown model
)
```

---

## Performance & Costs

### Embedding Costs

Azure OpenAI pricing (approximate):

| Model | Cost per 1M tokens |
|-------|-------------------|
| text-embedding-3-small | ~$0.02 |
| text-embedding-3-large | ~$0.13 |

**Cost reduction strategies:**
1. ✅ Use cache (default 1000 entries)
2. ✅ Batch operations (`embed_batch()`)
3. ✅ Use smaller model for dev/test
4. ✅ Only embed essential content

### Storage Costs

Vector embeddings increase storage:

| Component | Size per Message |
|-----------|------------------|
| Text | ~500 bytes |
| Embedding (3072-d) | ~12 KB |
| **Total** | **~12.5 KB** |

**Cosmos DB optimization:**
- `quantizedFlat` index compresses to 128 bytes
- Effective storage: ~600 bytes per message with index

---

## Graceful Degradation

The system automatically falls back when features unavailable:

```
Request: hybrid search
  ↓ (no embedding provider configured)
Fallback: full_text search

Request: semantic search
  ↓ (vector indexes not available)
Fallback: full_text search
```

**This ensures searches always work**, even in degraded mode.

---

## Upgrading from v0.1.0

See [UPGRADE_GUIDE.md](./docs/UPGRADE_GUIDE.md) for detailed migration instructions.

**Quick summary:**
1. v0.2.0 adds vector search capabilities
2. Recommended: Clear Cosmos containers and rebuild with embeddings
3. Alternative: Backfill embeddings for existing data
4. All basic sync operations remain backward compatible

---

## Documentation

- [Upgrade Guide](./docs/UPGRADE_GUIDE.md) - Migration from v0.1.0
- [Hybrid Search Guide](./docs/HYBRID_SEARCH_GUIDE.md) - Detailed search documentation
- [Architecture](./ARCHITECTURE.md) - System design and patterns

---

## Development

```bash
# Install all dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ --cov=amplifier_session_storage

# Linting and formatting
uv run ruff check .
uv run ruff format .

# Type checking
uv run pyright
```

---

## What's New in v0.2.0

### Core Enhancements
- ✅ Storage backend abstraction (Cosmos, DuckDB, SQLite)
- ✅ Hybrid search (full-text + semantic + MMR)
- ✅ Automatic embedding generation during ingestion
- ✅ LRU cache for query embeddings
- ✅ MMR re-ranking algorithm (ported from C# reference)

### Search Features
- ✅ Full-text search with role filtering
- ✅ Semantic search with vector similarity
- ✅ Hybrid search combining both approaches
- ✅ Event search by type, tool, level
- ✅ Advanced filtering (date, project, bundle)
- ✅ Cross-session analytics

### Developer Experience
- ✅ Multiple backend options (choose what fits)
- ✅ Optional dependencies (install only what you need)
- ✅ Graceful degradation (searches work without embeddings)
- ✅ Comprehensive test suite (64 tests passing)
- ✅ Detailed documentation and examples

---

## License

MIT

---

## Credits

- MMR algorithm ported from [AIGeekSquad/AIContext](https://github.com/AIGeekSquad/AIContext)
- Original MMR paper: Carbonell and Goldstein (1998)
