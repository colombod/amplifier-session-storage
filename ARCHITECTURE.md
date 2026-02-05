# Architecture

This document describes the architecture of the Amplifier Session Storage library.

## Overview

The library provides a unified storage abstraction for Amplifier session data with support for multiple backends (DuckDB, SQLite, Cosmos DB). It includes semantic search capabilities via multi-vector embeddings for user queries, assistant responses, thinking blocks, and tool outputs.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Application Layer                                │
│                                                                         │
│  Bundle Tools, Sync Daemons, Analytics, Search Interfaces               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         StorageBackend (Abstract)                        │
│                                                                         │
│  - sync_session_metadata()        - search_transcripts()                │
│  - sync_transcript_lines()        - search_events()                     │
│  - sync_event_lines()             - get_turn_context()                  │
│  - get_transcript_lines()         - get_session_statistics()            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│   DuckDBBackend      │ │   SQLiteBackend      │ │   CosmosBackend      │
│                      │ │                      │ │                      │
│ - HNSW vector index  │ │ - Simple local       │ │ - Azure cloud        │
│ - FTS5 full-text     │ │ - No vector ext      │ │ - Vector search      │
│ - Local file         │ │ - Portable           │ │ - Team-wide sync     │
└──────────────────────┘ └──────────────────────┘ └──────────────────────┘
```

## Design Principles

### 1. Backend Abstraction

All backends implement the `StorageBackend` protocol, enabling:
- Swappable storage without code changes
- Consistent API across local and cloud storage
- Backend-specific optimizations (e.g., vector indexes)

### 2. Multi-Vector Semantic Search

Each transcript message generates up to 4 vectors:
- `user_query_vector` - For user messages
- `assistant_response_vector` - For assistant responses
- `assistant_thinking_vector` - For thinking blocks
- `tool_output_vector` - For tool outputs

This enables targeted search (e.g., "find similar reasoning patterns").

### 3. Hybrid Search

Combines multiple search strategies:
- **Full-text search** - Keyword matching (fast, precise)
- **Semantic search** - Embedding similarity (meaning-based)
- **Hybrid search** - Best of both with MMR diversity ranking

### 4. Vector Safety

Vectors are **never returned** in read API responses to prevent:
- Bloating LLM context windows (3072-dim vectors are huge)
- Unnecessary data transfer
- Context overflow in agent tools

## Components

### StorageBackend (Abstract Base)

```python
class StorageBackend(ABC):
    # Lifecycle
    async def initialize() -> None
    async def close() -> None
    
    # Session metadata
    async def sync_session_metadata(user_id, host_id, metadata) -> None
    async def get_session(user_id, session_id) -> dict | None
    async def search_sessions(user_id, filters, limit) -> list[dict]
    async def delete_session(user_id, project_slug, session_id) -> bool
    
    # Transcripts
    async def sync_transcript_lines(user_id, host_id, project_slug, session_id, lines) -> int
    async def get_transcript_lines(user_id, project_slug, session_id, after_sequence) -> list[dict]
    async def search_transcripts(user_id, options, limit) -> list[SearchResult]
    async def get_turn_context(user_id, session_id, turn, before, after) -> TurnContext
    
    # Events
    async def sync_event_lines(user_id, host_id, project_slug, session_id, lines) -> int
    async def get_event_lines(user_id, project_slug, session_id, after_sequence) -> list[dict]
    async def search_events(user_id, options, limit) -> list[SearchResult]
    
    # Analytics
    async def get_session_statistics(user_id?, project_slug?) -> SessionStatistics
```

### Backend Configurations

```python
@dataclass
class DuckDBConfig:
    db_path: str = "~/.amplifier/sessions.duckdb"

@dataclass
class SQLiteConfig:
    db_path: str = "~/.amplifier/sessions.sqlite"

@dataclass
class CosmosConfig:
    endpoint: str
    database_name: str = "amplifier-db"
    auth_method: str = "default_credential"  # or "key"
    key: str | None = None
    enable_vector_search: bool = True
    
    @classmethod
    def from_env(cls) -> CosmosConfig
```

### Embedding Providers

```python
class EmbeddingProvider(ABC):
    async def embed(text: str) -> list[float]
    async def embed_batch(texts: list[str]) -> list[list[float]]

# Implementations
class AzureOpenAIEmbeddings(EmbeddingProvider)  # Azure AI Inference SDK
class OpenAIEmbeddings(EmbeddingProvider)        # OpenAI direct
```

### Identity Module

```python
class IdentityContext:
    @classmethod
    def get_user_id() -> str    # From config or system
    
    @classmethod  
    def get_host_id() -> str    # Machine identifier
```

## Data Flow

### Sync Flow (Local → Storage)

```
1. Sync daemon discovers local sessions
   └── Scans ~/.amplifier/projects/*/sessions/*/

2. For each session, get current sync status:
   └── storage.get_session() returns event_count, transcript_count

3. Calculate delta and sync new data:
   └── storage.sync_transcript_lines(lines[transcript_count:])
   └── storage.sync_event_lines(lines[event_count:])

4. Embedding provider generates vectors for new content:
   └── Parallel embedding of user/assistant/thinking/tool content
```

### Search Flow

```
1. User issues search query
   └── "How did we handle authentication errors?"

2. Search options determine strategy:
   └── search_type: "hybrid" | "semantic" | "full_text"
   └── search_in_user, search_in_assistant, search_in_thinking

3. Backend executes search:
   └── Full-text: SQL LIKE/CONTAINS queries
   └── Semantic: Vector similarity with HNSW/Cosmos
   └── Hybrid: Both + MMR diversity ranking

4. Results returned WITHOUT vectors:
   └── SearchResult(content, score, metadata, session_id, turn)
```

## Document Schemas

### Session Document

```json
{
  "id": "session-uuid",
  "user_id": "user-email",
  "host_id": "machine-uuid", 
  "project_slug": "my-project",
  "bundle": "foundation:main",
  "created": "2024-01-15T10:30:00Z",
  "updated": "2024-01-15T11:45:00Z",
  "turn_count": 15,
  "event_count": 234,
  "transcript_count": 45
}
```

### Transcript Document

```json
{
  "id": "transcript-line-uuid",
  "user_id": "user-email",
  "session_id": "session-uuid",
  "project_slug": "my-project",
  "sequence": 0,
  "turn": 1,
  "role": "user",
  "content": "Help me understand...",
  "ts": "2024-01-15T10:30:05Z",
  "user_query_vector": [0.1, 0.2, ...]  // NOT returned in read APIs
}
```

### Event Document

```json
{
  "id": "event-line-uuid",
  "user_id": "user-email",
  "session_id": "session-uuid",
  "project_slug": "my-project",
  "sequence": 0,
  "turn": 1,
  "event": "tool_call",
  "lvl": "INFO",
  "ts": "2024-01-15T10:30:10Z",
  "data_truncated": true,
  "data_size_bytes": 150000
}
```

## Backend Comparison

| Feature | DuckDB | SQLite | Cosmos DB |
|---------|--------|--------|-----------|
| **Storage** | Local file | Local file | Azure cloud |
| **Vector Search** | HNSW index | None | DiskANN |
| **Full-Text** | FTS extension | FTS5 | CONTAINS |
| **Concurrent** | Single process | Single process | Multi-region |
| **Team Sync** | No | No | Yes |
| **Best For** | Single-user, fast | Simple, portable | Team-wide |

## Authentication

### Azure (Cosmos + Azure OpenAI)

```bash
# RBAC authentication (recommended)
export AMPLIFIER_COSMOS_AUTH_METHOD="default_credential"
export AZURE_OPENAI_USE_RBAC="true"
az login  # Authenticates both services
```

### API Keys

```bash
# Cosmos with key
export AMPLIFIER_COSMOS_AUTH_METHOD="key"
export AMPLIFIER_COSMOS_KEY="your-key"

# OpenAI direct
export OPENAI_API_KEY="sk-..."
```

## Error Handling

All operations use typed exceptions:

```python
SessionStorageError        # Base exception
├── SessionNotFoundError   # Session doesn't exist
├── SessionExistsError     # Duplicate session
├── StorageIOError         # Read/write failures
├── StorageConnectionError # Connection issues
├── AuthenticationError    # Auth failures
├── ValidationError        # Invalid data
├── RewindError           # Rewind operation failed
└── EventTooLargeError    # Event exceeds size limit
```

## Testing

### Unit Tests (No External Dependencies)

```bash
pytest tests/test_duckdb.py tests/test_sqlite.py -v
```

### Integration Tests (Requires Azure)

```bash
# Set environment variables first
pytest tests/test_cosmos_backend.py -v
```

## Future Considerations

### Potential Enhancements
- PostgreSQL backend (pgvector)
- Compression for large events
- Cross-backend migration tools
- Real-time sync via WebSocket

### Not In Scope
- Direct CLI integration (handled by sync daemon)
- User authentication (handled by identity providers)
- Rate limiting (handled by Azure/application layer)
