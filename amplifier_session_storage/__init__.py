"""
Amplifier Session Storage

Enhanced session storage library with hybrid search capabilities.

Provides:
- Multiple storage backends (Cosmos DB, DuckDB, SQLite)
- Hybrid search (full-text + semantic + MMR re-ranking)
- Pluggable embedding providers (Azure OpenAI)
- LRU cache for hot query embeddings

Usage:

    >>> from amplifier_session_storage import CosmosBackend, AzureOpenAIEmbeddings
    >>> embeddings = AzureOpenAIEmbeddings.from_env()
    >>> async with CosmosBackend.create(embedding_provider=embeddings) as storage:
    ...     # Sync with automatic embedding generation
    ...     await storage.sync_transcript_lines(user_id, host_id, project, session, lines)
    ...
    ...     # Hybrid search with MMR re-ranking
    ...     results = await storage.search_transcripts(
    ...         user_id,
    ...         TranscriptSearchOptions(
    ...             query="vector search implementation",
    ...             search_type="hybrid",
    ...             mmr_lambda=0.7
    ...         )
    ...     )

Backend Selection:

    # Cosmos DB for cloud multi-device sync
    from amplifier_session_storage.backends import CosmosBackend, CosmosConfig

    # DuckDB for local analytics and development
    from amplifier_session_storage.backends import DuckDBBackend, DuckDBConfig

    # SQLite for embedded applications
    from amplifier_session_storage.backends import SQLiteBackend, SQLiteConfig

Embeddings:

    # Azure OpenAI
    from amplifier_session_storage.embeddings import AzureOpenAIEmbeddings

    # With caching
    embeddings = AzureOpenAIEmbeddings(
        endpoint=...,
        api_key=...,
        model="text-embedding-3-large",
        cache_size=1000
    )
"""

# Backend abstraction
from .backends import (
    EventSearchOptions,
    SearchFilters,
    SearchResult,
    StorageBackend,
    TranscriptSearchOptions,
)

# Embedding providers
from .embeddings import EmbeddingCache, EmbeddingProvider

# Exceptions
from .exceptions import (
    AuthenticationError,
    ChunkingError,
    ConflictError,
    EventNotFoundError,
    EventTooLargeError,
    PermissionDeniedError,
    RewindError,
    SessionExistsError,
    SessionNotFoundError,
    SessionStorageError,
    SessionValidationError,
    StorageConnectionError,
    StorageIOError,
    SyncError,
    ValidationError,
)

# Identity module
from .identity import (
    ConfigFileIdentityProvider,
    IdentityContext,
    IdentityProvider,
    UserIdentity,
)

# Search utilities
from .search import compute_mmr, cosine_similarity

# Conditional imports for optional backends
try:
    from .backends.cosmos import CosmosBackend, CosmosConfig  # noqa: F401

    _has_cosmos = True
except ImportError:
    _has_cosmos = False

try:
    from .backends.duckdb import DuckDBBackend, DuckDBConfig  # noqa: F401

    _has_duckdb = True
except ImportError:
    _has_duckdb = False

try:
    from .backends.sqlite import SQLiteBackend, SQLiteConfig  # noqa: F401

    _has_sqlite = True
except ImportError:
    _has_sqlite = False

# Conditional imports for embedding providers
try:
    from .embeddings.azure_openai import AzureOpenAIEmbeddings  # noqa: F401

    _has_azure_openai = True
except ImportError:
    _has_azure_openai = False

try:
    from .embeddings.openai import OpenAIEmbeddings  # noqa: F401

    _has_openai = True
except ImportError:
    _has_openai = False


__all__ = [
    # Core abstractions
    "StorageBackend",
    "SearchFilters",
    "SearchResult",
    "TranscriptSearchOptions",
    "EventSearchOptions",
    # Embeddings
    "EmbeddingProvider",
    "EmbeddingCache",
    # Search
    "compute_mmr",
    "cosine_similarity",
    # Identity
    "IdentityProvider",
    "UserIdentity",
    "ConfigFileIdentityProvider",
    "IdentityContext",
    # Exceptions
    "SessionStorageError",
    "SessionNotFoundError",
    "SessionValidationError",
    "SessionExistsError",
    "EventNotFoundError",
    "StorageIOError",
    "ChunkingError",
    "SyncError",
    "ConflictError",
    "StorageConnectionError",
    "AuthenticationError",
    "ValidationError",
    "RewindError",
    "EventTooLargeError",
    "PermissionDeniedError",
]

# Add optional exports
if _has_cosmos:
    __all__.extend(["CosmosBackend", "CosmosConfig"])

if _has_duckdb:
    __all__.extend(["DuckDBBackend", "DuckDBConfig"])

if _has_sqlite:
    __all__.extend(["SQLiteBackend", "SQLiteConfig"])

if _has_azure_openai:
    __all__.extend(["AzureOpenAIEmbeddings"])

if _has_openai:
    __all__.extend(["OpenAIEmbeddings"])

__version__ = "0.2.0"  # Bumped for major enhancements
