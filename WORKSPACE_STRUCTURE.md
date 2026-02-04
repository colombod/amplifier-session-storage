# Workspace Structure - Enhanced Session Storage

This repository uses a workspace structure to manage multiple packages with optional dependencies.

## Packages

### Core Package: `amplifier-session-storage`
**Location**: `./`  
**Dependencies**: Minimal (aiofiles, pydantic, pyyaml, numpy)  
**Contains**: Core abstractions, MMR, search interfaces

### Backend: `backends/cosmos`
**Location**: `./backends/cosmos/`  
**Dependencies**: azure-cosmos, azure-identity  
**Optional**: Install only if using Cosmos DB

### Backend: `backends/duckdb`
**Location**: `./backends/duckdb/`  
**Dependencies**: duckdb, duckdb-vss  
**Optional**: Install only if using DuckDB

### Backend: `backends/sqlite`
**Location**: `./backends/sqlite/`  
**Dependencies**: aiosqlite, sqlite-vss  
**Optional**: Install only if using SQLite

### Embeddings: `embeddings/azure-openai`
**Location**: `./embeddings/azure-openai/`  
**Dependencies**: azure-ai-inference  
**Optional**: Install only if using Azure OpenAI embeddings

## Installation Patterns

```bash
# Core only (no backends, no embeddings)
uv pip install amplifier-session-storage

# Core + Cosmos backend
uv pip install amplifier-session-storage[cosmos]

# Core + DuckDB backend
uv pip install amplifier-session-storage[duckdb]

# Core + Azure OpenAI embeddings
uv pip install amplifier-session-storage[azure-openai]

# Full installation (all backends and embeddings)
uv pip install amplifier-session-storage[all]

# Development (all packages)
uv sync --all-extras
```

## Workspace Benefits

1. **Minimal dependencies** - Users only install what they need
2. **Independent versioning** - Backends can evolve independently
3. **Clear boundaries** - Each package has well-defined scope
4. **Easy testing** - Test backends in isolation
5. **Flexible deployment** - Cloud users don't need DuckDB, local users don't need Cosmos
