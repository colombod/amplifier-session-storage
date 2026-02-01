# Amplifier Session Storage

A foundational library for session persistence in Amplifier applications.

## What This Library Is

This is a **foundational library** (like `amplifier-core` and `amplifier-foundation`) that provides session storage infrastructure for building Amplifier applications. It is **not** a dynamically-loaded module.

**How to use this library:**
- Applications **import** it directly: `from amplifier_session_storage import ...`
- Applications **instantiate** storage backends based on their configuration
- Applications **inject** storage where session persistence is needed

**This is NOT:**
- A tool module with `mount()` (that pattern is for plugins like `tool-filesystem`)
- Something you load via bundle YAML

## Features

- **Event-sourced architecture**: Sessions stored as immutable blocks for efficient sync
- **Offline-first**: Works locally, syncs when connected
- **Multi-device support**: Sequence-based merging handles concurrent edits
- **Multi-tenant**: Team/org visibility controls for shared sessions
- **Flexible authentication**: Supports key-based and Azure AD authentication
- **Drop-in compatible**: Works with existing Amplifier session file format

## Installation

```bash
# Basic installation (local storage only)
uv pip install git+https://github.com/colombod/amplifier-session-storage

# With Cosmos DB support (includes azure-identity for AAD auth)
uv pip install "amplifier-session-storage[cosmos] @ git+https://github.com/colombod/amplifier-session-storage"

# With real-time sync support
uv pip install "amplifier-session-storage[sync] @ git+https://github.com/colombod/amplifier-session-storage"

# Full installation (all features)
uv pip install "amplifier-session-storage[all] @ git+https://github.com/colombod/amplifier-session-storage"
```

### Development Setup

```bash
git clone https://github.com/colombod/amplifier-session-storage
cd amplifier-session-storage
uv sync --all-extras
uv run pytest tests/ -v
```

## Quick Start

### Local Storage

```python
from amplifier_session_storage import (
    LocalBlockStorage,
    StorageConfig,
    BlockWriter,
)

# Create storage config
config = StorageConfig(user_id="user-123")

# Create local storage
storage = LocalBlockStorage(config)

# Create a block writer for a session
writer = BlockWriter(
    session_id="sess-abc",
    user_id="user-123",
    device_id="device-001",
)

# Write session blocks
block = writer.create_session(project_slug="my-project", name="My Session")
await storage.write_block(block)

# Add messages
msg_block = writer.add_message(role="user", content="Hello!", turn=1)
await storage.write_block(msg_block)

# Read blocks back
blocks = await storage.read_blocks("sess-abc")

# Clean up
await storage.close()
```

### Cosmos DB Storage

```python
from amplifier_session_storage import (
    CosmosBlockStorage,
    StorageConfig,
    CosmosAuthMethod,
)

# Using Azure AD authentication (recommended)
config = StorageConfig(
    user_id="user-123",
    cosmos_endpoint="https://your-account.documents.azure.com:443/",
    cosmos_auth_method=CosmosAuthMethod.DEFAULT_CREDENTIAL,
    cosmos_database="amplifier-db",
    cosmos_container="items",
)

storage = CosmosBlockStorage(config)
# ... use like local storage
await storage.close()
```

## Authentication Methods

### 1. Azure AD DefaultAzureCredential (Recommended)

Uses the Azure Identity SDK to automatically try multiple authentication methods:
- Azure CLI credentials
- Environment variables
- Managed Identity
- Visual Studio Code credentials
- And more...

```python
config = StorageConfig(
    user_id="user-123",
    cosmos_endpoint="https://your-account.documents.azure.com:443/",
    cosmos_auth_method=CosmosAuthMethod.DEFAULT_CREDENTIAL,
)
```

**Required Azure RBAC role**: `Cosmos DB Built-in Data Contributor`

To assign the role:
```bash
# Get your user object ID
az ad signed-in-user show --query id -o tsv

# Assign role (replace with your values)
az cosmosdb sql role assignment create \
    --account-name your-cosmos-account \
    --resource-group your-resource-group \
    --role-definition-name "Cosmos DB Built-in Data Contributor" \
    --principal-id YOUR_USER_OBJECT_ID \
    --scope "/"
```

### 2. Managed Identity

For Azure-hosted services (App Service, Functions, AKS, VMs):

```python
# System-assigned managed identity
config = StorageConfig(
    user_id="user-123",
    cosmos_endpoint="https://your-account.documents.azure.com:443/",
    cosmos_auth_method=CosmosAuthMethod.MANAGED_IDENTITY,
)

# User-assigned managed identity
config = StorageConfig(
    user_id="user-123",
    cosmos_endpoint="https://your-account.documents.azure.com:443/",
    cosmos_auth_method=CosmosAuthMethod.MANAGED_IDENTITY,
    azure_client_id="your-managed-identity-client-id",
)
```

### 3. Service Principal

For CI/CD pipelines and automation:

```python
config = StorageConfig(
    user_id="user-123",
    cosmos_endpoint="https://your-account.documents.azure.com:443/",
    cosmos_auth_method=CosmosAuthMethod.SERVICE_PRINCIPAL,
    azure_tenant_id="your-tenant-id",
    azure_client_id="your-client-id",
    azure_client_secret="your-client-secret",  # Keep secure!
)
```

**⚠️ Security Note**: Never commit secrets to source control. Use environment variables or a secret manager.

### 4. Key-Based Authentication

For development/testing (if organization policy allows):

```python
config = StorageConfig(
    user_id="user-123",
    cosmos_endpoint="https://your-account.documents.azure.com:443/",
    cosmos_auth_method=CosmosAuthMethod.KEY,
    cosmos_key="your-cosmos-key",  # Keep secure!
)
```

## Environment Variables

All configuration can be provided via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `AMPLIFIER_COSMOS_ENDPOINT` | Cosmos DB endpoint URL | - |
| `AMPLIFIER_COSMOS_AUTH_METHOD` | Auth method: `default_credential`, `managed_identity`, `service_principal`, `key` | `default_credential` |
| `AMPLIFIER_COSMOS_KEY` | Cosmos DB key (for key auth) | - |
| `AMPLIFIER_COSMOS_DATABASE` | Database name | `amplifier-db` |
| `AMPLIFIER_COSMOS_CONTAINER` | Container name | `items` |
| `AMPLIFIER_COSMOS_PARTITION_KEY` | Partition key path | `/partitionKey` |
| `AMPLIFIER_LOCAL_STORAGE_PATH` | Local storage path | `~/.amplifier/sessions` |
| `AMPLIFIER_ENABLE_SYNC` | Enable cloud sync | `false` |
| `AZURE_TENANT_ID` | Azure tenant ID (for service principal) | - |
| `AZURE_CLIENT_ID` | Azure client ID (for service principal/managed identity) | - |
| `AZURE_CLIENT_SECRET` | Azure client secret (for service principal) | - |

Example using environment variables:

```python
from amplifier_session_storage import StorageConfig

# Load from environment
config = StorageConfig.from_environment(user_id="user-123")
```

## Cosmos DB Setup

### Container Configuration

The storage expects a container with the following configuration:

| Setting | Value |
|---------|-------|
| Partition Key | `/partitionKey` |
| Indexing Policy | Automatic (see below for optimization) |

### Recommended Indexing Policy

```json
{
    "indexingMode": "consistent",
    "automatic": true,
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [
        {"path": "/data/*"},
        {"path": "/\"_etag\"/?"}
    ],
    "compositeIndexes": [
        [
            {"path": "/user_id", "order": "ascending"},
            {"path": "/timestamp", "order": "descending"}
        ],
        [
            {"path": "/session_id", "order": "ascending"},
            {"path": "/sequence", "order": "ascending"}
        ],
        [
            {"path": "/org_id", "order": "ascending"},
            {"path": "/visibility", "order": "ascending"},
            {"path": "/timestamp", "order": "descending"}
        ]
    ]
}
```

## Session Visibility

Sessions can be shared with different visibility levels:

| Visibility | Who Can Access |
|------------|----------------|
| `private` | Only the owner |
| `team` | Owner + users in specified teams |
| `org` | All users in the organization |
| `public` | Anyone |

```python
# Create a team-visible session
block = writer.create_session(
    project_slug="my-project",
    name="Team Session",
    visibility="team",
    org_id="org-123",
    team_ids=["team-a", "team-b"],
)
```

## Migration from Legacy Sessions

Migrate existing `events.jsonl` sessions to block format:

```python
from pathlib import Path
from amplifier_session_storage import (
    SessionMigrator,
    LocalBlockStorage,
    StorageConfig,
)

# Create storage
config = StorageConfig(user_id="user-123")
storage = LocalBlockStorage(config)

# Create migrator
migrator = SessionMigrator(storage, user_id="user-123")

# Discover existing sessions
sources = await migrator.discover_sessions(
    Path("~/.amplifier/sessions").expanduser()
)

print(f"Found {len(sources)} sessions to migrate")

# Migrate with progress callback
def on_progress(result, index, total):
    print(f"[{index}/{total}] {result.session_id}: {result.status.value}")

batch = await migrator.migrate_batch(sources, on_progress=on_progress)

print(f"Completed: {batch.completed}, Failed: {batch.failed}")
```

## Architecture

### Block Types

| Type | Description |
|------|-------------|
| `SESSION_CREATED` | Initial session metadata |
| `SESSION_UPDATED` | Metadata changes |
| `MESSAGE` | Conversation messages |
| `EVENT` | Tool calls, LLM responses, etc. |
| `EVENT_CHUNK` | Large event continuations |
| `REWIND` | History truncation markers |
| `FORK` | Session fork points |

### Partition Strategy

Blocks are partitioned by `{user_id}_{session_id}`:
- All blocks for a session in one partition (efficient reads)
- User-scoped queries without cross-partition scans
- Team/org access via secondary queries

### Sync Protocol

1. Local writes are immediate
2. Background sync uploads new blocks to Cosmos
3. Sequence numbers enable conflict-free merging
4. Hybrid storage combines both for offline-first with cloud backup

## Building Amplifier Applications

This library is designed to be used by Amplifier applications (like `amplifier-app-cli`) to provide session persistence. Here's how to integrate it.

### Basic Application Integration

```python
"""Example: Building an Amplifier application with session storage."""

from amplifier_core import AmplifierSession
from amplifier_session_storage import (
    LocalBlockStorage,
    HybridBlockStorage,
    StorageConfig,
    SessionStore,
)
import os

def create_storage():
    """Create appropriate storage backend based on environment."""
    user_id = os.getenv("USER", "default-user")
    
    # Check if Cosmos DB is configured
    if os.getenv("AMPLIFIER_COSMOS_ENDPOINT"):
        # Hybrid mode: local + cloud sync
        config = StorageConfig(
            user_id=user_id,
            cosmos_endpoint=os.getenv("AMPLIFIER_COSMOS_ENDPOINT"),
            enable_sync=True,
        )
        return HybridBlockStorage(config)
    else:
        # Local only (default)
        config = StorageConfig(user_id=user_id)
        return LocalBlockStorage(config)

# In your application startup:
storage = create_storage()

# Use the drop-in compatible SessionStore for simple cases
store = SessionStore(storage=storage)

# Save a session
await store.save(
    session_id="sess-123",
    transcript=[{"role": "user", "content": "Hello"}],
    metadata={"project": "my-project"},
)

# Load a session
transcript, metadata = await store.load("sess-123")
```

### Advanced: Custom Session Tool for Agents

If you want agents to query session data, create a tool in your application:

```python
"""Example: Creating a session query tool for agents."""

from amplifier_session_storage.tools import SessionTool, SessionToolConfig

# Create the tool with your configuration
session_tool = SessionTool(
    SessionToolConfig(
        project_slug="my-project",
        max_results=50,
        max_excerpt_length=500,
    )
)

# Register with your agent/coordinator (application-specific)
# This depends on how your application handles tool registration
coordinator.register_tool(session_tool)
```

### Integration Pattern with amplifier-app-cli

For `amplifier-app-cli` or similar applications, the integration follows this pattern:

```python
"""Proposed integration pattern for amplifier-app-cli."""

from amplifier_session_storage import (
    StorageConfig,
    LocalBlockStorage,
    CosmosBlockStorage,
    HybridBlockStorage,
)

def load_storage_from_settings(settings: dict, user_id: str):
    """Load storage backend from settings.yaml configuration."""
    storage_config = settings.get("session_storage", {})
    mode = storage_config.get("mode", "local")
    
    config = StorageConfig(user_id=user_id)
    
    if mode == "local":
        return LocalBlockStorage(config)
    
    elif mode == "cloud":
        cloud = storage_config.get("cloud", {})
        config.cosmos_endpoint = cloud.get("endpoint")
        config.cosmos_database = cloud.get("database", "amplifier-db")
        config.cosmos_container = cloud.get("container", "items")
        return CosmosBlockStorage(config)
    
    elif mode == "hybrid":
        cloud = storage_config.get("cloud", {})
        config.cosmos_endpoint = cloud.get("endpoint")
        config.enable_sync = storage_config.get("sync", {}).get("enabled", True)
        return HybridBlockStorage(config)
    
    raise ValueError(f"Unknown storage mode: {mode}")
```

## Integration with Amplifier CLI (Proposed)

This section describes the proposed integration with `amplifier-app-cli` via `settings.yaml`. This will be implemented in a future pass.

### Settings.yaml Schema

```yaml
# ~/.amplifier/settings.yaml

providers:
  - module: provider-anthropic
    config:
      priority: 1

# Session storage configuration
session_storage:
  # Storage mode: "local", "cloud", or "hybrid"
  mode: hybrid
  
  # Local storage settings
  local:
    path: ~/.amplifier/sessions
    
  # Cloud storage settings (Cosmos DB)
  cloud:
    endpoint: https://your-account.documents.azure.com:443/
    database: amplifier-db
    container: items
    # Auth method: default_credential, managed_identity, service_principal
    auth_method: default_credential
  
  # Sync settings (for hybrid mode)
  sync:
    enabled: true
    interval: 5.0
    conflict_resolution: merge  # local_wins, remote_wins, merge
```

### Configuration Examples

#### Local Only (Default)

```yaml
session_storage:
  mode: local
```

#### Cloud with Azure AD (Recommended)

No secrets needed - uses `az login` credentials:

```yaml
session_storage:
  mode: cloud
  cloud:
    endpoint: https://your-account.documents.azure.com:443/
    # auth_method: default_credential (this is the default)
```

#### Hybrid for Multi-Device

```yaml
session_storage:
  mode: hybrid
  cloud:
    endpoint: https://your-account.documents.azure.com:443/
  sync:
    enabled: true
    conflict_resolution: merge
```

#### CI/CD with Service Principal

Secrets via environment variables:

```yaml
session_storage:
  mode: cloud
  cloud:
    endpoint: https://your-account.documents.azure.com:443/
    auth_method: service_principal
    tenant_id: ${AZURE_TENANT_ID}
    client_id: ${AZURE_CLIENT_ID}
    client_secret: ${AZURE_CLIENT_SECRET}
```

#### Azure-Hosted with Managed Identity

```yaml
session_storage:
  mode: cloud
  cloud:
    endpoint: https://your-account.documents.azure.com:443/
    auth_method: managed_identity
    # For user-assigned identity:
    # client_id: your-managed-identity-client-id
```

### Authentication Methods

| Method | Use Case | Secrets Required |
|--------|----------|------------------|
| `default_credential` | Developer workstation | None (uses `az login`) |
| `managed_identity` | Azure App Service, Functions, AKS | None |
| `service_principal` | CI/CD pipelines | Via `${ENV_VAR}` |

### Security Notes

**Safe to put in settings.yaml:**
- Cosmos DB endpoint URLs
- Database/container names
- Auth method selection
- Environment variable references (`${VAR}`)

**Never put in settings.yaml:**
- Cosmos DB keys
- Client secrets
- Any raw credentials

### Required Azure Setup

For Azure AD authentication, assign the Cosmos DB RBAC role:

```bash
# Get your user/service principal object ID
az ad signed-in-user show --query id -o tsv

# Assign Cosmos DB Built-in Data Contributor role
az cosmosdb sql role assignment create \
    --account-name your-cosmos-account \
    --resource-group your-resource-group \
    --role-definition-name "Cosmos DB Built-in Data Contributor" \
    --principal-id YOUR_OBJECT_ID \
    --scope "/"
```

### Implementation Status

- [x] Core storage backends (Local, Cosmos, Hybrid)
- [x] Azure AD authentication support
- [x] Environment variable configuration
- [ ] Settings.yaml loader integration
- [ ] CLI commands for storage management

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
ruff check .
ruff format .

# Type checking
pyright
```

## License

MIT
