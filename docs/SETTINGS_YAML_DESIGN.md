# Settings.yaml Integration Design

This document describes how session storage integrates with Amplifier's `settings.yaml` configuration pattern.

## Overview

Amplifier uses `~/.amplifier/settings.yaml` for user configuration. Session storage adds a `session_storage` section that mirrors the existing provider/module pattern.

## Configuration Schema

```yaml
# ~/.amplifier/settings.yaml

# Existing amplifier settings...
providers:
  - module: provider-anthropic
    config:
      priority: 1

# NEW: Session storage configuration
session_storage:
  # Storage mode: "local", "cloud", or "hybrid" (default: "local")
  mode: hybrid
  
  # Local storage settings (always available)
  local:
    path: ~/.amplifier/sessions  # Default location
    
  # Cloud storage settings (Cosmos DB)
  cloud:
    # Cosmos DB endpoint (required for cloud/hybrid mode)
    endpoint: https://your-account.documents.azure.com:443/
    
    # Database and container (have sensible defaults)
    database: amplifier-db
    container: items
    
    # Authentication method (default: default_credential)
    # Options: default_credential, managed_identity, service_principal, key
    auth_method: default_credential
    
    # For managed_identity with user-assigned identity:
    # client_id: your-managed-identity-client-id
    
    # For service_principal (secrets should use env vars):
    # tenant_id: ${AZURE_TENANT_ID}
    # client_id: ${AZURE_CLIENT_ID}
    # client_secret: ${AZURE_CLIENT_SECRET}
    
    # For key auth (not recommended, use env var):
    # key: ${AMPLIFIER_COSMOS_KEY}
  
  # Sync settings (for hybrid mode)
  sync:
    enabled: true
    interval: 5.0  # seconds between sync attempts
    conflict_resolution: merge  # local_wins, remote_wins, merge, manual
    
  # Real-time sync (optional)
  realtime:
    enabled: false
    # url: wss://your-sync-server.example.com/sync
```

## Minimal Configurations

### Local Only (Default)

```yaml
session_storage:
  mode: local
```

Or simply omit the section entirely - local mode is the default.

### Cloud Only (Azure AD)

```yaml
session_storage:
  mode: cloud
  cloud:
    endpoint: https://your-account.documents.azure.com:443/
```

Uses `default_credential` which picks up Azure CLI login, managed identity, or environment variables automatically.

### Hybrid (Recommended for Multi-Device)

```yaml
session_storage:
  mode: hybrid
  cloud:
    endpoint: https://your-account.documents.azure.com:443/
  sync:
    enabled: true
    conflict_resolution: merge
```

## Environment Variable Substitution

Settings support `${ENV_VAR}` syntax for secrets:

```yaml
session_storage:
  mode: cloud
  cloud:
    endpoint: ${AMPLIFIER_COSMOS_ENDPOINT}
    auth_method: service_principal
    tenant_id: ${AZURE_TENANT_ID}
    client_id: ${AZURE_CLIENT_ID}
    client_secret: ${AZURE_CLIENT_SECRET}
```

## Authentication Patterns

### 1. Developer Workstation (Recommended)

Uses Azure CLI credentials - no configuration needed beyond endpoint:

```yaml
session_storage:
  mode: hybrid
  cloud:
    endpoint: https://your-account.documents.azure.com:443/
    # auth_method defaults to default_credential
    # Automatically uses: az login credentials
```

**Setup:**
```bash
az login
az cosmosdb sql role assignment create \
  --account-name your-account \
  --resource-group your-rg \
  --role-definition-name "Cosmos DB Built-in Data Contributor" \
  --principal-id $(az ad signed-in-user show --query id -o tsv) \
  --scope "/"
```

### 2. Azure-Hosted Service (App Service, Functions, AKS)

Uses managed identity - zero secrets:

```yaml
session_storage:
  mode: cloud
  cloud:
    endpoint: https://your-account.documents.azure.com:443/
    auth_method: managed_identity
    # For user-assigned identity, add:
    # client_id: your-managed-identity-client-id
```

**Setup:**
- Enable managed identity on your Azure resource
- Assign `Cosmos DB Built-in Data Contributor` role to the identity

### 3. CI/CD Pipeline

Uses service principal with secrets in environment:

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

**Setup:**
- Create service principal: `az ad sp create-for-rbac --name amplifier-ci`
- Store credentials in CI/CD secrets
- Assign Cosmos DB role to the service principal

### 4. Key-Based (Development Only)

Not recommended for production:

```yaml
session_storage:
  mode: cloud
  cloud:
    endpoint: https://your-account.documents.azure.com:443/
    auth_method: key
    key: ${AMPLIFIER_COSMOS_KEY}
```

## Implementation

### Loading Configuration

```python
# In amplifier-app-cli or similar

from pathlib import Path
import yaml
from amplifier_session_storage import (
    StorageConfig,
    CosmosAuthMethod,
    LocalBlockStorage,
    CosmosBlockStorage,
    HybridBlockStorage,
)

def load_session_storage(settings_path: Path, user_id: str):
    """Load session storage from settings.yaml."""
    
    with open(settings_path) as f:
        settings = yaml.safe_load(f)
    
    storage_config = settings.get("session_storage", {})
    mode = storage_config.get("mode", "local")
    
    # Build StorageConfig
    cloud = storage_config.get("cloud", {})
    sync = storage_config.get("sync", {})
    local = storage_config.get("local", {})
    
    config = StorageConfig(
        user_id=user_id,
        # Local settings
        local_path=local.get("path"),
        # Cloud settings
        cosmos_endpoint=_expand_env(cloud.get("endpoint")),
        cosmos_database=cloud.get("database", "amplifier-db"),
        cosmos_container=cloud.get("container", "items"),
        cosmos_auth_method=CosmosAuthMethod(
            cloud.get("auth_method", "default_credential")
        ),
        cosmos_key=_expand_env(cloud.get("key")),
        azure_tenant_id=_expand_env(cloud.get("tenant_id")),
        azure_client_id=_expand_env(cloud.get("client_id")),
        azure_client_secret=_expand_env(cloud.get("client_secret")),
        # Sync settings
        enable_sync=sync.get("enabled", False),
        sync_interval=sync.get("interval", 5.0),
    )
    
    # Create appropriate storage backend
    if mode == "local":
        return LocalBlockStorage(config)
    elif mode == "cloud":
        return CosmosBlockStorage(config)
    elif mode == "hybrid":
        return HybridBlockStorage(config)
    else:
        raise ValueError(f"Unknown storage mode: {mode}")


def _expand_env(value: str | None) -> str | None:
    """Expand ${ENV_VAR} references in a value."""
    if value is None:
        return None
    if value.startswith("${") and value.endswith("}"):
        import os
        env_var = value[2:-1]
        return os.environ.get(env_var)
    return value
```

### Integration with IdentityContext

The user_id comes from Amplifier's identity system:

```python
from amplifier_session_storage import IdentityContext

# Initialize identity (done once at app startup)
IdentityContext.initialize()

# Get user_id for storage config
user_id = IdentityContext.get_user_id()

# Load storage with user context
storage = load_session_storage(settings_path, user_id)
```

## Security Considerations

### What Goes in settings.yaml (Safe)

- Endpoints (public URLs)
- Database/container names
- Auth method selection
- Sync preferences
- References to environment variables (`${VAR}`)

### What NEVER Goes in settings.yaml

- Cosmos DB keys
- Client secrets
- Any credential that could be used to access data

### Recommended Pattern

```yaml
# settings.yaml - checked into dotfiles, safe to share
session_storage:
  mode: hybrid
  cloud:
    endpoint: https://my-account.documents.azure.com:443/
    auth_method: default_credential  # Uses Azure CLI login
```

No secrets needed! Azure AD handles authentication securely.

## Migration Path

### From No Cloud Storage

1. Add `session_storage` section to settings.yaml
2. Set up Cosmos DB with RBAC
3. Run migration tool to sync existing sessions

### From Environment Variables

If already using env vars, settings.yaml can reference them:

```yaml
session_storage:
  mode: cloud
  cloud:
    endpoint: ${AMPLIFIER_COSMOS_ENDPOINT}
```

Both approaches work - choose based on preference.

## Example: Your Current Setup

Based on your provisioned Cosmos DB:

```yaml
# ~/.amplifier/settings.yaml

providers:
  - module: provider-anthropic
    config:
      priority: 1

session_storage:
  mode: hybrid
  cloud:
    endpoint: https://your-cosmos-account.documents.azure.com:443/
    database: amplifier-db
    container: items
    # auth_method: default_credential (this is the default)
  sync:
    enabled: true
    conflict_resolution: merge
```

No secrets needed - uses your `az login` credentials automatically.
