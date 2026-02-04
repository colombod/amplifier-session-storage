# Azure OpenAI Configuration Guide

## Overview

The Azure OpenAI embedding provider supports multiple authentication methods and endpoint types. Configuration is flexible to work with different Azure setups.

## Configuration via Environment Variables

### Foundry AI Services Endpoint (Current Setup)

```bash
# Base endpoint (no deployment path)
export AZURE_OPENAI_ENDPOINT="https://your-openai-resource.openai.azure.com"

# Model and deployment
export AZURE_OPENAI_EMBEDDING_MODEL="text-embedding-3-large"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-large"

# Authentication method
export AZURE_OPENAI_USE_RBAC="true"  # Use DefaultAzureCredential

# Optional: Vector dimensions (auto-detected)
export AZURE_OPENAI_EMBEDDING_DIMENSIONS="3072"

# Optional: Cache size
export AZURE_OPENAI_EMBEDDING_CACHE_SIZE="1000"

# Then authenticate with Azure CLI
az login
```

### Standard Azure OpenAI Resource (API Key)

```bash
# Base endpoint
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"

# Model and deployment
export AZURE_OPENAI_EMBEDDING_MODEL="text-embedding-3-large"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-large"

# API Key authentication
export AZURE_OPENAI_USE_RBAC="false"
export AZURE_OPENAI_API_KEY="your-api-key-here"
```

## Usage in Code

### From Environment

```python
from amplifier_session_storage.embeddings.azure_openai import AzureOpenAIEmbeddings

# Load from environment variables
embeddings = AzureOpenAIEmbeddings.from_env()

# Use in storage backend
async with DuckDBBackend.create(embedding_provider=embeddings) as storage:
    await storage.sync_transcript_lines(...)  # Embeddings generated automatically
```

### Direct Configuration

```python
from amplifier_session_storage.embeddings.azure_openai import AzureOpenAIEmbeddings

# With RBAC (recommended)
embeddings = AzureOpenAIEmbeddings(
    endpoint="https://your-openai-resource.openai.azure.com",
    model="text-embedding-3-large",
    deployment="text-embedding-3-large",
    use_default_credential=True,
    cache_size=1000
)

# With API Key
embeddings = AzureOpenAIEmbeddings(
    endpoint="https://your-resource.openai.azure.com",
    model="text-embedding-3-large",
    deployment="text-embedding-3-large",
    api_key="your-key",
    use_default_credential=False,
    cache_size=1000
)
```

## Testing Configuration

### Running Tests Without Azure OpenAI

Tests use mock embeddings by default:

```bash
# All tests pass without any Azure configuration
uv run pytest tests/ -v -m "not integration"
```

### Running Tests With Real Embeddings

Set environment variables and tests will use real Azure OpenAI:

```bash
# Configure Azure OpenAI
export AZURE_OPENAI_ENDPOINT="https://your-openai-resource.openai.azure.com"
export AZURE_OPENAI_EMBEDDING_MODEL="text-embedding-3-large"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-large"
export AZURE_OPENAI_USE_RBAC="true"

# Authenticate
az login

# Run tests (will use real embeddings)
uv run pytest tests/ -v -m "not integration"
```

Tests automatically detect configuration and use real embeddings when available, falling back to mock otherwise.

## Troubleshooting

### "Resource not found" (404)

**Possible causes**:
1. Incorrect endpoint format
2. Deployment name mismatch
3. Resource not accessible

**Verification steps**:

```bash
# 1. Verify resource exists
az cognitiveservices account show \
    --name your-openai-resource \
    --resource-group your-resource-group

# 2. Verify deployment exists
az cognitiveservices account deployment list \
    --name your-openai-resource \
    --resource-group your-resource-group

# 3. Verify RBAC access
az role assignment list \
    --assignee $(az ad signed-in-user show --query id -o tsv) \
    --scope /subscriptions/your-subscription-id/resourceGroups/your-resource-group/providers/Microsoft.CognitiveServices/accounts/your-openai-resource
```

### "Unauthorized" with RBAC

**Required role**: `Cognitive Services OpenAI User` or `Cognitive Services User`

```bash
# Assign role
az role assignment create \
    --role "Cognitive Services OpenAI User" \
    --assignee $(az ad signed-in-user show --query id -o tsv) \
    --scope /subscriptions/your-subscription-id/resourceGroups/your-resource-group/providers/Microsoft.CognitiveServices/accounts/your-openai-resource
```

### Tests Pass but Real Embeddings Don't Work

**This is expected behavior** - Tests use mock embeddings by default to avoid:
- API costs during development
- Dependency on external services
- Authentication setup requirements

The library works correctly; Azure OpenAI configuration is optional for testing purposes.

## Alternative: Use Mock Embeddings for Development

For cost-effective development and testing:

```python
# Use the mock provider from conftest
from tests.conftest import MockEmbeddingProvider

embeddings = MockEmbeddingProvider(dimensions=3072)

async with DuckDBBackend.create(embedding_provider=embeddings) as storage:
    # All functionality works with deterministic mock embeddings
    await storage.sync_transcript_lines(...)
    results = await storage.search_transcripts(
        options=TranscriptSearchOptions(search_type="hybrid")
    )
```

**Benefits**:
- Zero API costs
- No network dependencies
- Deterministic results for testing
- Fast execution

**When to use real embeddings**:
- Production ingestion pipelines
- Validating search quality
- Benchmarking performance
- Integration testing

## Summary

**For Development**: Mock embeddings (default, no config needed)
**For Testing**: Mock embeddings (tests pass without Azure)
**For Production**: Azure OpenAI with RBAC (requires proper role assignment)

The library is designed to work seamlessly in all three scenarios with graceful degradation.
