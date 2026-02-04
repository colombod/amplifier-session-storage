# Cosmos DB Setup for Testing

This document explains how to set up Azure Cosmos DB for testing the enhanced session storage.

## Test Account Created

A Cosmos DB account has been provisioned for testing:

```
Account Name: your-cosmos-account
Resource Group: your-resource-group
Subscription: your-subscription-id
Endpoint: https://your-cosmos-account.documents.azure.com:443/
Capabilities: EnableServerless, EnableNoSQLVectorSearch
Region: eastus2
```

## Environment Variables for Testing

```bash
# Set for integration tests
export AMPLIFIER_COSMOS_ENDPOINT="https://your-cosmos-account.documents.azure.com:443/"
export AMPLIFIER_COSMOS_DATABASE="your-database"
export AMPLIFIER_COSMOS_AUTH_METHOD="default_credential"
export AMPLIFIER_COSMOS_ENABLE_VECTOR="true"
```

## Running Integration Tests

```bash
# Run integration tests (requires Cosmos DB access)
uv run pytest tests/ -m integration -v

# Run all tests including integration
uv run pytest tests/ -v

# Run only unit tests (no Cosmos DB required)
uv run pytest tests/ -m "not integration" -v
```

## Authentication

The test account uses **DefaultAzureCredential** (recommended approach):

```bash
# Login with Azure CLI
az login

# Verify you have access
az account show

# Should show subscription: OCTO - MADE Explorations
```

**Required RBAC Role**: `Cosmos DB Built-in Data Contributor`

## Assigning RBAC Role

If you need to grant access to another user:

```bash
az cosmosdb sql role assignment create \
    --account-name your-cosmos-account \
    --resource-group your-resource-group \
    --subscription your-subscription-id \
    --role-definition-name "Cosmos DB Built-in Data Contributor" \
    --principal-id <USER_PRINCIPAL_ID> \
    --scope "/"
```

## Vector Search Configuration

The account has been configured with:
- ✅ **EnableNoSQLVectorSearch** capability enabled
- ✅ **Serverless** tier for cost-effective testing
- 📋 Vector indexes will be created automatically by the library on first use

## Container Schema

The library creates three containers:

| Container | Partition Key | Vector Index | Purpose |
|-----------|---------------|--------------|---------|
| `sessions` | `/user_id` | No | Session metadata |
| `transcripts` | `/partition_key` | **Yes (3072-d)** | Messages with embeddings |
| `events` | `/partition_key` | No | Event logs |

## Vector Index Configuration

When vector search is enabled, the `transcripts` container gets:

```json
{
  "vectorIndexes": [
    {
      "path": "/embedding",
      "type": "quantizedFlat",
      "quantizationByteSize": 128,
      "vectorDataType": "float32",
      "dimensions": 3072
    }
  ]
}
```

**Note**: For datasets >100k documents, consider switching to `diskANN` index type for better performance.

## Cost Considerations

**Serverless Tier Pricing** (approximate):
- Storage: $0.25/GB per month
- Request Units: $0.28 per million RUs
- Vector searches: ~10-50 RUs per query depending on dataset size

**Expected test costs**: <$5/month for light testing

## Cleanup

To delete the test account when done:

```bash
az cosmosdb delete \
    --name your-cosmos-account \
    --resource-group your-resource-group \
    --subscription your-subscription-id \
    --yes
```

## Troubleshooting

### "RequestDisallowedByPolicy" error

If you see policy violations when creating Cosmos DB:

1. Check resource group tags (needs `ringValue: r0/r1/r2`)
2. Add skip tag: `skip-CloudGov-CosmosDb-SS=true`
3. Verify serverless is allowed by org policy

### "Vector search not available"

If vector search doesn't work:

1. Verify capability: `az cosmosdb show --name ... --query "capabilities[].name"`
2. Should include: `EnableNoSQLVectorSearch`
3. Enable via portal if missing: Features → Vector Search

### Authentication failures

```bash
# Verify you're logged in
az account show

# Verify subscription is correct
az account set --subscription your-subscription-id

# Test RBAC access
az cosmosdb database list \
    --name your-cosmos-account \
    --resource-group your-resource-group
```

## Development vs Production

**Test Account** (your-cosmos-account):
- Use for: Development, integration testing, experimentation
- Serverless tier (cost-effective)
- Can be deleted and recreated
- No data retention guarantees

**Production Account** (separate):
- Use for: Real data, production workloads
- Provisioned throughput or autoscale
- Geo-replication for HA
- Backup and retention policies
