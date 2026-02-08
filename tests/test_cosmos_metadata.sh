#!/bin/bash
# Test metadata preservation against Cosmos DB test environment

export AMPLIFIER_COSMOS_ENDPOINT="https://amplifier-session-storage-test.documents.azure.com:443/"
export AMPLIFIER_COSMOS_DATABASE="amplifier-test-db"
export AMPLIFIER_COSMOS_ENABLE_VECTOR="true"
export AMPLIFIER_SYNC_SERVER_STORAGE_BACKEND="cosmos"
export AZURE_OPENAI_ENDPOINT="https://amplifier-teamtracking-foundry.cognitiveservices.azure.com/openai/deployments/text-embedding-3-large"
export AZURE_OPENAI_EMBEDDING_MODEL="text-embedding-3-large"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-large"
export AZURE_OPENAI_EMBEDDING_DIMENSIONS="3072"
export AZURE_OPENAI_USE_RBAC=true

# Run only Cosmos-specific tests
pytest tests/test_metadata_preservation.py -v -m cosmos
