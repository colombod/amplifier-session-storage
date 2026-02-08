"""Real Cosmos DB connection test.

Tests the actual changes against the test Cosmos DB instance.
"""

import pytest
import os
from datetime import datetime, timedelta, UTC

from amplifier_session_storage.backends.cosmos import CosmosBackend, CosmosConfig


@pytest.mark.asyncio
async def test_metadata_preservation_real_cosmos():
    """Test metadata preservation with real Cosmos DB."""
    # Use environment variables for test connection
    config = CosmosConfig(
        endpoint=os.environ.get("AMPLIFIER_COSMOS_ENDPOINT", ""),
        database_name=os.environ.get("AMPLIFIER_COSMOS_DATABASE", "amplifier-test-db"),
        auth_method="default_credential",
        enable_vector_search=os.environ.get("AMPLIFIER_COSMOS_ENABLE_VECTOR", "false").lower() == "true",
    )
    
    backend = await CosmosBackend.create(config=config, embedding_provider=None)
    
    try:
        user_id = f"test_user_{datetime.now(UTC).timestamp()}"
        session_id = f"test_session_{datetime.now(UTC).timestamp()}"
        
        # Test 1: Store message with metadata
        messages = [
            {
                "role": "user",
                "content": "Test message",
                "metadata": {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "source": "cli",
                    "confidence": 0.95,
                },
            }
        ]
        
        await backend.sync_transcript_lines(
            user_id=user_id,
            host_id="test_host",
            project_slug="test_project",
            session_id=session_id,
            lines=messages,
            start_sequence=0,
        )
        
        # Test 2: Read back and verify metadata preserved
        retrieved = await backend.get_transcript_lines(
            user_id=user_id,
            project_slug="test_project",
            session_id=session_id,
        )
        
        assert len(retrieved) == 1
        assert "metadata" in retrieved[0], "Metadata field missing from query results!"
        assert retrieved[0]["metadata"]["timestamp"] == "2024-01-01T00:00:00Z"
        assert retrieved[0]["metadata"]["source"] == "cli"
        assert retrieved[0]["metadata"]["confidence"] == 0.95
        
        print("✓ Metadata preservation test PASSED")
        
        # Test 3: get_active_sessions
        await backend.upsert_session_metadata(
            user_id=user_id,
            host_id="test_host",
            session_id=session_id,
            project_slug="test_project",
            bundle="foundation",
            metadata={},
        )
        
        cutoff_date = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        active = await backend.get_active_sessions(
            user_id=user_id,
            start_date=cutoff_date,
            limit=10,
        )
        
        assert len(active) >= 1
        print(f"✓ get_active_sessions test PASSED - found {len(active)} session(s)")
        
    finally:
        await backend.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_metadata_preservation_real_cosmos())
