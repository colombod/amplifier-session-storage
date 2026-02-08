"""Live test against Cosmos DB test environment.

Run without pytest - direct execution.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta, UTC


async def test_metadata_preservation():
    """Test that metadata field is preserved in write/read cycle."""
    from amplifier_session_storage.backends.cosmos import CosmosBackend, CosmosConfig
    
    print("=" * 80)
    print("TEST 1: Metadata Preservation")
    print("=" * 80)
    
    config = CosmosConfig(
        endpoint=os.environ["AMPLIFIER_COSMOS_ENDPOINT"],
        database_name=os.environ["AMPLIFIER_COSMOS_DATABASE"],
        auth_method="default_credential",
        enable_vector_search=os.environ.get("AMPLIFIER_COSMOS_ENABLE_VECTOR", "false").lower() == "true",
    )
    
    print(f"Connecting to: {config.endpoint}")
    print(f"Database: {config.database_name}")
    
    backend = await CosmosBackend.create(config=config, embedding_provider=None)
    
    try:
        # Use timestamp-based IDs for uniqueness
        test_timestamp = int(datetime.now(UTC).timestamp() * 1000)
        user_id = f"test_user_{test_timestamp}"
        session_id = f"test_session_{test_timestamp}"
        
        print(f"\nTest session: {session_id}")
        
        # Test message with rich metadata
        messages = [
            {
                "role": "user",
                "content": "Test message with metadata",
                "metadata": {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "source": "cli",
                    "confidence": 0.95,
                    "test_field": "custom_value",
                },
            }
        ]
        
        print("\n1. Writing message with metadata...")
        await backend.sync_transcript_lines(
            user_id=user_id,
            host_id="test_host",
            project_slug="test_project",
            session_id=session_id,
            lines=messages,
            start_sequence=0,
        )
        print("   ✓ Written successfully")
        
        print("\n2. Reading back from Cosmos DB...")
        retrieved = await backend.get_transcript_lines(
            user_id=user_id,
            project_slug="test_project",
            session_id=session_id,
        )
        
        print(f"   ✓ Retrieved {len(retrieved)} message(s)")
        
        # Validate metadata preservation
        if len(retrieved) != 1:
            print(f"   ✗ FAILED: Expected 1 message, got {len(retrieved)}")
            return False
        
        msg = retrieved[0]
        
        print("\n3. Validating metadata fields...")
        if "metadata" not in msg:
            print("   ✗ FAILED: metadata field missing from query results!")
            print(f"   Available fields: {list(msg.keys())}")
            return False
        
        metadata = msg["metadata"]
        
        # Check all expected fields
        checks = [
            ("timestamp", "2024-01-01T00:00:00Z"),
            ("source", "cli"),
            ("confidence", 0.95),
            ("test_field", "custom_value"),
        ]
        
        for field, expected in checks:
            if field not in metadata:
                print(f"   ✗ FAILED: metadata.{field} missing")
                return False
            if metadata[field] != expected:
                print(f"   ✗ FAILED: metadata.{field} = {metadata[field]}, expected {expected}")
                return False
            print(f"   ✓ metadata.{field} = {expected}")
        
        print("\n✓ METADATA PRESERVATION TEST PASSED")
        return True
        
    finally:
        await backend.close()


async def test_active_sessions():
    """Test get_active_sessions with date filtering."""
    from amplifier_session_storage.backends.cosmos import CosmosBackend, CosmosConfig
    
    print("\n" + "=" * 80)
    print("TEST 2: Active Sessions Query")
    print("=" * 80)
    
    config = CosmosConfig(
        endpoint=os.environ["AMPLIFIER_COSMOS_ENDPOINT"],
        database_name=os.environ["AMPLIFIER_COSMOS_DATABASE"],
        auth_method="default_credential",
        enable_vector_search=False,  # Not needed for this test
    )
    
    backend = await CosmosBackend.create(config=config, embedding_provider=None)
    
    try:
        test_timestamp = int(datetime.now(UTC).timestamp() * 1000)
        user_id = f"test_user_{test_timestamp}"
        
        # Create test session
        session_id = f"active_test_{test_timestamp}"
        
        print(f"\n1. Creating test session: {session_id}")
        await backend.upsert_session_metadata(
            user_id=user_id,
            host_id="test_host",
            metadata={
                "session_id": session_id,
                "project_slug": "test_project_active",
                "bundle": "foundation",
            },
        )
        
        # Add transcript messages to create activity
        messages = [
            {
                "role": "user",
                "content": "Test activity",
                "metadata": {"timestamp": datetime.now(UTC).isoformat()},
            }
        ]
        await backend.sync_transcript_lines(
            user_id=user_id,
            host_id="test_host",
            project_slug="test_project_active",
            session_id=session_id,
            lines=messages,
            start_sequence=0,
        )
        print("   ✓ Session created")
        

        # Important: This tests activity-based detection
        # The session was just created, so it should appear in "recent activity" queries
        # Query recent sessions (last 1 hour)
        cutoff_date = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        
        print(f"\n2. Querying active sessions since {cutoff_date}...")
        
        # First check: Can we find the transcript we just created?
        print("   DEBUG: Checking if transcript was stored...")
        transcripts = await backend.get_transcript_lines(
            user_id=user_id,
            project_slug="test_project_active",
            session_id=session_id,
        )
        print(f"   DEBUG: Found {len(transcripts)} transcript(s)")
        if transcripts:
            print(f"   DEBUG: Transcript ts: {transcripts[0].get('ts', 'NO TS')}")
        
        active = await backend.get_active_sessions(
            user_id=user_id,
            start_date=cutoff_date,
            limit=10,
        )
        
        print(f"   ✓ Found {len(active)} session(s)")
        if active:
            for s in active:
                print(f"   DEBUG: Session {s['session_id']}: created={s.get('created', 'N/A')}")
        
        # Validate our session is in results
        session_ids = [s["session_id"] for s in active]
        if session_id in session_ids:
            print(f"   ✓ Test session found in results")
        else:
            print(f"   ✗ FAILED: Test session not found in results")
            print(f"   Results: {session_ids}")
            return False
        
        # Test project filtering
        print(f"\n3. Testing project filtering...")
        project_filtered = await backend.get_active_sessions(
            user_id=user_id,
            project_slug="test_project_active",
            limit=10,
        )
        
        print(f"   ✓ Found {len(project_filtered)} session(s) in project")
        
        if session_id in [s["session_id"] for s in project_filtered]:
            print(f"   ✓ Session found with project filter")
        else:
            print(f"   ✗ FAILED: Session not found with project filter")
            return False
        
        print("\n✓ ACTIVE SESSIONS QUERY TEST PASSED")
        return True
        
    finally:
        await backend.close()


async def main():
    """Run all tests."""
    print("\nCosmos DB Test Environment Validation")
    print("=" * 80)
    print(f"Endpoint: {os.environ.get('AMPLIFIER_COSMOS_ENDPOINT', 'NOT SET')}")
    print(f"Database: {os.environ.get('AMPLIFIER_COSMOS_DATABASE', 'NOT SET')}")
    print(f"Vector enabled: {os.environ.get('AMPLIFIER_COSMOS_ENABLE_VECTOR', 'NOT SET')}")
    print(f"RBAC: {os.environ.get('AZURE_OPENAI_USE_RBAC', 'NOT SET')}")
    print("=" * 80)
    
    results = []
    
    # Test 1: Metadata preservation
    try:
        result1 = await test_metadata_preservation()
        results.append(("Metadata Preservation", result1))
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Metadata Preservation", False))
    
    # Test 2: Active sessions
    try:
        result2 = await test_active_sessions()
        results.append(("Active Sessions", result2))
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Active Sessions", False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(r[1] for r in results)
    print("=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
