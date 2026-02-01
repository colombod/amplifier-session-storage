"""Test Cosmos DB connection with Azure AD authentication."""

import asyncio

from amplifier_session_storage import (
    BlockWriter,
    CosmosAuthMethod,
    CosmosBlockStorage,
    StorageConfig,
)


async def test_connection():
    """Test basic Cosmos DB operations."""

    # Configuration for the provisioned instance
    config = StorageConfig(
        user_id="dicolomb-test",
        cosmos_endpoint="https://dicolomb-amplifier-storage-test.documents.azure.com:443/",
        cosmos_auth_method=CosmosAuthMethod.DEFAULT_CREDENTIAL,
        cosmos_database="amplifier-db",
        cosmos_container="items",
        cosmos_partition_key_path="/partitionKey",
    )

    print(f"Connecting to: {config.cosmos_endpoint}")
    print(f"Database: {config.cosmos_database}")
    print(f"Container: {config.cosmos_container}")
    print(f"Auth method: {config.cosmos_auth_method.value}")
    print()

    storage = CosmosBlockStorage(config)

    try:
        # Test 1: Write a test block
        print("1. Creating test session block...")
        writer = BlockWriter(
            session_id="test-session-001",
            user_id="dicolomb-test",
            device_id="test-device",
        )

        block = writer.create_session(
            project_slug="cosmos-test",
            name="Test Session",
            description="Testing Cosmos DB connection",
        )

        await storage.write_block(block)
        print("   ✓ Block written successfully")

        # Test 2: Read blocks back
        print("2. Reading blocks...")
        blocks = await storage.read_blocks("test-session-001")
        print(f"   ✓ Found {len(blocks)} block(s)")

        if blocks:
            b = blocks[0]
            print(f"   - Block ID: {b.block_id}")
            print(f"   - Type: {b.block_type.value}")
            print(f"   - Sequence: {b.sequence}")

        # Test 3: List sessions
        print("3. Listing sessions...")
        sessions = await storage.list_sessions()
        print(f"   ✓ Found {len(sessions)} session(s)")

        # Test 4: Get latest sequence
        print("4. Getting latest sequence...")
        seq = await storage.get_latest_sequence("test-session-001")
        print(f"   ✓ Latest sequence: {seq}")

        # Test 5: Add a message
        print("5. Adding a message...")
        msg_block = writer.add_message(role="user", content="Hello Cosmos!", turn=1)
        await storage.write_block(msg_block)
        print("   ✓ Message block written")

        # Test 6: Verify message was added
        print("6. Verifying message...")
        blocks = await storage.read_blocks("test-session-001")
        print(f"   ✓ Now have {len(blocks)} block(s)")

        # Test 7: Clean up - delete test session
        print("7. Cleaning up test session...")
        await storage.delete_session("test-session-001")
        print("   ✓ Test session deleted")

        print()
        print("=" * 50)
        print("ALL TESTS PASSED! Cosmos DB connection working.")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        raise
    finally:
        await storage.close()


if __name__ == "__main__":
    asyncio.run(test_connection())
