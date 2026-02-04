"""
Test Cosmos DB storage operations with real session data.

Tests:
1. Connection and authentication (RBAC via DefaultAzureCredential)
2. Write operations (upsert sessions, transcripts, events)
3. Read operations (query sessions, search transcripts)
4. Vector search with Azure OpenAI embeddings
5. Hybrid search with MMR

Usage:
    # Set environment (uses RBAC by default)
    export AMPLIFIER_COSMOS_ENDPOINT="https://your-cosmos-account.documents.azure.com:443/"
    export AMPLIFIER_COSMOS_DATABASE="your-database"
    export AMPLIFIER_COSMOS_AUTH_METHOD="default_credential"
    export AMPLIFIER_COSMOS_ENABLE_VECTOR="true"

    # Azure OpenAI for embeddings
    export AZURE_OPENAI_ENDPOINT="https://your-openai-resource.openai.azure.com/openai/deployments/text-embedding-3-large"
    export AZURE_OPENAI_API_KEY="<your-key>"

    python scripts/test_cosmos_operations.py
"""

import asyncio
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from amplifier_session_storage.backends.base import SearchFilters, TranscriptSearchOptions
from amplifier_session_storage.backends.cosmos import CosmosBackend, CosmosConfig
from amplifier_session_storage.embeddings.azure_openai import AzureOpenAIEmbeddings
from amplifier_session_storage.sanitization import (
    sanitize_event,
    sanitize_session_metadata,
    sanitize_transcript_message,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def create_embedding_provider() -> AzureOpenAIEmbeddings | None:
    """Create Azure OpenAI embedding provider from environment."""
    try:
        provider = AzureOpenAIEmbeddings.from_env()

        # Test connection
        test_embedding = await provider.embed_text("test")
        logger.info(
            f"✓ Embedding provider ready: {provider.model_name}, {len(test_embedding)} dimensions"
        )
        return provider
    except Exception as e:
        logger.warning(f"⚠ Embedding provider not available: {e}")
        return None


async def load_project_sessions(project_path: Path, limit: int = 2) -> list[dict]:
    """Load sessions from an Amplifier project directory."""
    sessions = []
    sessions_dir = project_path / "sessions"

    if not sessions_dir.exists():
        logger.warning(f"No sessions directory at {sessions_dir}")
        return sessions

    for session_dir in sorted(sessions_dir.iterdir())[:limit]:
        if not session_dir.is_dir():
            continue

        session_data = {
            "session_id": session_dir.name,
            "metadata": None,
            "transcript": [],
            "events": [],
        }

        # Load metadata
        metadata_file = session_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                session_data["metadata"] = sanitize_session_metadata(json.load(f))

        # Load transcript (limit lines for test)
        transcript_file = session_dir / "transcript.jsonl"
        if transcript_file.exists():
            with open(transcript_file) as f:
                for i, line in enumerate(f):
                    if i >= 50:  # Limit for testing
                        break
                    msg = sanitize_transcript_message(json.loads(line.strip()))
                    session_data["transcript"].append(msg)

        # Load events (limit for test)
        events_file = session_dir / "events.jsonl"
        if events_file.exists():
            with open(events_file) as f:
                for i, line in enumerate(f):
                    if i >= 100:  # Limit for testing
                        break
                    try:
                        event = sanitize_event(json.loads(line.strip()))
                        session_data["events"].append(event)
                    except json.JSONDecodeError:
                        continue

        sessions.append(session_data)
        logger.info(
            f"  Loaded session: {session_dir.name} "
            f"(transcript: {len(session_data['transcript'])}, events: {len(session_data['events'])})"
        )

    return sessions


async def test_cosmos_connection(storage: CosmosBackend) -> bool:
    """Test basic Cosmos DB connectivity."""
    print("\n" + "=" * 60)
    print("TEST 1: Cosmos DB Connection")
    print("=" * 60)

    try:
        # Check connection by getting statistics
        stats = await storage.get_session_statistics("test-user")
        print("✓ Connected to Cosmos DB")
        print(f"  Endpoint: {storage.config.endpoint}")
        print(f"  Database: {storage.config.database_name}")
        print(f"  Vector search enabled: {storage.config.enable_vector_search}")
        print(f"  Current stats: {stats}")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


async def test_write_operations(
    storage: CosmosBackend,
    sessions: list[dict],
    user_id: str = "test-user",
    host_id: str = "test-host",
) -> bool:
    """Test write operations: upsert sessions, transcripts, events."""
    print("\n" + "=" * 60)
    print("TEST 2: Write Operations")
    print("=" * 60)

    try:
        for session in sessions:
            session_id = session["session_id"]
            metadata = session["metadata"]

            if not metadata:
                logger.warning(f"Skipping {session_id} - no metadata")
                continue

            project_slug = metadata.get("project_slug", "test-project")

            # 1. Upsert session metadata
            print(f"\n  Writing session: {session_id}")
            await storage.upsert_session_metadata(
                user_id=user_id,
                host_id=host_id,
                metadata=metadata,
            )
            print("    ✓ Metadata written")

            # 2. Upsert transcript lines
            transcript = session["transcript"]
            if transcript:
                written = await storage.sync_transcript_lines(
                    user_id=user_id,
                    host_id=host_id,
                    project_slug=project_slug,
                    session_id=session_id,
                    lines=transcript,
                )
                print(f"    ✓ Transcript written: {written} messages")

            # 3. Upsert events
            events = session["events"]
            if events:
                written = await storage.sync_event_lines(
                    user_id=user_id,
                    host_id=host_id,
                    project_slug=project_slug,
                    session_id=session_id,
                    lines=events,
                )
                print(f"    ✓ Events written: {written} events")

        print(f"\n✓ Write operations successful: {len(sessions)} sessions")
        return True

    except Exception as e:
        print(f"✗ Write operations failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_read_operations(
    storage: CosmosBackend,
    user_id: str = "test-user",
) -> bool:
    """Test read operations: search sessions, query transcripts."""
    print("\n" + "=" * 60)
    print("TEST 3: Read Operations")
    print("=" * 60)

    try:
        # 1. Get session statistics
        stats = await storage.get_session_statistics(user_id)
        print("\n  Session statistics:")
        print(f"    Total sessions: {stats['total_sessions']}")
        print(f"    By project: {stats['sessions_by_project']}")
        print(f"    By bundle: {stats['sessions_by_bundle']}")

        # 2. Search sessions
        sessions = await storage.search_sessions(
            user_id=user_id,
            filters=SearchFilters(),
            limit=10,
        )
        print(f"\n  Found {len(sessions)} sessions")
        for s in sessions[:3]:
            print(
                f"    - {s.get('session_id', 'unknown')[:40]}... (turns: {s.get('turn_count', 0)})"
            )

        # 3. Full-text search in transcripts
        if sessions:
            first_session = sessions[0]
            project_slug = first_session.get("project_slug", "test-project")

            results = await storage.search_transcripts(
                user_id=user_id,
                options=TranscriptSearchOptions(
                    query="test",
                    search_type="full_text",
                    filters=SearchFilters(project_slug=project_slug),
                ),
                limit=5,
            )
            print(f"\n  Full-text search for 'test': {len(results)} results")
            for r in results[:2]:
                content_preview = r.content[:80] if r.content else ""
                print(f"    - Score: {r.score:.3f}, Content: {content_preview}...")

        print("\n✓ Read operations successful")
        return True

    except Exception as e:
        print(f"✗ Read operations failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_vector_search(
    storage: CosmosBackend,
    user_id: str = "test-user",
) -> bool:
    """Test vector search with embeddings."""
    print("\n" + "=" * 60)
    print("TEST 4: Vector Search")
    print("=" * 60)

    if not storage.embedding_provider:
        print("⚠ Skipping vector search - no embedding provider")
        return True

    try:
        # Check if vector search is supported
        vector_supported = await storage.supports_vector_search()
        print(f"  Vector search supported: {vector_supported}")

        if not vector_supported:
            print("⚠ Vector search not available (check container configuration)")
            return True

        # Generate query embedding
        query = "How to implement authentication?"
        print(f"\n  Query: '{query}'")

        query_vector = await storage.embedding_provider.embed_text(query)
        print(f"  Query vector: {len(query_vector)} dimensions")

        # Perform vector search
        results = await storage.vector_search(
            user_id=user_id,
            query_vector=query_vector,
            filters=SearchFilters(),
            top_k=5,
        )

        print(f"\n  Vector search results: {len(results)}")
        for r in results[:3]:
            content_preview = r.content[:80] if r.content else "(no content)"
            print(f"    - Score: {r.score:.3f}, Source: {r.source}")
            print(f"      Content: {content_preview}...")

        print("\n✓ Vector search successful")
        return True

    except Exception as e:
        print(f"✗ Vector search failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_hybrid_search(
    storage: CosmosBackend,
    user_id: str = "test-user",
) -> bool:
    """Test hybrid search (keyword + semantic + MMR)."""
    print("\n" + "=" * 60)
    print("TEST 5: Hybrid Search with MMR")
    print("=" * 60)

    if not storage.embedding_provider:
        print("⚠ Skipping hybrid search - no embedding provider")
        return True

    try:
        query = "vector database implementation"
        print(f"\n  Query: '{query}'")

        # Test hybrid search
        results = await storage.search_transcripts(
            user_id=user_id,
            options=TranscriptSearchOptions(
                query=query,
                search_type="hybrid",
                mmr_lambda=0.7,  # Balance relevance and diversity
                filters=SearchFilters(),
            ),
            limit=5,
        )

        print(f"\n  Hybrid search results: {len(results)}")
        for r in results[:3]:
            content_preview = r.content[:80] if r.content else "(no content)"
            print(f"    - Score: {r.score:.3f}, Source: {r.source}")
            print(f"      Content: {content_preview}...")

        print("\n✓ Hybrid search successful")
        return True

    except Exception as e:
        print(f"✗ Hybrid search failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all Cosmos DB tests."""
    print("=" * 60)
    print("COSMOS DB STORAGE TEST")
    print("=" * 60)
    print(f"Started: {datetime.now(UTC).isoformat()}")

    # Configuration
    user_id = "test-user"
    host_id = "test-host"

    # Find a project to load - prefer one with actual transcript data
    amplifier_dir = Path.home() / ".amplifier" / "projects"
    projects = list(amplifier_dir.iterdir()) if amplifier_dir.exists() else []

    if not projects:
        print(f"\n⚠ No projects found in {amplifier_dir}")
        print("  Will run with synthetic test data")
        project_path = None
    else:
        # Pick a project that has sessions WITH transcript files
        project_path = None
        for p in sorted(projects):
            sessions_dir = p / "sessions"
            if sessions_dir.exists():
                # Check if any session has a transcript file
                for session_dir in sessions_dir.iterdir():
                    if session_dir.is_dir() and (session_dir / "transcript.jsonl").exists():
                        project_path = p
                        break
                if project_path:
                    break

        if project_path:
            print(f"\n📁 Using project: {project_path.name}")

    # Create embedding provider
    print("\n🔧 Creating embedding provider...")
    embedding_provider = await create_embedding_provider()

    # Create Cosmos backend
    print("\n🔧 Creating Cosmos backend...")
    try:
        config = CosmosConfig(
            endpoint=os.environ.get(
                "AMPLIFIER_COSMOS_ENDPOINT",
                "https://your-cosmos-account.documents.azure.com:443/",
            ),
            database_name=os.environ.get("AMPLIFIER_COSMOS_DATABASE", "your-database"),
            auth_method=os.environ.get("AMPLIFIER_COSMOS_AUTH_METHOD", "default_credential"),
            enable_vector_search=True,
        )

        storage = await CosmosBackend.create(
            config=config,
            embedding_provider=embedding_provider,
        )
        print("✓ Cosmos backend created")

    except Exception as e:
        print(f"✗ Failed to create Cosmos backend: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Load test data
    sessions = []
    if project_path:
        print(f"\n📥 Loading sessions from {project_path.name}...")
        sessions = await load_project_sessions(project_path, limit=2)
        print(f"  Loaded {len(sessions)} sessions")

    if not sessions:
        # Create synthetic test data
        print("\n📝 Creating synthetic test data...")
        sessions = [
            {
                "session_id": f"test-session-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
                "metadata": {
                    "session_id": f"test-session-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
                    "project_slug": "test-project",
                    "bundle": "test-bundle",
                    "created": datetime.now(UTC).isoformat(),
                    "turn_count": 3,
                },
                "transcript": [
                    {
                        "role": "user",
                        "content": "How do I implement vector search in Cosmos DB?",
                        "turn": 1,
                        "ts": datetime.now(UTC).isoformat(),
                    },
                    {
                        "role": "assistant",
                        "content": "To implement vector search in Cosmos DB, you need to enable the NoSQLVectorSearch capability and create vector indexes on your container.",
                        "turn": 1,
                        "ts": datetime.now(UTC).isoformat(),
                    },
                    {
                        "role": "user",
                        "content": "What about authentication and RBAC?",
                        "turn": 2,
                        "ts": datetime.now(UTC).isoformat(),
                    },
                ],
                "events": [
                    {
                        "event": "session_start",
                        "ts": datetime.now(UTC).isoformat(),
                        "session_id": f"test-session-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
                        "lvl": "info",
                        "sequence": 0,
                    },
                ],
            }
        ]
        print(f"  Created {len(sessions)} synthetic sessions")

    # Run tests
    results = {}

    try:
        results["connection"] = await test_cosmos_connection(storage)

        if results["connection"]:
            results["write"] = await test_write_operations(storage, sessions, user_id, host_id)
            results["read"] = await test_read_operations(storage, user_id)
            results["vector"] = await test_vector_search(storage, user_id)
            results["hybrid"] = await test_hybrid_search(storage, user_id)

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        await storage.close()
        print("  ✓ Connections closed")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"Completed: {datetime.now(UTC).isoformat()}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
