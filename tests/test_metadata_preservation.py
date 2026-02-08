"""Tests for metadata field preservation in transcript messages.

Verifies that the metadata field is:
1. Stored correctly in Cosmos DB
2. Retrieved correctly in queries
3. Preserved through round-trip operations
"""

import pytest

from amplifier_session_storage.backends.cosmos import CosmosBackend, CosmosConfig


@pytest.mark.asyncio
@pytest.mark.cosmos
async def test_metadata_preserved_in_write_read(cosmos_backend: CosmosBackend):
    """Test that metadata field is preserved in write/read operations."""
    user_id = "test_user"
    host_id = "test_host"
    project_slug = "test_project"
    session_id = "test_session_metadata"
    
    # Create message with metadata
    messages = [
        {
            "role": "user",
            "content": "Hello",
            "metadata": {
                "timestamp": "2024-01-01T00:00:00Z",
                "source": "cli",
                "confidence": 0.95,
            },
        },
        {
            "role": "assistant",
            "content": "Hi there",
            "metadata": {
                "timestamp": "2024-01-01T00:00:01Z",
                "source": "api",
                "model": "gpt-4",
            },
        },
    ]
    
    # Store messages
    await cosmos_backend.sync_transcript_lines(
        user_id=user_id,
        host_id=host_id,
        project_slug=project_slug,
        session_id=session_id,
        lines=messages,
        start_sequence=0,
    )
    
    # Read back
    retrieved = await cosmos_backend.get_transcript_lines(
        user_id=user_id,
        project_slug=project_slug,
        session_id=session_id,
    )
    
    assert len(retrieved) == 2
    
    # Check first message metadata preserved
    assert "metadata" in retrieved[0]
    assert retrieved[0]["metadata"]["timestamp"] == "2024-01-01T00:00:00Z"
    assert retrieved[0]["metadata"]["source"] == "cli"
    assert retrieved[0]["metadata"]["confidence"] == 0.95
    
    # Check second message metadata preserved
    assert "metadata" in retrieved[1]
    assert retrieved[1]["metadata"]["timestamp"] == "2024-01-01T00:00:01Z"
    assert retrieved[1]["metadata"]["source"] == "api"
    assert retrieved[1]["metadata"]["model"] == "gpt-4"


@pytest.mark.asyncio
@pytest.mark.cosmos
async def test_backward_compat_old_format_still_works(cosmos_backend: CosmosBackend):
    """Test that old format (no metadata) still works."""
    user_id = "test_user"
    host_id = "test_host"
    project_slug = "test_project"
    session_id = "test_session_old_format"
    
    # Old format message (no metadata field)
    messages = [
        {
            "role": "user",
            "content": "Hello",
            "timestamp": "2024-01-01T00:00:00Z",  # Old top-level timestamp
        }
    ]
    
    # Store
    await cosmos_backend.sync_transcript_lines(
        user_id=user_id,
        host_id=host_id,
        project_slug=project_slug,
        session_id=session_id,
        lines=messages,
        start_sequence=0,
    )
    
    # Read back
    retrieved = await cosmos_backend.get_transcript_lines(
        user_id=user_id,
        project_slug=project_slug,
        session_id=session_id,
    )
    
    assert len(retrieved) == 1
    # Should have ts field (normalized)
    assert retrieved[0]["ts"] == "2024-01-01T00:00:00Z"


@pytest.mark.asyncio
@pytest.mark.cosmos
async def test_get_active_sessions_date_filtering(cosmos_backend: CosmosBackend):
    """Test get_active_sessions with date range filtering."""
    from datetime import datetime, timedelta, UTC
    
    user_id = "test_user"
    
    # Create sessions with different dates
    now = datetime.now(UTC)
    old_date = now - timedelta(days=10)
    recent_date = now - timedelta(days=2)
    
    sessions = [
        {
            "user_id": user_id,
            "session_id": "old_session",
            "project_slug": "project1",
            "bundle": "foundation",
            "created": old_date.isoformat(),
            "turn_count": 5,
        },
        {
            "user_id": user_id,
            "session_id": "recent_session",
            "project_slug": "project1",
            "bundle": "foundation",
            "created": recent_date.isoformat(),
            "turn_count": 10,
        },
    ]
    
    # Store sessions
    for sess in sessions:
        await cosmos_backend.upsert_session(
            user_id=sess["user_id"],
            host_id="test_host",
            session_id=sess["session_id"],
            project_slug=sess["project_slug"],
            bundle=sess["bundle"],
            metadata={},
        )
    
    # Query recent sessions only (last 5 days)
    cutoff_date = (now - timedelta(days=5)).isoformat()
    active = await cosmos_backend.get_active_sessions(
        user_id=user_id,
        start_date=cutoff_date,
        limit=100,
    )
    
    # Should only get recent session
    assert len(active) == 1
    assert active[0]["session_id"] == "recent_session"


@pytest.mark.asyncio
@pytest.mark.cosmos
async def test_get_active_sessions_turn_count_filtering(cosmos_backend: CosmosBackend):
    """Test get_active_sessions with min_turn_count filtering."""
    user_id = "test_user"
    
    # Create test sessions via upsert
    sessions_data = [
        ("low_activity", 2),
        ("medium_activity", 15),
        ("high_activity", 50),
    ]
    
    for session_id, turns in sessions_data:
        await cosmos_backend.upsert_session(
            user_id=user_id,
            host_id="test_host",
            session_id=session_id,
            project_slug="activity_test",
            bundle="foundation",
            metadata={"turn_count": turns},
        )
    
    # Query sessions with at least 10 turns
    active = await cosmos_backend.get_active_sessions(
        user_id=user_id,
        project_slug="activity_test",
        min_turn_count=10,
        limit=100,
    )
    
    # Should get medium and high activity sessions
    session_ids = {s["session_id"] for s in active}
    assert "medium_activity" in session_ids
    assert "high_activity" in session_ids
    assert "low_activity" not in session_ids
