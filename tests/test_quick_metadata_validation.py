"""Quick validation test for metadata preservation.

Can run without Cosmos DB connection to verify logic.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


def test_transcript_projection_includes_metadata():
    """Verify TRANSCRIPT_PROJECTION contains c.metadata field."""
    from amplifier_session_storage.backends.cosmos import TRANSCRIPT_PROJECTION
    
    assert "c.metadata" in TRANSCRIPT_PROJECTION
    assert "c.ts" in TRANSCRIPT_PROJECTION
    assert "c.role" in TRANSCRIPT_PROJECTION
    assert "c.content" in TRANSCRIPT_PROJECTION


def test_base_protocol_has_get_active_sessions():
    """Verify base protocol includes get_active_sessions method."""
    from amplifier_session_storage.backends.base import StorageReader
    import inspect
    
    # Check that the method exists
    assert hasattr(StorageReader, "get_active_sessions")
    
    # Check it's abstract
    method = getattr(StorageReader, "get_active_sessions")
    assert inspect.iscoroutinefunction(method)
