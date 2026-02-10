"""Verify the StorageBackend protocol split hierarchy.

The backend protocol is split into focused ABCs:
- _StorageLifecycle: shared initialize/close lifecycle
- StorageReader: read, query, and search operations
- StorageWriter: write and sync operations
- StorageAdmin: administrative operations
- StorageBackend: composed type combining all three

This test ensures the class hierarchy is correct and that concrete
backends inherit all protocol ABCs through StorageBackend.
"""

import pytest

from amplifier_session_storage.backends.base import (
    StorageAdmin,
    StorageBackend,
    StorageReader,
    StorageWriter,
    _StorageLifecycle,
)
from amplifier_session_storage.backends.duckdb import DuckDBBackend
from amplifier_session_storage.backends.sqlite import SQLiteBackend


class TestProtocolHierarchy:
    """StorageBackend composes all three protocol ABCs."""

    def test_storage_backend_is_subclass_of_reader(self) -> None:
        assert issubclass(StorageBackend, StorageReader)

    def test_storage_backend_is_subclass_of_writer(self) -> None:
        assert issubclass(StorageBackend, StorageWriter)

    def test_storage_backend_is_subclass_of_admin(self) -> None:
        assert issubclass(StorageBackend, StorageAdmin)

    def test_all_abcs_share_lifecycle(self) -> None:
        assert issubclass(StorageReader, _StorageLifecycle)
        assert issubclass(StorageWriter, _StorageLifecycle)
        assert issubclass(StorageAdmin, _StorageLifecycle)


class TestConcreteBackendsInheritProtocols:
    """Concrete backends inherit all protocols through StorageBackend."""

    @pytest.mark.parametrize("backend_cls", [DuckDBBackend, SQLiteBackend])
    def test_backend_is_storage_reader(self, backend_cls: type) -> None:
        assert issubclass(backend_cls, StorageReader)

    @pytest.mark.parametrize("backend_cls", [DuckDBBackend, SQLiteBackend])
    def test_backend_is_storage_writer(self, backend_cls: type) -> None:
        assert issubclass(backend_cls, StorageWriter)

    @pytest.mark.parametrize("backend_cls", [DuckDBBackend, SQLiteBackend])
    def test_backend_is_storage_admin(self, backend_cls: type) -> None:
        assert issubclass(backend_cls, StorageAdmin)

    @pytest.mark.parametrize("backend_cls", [DuckDBBackend, SQLiteBackend])
    def test_backend_is_storage_backend(self, backend_cls: type) -> None:
        assert issubclass(backend_cls, StorageBackend)


class TestMethodDistribution:
    """Each protocol ABC declares the expected abstract methods."""

    def _abstract_methods(self, cls: type) -> set[str]:
        """Get abstract methods defined directly on a class (not inherited)."""
        own = set()
        for name in vars(cls):
            obj = getattr(cls, name, None)
            if getattr(obj, "__isabstractmethod__", False):
                own.add(name)
        return own

    def test_lifecycle_methods(self) -> None:
        methods = self._abstract_methods(_StorageLifecycle)
        assert methods == {"initialize", "close"}

    def test_reader_methods(self) -> None:
        methods = self._abstract_methods(StorageReader)
        expected = {
            "get_session_metadata",
            "get_transcript_lines",
            "get_turn_context",
            "get_message_context",
            "get_event_lines",
            "search_sessions",
            "search_transcripts",
            "search_events",
            "supports_vector_search",
            "vector_search",
            "list_users",
            "list_projects",
            "list_sessions",
            "get_active_sessions",
            "get_session_statistics",
        }
        assert methods == expected

    def test_writer_methods(self) -> None:
        methods = self._abstract_methods(StorageWriter)
        expected = {
            "upsert_session_metadata",
            "sync_transcript_lines",
            "sync_event_lines",
            "upsert_embeddings",
            "get_session_sync_stats",
        }
        assert methods == expected

    def test_admin_methods(self) -> None:
        methods = self._abstract_methods(StorageAdmin)
        assert methods == {"delete_session", "backfill_embeddings", "rebuild_vectors"}

    def test_storage_backend_adds_no_new_methods(self) -> None:
        methods = self._abstract_methods(StorageBackend)
        assert methods == set()

    def test_total_abstract_method_count(self) -> None:
        """25 abstract methods total across the hierarchy."""
        all_abstract = set()
        for cls in (_StorageLifecycle, StorageReader, StorageWriter, StorageAdmin):
            all_abstract |= self._abstract_methods(cls)
        assert len(all_abstract) == 25
