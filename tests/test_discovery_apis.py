"""
Tests for Discovery APIs across all storage backends.

Tests list_users, list_projects, list_sessions, and get_message_context.
"""

from __future__ import annotations

import pytest

from amplifier_session_storage.backends import (
    SearchFilters,
)

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_sessions():
    """Sample session data for testing discovery APIs."""
    return [
        {
            "session_id": "session-1",
            "user_id": "user-alice",
            "project_slug": "project-alpha",
            "bundle": "foundation",
            "created": "2024-01-15T10:00:00Z",
            "turn_count": 5,
        },
        {
            "session_id": "session-2",
            "user_id": "user-alice",
            "project_slug": "project-beta",
            "bundle": "design",
            "created": "2024-01-16T10:00:00Z",
            "turn_count": 10,
        },
        {
            "session_id": "session-3",
            "user_id": "user-bob",
            "project_slug": "project-alpha",
            "bundle": "foundation",
            "created": "2024-01-17T10:00:00Z",
            "turn_count": 3,
        },
        {
            "session_id": "session-4",
            "user_id": "user-charlie",
            "project_slug": "project-gamma",
            "bundle": "recipes",
            "created": "2024-01-18T10:00:00Z",
            "turn_count": 8,
        },
    ]


@pytest.fixture
def sample_transcripts():
    """Sample transcript data with some null turns for testing."""
    return [
        # Session 1 - all turns defined
        {"session_id": "session-1", "sequence": 0, "turn": 1, "role": "user", "content": "Hello"},
        {
            "session_id": "session-1",
            "sequence": 1,
            "turn": 1,
            "role": "assistant",
            "content": "Hi there!",
        },
        {
            "session_id": "session-1",
            "sequence": 2,
            "turn": 2,
            "role": "user",
            "content": "How are you?",
        },
        {
            "session_id": "session-1",
            "sequence": 3,
            "turn": 2,
            "role": "assistant",
            "content": "I'm doing well!",
        },
        {"session_id": "session-1", "sequence": 4, "turn": 3, "role": "user", "content": "Great!"},
        # Session 2 - some null turns (common in real data)
        {
            "session_id": "session-2",
            "sequence": 0,
            "turn": None,
            "role": "system",
            "content": "System init",
        },
        {"session_id": "session-2", "sequence": 1, "turn": 1, "role": "user", "content": "Start"},
        {
            "session_id": "session-2",
            "sequence": 2,
            "turn": 1,
            "role": "tool",
            "content": "Tool output",
        },
        {
            "session_id": "session-2",
            "sequence": 3,
            "turn": 1,
            "role": "assistant",
            "content": "Done",
        },
        {
            "session_id": "session-2",
            "sequence": 4,
            "turn": None,
            "role": "tool",
            "content": "Another tool",
        },
        {
            "session_id": "session-2",
            "sequence": 5,
            "turn": 2,
            "role": "user",
            "content": "Continue",
        },
    ]


# =============================================================================
# DuckDB Backend Tests
# =============================================================================


class TestDuckDBDiscoveryAPIs:
    """Test discovery APIs for DuckDB backend."""

    @pytest.fixture
    async def backend(self, sample_sessions, sample_transcripts):
        """Create DuckDB backend with sample data."""
        from amplifier_session_storage.backends.duckdb import DuckDBBackend, DuckDBConfig

        config = DuckDBConfig(db_path=":memory:")
        backend = await DuckDBBackend.create(config=config)

        # Insert sample sessions
        for session in sample_sessions:
            await backend.upsert_session_metadata(
                user_id=session["user_id"],
                host_id="test-host",
                metadata=session,
            )

        # Insert sample transcripts
        for user_id in ["user-alice", "user-bob"]:
            user_transcripts = [
                t
                for t in sample_transcripts
                if t["session_id"] in ["session-1", "session-2"]
                and user_id == "user-alice"
                or t["session_id"] == "session-3"
                and user_id == "user-bob"
            ]
            if user_transcripts:
                session_id = user_transcripts[0]["session_id"]
                project_slug = (
                    "project-alpha" if session_id in ["session-1", "session-3"] else "project-beta"
                )
                await backend.sync_transcript_lines(
                    user_id=user_id,
                    host_id="test-host",
                    project_slug=project_slug,
                    session_id=session_id,
                    lines=user_transcripts,
                )

        yield backend
        await backend.close()

    @pytest.mark.asyncio
    async def test_list_users_all(self, backend):
        """Test listing all users."""
        users = await backend.list_users()
        assert len(users) >= 3
        assert "user-alice" in users
        assert "user-bob" in users
        assert "user-charlie" in users
        # Should be sorted
        assert users == sorted(users)

    @pytest.mark.asyncio
    async def test_list_users_with_project_filter(self, backend):
        """Test listing users filtered by project."""
        users = await backend.list_users(filters=SearchFilters(project_slug="project-alpha"))
        assert "user-alice" in users
        assert "user-bob" in users
        assert "user-charlie" not in users

    @pytest.mark.asyncio
    async def test_list_users_with_bundle_filter(self, backend):
        """Test listing users filtered by bundle."""
        users = await backend.list_users(filters=SearchFilters(bundle="foundation"))
        assert "user-alice" in users
        assert "user-bob" in users
        assert "user-charlie" not in users

    @pytest.mark.asyncio
    async def test_list_projects_all(self, backend):
        """Test listing all projects."""
        projects = await backend.list_projects()
        assert len(projects) >= 3
        assert "project-alpha" in projects
        assert "project-beta" in projects
        assert "project-gamma" in projects
        # Should be sorted
        assert projects == sorted(projects)

    @pytest.mark.asyncio
    async def test_list_projects_for_user(self, backend):
        """Test listing projects for a specific user."""
        projects = await backend.list_projects(user_id="user-alice")
        assert "project-alpha" in projects
        assert "project-beta" in projects
        assert "project-gamma" not in projects

    @pytest.mark.asyncio
    async def test_list_projects_team_wide(self, backend):
        """Test listing projects without user filter (team-wide)."""
        projects = await backend.list_projects(user_id="")
        assert len(projects) >= 3

    @pytest.mark.asyncio
    async def test_list_sessions_all(self, backend):
        """Test listing all sessions."""
        sessions = await backend.list_sessions()
        assert len(sessions) >= 4
        # Should be ordered by created DESC (newest first)
        dates = [s["created"] for s in sessions]
        assert dates == sorted(dates, reverse=True)

    @pytest.mark.asyncio
    async def test_list_sessions_for_user(self, backend):
        """Test listing sessions for a specific user."""
        sessions = await backend.list_sessions(user_id="user-alice")
        assert len(sessions) == 2
        for s in sessions:
            assert s["user_id"] == "user-alice"

    @pytest.mark.asyncio
    async def test_list_sessions_for_project(self, backend):
        """Test listing sessions for a specific project."""
        sessions = await backend.list_sessions(project_slug="project-alpha")
        assert len(sessions) == 2
        for s in sessions:
            assert s["project_slug"] == "project-alpha"

    @pytest.mark.asyncio
    async def test_list_sessions_pagination(self, backend):
        """Test session listing pagination."""
        page1 = await backend.list_sessions(limit=2, offset=0)
        page2 = await backend.list_sessions(limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) >= 1

        # No overlap between pages
        page1_ids = {s["session_id"] for s in page1}
        page2_ids = {s["session_id"] for s in page2}
        assert page1_ids.isdisjoint(page2_ids)


class TestDuckDBMessageContext:
    """Test get_message_context for DuckDB backend."""

    @pytest.fixture
    async def backend_with_transcripts(self):
        """Create DuckDB backend with transcript data."""
        from amplifier_session_storage.backends.duckdb import DuckDBBackend, DuckDBConfig

        config = DuckDBConfig(db_path=":memory:")
        backend = await DuckDBBackend.create(config=config)

        # Insert session
        await backend.upsert_session_metadata(
            user_id="test-user",
            host_id="test-host",
            metadata={
                "session_id": "test-session",
                "user_id": "test-user",
                "project_slug": "test-project",
                "bundle": "test",
                "created": "2024-01-15T10:00:00Z",
            },
        )

        # Insert transcripts with varying turn values (some null)
        lines = [
            {"sequence": 0, "turn": None, "role": "system", "content": "Init"},
            {"sequence": 1, "turn": 1, "role": "user", "content": "Hello"},
            {"sequence": 2, "turn": 1, "role": "assistant", "content": "Hi!"},
            {"sequence": 3, "turn": 2, "role": "user", "content": "What time?"},
            {"sequence": 4, "turn": 2, "role": "tool", "content": "Tool result"},
            {"sequence": 5, "turn": 2, "role": "assistant", "content": "It's noon"},
            {"sequence": 6, "turn": 3, "role": "user", "content": "Thanks"},
            {"sequence": 7, "turn": 3, "role": "assistant", "content": "Welcome!"},
            {"sequence": 8, "turn": None, "role": "tool", "content": "Background"},
            {"sequence": 9, "turn": 4, "role": "user", "content": "Bye"},
        ]

        await backend.sync_transcript_lines(
            user_id="test-user",
            host_id="test-host",
            project_slug="test-project",
            session_id="test-session",
            lines=lines,
        )

        yield backend
        await backend.close()

    @pytest.mark.asyncio
    async def test_get_message_context_middle(self, backend_with_transcripts):
        """Test getting context around a middle message."""
        ctx = await backend_with_transcripts.get_message_context(
            session_id="test-session",
            sequence=5,
            before=2,
            after=2,
        )

        assert ctx.session_id == "test-session"
        assert ctx.target_sequence == 5
        assert ctx.current is not None
        # Content may be JSON-escaped depending on backend
        assert "noon" in ctx.current.content

        # Should have 2 messages before
        assert len(ctx.previous) == 2
        assert ctx.previous[0].sequence == 3
        assert ctx.previous[1].sequence == 4

        # Should have 2 messages after
        assert len(ctx.following) == 2
        assert ctx.following[0].sequence == 6
        assert ctx.following[1].sequence == 7

    @pytest.mark.asyncio
    async def test_get_message_context_start(self, backend_with_transcripts):
        """Test getting context at the start of session."""
        ctx = await backend_with_transcripts.get_message_context(
            session_id="test-session",
            sequence=0,
            before=3,
            after=2,
        )

        assert ctx.target_sequence == 0
        assert ctx.current is not None
        assert len(ctx.previous) == 0  # Nothing before sequence 0
        assert len(ctx.following) == 2
        assert ctx.has_more_before is False
        assert ctx.has_more_after is True

    @pytest.mark.asyncio
    async def test_get_message_context_end(self, backend_with_transcripts):
        """Test getting context at the end of session."""
        ctx = await backend_with_transcripts.get_message_context(
            session_id="test-session",
            sequence=9,
            before=2,
            after=3,
        )

        assert ctx.target_sequence == 9
        assert ctx.current is not None
        assert len(ctx.following) == 0  # Nothing after last message
        assert ctx.has_more_after is False
        assert ctx.has_more_before is True

    @pytest.mark.asyncio
    async def test_get_message_context_exclude_tools(self, backend_with_transcripts):
        """Test excluding tool outputs from context."""
        ctx = await backend_with_transcripts.get_message_context(
            session_id="test-session",
            sequence=5,
            before=3,
            after=3,
            include_tool_outputs=False,
        )

        # Tool messages should be excluded
        all_roles = [m.role for m in ctx.previous + ctx.following]
        if ctx.current:
            all_roles.append(ctx.current.role)
        assert "tool" not in all_roles

    @pytest.mark.asyncio
    async def test_get_message_context_null_turns(self, backend_with_transcripts):
        """Test context retrieval handles null turns gracefully."""
        # Sequence 0 has turn=None
        ctx = await backend_with_transcripts.get_message_context(
            session_id="test-session",
            sequence=0,
            before=0,
            after=2,
        )

        assert ctx.current is not None
        assert ctx.current.turn is None
        assert ctx.current.sequence == 0

    @pytest.mark.asyncio
    async def test_get_message_context_navigation_metadata(self, backend_with_transcripts):
        """Test navigation metadata is correct."""
        ctx = await backend_with_transcripts.get_message_context(
            session_id="test-session",
            sequence=5,
            before=2,
            after=2,
        )

        assert ctx.first_sequence == 0
        assert ctx.last_sequence == 9
        assert ctx.has_more_before is True  # sequence 0-2 exist
        assert ctx.has_more_after is True  # sequence 8-9 exist


# =============================================================================
# SQLite Backend Tests
# =============================================================================


class TestSQLiteDiscoveryAPIs:
    """Test discovery APIs for SQLite backend."""

    @pytest.fixture
    async def backend(self, sample_sessions):
        """Create SQLite backend with sample data."""
        from amplifier_session_storage.backends.sqlite import SQLiteBackend, SQLiteConfig

        config = SQLiteConfig(db_path=":memory:")
        backend = await SQLiteBackend.create(config=config)

        # Insert sample sessions
        for session in sample_sessions:
            await backend.upsert_session_metadata(
                user_id=session["user_id"],
                host_id="test-host",
                metadata=session,
            )

        yield backend
        await backend.close()

    @pytest.mark.asyncio
    async def test_list_users_all(self, backend):
        """Test listing all users."""
        users = await backend.list_users()
        assert len(users) >= 3
        assert "user-alice" in users
        assert "user-bob" in users
        assert "user-charlie" in users

    @pytest.mark.asyncio
    async def test_list_projects_all(self, backend):
        """Test listing all projects."""
        projects = await backend.list_projects()
        assert len(projects) >= 3

    @pytest.mark.asyncio
    async def test_list_sessions_pagination(self, backend):
        """Test session listing pagination."""
        page1 = await backend.list_sessions(limit=2)
        assert len(page1) == 2


# =============================================================================
# Cosmos Backend Tests (Integration - requires Azure resources)
# =============================================================================


@pytest.mark.skipif(
    True,  # Skip by default - enable when Azure resources available
    reason="Cosmos tests require Azure resources",
)
class TestCosmosDiscoveryAPIs:
    """Test discovery APIs for Cosmos backend."""

    @pytest.fixture
    async def backend(self):
        """Create Cosmos backend for testing."""
        from amplifier_session_storage.backends.cosmos import CosmosBackend, CosmosConfig

        config = CosmosConfig.from_env()
        backend = await CosmosBackend.create(config=config)
        yield backend
        await backend.close()

    @pytest.mark.asyncio
    async def test_list_users(self, backend):
        """Test listing users from Cosmos."""
        users = await backend.list_users()
        assert isinstance(users, list)

    @pytest.mark.asyncio
    async def test_list_projects(self, backend):
        """Test listing projects from Cosmos."""
        projects = await backend.list_projects()
        assert isinstance(projects, list)


# =============================================================================
# Cross-Backend Contract Tests
# =============================================================================


class TestDiscoveryAPIContract:
    """
    Tests that verify all backends implement the discovery API contract correctly.

    These tests ensure consistency across DuckDB, SQLite, and Cosmos backends.
    """

    @pytest.fixture(params=["duckdb", "sqlite"])
    async def backend(self, request, sample_sessions):
        """Parameterized backend fixture for contract testing."""
        if request.param == "duckdb":
            from amplifier_session_storage.backends.duckdb import DuckDBBackend, DuckDBConfig

            config = DuckDBConfig(db_path=":memory:")
            backend = await DuckDBBackend.create(config=config)
        else:
            from amplifier_session_storage.backends.sqlite import SQLiteBackend, SQLiteConfig

            config = SQLiteConfig(db_path=":memory:")
            backend = await SQLiteBackend.create(config=config)

        # Insert sample data
        for session in sample_sessions:
            await backend.upsert_session_metadata(
                user_id=session["user_id"],
                host_id="test-host",
                metadata=session,
            )

        yield backend
        await backend.close()

    @pytest.mark.asyncio
    async def test_list_users_returns_sorted_list(self, backend):
        """list_users should return a sorted list of strings."""
        users = await backend.list_users()
        assert isinstance(users, list)
        assert all(isinstance(u, str) for u in users)
        assert users == sorted(users)

    @pytest.mark.asyncio
    async def test_list_projects_returns_sorted_list(self, backend):
        """list_projects should return a sorted list of strings."""
        projects = await backend.list_projects()
        assert isinstance(projects, list)
        assert all(isinstance(p, str) for p in projects)
        assert projects == sorted(projects)

    @pytest.mark.asyncio
    async def test_list_sessions_returns_expected_fields(self, backend):
        """list_sessions should return dicts with expected fields."""
        sessions = await backend.list_sessions(limit=1)
        assert len(sessions) > 0

        session = sessions[0]
        required_fields = ["session_id", "user_id", "project_slug", "bundle", "created"]
        for field in required_fields:
            assert field in session, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_empty_user_id_means_team_wide(self, backend):
        """Empty user_id should return results from all users."""
        # Team-wide
        all_projects = await backend.list_projects(user_id="")

        # User-specific
        alice_projects = await backend.list_projects(user_id="user-alice")

        # Team-wide should have at least as many as user-specific
        assert len(all_projects) >= len(alice_projects)

    @pytest.mark.asyncio
    async def test_filters_reduce_results(self, backend):
        """Filters should reduce results appropriately."""
        all_users = await backend.list_users()
        filtered_users = await backend.list_users(
            filters=SearchFilters(project_slug="project-alpha")
        )

        # Filtered should have fewer or equal users
        assert len(filtered_users) <= len(all_users)
