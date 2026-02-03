"""
Hybrid file storage implementation.

Provides offline-first session storage that:
- Uses local file storage (CLI format) for all operations
- Background syncs to Cosmos DB for cloud backup
- Supports directory exclusion patterns for local-only storage
- Handles cloud auth failures gracefully (local-only fallback)
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..cosmos.file_storage import CosmosFileConfig, CosmosFileStorage
from ..local.file_ops import file_exists, read_json, read_jsonl

logger = logging.getLogger(__name__)

# Default base path for Amplifier sessions (CLI format)
DEFAULT_BASE_PATH = Path.home() / ".amplifier" / "projects"


@dataclass
class SyncResult:
    """Result of a sync operation."""

    success: bool
    sessions_synced: int = 0
    messages_synced: int = 0
    events_synced: int = 0
    errors: list[str] = field(default_factory=list)
    skipped_excluded: int = 0


@dataclass
class HybridFileStorageConfig:
    """Configuration for hybrid file storage.

    Attributes:
        base_path: Local base path for sessions
        cosmos_config: Optional Cosmos configuration for cloud sync
        exclusion_patterns: Glob patterns for directories to exclude from sync
        sync_on_write: Whether to sync immediately on writes
        user_id: User ID for cloud storage partitioning
    """

    base_path: Path = field(default_factory=lambda: DEFAULT_BASE_PATH)
    cosmos_config: CosmosFileConfig | None = None
    exclusion_patterns: list[str] = field(default_factory=list)
    sync_on_write: bool = False
    user_id: str = "local"

    def is_excluded(self, project_slug: str) -> bool:
        """Check if a project matches any exclusion pattern.

        Args:
            project_slug: Project slug to check

        Returns:
            True if project should be excluded from cloud sync
        """
        for pattern in self.exclusion_patterns:
            if fnmatch.fnmatch(project_slug, pattern):
                return True
        return False


class HybridFileStorage:
    """Hybrid storage combining local files with cloud sync.

    This storage is designed specifically for CLI compatibility:
    - All operations use local file storage first (offline-first)
    - Background sync uploads to Cosmos DB
    - Exclusion patterns allow local-only storage for certain directories
    - Cloud auth failures gracefully degrade to local-only mode

    Directory structure (unchanged from CLI):
        {base_path}/{project_slug}/sessions/{session_id}/
            metadata.json    - Session metadata
            transcript.jsonl - Conversation transcript
            events.jsonl     - Event log
    """

    def __init__(self, config: HybridFileStorageConfig):
        """Initialize hybrid storage.

        Args:
            config: Hybrid storage configuration
        """
        self.config = config
        self._cosmos: CosmosFileStorage | None = None
        self._cosmos_available = False
        self._sync_state: dict[str, dict[str, int]] = {}  # session_id -> {transcript: N, events: N}

    async def initialize(self) -> None:
        """Initialize storage, including optional cloud connection."""
        if self.config.cosmos_config:
            try:
                self._cosmos = CosmosFileStorage(self.config.cosmos_config)
                await self._cosmos.initialize()
                self._cosmos_available = True
                logger.info("Hybrid storage initialized with cloud sync enabled")
            except Exception as e:
                logger.warning(f"Cloud sync unavailable, running in local-only mode: {e}")
                self._cosmos_available = False
        else:
            logger.info("Hybrid storage initialized in local-only mode (no cosmos config)")

    async def close(self) -> None:
        """Close storage connections."""
        if self._cosmos:
            await self._cosmos.close()
            self._cosmos = None
            self._cosmos_available = False

    async def __aenter__(self) -> HybridFileStorage:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    @property
    def cloud_available(self) -> bool:
        """Check if cloud sync is available."""
        return self._cosmos_available and self._cosmos is not None

    # =========================================================================
    # Local Path Helpers (CLI-compatible)
    # =========================================================================

    def _session_dir(self, project_slug: str, session_id: str) -> Path:
        """Get local session directory path."""
        return self.config.base_path / project_slug / "sessions" / session_id

    def _metadata_path(self, project_slug: str, session_id: str) -> Path:
        """Get local metadata.json path."""
        return self._session_dir(project_slug, session_id) / "metadata.json"

    def _transcript_path(self, project_slug: str, session_id: str) -> Path:
        """Get local transcript.jsonl path."""
        return self._session_dir(project_slug, session_id) / "transcript.jsonl"

    def _events_path(self, project_slug: str, session_id: str) -> Path:
        """Get local events.jsonl path."""
        return self._session_dir(project_slug, session_id) / "events.jsonl"

    # =========================================================================
    # Sync Operations
    # =========================================================================

    async def sync_session(
        self,
        project_slug: str,
        session_id: str,
        force: bool = False,
    ) -> SyncResult:
        """Sync a single session to cloud.

        Args:
            project_slug: Project slug
            session_id: Session ID
            force: Force full resync (ignore sync state)

        Returns:
            SyncResult with details
        """
        result = SyncResult(success=True)

        # Check exclusion patterns
        if self.config.is_excluded(project_slug):
            result.skipped_excluded = 1
            logger.debug(f"Session {session_id} excluded from sync (project: {project_slug})")
            return result

        # Check cloud availability
        if not self.cloud_available or self._cosmos is None:
            result.success = False
            result.errors.append("Cloud storage not available")
            return result

        try:
            # Sync metadata
            metadata_path = self._metadata_path(project_slug, session_id)
            metadata = await read_json(metadata_path)
            if metadata:
                await self._cosmos.upsert_session_metadata(
                    user_id=self.config.user_id,
                    metadata=metadata,
                )
                result.sessions_synced = 1

            # Get sync state for this session
            sync_key = f"{project_slug}/{session_id}"
            state = self._sync_state.get(sync_key, {"transcript": 0, "events": 0})
            if force:
                state = {"transcript": 0, "events": 0}

            # Sync transcript (incremental)
            transcript_path = self._transcript_path(project_slug, session_id)
            if await file_exists(transcript_path):
                all_messages = await read_jsonl(transcript_path)
                new_messages = all_messages[state["transcript"] :]
                if new_messages:
                    synced = await self._cosmos.sync_transcript_lines(
                        user_id=self.config.user_id,
                        project_slug=project_slug,
                        session_id=session_id,
                        lines=new_messages,
                        start_sequence=state["transcript"],
                    )
                    result.messages_synced = synced
                    state["transcript"] = len(all_messages)

            # Sync events (incremental)
            events_path = self._events_path(project_slug, session_id)
            if await file_exists(events_path):
                all_events = await read_jsonl(events_path)
                new_events = all_events[state["events"] :]
                if new_events:
                    synced = await self._cosmos.sync_event_lines(
                        user_id=self.config.user_id,
                        project_slug=project_slug,
                        session_id=session_id,
                        lines=new_events,
                        start_sequence=state["events"],
                    )
                    result.events_synced = synced
                    state["events"] = len(all_events)

            # Update sync state
            self._sync_state[sync_key] = state

        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            logger.error(f"Error syncing session {session_id}: {e}")

        return result

    async def sync_project(
        self,
        project_slug: str,
        force: bool = False,
    ) -> SyncResult:
        """Sync all sessions in a project to cloud.

        Args:
            project_slug: Project slug
            force: Force full resync

        Returns:
            Combined SyncResult
        """
        result = SyncResult(success=True)

        # Check exclusion
        if self.config.is_excluded(project_slug):
            result.skipped_excluded = 1
            return result

        # List sessions in project
        sessions_dir = self.config.base_path / project_slug / "sessions"
        if not sessions_dir.exists():
            return result

        for session_dir in sessions_dir.iterdir():
            if session_dir.is_dir():
                session_result = await self.sync_session(
                    project_slug=project_slug,
                    session_id=session_dir.name,
                    force=force,
                )
                result.sessions_synced += session_result.sessions_synced
                result.messages_synced += session_result.messages_synced
                result.events_synced += session_result.events_synced
                result.errors.extend(session_result.errors)
                if not session_result.success:
                    result.success = False

        return result

    async def sync_all(self, force: bool = False) -> SyncResult:
        """Sync all projects to cloud.

        Args:
            force: Force full resync

        Returns:
            Combined SyncResult
        """
        result = SyncResult(success=True)

        if not self.config.base_path.exists():
            return result

        for project_dir in self.config.base_path.iterdir():
            if project_dir.is_dir():
                project_result = await self.sync_project(
                    project_slug=project_dir.name,
                    force=force,
                )
                result.sessions_synced += project_result.sessions_synced
                result.messages_synced += project_result.messages_synced
                result.events_synced += project_result.events_synced
                result.skipped_excluded += project_result.skipped_excluded
                result.errors.extend(project_result.errors)
                if not project_result.success:
                    result.success = False

        return result

    # =========================================================================
    # Query Operations (Cloud)
    # =========================================================================

    async def list_cloud_sessions(
        self,
        project_slug: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List sessions from cloud storage.

        Args:
            project_slug: Optional project filter
            limit: Maximum sessions to return

        Returns:
            List of session metadata dicts
        """
        if not self.cloud_available or self._cosmos is None:
            return []

        return await self._cosmos.list_sessions(
            user_id=self.config.user_id,
            project_slug=project_slug,
            limit=limit,
        )

    async def get_cloud_session(
        self,
        session_id: str,
    ) -> dict[str, Any] | None:
        """Get session metadata from cloud.

        Args:
            session_id: Session ID

        Returns:
            Session metadata dict or None
        """
        if not self.cloud_available or self._cosmos is None:
            return None

        return await self._cosmos.get_session_metadata(
            user_id=self.config.user_id,
            session_id=session_id,
        )

    # =========================================================================
    # Sync State Management
    # =========================================================================

    def get_sync_state(self, project_slug: str, session_id: str) -> dict[str, int]:
        """Get sync state for a session.

        Returns:
            Dict with 'transcript' and 'events' counts
        """
        sync_key = f"{project_slug}/{session_id}"
        return self._sync_state.get(sync_key, {"transcript": 0, "events": 0})

    def reset_sync_state(
        self, project_slug: str | None = None, session_id: str | None = None
    ) -> None:
        """Reset sync state to force full resync.

        Args:
            project_slug: Optional project to reset (None = all)
            session_id: Optional session to reset (requires project_slug)
        """
        if project_slug is None:
            self._sync_state.clear()
        elif session_id is None:
            # Reset all sessions in project
            prefix = f"{project_slug}/"
            to_remove = [k for k in self._sync_state if k.startswith(prefix)]
            for k in to_remove:
                del self._sync_state[k]
        else:
            # Reset specific session
            sync_key = f"{project_slug}/{session_id}"
            if sync_key in self._sync_state:
                del self._sync_state[sync_key]

    # =========================================================================
    # Background Sync Task
    # =========================================================================

    async def start_background_sync(
        self,
        interval_seconds: float = 60.0,
    ) -> asyncio.Task[None]:
        """Start a background task that syncs periodically.

        Args:
            interval_seconds: Sync interval

        Returns:
            The background task
        """

        async def sync_loop() -> None:
            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    if self.cloud_available:
                        result = await self.sync_all()
                        if result.sessions_synced > 0:
                            logger.debug(
                                f"Background sync: {result.sessions_synced} sessions, "
                                f"{result.messages_synced} messages, "
                                f"{result.events_synced} events"
                            )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Background sync error: {e}")

        return asyncio.create_task(sync_loop())

    # =========================================================================
    # Exclusion Pattern Management
    # =========================================================================

    def add_exclusion_pattern(self, pattern: str) -> None:
        """Add a directory exclusion pattern.

        Args:
            pattern: Glob pattern (e.g., "*-temp", "test-*")
        """
        if pattern not in self.config.exclusion_patterns:
            self.config.exclusion_patterns.append(pattern)

    def remove_exclusion_pattern(self, pattern: str) -> None:
        """Remove a directory exclusion pattern."""
        if pattern in self.config.exclusion_patterns:
            self.config.exclusion_patterns.remove(pattern)

    def list_exclusion_patterns(self) -> list[str]:
        """Get current exclusion patterns."""
        return list(self.config.exclusion_patterns)
