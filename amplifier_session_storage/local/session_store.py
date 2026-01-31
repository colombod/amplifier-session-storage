"""
Session persistence management compatible with amplifier-app-cli.

This module provides a SessionStore class that is a drop-in replacement for
the SessionStore in amplifier-app-cli. It maintains exact compatibility with:
- File paths: ~/.amplifier/projects/{project}/sessions/{session_id}/
- File formats: metadata.json, transcript.jsonl, events.jsonl
- session-analyst agent expectations

The session-analyst agent depends on these exact paths and formats to:
- Search sessions by metadata
- Read transcript content
- Safely extract from events.jsonl (surgical patterns only)
- Repair/rewind sessions by truncating files
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..exceptions import SessionNotFoundError, SessionValidationError

logger = logging.getLogger(__name__)

# Prefix used to identify bundle-based sessions in metadata
BUNDLE_PREFIX = "bundle:"


def is_top_level_session(session_id: str) -> bool:
    """Check if a session ID is a top-level (main) session.

    Spawned sub-sessions have IDs in the format: {parent_id}_{agent_name}
    Top-level sessions are just UUIDs without underscores.

    Args:
        session_id: Session ID to check

    Returns:
        True if this is a top-level session, False if spawned
    """
    return "_" not in session_id


def extract_session_mode(metadata: dict[str, Any]) -> tuple[str | None, None]:
    """Extract bundle name from session metadata.

    Sessions are created with a bundle (e.g., "foundation").
    This function extracts the bundle name for session resumption.

    Args:
        metadata: Session metadata dict containing "bundle" key

    Returns:
        (bundle_name, None) tuple. Returns (None, None) if no bundle found.
    """
    bundle_value = metadata.get("bundle")
    if bundle_value and bundle_value != "unknown":
        if bundle_value.startswith(BUNDLE_PREFIX):
            return (bundle_value[len(BUNDLE_PREFIX) :], None)
        return (bundle_value, None)

    return (None, None)


def _sanitize_message(message: dict[str, Any] | Any) -> dict[str, Any]:
    """Sanitize a message for JSON serialization.

    This is a simplified version - the full implementation would use
    amplifier_foundation.sanitize_message when available.

    Args:
        message: Message dict or object with model_dump()

    Returns:
        Sanitized message dict safe for JSON serialization
    """
    if hasattr(message, "model_dump"):
        msg_dict = message.model_dump()
    elif isinstance(message, dict):
        msg_dict = message.copy()
    else:
        msg_dict = {"content": str(message)}

    # Ensure all values are JSON-serializable
    def _sanitize_value(v: Any) -> Any:
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, dict):
            return {k: _sanitize_value(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_sanitize_value(item) for item in v]
        # Convert other types to string
        return str(v)

    return {k: _sanitize_value(v) for k, v in msg_dict.items()}


def _write_with_backup(path: Path, content: str) -> None:
    """Write content to file with atomic write and backup.

    This is a simplified version - the full implementation would use
    amplifier_foundation.write_with_backup when available.

    Args:
        path: Path to write to
        content: Content to write
    """
    # Create backup of existing file
    if path.exists():
        backup_path = path.with_suffix(path.suffix + ".backup")
        shutil.copy2(path, backup_path)

    # Write to temp file then rename (atomic on POSIX)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(path)
    except Exception:
        # Clean up temp file on failure
        if temp_path.exists():
            temp_path.unlink()
        raise


class SessionStore:
    """
    Manages session persistence to filesystem.

    Compatible with amplifier-app-cli SessionStore and session-analyst agent.

    Contract:
    - Inputs: session_id (str), transcript (list), metadata (dict)
    - Outputs: Saved files or loaded data tuples
    - Side Effects: Filesystem writes to base_dir/{session_id}/
    - Files created: transcript.jsonl, metadata.json
    """

    def __init__(
        self,
        base_dir: Path | None = None,
        *,
        project_slug: str | None = None,
    ):
        """Initialize with base directory for sessions.

        Args:
            base_dir: Base directory for session storage.
                     If None, uses ~/.amplifier/projects/{project_slug}/sessions/
            project_slug: Project identifier. Used if base_dir is None.
                         Defaults to "default" if not provided.
        """
        self.project_slug = project_slug or "default"

        if base_dir is None:
            base_dir = Path.home() / ".amplifier" / "projects" / self.project_slug / "sessions"

        self.base_dir = Path(base_dir).expanduser()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Ensure metadata has all required fields with defaults.

        This ensures parity between local storage and Cosmos DB schema.
        Missing fields are set to sensible defaults.

        Args:
            metadata: Input metadata dictionary

        Returns:
            Normalized metadata with all required fields
        """
        now = datetime.now(UTC).isoformat(timespec="milliseconds")

        # Required fields with defaults
        normalized = {
            "session_id": metadata.get("session_id", ""),
            "project_slug": metadata.get("project_slug", self.project_slug),
            "created": metadata.get("created", now),
            "updated": metadata.get("updated", now),
            "turn_count": metadata.get("turn_count", 0),
            "message_count": metadata.get("message_count", 0),
            "event_count": metadata.get("event_count", 0),
        }

        # Optional fields (only include if present or explicitly set)
        optional_fields = [
            "name",
            "description",
            "bundle",
            "model",
            "parent_id",
            "forked_from_turn",
            "tags",
        ]
        for field in optional_fields:
            if field in metadata:
                normalized[field] = metadata[field]

        # Preserve any additional fields from the input
        for key, value in metadata.items():
            if key not in normalized:
                normalized[key] = value

        return normalized

    def _validate_session_id(self, session_id: str) -> None:
        """Validate session ID to prevent path traversal attacks.

        Args:
            session_id: Session ID to validate

        Raises:
            SessionValidationError: If session_id is invalid
        """
        if not session_id or not session_id.strip():
            raise SessionValidationError("session_id cannot be empty")

        if "/" in session_id or "\\" in session_id or session_id in (".", ".."):
            raise SessionValidationError(f"Invalid session_id: {session_id}")

    def save(
        self, session_id: str, transcript: list[dict[str, Any]], metadata: dict[str, Any]
    ) -> None:
        """Save session state atomically with backup.

        Args:
            session_id: Unique session identifier
            transcript: List of message objects for the session
            metadata: Session metadata dictionary

        Raises:
            SessionValidationError: If session_id is invalid
            IOError: If unable to write files
        """
        self._validate_session_id(session_id)

        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Ensure session_id is in metadata
        metadata["session_id"] = session_id

        # Normalize metadata to ensure all required fields are present
        normalized_metadata = self._normalize_metadata(metadata)

        # Save transcript with atomic write
        self._save_transcript(session_dir, transcript)

        # Save metadata with atomic write
        self._save_metadata(session_dir, normalized_metadata)

        logger.debug(f"Session {session_id} saved successfully")

    def _save_transcript(self, session_dir: Path, transcript: list[dict[str, Any]]) -> None:
        """Save transcript with atomic write and backup.

        Args:
            session_dir: Directory for this session
            transcript: List of message objects
        """
        transcript_file = session_dir / "transcript.jsonl"

        # Build JSONL content
        lines = []
        for message in transcript:
            # Skip system and developer role messages from transcript
            msg_dict = message if isinstance(message, dict) else message.model_dump()
            if msg_dict.get("role") in ("system", "developer"):
                continue

            # Sanitize message to ensure it's JSON-serializable
            sanitized_msg = _sanitize_message(message)

            # Add timestamp if not present
            if "timestamp" not in sanitized_msg:
                sanitized_msg["timestamp"] = datetime.now(UTC).isoformat(timespec="milliseconds")

            lines.append(json.dumps(sanitized_msg, ensure_ascii=False))

        content = "\n".join(lines) + "\n" if lines else ""
        _write_with_backup(transcript_file, content)

    def _save_metadata(self, session_dir: Path, metadata: dict[str, Any]) -> None:
        """Save metadata with atomic write and backup.

        Args:
            session_dir: Directory for this session
            metadata: Metadata dictionary
        """
        metadata_file = session_dir / "metadata.json"
        content = json.dumps(metadata, indent=2, ensure_ascii=False)
        _write_with_backup(metadata_file, content)

    def load(self, session_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Load session state with corruption recovery.

        Args:
            session_id: Session identifier to load

        Returns:
            Tuple of (transcript, metadata)

        Raises:
            SessionNotFoundError: If session does not exist
            SessionValidationError: If session_id is invalid
        """
        self._validate_session_id(session_id)

        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            raise SessionNotFoundError(f"Session '{session_id}' not found")

        # Load transcript with recovery
        transcript = self._load_transcript(session_dir)

        # Load metadata with recovery
        metadata = self._load_metadata(session_dir)

        logger.debug(f"Session {session_id} loaded successfully")
        return transcript, metadata

    def _load_transcript(self, session_dir: Path) -> list[dict[str, Any]]:
        """Load transcript with corruption recovery.

        Args:
            session_dir: Directory for this session

        Returns:
            List of message objects (empty list if no transcript exists)
        """
        transcript_file = session_dir / "transcript.jsonl"
        backup_file = session_dir / "transcript.jsonl.backup"

        # If neither file exists, return empty list
        if not transcript_file.exists() and not backup_file.exists():
            return []

        # Try main file first
        if transcript_file.exists():
            try:
                transcript = []
                with open(transcript_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            transcript.append(json.loads(line))
                return transcript
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load transcript, trying backup: {e}")

        # Try backup if main file failed or missing
        if backup_file.exists():
            try:
                transcript = []
                with open(backup_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            transcript.append(json.loads(line))
                logger.info("Loaded transcript from backup")
                return transcript
            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"Backup also corrupted: {e}")

        logger.warning("Both transcript files corrupted, returning empty transcript")
        return []

    def _load_metadata(self, session_dir: Path) -> dict[str, Any]:
        """Load metadata with corruption recovery.

        Args:
            session_dir: Directory for this session

        Returns:
            Metadata dictionary (empty dict if no metadata exists)
        """
        metadata_file = session_dir / "metadata.json"
        backup_file = session_dir / "metadata.json.backup"

        # If neither file exists, return empty dict
        if not metadata_file.exists() and not backup_file.exists():
            return {}

        # Try main file first
        if metadata_file.exists():
            try:
                with open(metadata_file, encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load metadata, trying backup: {e}")

        # Try backup if main file failed
        if backup_file.exists():
            try:
                with open(backup_file, encoding="utf-8") as f:
                    metadata = json.load(f)
                logger.info("Loaded metadata from backup")
                return metadata
            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"Backup also corrupted: {e}")

        logger.warning("Both metadata files corrupted, returning minimal metadata")
        return {
            "session_id": session_dir.name,
            "recovered": True,
            "recovery_time": datetime.now(UTC).isoformat(),
        }

    def exists(self, session_id: str) -> bool:
        """Check if session exists.

        Args:
            session_id: Session identifier to check

        Returns:
            True if session exists, False otherwise
        """
        if not session_id or not session_id.strip():
            return False

        if "/" in session_id or "\\" in session_id or session_id in (".", ".."):
            return False

        session_dir = self.base_dir / session_id
        return session_dir.exists() and session_dir.is_dir()

    def find_session(self, partial_id: str, *, top_level_only: bool = True) -> str:
        """Find session by partial ID prefix.

        Args:
            partial_id: Partial session ID (prefix match)
            top_level_only: If True, only match top-level sessions

        Returns:
            Full session ID if exactly one match

        Raises:
            SessionNotFoundError: If no sessions match
            SessionValidationError: If multiple sessions match (ambiguous)
        """
        if not partial_id or not partial_id.strip():
            raise SessionValidationError("Session ID cannot be empty")

        partial_id = partial_id.strip()

        # Check for exact match first
        if self.exists(partial_id):
            if not top_level_only or is_top_level_session(partial_id):
                return partial_id

        # Find prefix matches
        matches = [
            sid
            for sid in self.list_sessions(top_level_only=top_level_only)
            if sid.startswith(partial_id)
        ]

        if not matches:
            raise SessionNotFoundError(f"No session found matching '{partial_id}'")
        if len(matches) > 1:
            raise SessionValidationError(
                f"Ambiguous session ID '{partial_id}' matches {len(matches)} sessions: "
                f"{', '.join(m[:12] + '...' for m in matches[:3])}"
                + (f" and {len(matches) - 3} more" if len(matches) > 3 else "")
            )
        return matches[0]

    def list_sessions(self, *, top_level_only: bool = True) -> list[str]:
        """List session IDs sorted by modification time (newest first).

        Args:
            top_level_only: If True, return only top-level sessions

        Returns:
            List of session identifiers
        """
        if not self.base_dir.exists():
            return []

        sessions = []
        for session_dir in self.base_dir.iterdir():
            if session_dir.is_dir() and not session_dir.name.startswith("."):
                session_name = session_dir.name

                if top_level_only and not is_top_level_session(session_name):
                    continue

                try:
                    mtime = session_dir.stat().st_mtime
                    sessions.append((session_name, mtime))
                except Exception:
                    sessions.append((session_name, 0))

        # Sort by modification time (newest first)
        sessions.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in sessions]

    def get_metadata(self, session_id: str) -> dict[str, Any]:
        """Get session metadata without loading transcript.

        Args:
            session_id: Session identifier

        Returns:
            Metadata dictionary

        Raises:
            SessionNotFoundError: If session does not exist
        """
        self._validate_session_id(session_id)

        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            raise SessionNotFoundError(f"Session '{session_id}' not found")

        return self._load_metadata(session_dir)

    def update_metadata(self, session_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        """Update specific fields in session metadata.

        Args:
            session_id: Session identifier
            updates: Dictionary of fields to update

        Returns:
            Updated metadata dictionary

        Raises:
            SessionNotFoundError: If session does not exist
        """
        self._validate_session_id(session_id)

        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            raise SessionNotFoundError(f"Session '{session_id}' not found")

        metadata = self._load_metadata(session_dir)
        metadata.update(updates)
        self._save_metadata(session_dir, metadata)

        logger.debug(f"Session {session_id} metadata updated: {list(updates.keys())}")
        return metadata

    def save_config_snapshot(self, session_id: str, config: dict[str, Any]) -> None:
        """Save config snapshot used for session.

        Args:
            session_id: Session identifier
            config: Bundle configuration dictionary
        """
        self._validate_session_id(session_id)

        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        config_file = session_dir / "config.md"

        # Convert config dict to Markdown+YAML frontmatter
        import yaml

        yaml_content = yaml.dump(config, default_flow_style=False, sort_keys=False)
        content = f"---\n{yaml_content}---\n\nConfig snapshot for session {session_id}\n"
        _write_with_backup(config_file, content)

        logger.debug(f"Config saved for session {session_id}")

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its files.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if it didn't exist
        """
        self._validate_session_id(session_id)

        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            return False

        shutil.rmtree(session_dir)
        logger.info(f"Deleted session: {session_id}")
        return True

    def cleanup_old_sessions(self, days: int = 30) -> int:
        """Remove sessions older than specified days.

        Args:
            days: Number of days to keep sessions (default 30)

        Returns:
            Number of sessions removed
        """
        if days < 0:
            raise ValueError("days must be non-negative")

        if not self.base_dir.exists():
            return 0

        from datetime import timedelta

        cutoff_time = datetime.now(UTC) - timedelta(days=days)
        cutoff_timestamp = cutoff_time.timestamp()

        removed = 0
        for session_dir in self.base_dir.iterdir():
            if not session_dir.is_dir() or session_dir.name.startswith("."):
                continue

            try:
                mtime = session_dir.stat().st_mtime
                if mtime < cutoff_timestamp:
                    shutil.rmtree(session_dir)
                    logger.info(f"Removed old session: {session_dir.name}")
                    removed += 1
            except Exception as e:
                logger.error(f"Failed to remove session {session_dir.name}: {e}")

        if removed > 0:
            logger.info(f"Cleaned up {removed} old sessions")

        return removed

    def get_session_dir(self, session_id: str) -> Path:
        """Get the directory path for a session.

        Args:
            session_id: Session identifier

        Returns:
            Path to the session directory
        """
        self._validate_session_id(session_id)
        return self.base_dir / session_id
