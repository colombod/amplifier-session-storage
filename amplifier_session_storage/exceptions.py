"""
Custom exceptions for session storage.

All storage implementations should raise these exceptions
for consistent error handling across backends.
"""


class SessionStorageError(Exception):
    """Base exception for all session storage errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class SessionNotFoundError(SessionStorageError):
    """Raised when a session is not found."""

    def __init__(self, message_or_session_id: str, user_id: str | None = None):
        # Support both simple message and structured session_id
        if user_id is not None or not message_or_session_id.startswith("Session"):
            # Structured form: session_id passed directly
            session_id = message_or_session_id
            details = {"session_id": session_id}
            if user_id:
                details["user_id"] = user_id
            super().__init__(f"Session not found: {session_id}", details)
            self.session_id = session_id
        else:
            # Simple message form: message passed directly
            super().__init__(message_or_session_id, {})
            # Try to extract session_id from message
            self.session_id = message_or_session_id.replace("Session '", "").replace(
                "' not found", ""
            )
        self.user_id = user_id


class SessionValidationError(SessionStorageError):
    """Raised when session validation fails (e.g., invalid session_id)."""

    def __init__(self, message: str, field: str | None = None):
        details = {}
        if field:
            details["field"] = field
        super().__init__(message, details)
        self.field = field


class SessionExistsError(SessionStorageError):
    """Raised when trying to create a session that already exists."""

    def __init__(self, session_id: str):
        super().__init__(f"Session already exists: {session_id}", {"session_id": session_id})
        self.session_id = session_id


class EventNotFoundError(SessionStorageError):
    """Raised when an event is not found."""

    def __init__(self, event_id: str, session_id: str | None = None):
        details = {"event_id": event_id}
        if session_id:
            details["session_id"] = session_id
        super().__init__(f"Event not found: {event_id}", details)
        self.event_id = event_id
        self.session_id = session_id


class StorageIOError(SessionStorageError):
    """Raised when a storage I/O operation fails."""

    def __init__(self, operation: str, path: str | None = None, cause: Exception | None = None):
        details = {"operation": operation}
        if path:
            details["path"] = path
        if cause:
            details["cause"] = str(cause)
        message = f"Storage I/O error during {operation}"
        if path:
            message += f": {path}"
        super().__init__(message, details)
        self.operation = operation
        self.path = path
        self.cause = cause


class ChunkingError(SessionStorageError):
    """Raised when event chunking or reassembly fails."""

    def __init__(self, event_id: str, reason: str):
        super().__init__(
            f"Chunking error for event {event_id}: {reason}",
            {"event_id": event_id, "reason": reason},
        )
        self.event_id = event_id
        self.reason = reason


class SyncError(SessionStorageError):
    """Raised when synchronization fails."""

    def __init__(self, message: str, session_id: str | None = None, cause: Exception | None = None):
        details: dict = {}
        if session_id:
            details["session_id"] = session_id
        if cause:
            details["cause"] = str(cause)
        super().__init__(message, details)
        self.session_id = session_id
        self.cause = cause


class ConflictError(SessionStorageError):
    """Raised when a sync conflict cannot be automatically resolved."""

    def __init__(
        self,
        session_id: str,
        conflict_type: str,
        local_version: str | None = None,
        remote_version: str | None = None,
    ):
        details = {
            "session_id": session_id,
            "conflict_type": conflict_type,
        }
        if local_version:
            details["local_version"] = local_version
        if remote_version:
            details["remote_version"] = remote_version
        super().__init__(
            f"Sync conflict in session {session_id}: {conflict_type}",
            details,
        )
        self.session_id = session_id
        self.conflict_type = conflict_type
        self.local_version = local_version
        self.remote_version = remote_version


class StorageConnectionError(SessionStorageError):
    """Raised when connection to remote storage fails.

    Note: Named StorageConnectionError to avoid shadowing the builtin ConnectionError.
    """

    def __init__(self, endpoint: str, cause: Exception | None = None):
        details = {"endpoint": endpoint}
        if cause:
            details["cause"] = str(cause)
        super().__init__(f"Connection failed to {endpoint}", details)
        self.endpoint = endpoint
        self.cause = cause


class AuthenticationError(SessionStorageError):
    """Raised when authentication to remote storage fails."""

    def __init__(self, endpoint: str, reason: str | None = None):
        details = {"endpoint": endpoint}
        if reason:
            details["reason"] = reason
        super().__init__(f"Authentication failed for {endpoint}", details)
        self.endpoint = endpoint
        self.reason = reason


class ValidationError(SessionStorageError):
    """Raised when data validation fails."""

    def __init__(self, field: str, reason: str, value: str | None = None):
        details = {"field": field, "reason": reason}
        if value is not None:
            details["value"] = value
        super().__init__(f"Validation failed for {field}: {reason}", details)
        self.field = field
        self.reason = reason
        self.value = value


class RewindError(SessionStorageError):
    """Raised when a rewind operation fails."""

    def __init__(self, session_id: str, reason: str, partial: bool = False):
        details = {"session_id": session_id, "reason": reason, "partial": partial}
        super().__init__(f"Rewind failed for session {session_id}: {reason}", details)
        self.session_id = session_id
        self.reason = reason
        self.partial = partial


class EventTooLargeError(SessionStorageError):
    """Raised when an event exceeds maximum size limits."""

    def __init__(self, event_id: str, size_bytes: int, max_bytes: int):
        details = {
            "event_id": event_id,
            "size_bytes": size_bytes,
            "max_bytes": max_bytes,
        }
        super().__init__(
            f"Event {event_id} exceeds maximum size: {size_bytes} > {max_bytes} bytes",
            details,
        )
        self.event_id = event_id
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes


class PermissionDeniedError(SessionStorageError):
    """User does not have permission for the requested operation."""

    def __init__(self, user_id: str, session_id: str, reason: str):
        details = {"user_id": user_id, "session_id": session_id, "reason": reason}
        super().__init__(
            f"Permission denied for user {user_id} on session {session_id}: {reason}",
            details,
        )
        self.user_id = user_id
        self.session_id = session_id
        self.reason = reason


# Alias for backward compatibility
SyncConflictError = ConflictError
