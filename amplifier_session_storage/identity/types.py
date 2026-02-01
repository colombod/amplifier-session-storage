"""
Identity types and data classes.

Defines the core types for user identity, device information,
and authentication state.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class AuthProvider(Enum):
    """Supported authentication providers."""

    CONFIG = "config"  # Local config file (dev/offline)
    ENTRA = "entra"  # Azure Entra ID (enterprise)
    OAUTH = "oauth"  # Generic OAuth (GitHub, Google)


@dataclass
class DeviceInfo:
    """Information about the current device.

    Each device has a unique ID that persists across sessions.
    This enables multi-machine sync and conflict resolution.
    """

    device_id: str
    device_name: str
    os_type: str  # windows, macos, linux
    first_seen: datetime
    last_seen: datetime

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "os_type": self.os_type,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeviceInfo":
        """Deserialize from dictionary."""
        return cls(
            device_id=data["device_id"],
            device_name=data["device_name"],
            os_type=data["os_type"],
            first_seen=datetime.fromisoformat(data["first_seen"]),
            last_seen=datetime.fromisoformat(data["last_seen"]),
        )


@dataclass
class UserIdentity:
    """Complete identity of the current user.

    This is the single source of truth for "who am I?" in the system.
    All storage operations use this identity for access control.
    """

    # Core identity
    user_id: str
    display_name: str
    email: str | None = None

    # Organization context
    org_id: str | None = None
    team_ids: list[str] = field(default_factory=list)
    role: str = "member"  # "owner", "admin", "member"

    # Device context
    device: DeviceInfo | None = None

    # Auth context (for cloud operations)
    auth_provider: AuthProvider = AuthProvider.CONFIG
    auth_token: str | None = None
    token_expiry: datetime | None = None

    def is_authenticated(self) -> bool:
        """Check if we have valid authentication.

        Config provider is always "authenticated" (local-only).
        Token-based providers check expiry.
        """
        if self.auth_token is None:
            return self.auth_provider == AuthProvider.CONFIG
        if self.token_expiry is None:
            return True
        return datetime.now(UTC) < self.token_expiry

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "email": self.email,
            "org_id": self.org_id,
            "team_ids": self.team_ids,
            "role": self.role,
            "device": self.device.to_dict() if self.device else None,
            "auth_provider": self.auth_provider.value,
            "token_expiry": self.token_expiry.isoformat() if self.token_expiry else None,
            # Note: auth_token intentionally excluded for security
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserIdentity":
        """Deserialize from dictionary."""
        device = None
        if data.get("device"):
            device = DeviceInfo.from_dict(data["device"])

        token_expiry = None
        if data.get("token_expiry"):
            token_expiry = datetime.fromisoformat(data["token_expiry"])

        return cls(
            user_id=data["user_id"],
            display_name=data["display_name"],
            email=data.get("email"),
            org_id=data.get("org_id"),
            team_ids=data.get("team_ids", []),
            role=data.get("role", "member"),
            device=device,
            auth_provider=AuthProvider(data.get("auth_provider", "config")),
            token_expiry=token_expiry,
        )


# Exceptions


class AuthenticationError(Exception):
    """Base class for authentication errors."""

    pass


class AuthenticationRequiredError(AuthenticationError):
    """Raised when authentication is required but not present."""

    def __init__(self, message: str = "Authentication required"):
        super().__init__(message)
        self.message = message
