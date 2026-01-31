"""Permission types for access control."""

from dataclasses import dataclass
from enum import Enum


class Permission(Enum):
    """Permission levels for session access."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"


@dataclass
class AccessDecision:
    """Result of an access control check."""

    allowed: bool
    reason: str
    permission_level: Permission | None = None
