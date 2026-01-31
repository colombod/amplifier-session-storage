"""
Identity management for session storage.

Provides abstractions for user identity, device tracking,
and authentication across different providers.
"""

from .config_provider import ConfigFileIdentityProvider
from .context import IdentityContext
from .provider import IdentityProvider
from .types import (
    AuthenticationError,
    AuthenticationRequiredError,
    AuthProvider,
    DeviceInfo,
    UserIdentity,
)

__all__ = [
    # Types
    "AuthProvider",
    "DeviceInfo",
    "UserIdentity",
    # Errors
    "AuthenticationError",
    "AuthenticationRequiredError",
    # Provider
    "IdentityProvider",
    "ConfigFileIdentityProvider",
    # Context
    "IdentityContext",
]
