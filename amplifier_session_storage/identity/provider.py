"""
Identity provider abstract interface.

Defines the contract that all identity providers must implement.
"""

from abc import ABC, abstractmethod

from .types import AuthProvider, UserIdentity


class IdentityProvider(ABC):
    """Abstract identity provider.

    Implementations handle authentication and identity resolution
    for different auth providers (config file, Entra ID, OAuth, etc.).

    The provider is responsible for:
    - Resolving current user identity
    - Managing device registration
    - Token refresh (for token-based providers)
    - Sign out / credential clearing
    """

    @abstractmethod
    async def get_current_identity(self) -> UserIdentity:
        """Get the current authenticated user identity.

        Returns:
            UserIdentity with all context filled in

        Raises:
            AuthenticationRequiredError: If not authenticated
            AuthenticationError: If authentication fails
        """
        ...

    @abstractmethod
    async def refresh_token(self) -> UserIdentity:
        """Refresh authentication token if expired.

        For providers that don't use tokens (e.g., config),
        this should just return the current identity.

        Returns:
            Updated UserIdentity with fresh token

        Raises:
            AuthenticationError: If refresh fails
        """
        ...

    @abstractmethod
    async def sign_out(self) -> None:
        """Sign out and clear cached credentials.

        After sign out, get_current_identity() will raise
        AuthenticationRequiredError until re-authenticated.
        """
        ...

    @property
    @abstractmethod
    def provider_type(self) -> AuthProvider:
        """Get the provider type."""
        ...

    @abstractmethod
    async def get_device_id(self) -> str:
        """Get the unique device identifier.

        The device ID is persistent across sessions and
        used for multi-machine sync and conflict resolution.
        """
        ...
