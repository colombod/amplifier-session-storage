"""
Azure CLI Identity Provider.

Resolves user identity from the current Azure CLI login (az login).
This is useful for development and local scenarios where the user
is already authenticated via `az login`.
"""

import json
import logging
import subprocess

from .provider import IdentityProvider
from .types import AuthProvider, UserIdentity

logger = logging.getLogger(__name__)


class AzureCliIdentityProvider(IdentityProvider):
    """Identity provider using Azure CLI credentials.

    Gets the current user from `az account show` command.
    The user_id is extracted from the email (part before @).

    Usage:
        provider = AzureCliIdentityProvider()
        identity = await provider.get_current_identity()
        print(f"Current user: {identity.user_id}")
    """

    def __init__(self) -> None:
        """Initialize the Azure CLI identity provider."""
        self._cached_identity: UserIdentity | None = None
        self._device_id: str | None = None

    async def get_current_identity(self) -> UserIdentity:
        """Get identity from Azure CLI.

        Returns:
            UserIdentity with user_id extracted from Azure login

        Raises:
            AuthenticationError: If not logged in or az CLI not available
        """
        if self._cached_identity:
            return self._cached_identity

        try:
            result = subprocess.run(
                ["az", "account", "show", "--query", "user", "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                from .types import AuthenticationRequiredError

                raise AuthenticationRequiredError(
                    "Not logged in to Azure CLI. Run 'az login' first."
                )

            user_info = json.loads(result.stdout)
            email = user_info.get("name", "")

            # Extract user_id (part before @)
            if "@" in email:
                user_id = email.split("@")[0]
            else:
                user_id = email

            self._cached_identity = UserIdentity(
                user_id=user_id,
                display_name=email,
                email=email,
                auth_provider=AuthProvider.ENTRA,
            )

            logger.info(f"Azure CLI identity resolved: {user_id}")
            return self._cached_identity

        except FileNotFoundError as err:
            from .types import AuthenticationError

            raise AuthenticationError("Azure CLI (az) not found. Install Azure CLI first.") from err
        except subprocess.TimeoutExpired as err:
            from .types import AuthenticationError

            raise AuthenticationError("Azure CLI command timed out") from err
        except json.JSONDecodeError as err:
            from .types import AuthenticationError

            raise AuthenticationError(f"Failed to parse Azure CLI output: {err}") from err

    async def refresh_token(self) -> UserIdentity:
        """Refresh by re-reading from Azure CLI."""
        self._cached_identity = None
        return await self.get_current_identity()

    async def sign_out(self) -> None:
        """Clear cached identity (doesn't sign out of Azure CLI)."""
        self._cached_identity = None

    @property
    def provider_type(self) -> AuthProvider:
        """Return Entra as the provider type."""
        return AuthProvider.ENTRA

    async def get_device_id(self) -> str:
        """Get device ID from hostname."""
        if self._device_id:
            return self._device_id

        import socket

        self._device_id = socket.gethostname()
        return self._device_id


def get_current_user_id() -> str | None:
    """Convenience function to get current Azure CLI user_id synchronously.

    Returns:
        User ID (part before @ in email) or None if not logged in

    Example:
        >>> user_id = get_current_user_id()
        >>> print(user_id)  # "SC-dc174"
    """
    try:
        result = subprocess.run(
            ["az", "account", "show", "--query", "user.name", "-o", "tsv"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return None

        email = result.stdout.strip()
        if "@" in email:
            return email.split("@")[0]
        return email if email else None

    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
