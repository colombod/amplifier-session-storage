"""
Identity context singleton.

Provides a global access point for identity management.
Bridges identity providers with storage operations.
"""

from pathlib import Path
from typing import Any

import yaml

from .config_provider import ConfigFileIdentityProvider
from .provider import IdentityProvider
from .types import UserIdentity


class IdentityContext:
    """Manages identity context for storage operations.

    This is the bridge between identity providers and storage.
    It maintains a singleton instance of the configured identity provider.

    Usage:
        # Initialize once at startup
        await IdentityContext.initialize()

        # Get identity anywhere in the app
        identity = await IdentityContext.get_identity()
        user_id = await IdentityContext.get_user_id()
        device_id = await IdentityContext.get_device_id()
    """

    _instance: "IdentityContext | None" = None
    _provider: IdentityProvider | None = None
    _config_path: Path | None = None

    def __init__(self) -> None:
        """Private constructor. Use initialize() instead."""
        pass

    @classmethod
    async def initialize(
        cls,
        config_path: Path | None = None,
        provider: IdentityProvider | None = None,
    ) -> "IdentityContext":
        """Initialize identity context from configuration.

        Args:
            config_path: Path to settings.yaml. Defaults to ~/.amplifier/settings.yaml
            provider: Optional pre-configured provider (for testing)

        Returns:
            The initialized IdentityContext instance
        """
        if provider is not None:
            # Use provided provider (for testing)
            cls._provider = provider
            cls._instance = cls()
            return cls._instance

        cls._config_path = config_path or Path.home() / ".amplifier" / "settings.yaml"
        config = cls._load_config(cls._config_path)
        provider_type = config.get("identity", {}).get("provider", "config")

        if provider_type == "config":
            cls._provider = ConfigFileIdentityProvider(cls._config_path)
        elif provider_type == "entra":
            # Entra provider would be imported and instantiated here
            # For now, fall back to config
            raise NotImplementedError(
                "Entra identity provider not yet implemented. "
                "Use 'config' provider or set AMPLIFIER_IDENTITY_PROVIDER=config"
            )
        elif provider_type == "oauth":
            raise NotImplementedError(
                "OAuth identity provider not yet implemented. "
                "Use 'config' provider or set AMPLIFIER_IDENTITY_PROVIDER=config"
            )
        else:
            raise ValueError(f"Unknown identity provider: {provider_type}")

        cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the context (primarily for testing)."""
        cls._instance = None
        cls._provider = None
        cls._config_path = None

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the context has been initialized."""
        return cls._provider is not None

    @classmethod
    async def get_identity(cls) -> UserIdentity:
        """Get current user identity.

        Returns:
            The current UserIdentity

        Raises:
            RuntimeError: If context not initialized
            AuthenticationRequiredError: If not authenticated
        """
        if cls._provider is None:
            raise RuntimeError(
                "IdentityContext not initialized. Call 'await IdentityContext.initialize()' first."
            )
        return await cls._provider.get_current_identity()

    @classmethod
    async def get_user_id(cls) -> str:
        """Convenience: get just user_id."""
        identity = await cls.get_identity()
        return identity.user_id

    @classmethod
    async def get_device_id(cls) -> str:
        """Convenience: get just device_id."""
        if cls._provider is None:
            raise RuntimeError(
                "IdentityContext not initialized. Call 'await IdentityContext.initialize()' first."
            )
        return await cls._provider.get_device_id()

    @classmethod
    async def get_org_id(cls) -> str | None:
        """Convenience: get just org_id."""
        identity = await cls.get_identity()
        return identity.org_id

    @classmethod
    async def get_team_ids(cls) -> list[str]:
        """Convenience: get just team_ids."""
        identity = await cls.get_identity()
        return identity.team_ids

    @classmethod
    def get_provider(cls) -> IdentityProvider:
        """Get the underlying identity provider.

        Useful for provider-specific operations like token refresh.
        """
        if cls._provider is None:
            raise RuntimeError(
                "IdentityContext not initialized. Call 'await IdentityContext.initialize()' first."
            )
        return cls._provider

    @classmethod
    async def refresh(cls) -> UserIdentity:
        """Refresh authentication if needed.

        For token-based providers, this refreshes the token.
        For config provider, this just returns current identity.
        """
        if cls._provider is None:
            raise RuntimeError(
                "IdentityContext not initialized. Call 'await IdentityContext.initialize()' first."
            )
        return await cls._provider.refresh_token()

    @classmethod
    async def sign_out(cls) -> None:
        """Sign out the current user."""
        if cls._provider is None:
            return
        await cls._provider.sign_out()

    @classmethod
    def _load_config(cls, config_path: Path) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if not config_path.exists():
            return {}

        try:
            content = config_path.read_text()
            return yaml.safe_load(content) or {}
        except Exception:
            return {}
