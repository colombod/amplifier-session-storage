"""
Config file identity provider.

Reads identity from local configuration file for development
and offline-first usage.
"""

import platform
import socket
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .provider import IdentityProvider
from .types import AuthProvider, DeviceInfo, UserIdentity


class ConfigFileIdentityProvider(IdentityProvider):
    """Identity provider that reads from local config.

    Configuration in ~/.amplifier/settings.yaml:

    ```yaml
    identity:
      provider: config
      user_id: "user-abc123"
      display_name: "Alice Developer"
      email: "alice@example.com"
      org_id: "org-xyz"
      team_ids: ["backend", "platform"]
      role: "member"
      device_name: "Alice's Laptop"  # Optional, auto-detected if not set
    ```

    If identity section is missing, generates a local identity
    based on the device ID.
    """

    def __init__(self, config_path: Path | None = None):
        """Initialize the config file provider.

        Args:
            config_path: Path to settings.yaml. Defaults to ~/.amplifier/settings.yaml
        """
        self.config_path = config_path or Path.home() / ".amplifier" / "settings.yaml"
        self._identity: UserIdentity | None = None
        self._device_id: str | None = None

    async def get_current_identity(self) -> UserIdentity:
        """Get the current user identity from config.

        Returns cached identity if available, otherwise loads from config.
        """
        if self._identity is not None:
            # Update last_seen on the device
            if self._identity.device:
                self._identity.device.last_seen = datetime.utcnow()
            return self._identity

        # Load config
        config = self._load_config()
        identity_config = config.get("identity", {})

        # Get or create device ID
        device_id = await self.get_device_id()

        # Build device info
        device = DeviceInfo(
            device_id=device_id,
            device_name=identity_config.get("device_name", self._get_hostname()),
            os_type=self._get_os_type(),
            first_seen=datetime.utcnow(),
            last_seen=datetime.utcnow(),
        )

        # Build identity - use config values or generate defaults
        self._identity = UserIdentity(
            user_id=identity_config.get("user_id", f"local-{device_id[:8]}"),
            display_name=identity_config.get("display_name", self._get_default_display_name()),
            email=identity_config.get("email"),
            org_id=identity_config.get("org_id"),
            team_ids=identity_config.get("team_ids", []),
            role=identity_config.get("role", "member"),
            device=device,
            auth_provider=AuthProvider.CONFIG,
            auth_token=None,
            token_expiry=None,
        )

        return self._identity

    async def refresh_token(self) -> UserIdentity:
        """Config provider has no token to refresh.

        Just returns the current identity.
        """
        return await self.get_current_identity()

    async def sign_out(self) -> None:
        """Clear cached identity.

        For config provider, this just clears the cache.
        The config file is not modified.
        """
        self._identity = None

    @property
    def provider_type(self) -> AuthProvider:
        """Return CONFIG provider type."""
        return AuthProvider.CONFIG

    async def get_device_id(self) -> str:
        """Get or create persistent device ID.

        The device ID is stored in ~/.amplifier/.device_id
        and persists across sessions.
        """
        if self._device_id is not None:
            return self._device_id

        device_file = self.config_path.parent / ".device_id"

        if device_file.exists():
            self._device_id = device_file.read_text().strip()
            return self._device_id

        # Generate new device ID
        self._device_id = str(uuid.uuid4())

        # Persist it
        device_file.parent.mkdir(parents=True, exist_ok=True)
        device_file.write_text(self._device_id)

        return self._device_id

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            return {}

        try:
            content = self.config_path.read_text()
            return yaml.safe_load(content) or {}
        except Exception:
            return {}

    def _get_hostname(self) -> str:
        """Get the hostname for device name."""
        try:
            return socket.gethostname()
        except Exception:
            return "unknown-device"

    def _get_os_type(self) -> str:
        """Get the OS type (windows, macos, linux)."""
        system = platform.system().lower()
        if system == "darwin":
            return "macos"
        return system

    def _get_default_display_name(self) -> str:
        """Generate a default display name."""
        try:
            # Try to get the system username
            import getpass

            username = getpass.getuser()
            return username.title()
        except Exception:
            return "Local User"

    async def update_config(
        self,
        user_id: str | None = None,
        display_name: str | None = None,
        email: str | None = None,
        org_id: str | None = None,
        team_ids: list[str] | None = None,
    ) -> UserIdentity:
        """Update identity configuration.

        Updates the settings.yaml file and returns the new identity.
        Only provided values are updated; None values are left unchanged.
        """
        config = self._load_config()

        if "identity" not in config:
            config["identity"] = {"provider": "config"}

        identity_config: dict[str, Any] = config["identity"]

        if user_id is not None:
            identity_config["user_id"] = user_id
        if display_name is not None:
            identity_config["display_name"] = display_name
        if email is not None:
            identity_config["email"] = email
        if org_id is not None:
            identity_config["org_id"] = org_id
        if team_ids is not None:
            identity_config["team_ids"] = team_ids

        # Write back
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(yaml.safe_dump(config, default_flow_style=False))

        # Clear cache and reload
        self._identity = None
        return await self.get_current_identity()
