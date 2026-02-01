"""Tests for identity module."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

from amplifier_session_storage.identity import (
    AuthProvider,
    ConfigFileIdentityProvider,
    DeviceInfo,
    IdentityContext,
    UserIdentity,
)

if TYPE_CHECKING:
    pass


class TestDeviceInfo:
    """Tests for DeviceInfo dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        now = datetime.now(UTC)
        device = DeviceInfo(
            device_id="dev-123",
            device_name="Test Device",
            os_type="linux",
            first_seen=now,
            last_seen=now,
        )

        data = device.to_dict()

        assert data["device_id"] == "dev-123"
        assert data["device_name"] == "Test Device"
        assert data["os_type"] == "linux"
        assert data["first_seen"] == now.isoformat()
        assert data["last_seen"] == now.isoformat()

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        now = datetime.now(UTC)
        data = {
            "device_id": "dev-456",
            "device_name": "Another Device",
            "os_type": "macos",
            "first_seen": now.isoformat(),
            "last_seen": now.isoformat(),
        }

        device = DeviceInfo.from_dict(data)

        assert device.device_id == "dev-456"
        assert device.device_name == "Another Device"
        assert device.os_type == "macos"

    def test_roundtrip(self) -> None:
        """Test serialization roundtrip."""
        now = datetime.now(UTC)
        original = DeviceInfo(
            device_id="dev-789",
            device_name="Roundtrip Device",
            os_type="windows",
            first_seen=now,
            last_seen=now,
        )

        data = original.to_dict()
        restored = DeviceInfo.from_dict(data)

        assert restored.device_id == original.device_id
        assert restored.device_name == original.device_name
        assert restored.os_type == original.os_type


class TestUserIdentity:
    """Tests for UserIdentity dataclass."""

    def test_minimal_identity(self) -> None:
        """Test identity with minimal fields."""
        identity = UserIdentity(
            user_id="user-123",
            display_name="Test User",
        )

        assert identity.user_id == "user-123"
        assert identity.display_name == "Test User"
        assert identity.email is None
        assert identity.org_id is None
        assert identity.team_ids == []
        assert identity.role == "member"
        assert identity.auth_provider == AuthProvider.CONFIG

    def test_full_identity(self) -> None:
        """Test identity with all fields."""
        now = datetime.now(UTC)
        device = DeviceInfo(
            device_id="dev-123",
            device_name="Test Device",
            os_type="linux",
            first_seen=now,
            last_seen=now,
        )

        identity = UserIdentity(
            user_id="user-456",
            display_name="Full User",
            email="user@example.com",
            org_id="org-789",
            team_ids=["team-a", "team-b"],
            role="admin",
            device=device,
            auth_provider=AuthProvider.ENTRA,
            auth_token="token-123",
            token_expiry=now,
        )

        assert identity.user_id == "user-456"
        assert identity.email == "user@example.com"
        assert identity.org_id == "org-789"
        assert identity.team_ids == ["team-a", "team-b"]
        assert identity.role == "admin"
        assert identity.device is not None
        assert identity.auth_provider == AuthProvider.ENTRA

    def test_is_authenticated_config_provider(self) -> None:
        """Test authentication check for config provider."""
        identity = UserIdentity(
            user_id="user-123",
            display_name="Config User",
            auth_provider=AuthProvider.CONFIG,
            auth_token=None,
        )

        # Config provider is always authenticated
        assert identity.is_authenticated() is True

    def test_is_authenticated_with_valid_token(self) -> None:
        """Test authentication check with valid token."""
        from datetime import timedelta

        future = datetime.now(UTC) + timedelta(hours=1)
        identity = UserIdentity(
            user_id="user-123",
            display_name="Token User",
            auth_provider=AuthProvider.ENTRA,
            auth_token="valid-token",
            token_expiry=future,
        )

        assert identity.is_authenticated() is True

    def test_is_authenticated_with_expired_token(self) -> None:
        """Test authentication check with expired token."""
        from datetime import timedelta

        past = datetime.now(UTC) - timedelta(hours=1)
        identity = UserIdentity(
            user_id="user-123",
            display_name="Expired User",
            auth_provider=AuthProvider.ENTRA,
            auth_token="expired-token",
            token_expiry=past,
        )

        assert identity.is_authenticated() is False

    def test_to_dict_excludes_token(self) -> None:
        """Test that auth_token is excluded from serialization."""
        identity = UserIdentity(
            user_id="user-123",
            display_name="Secret User",
            auth_token="super-secret-token",
        )

        data = identity.to_dict()

        assert "auth_token" not in data
        assert data["user_id"] == "user-123"

    def test_roundtrip(self) -> None:
        """Test serialization roundtrip."""
        now = datetime.now(UTC)
        device = DeviceInfo(
            device_id="dev-123",
            device_name="Test Device",
            os_type="linux",
            first_seen=now,
            last_seen=now,
        )

        original = UserIdentity(
            user_id="user-roundtrip",
            display_name="Roundtrip User",
            email="roundtrip@example.com",
            org_id="org-123",
            team_ids=["team-1", "team-2"],
            role="member",
            device=device,
            auth_provider=AuthProvider.CONFIG,
        )

        data = original.to_dict()
        restored = UserIdentity.from_dict(data)

        assert restored.user_id == original.user_id
        assert restored.display_name == original.display_name
        assert restored.email == original.email
        assert restored.org_id == original.org_id
        assert restored.team_ids == original.team_ids


class TestConfigFileIdentityProvider:
    """Tests for ConfigFileIdentityProvider."""

    @pytest.fixture
    def temp_dir(self) -> Iterator[Path]:
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config_path(self, temp_dir: Path) -> Path:
        """Create a config file path in temp directory."""
        return temp_dir / "settings.yaml"

    @pytest.mark.asyncio
    async def test_get_identity_no_config(self, config_path: Path) -> None:
        """Test getting identity when no config exists."""
        provider = ConfigFileIdentityProvider(config_path)

        identity = await provider.get_current_identity()

        # Should generate a local identity
        assert identity.user_id.startswith("local-")
        assert identity.display_name is not None
        assert identity.auth_provider == AuthProvider.CONFIG
        assert identity.device is not None

    @pytest.mark.asyncio
    async def test_get_identity_with_config(self, config_path: Path) -> None:
        """Test getting identity from config file."""
        # Create config
        config = {
            "identity": {
                "provider": "config",
                "user_id": "configured-user",
                "display_name": "Configured User",
                "email": "configured@example.com",
                "org_id": "configured-org",
                "team_ids": ["team-x", "team-y"],
            }
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config))

        provider = ConfigFileIdentityProvider(config_path)
        identity = await provider.get_current_identity()

        assert identity.user_id == "configured-user"
        assert identity.display_name == "Configured User"
        assert identity.email == "configured@example.com"
        assert identity.org_id == "configured-org"
        assert identity.team_ids == ["team-x", "team-y"]

    @pytest.mark.asyncio
    async def test_get_identity_caching(self, config_path: Path) -> None:
        """Test that identity is cached."""
        provider = ConfigFileIdentityProvider(config_path)

        identity1 = await provider.get_current_identity()
        identity2 = await provider.get_current_identity()

        # Should be the same object (cached)
        assert identity1 is identity2

    @pytest.mark.asyncio
    async def test_sign_out_clears_cache(self, config_path: Path) -> None:
        """Test that sign out clears the cached identity."""
        provider = ConfigFileIdentityProvider(config_path)

        identity1 = await provider.get_current_identity()
        await provider.sign_out()
        identity2 = await provider.get_current_identity()

        # Should be different objects after sign out
        assert identity1 is not identity2
        # But same user_id (from config)
        assert identity1.user_id == identity2.user_id

    @pytest.mark.asyncio
    async def test_device_id_persistence(self, config_path: Path) -> None:
        """Test that device ID is persisted across provider instances."""
        provider1 = ConfigFileIdentityProvider(config_path)
        device_id1 = await provider1.get_device_id()

        # Create a new provider instance
        provider2 = ConfigFileIdentityProvider(config_path)
        device_id2 = await provider2.get_device_id()

        # Should be the same device ID
        assert device_id1 == device_id2

        # Verify file exists
        device_file = config_path.parent / ".device_id"
        assert device_file.exists()
        assert device_file.read_text().strip() == device_id1

    @pytest.mark.asyncio
    async def test_provider_type(self, config_path: Path) -> None:
        """Test provider type property."""
        provider = ConfigFileIdentityProvider(config_path)

        assert provider.provider_type == AuthProvider.CONFIG

    @pytest.mark.asyncio
    async def test_refresh_token_returns_identity(self, config_path: Path) -> None:
        """Test that refresh_token returns identity (no-op for config)."""
        provider = ConfigFileIdentityProvider(config_path)

        identity = await provider.refresh_token()

        assert identity is not None
        assert identity.user_id is not None

    @pytest.mark.asyncio
    async def test_update_config(self, config_path: Path) -> None:
        """Test updating identity configuration."""
        provider = ConfigFileIdentityProvider(config_path)

        # Update config
        identity = await provider.update_config(
            user_id="updated-user",
            display_name="Updated User",
            org_id="updated-org",
            team_ids=["new-team"],
        )

        assert identity.user_id == "updated-user"
        assert identity.display_name == "Updated User"
        assert identity.org_id == "updated-org"
        assert identity.team_ids == ["new-team"]

        # Verify persisted
        config = yaml.safe_load(config_path.read_text())
        assert config["identity"]["user_id"] == "updated-user"


class TestIdentityContext:
    """Tests for IdentityContext singleton."""

    @pytest.fixture
    def temp_dir(self) -> Iterator[Path]:
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config_path(self, temp_dir: Path) -> Path:
        """Create a config file path in temp directory."""
        return temp_dir / "settings.yaml"

    @pytest.fixture(autouse=True)
    def reset_context(self) -> Iterator[None]:
        """Reset the context before and after each test."""
        IdentityContext.reset()
        yield
        IdentityContext.reset()

    @pytest.mark.asyncio
    async def test_initialize_default(self, config_path: Path) -> None:
        """Test default initialization."""
        await IdentityContext.initialize(config_path)

        assert IdentityContext.is_initialized()

    @pytest.mark.asyncio
    async def test_initialize_with_provider(self, config_path: Path) -> None:
        """Test initialization with custom provider."""
        provider = ConfigFileIdentityProvider(config_path)
        await IdentityContext.initialize(provider=provider)

        assert IdentityContext.is_initialized()
        assert IdentityContext.get_provider() is provider

    @pytest.mark.asyncio
    async def test_get_identity(self, config_path: Path) -> None:
        """Test getting identity from context."""
        config = {
            "identity": {
                "user_id": "context-user",
                "display_name": "Context User",
            }
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config))

        await IdentityContext.initialize(config_path)
        identity = await IdentityContext.get_identity()

        assert identity.user_id == "context-user"
        assert identity.display_name == "Context User"

    @pytest.mark.asyncio
    async def test_get_user_id(self, config_path: Path) -> None:
        """Test getting user_id from context."""
        config = {"identity": {"user_id": "shortcut-user"}}
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config))

        await IdentityContext.initialize(config_path)
        user_id = await IdentityContext.get_user_id()

        assert user_id == "shortcut-user"

    @pytest.mark.asyncio
    async def test_get_device_id(self, config_path: Path) -> None:
        """Test getting device_id from context."""
        await IdentityContext.initialize(config_path)
        device_id = await IdentityContext.get_device_id()

        assert device_id is not None
        assert len(device_id) > 0

    @pytest.mark.asyncio
    async def test_get_org_id(self, config_path: Path) -> None:
        """Test getting org_id from context."""
        config = {"identity": {"org_id": "context-org"}}
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config))

        await IdentityContext.initialize(config_path)
        org_id = await IdentityContext.get_org_id()

        assert org_id == "context-org"

    @pytest.mark.asyncio
    async def test_get_team_ids(self, config_path: Path) -> None:
        """Test getting team_ids from context."""
        config = {"identity": {"team_ids": ["team-1", "team-2"]}}
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config))

        await IdentityContext.initialize(config_path)
        team_ids = await IdentityContext.get_team_ids()

        assert team_ids == ["team-1", "team-2"]

    @pytest.mark.asyncio
    async def test_not_initialized_error(self) -> None:
        """Test error when context not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await IdentityContext.get_identity()

    @pytest.mark.asyncio
    async def test_reset(self, config_path: Path) -> None:
        """Test resetting the context."""
        await IdentityContext.initialize(config_path)
        assert IdentityContext.is_initialized()

        IdentityContext.reset()

        assert not IdentityContext.is_initialized()

    @pytest.mark.asyncio
    async def test_sign_out(self, config_path: Path) -> None:
        """Test signing out from context."""
        await IdentityContext.initialize(config_path)

        # Should not raise
        await IdentityContext.sign_out()

    @pytest.mark.asyncio
    async def test_refresh(self, config_path: Path) -> None:
        """Test refreshing identity."""
        await IdentityContext.initialize(config_path)

        identity = await IdentityContext.refresh()

        assert identity is not None
