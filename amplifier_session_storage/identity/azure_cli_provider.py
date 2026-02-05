"""
Auth Token Identity Provider.

Resolves user identity from the daemon's cached auth token at ~/.amplifier/.auth-token.
The sync daemon (amplifier-session-sync) handles the OAuth2 Authorization Code Flow
with PKCE and caches the token. This provider simply reads the cached result.

Fallback: If the auth token file doesn't exist, attempts to extract the email
from the JWT access token claims (same approach as the daemon's Rust code).
"""

import base64
import json
import logging
from pathlib import Path

from .provider import IdentityProvider
from .types import AuthProvider, UserIdentity

logger = logging.getLogger(__name__)

# Same path the daemon uses: ~/.amplifier/.auth-token
AUTH_TOKEN_PATH = Path.home() / ".amplifier" / ".auth-token"


def _extract_user_id_from_email(email: str) -> str:
    """Extract user_id from email by stripping domain.

    Same logic as daemon's extract_user_id_from_email() in config/mod.rs.
    """
    if "@" in email:
        return email.split("@")[0]
    return email


def _extract_email_from_jwt(token: str) -> str | None:
    """Extract email from JWT access token claims without verification.

    Same logic as daemon's extract_email_from_jwt() in auth/mod.rs:
    tries 'email', then 'preferred_username', then 'upn' claims.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        # Decode payload (second part) - base64url without padding
        payload_b64 = parts[1]
        # Add padding if needed
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding

        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        claims = json.loads(payload_bytes)

        # Try claims in same order as daemon: email > preferred_username > upn
        for claim_key in ("email", "preferred_username", "upn"):
            value = claims.get(claim_key)
            if value and isinstance(value, str):
                return value

        return None
    except Exception:
        return None


def _read_auth_token_file() -> dict | None:
    """Read the daemon's cached auth token file.

    Returns:
        Parsed JSON dict or None if file doesn't exist or is invalid.
    """
    try:
        if not AUTH_TOKEN_PATH.exists():
            return None

        data = json.loads(AUTH_TOKEN_PATH.read_text())
        return data
    except (json.JSONDecodeError, OSError, PermissionError) as e:
        logger.debug(f"Failed to read auth token file: {e}")
        return None


class AuthTokenIdentityProvider(IdentityProvider):
    """Identity provider using the daemon's cached auth token.

    Reads ~/.amplifier/.auth-token which is created and maintained by
    the amplifier-session-sync daemon via OAuth2 Authorization Code Flow + PKCE.

    The token file contains:
    - access_token: JWT with user claims
    - refresh_token: For token renewal
    - expires_at_unix: Expiration timestamp
    - user_email: Pre-extracted email from the token

    Usage:
        provider = AuthTokenIdentityProvider()
        identity = await provider.get_current_identity()
        print(f"Current user: {identity.user_id}")
    """

    def __init__(self) -> None:
        """Initialize the auth token identity provider."""
        self._cached_identity: UserIdentity | None = None
        self._device_id: str | None = None

    async def get_current_identity(self) -> UserIdentity:
        """Get identity from the daemon's cached auth token.

        Resolution order:
        1. Read user_email from ~/.amplifier/.auth-token
        2. If no user_email field, decode JWT access_token claims
        3. Extract user_id from email (strip domain)

        Returns:
            UserIdentity with user_id extracted from auth token

        Raises:
            AuthenticationRequiredError: If auth token file missing or invalid
        """
        if self._cached_identity:
            return self._cached_identity

        token_data = _read_auth_token_file()

        if token_data is None:
            from .types import AuthenticationRequiredError

            raise AuthenticationRequiredError(
                "Auth token not found at ~/.amplifier/.auth-token. "
                "Run the amplifier-session-sync daemon with --setup-auth to authenticate."
            )

        # Try user_email field first (daemon writes this directly)
        email = token_data.get("user_email", "")

        # Fallback: decode JWT to extract email from claims
        if not email:
            access_token = token_data.get("access_token", "")
            if access_token:
                email = _extract_email_from_jwt(access_token) or ""

        if not email:
            from .types import AuthenticationError

            raise AuthenticationError(
                "Auth token exists but contains no user email. "
                "Token may be corrupted. Re-run daemon --setup-auth."
            )

        user_id = _extract_user_id_from_email(email)

        self._cached_identity = UserIdentity(
            user_id=user_id,
            display_name=email,
            email=email,
            auth_provider=AuthProvider.ENTRA,
            auth_token=token_data.get("access_token"),
        )

        logger.info(f"Auth token identity resolved: {user_id}")
        return self._cached_identity

    async def refresh_token(self) -> UserIdentity:
        """Refresh by re-reading the auth token file."""
        self._cached_identity = None
        return await self.get_current_identity()

    async def sign_out(self) -> None:
        """Clear cached identity."""
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


# Keep old name as alias for backward compatibility
AzureCliIdentityProvider = AuthTokenIdentityProvider


def get_current_user_id() -> str | None:
    """Get current user_id from the daemon's cached auth token.

    Reads ~/.amplifier/.auth-token (created by amplifier-session-sync daemon)
    and extracts the user_id from the stored email.

    Returns:
        User ID (part before @ in email) or None if not authenticated

    Example:
        >>> user_id = get_current_user_id()
        >>> print(user_id)  # "SC-dc174"
    """
    token_data = _read_auth_token_file()
    if token_data is None:
        logger.debug("No auth token file found at ~/.amplifier/.auth-token")
        return None

    # Try user_email field first (daemon writes this directly)
    email = token_data.get("user_email", "")

    # Fallback: decode JWT to extract email from claims
    if not email:
        access_token = token_data.get("access_token", "")
        if access_token:
            email = _extract_email_from_jwt(access_token) or ""

    if not email:
        logger.debug("Auth token exists but no email found")
        return None

    user_id = _extract_user_id_from_email(email)
    return user_id if user_id else None
