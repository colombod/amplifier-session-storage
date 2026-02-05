"""
OAuth2 Login Utility for Amplifier Session Storage.

Implements the same OAuth2 Authorization Code Flow + PKCE as the
amplifier-session-sync daemon (Rust), but in Python using only stdlib.

Usage:
    # From command line:
    python -m amplifier_session_storage.identity.login

    # Force re-login (select different account):
    python -m amplifier_session_storage.identity.login --force

    # Programmatic:
    from amplifier_session_storage.identity.login import login
    user_id = login(force=True)
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import platform
import secrets
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

# ---------------------------------------------------------------------------
# Azure AD configuration - same as daemon (config/mod.rs lines 132-139)
# ---------------------------------------------------------------------------
CLIENT_ID = "81bb79aa-a4a0-440b-8488-9987ee40c4fa"  # Public client app
API_RESOURCE_ID = "e2fa5d4f-4eed-42f3-bc1e-4869ed8714ba"  # API app (scope target)
TENANT_ID = "72f988bf-86f1-41af-91ab-2d7cd011db47"
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0"
AUTHORIZE_URL = f"{AUTHORITY}/authorize"
TOKEN_URL = f"{AUTHORITY}/token"
SCOPES = f"api://{API_RESOURCE_ID}/access_as_user offline_access openid profile email"

# Same token path as daemon
AUTH_TOKEN_PATH = Path.home() / ".amplifier" / ".auth-token"

# Ports to try for the local callback server (same as daemon)
CALLBACK_PORTS = [8080, 8081, 8082]


def _is_wsl() -> bool:
    """Detect if running inside WSL."""
    try:
        return "microsoft" in platform.release().lower()
    except Exception:
        return False


def _open_browser(url: str) -> None:
    """Open URL in browser, with WSL2 support."""
    if _is_wsl():
        # WSL2: use wslview (from wslu) or powershell.exe as fallback
        if shutil.which("wslview"):
            subprocess.Popen(["wslview", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        # Fallback: powershell Start-Process
        ps = shutil.which("powershell.exe")
        if ps:
            subprocess.Popen(
                [ps, "-NoProfile", "-Command", f'Start-Process "{url}"'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

    # Standard: use Python's webbrowser module
    webbrowser.open(url)


def _base64url_encode(data: bytes) -> str:
    """Base64url encode without padding (RFC 7636)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256).

    Same approach as daemon's auth/mod.rs.
    """
    # 32 random bytes -> base64url = 43 char verifier
    code_verifier = _base64url_encode(secrets.token_bytes(32))
    # S256: SHA256(verifier) -> base64url
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = _base64url_encode(digest)
    return code_verifier, code_challenge


def _extract_email_from_jwt(token: str) -> str | None:
    """Extract email from JWT access token claims.

    Same logic as daemon's extract_email_from_jwt() in auth/mod.rs:
    tries 'email', then 'preferred_username', then 'upn'.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload_b64 = parts[1]
        # Add padding
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        claims = json.loads(payload_bytes)
        for key in ("email", "preferred_username", "upn"):
            value = claims.get(key)
            if value and isinstance(value, str):
                return value
        return None
    except Exception:
        return None


def _extract_user_id_from_email(email: str) -> str:
    """Extract user_id from email (strip domain).

    Same as daemon's extract_user_id_from_email() in config/mod.rs.
    """
    return email.split("@")[0] if "@" in email else email


def _save_token(
    access_token: str,
    refresh_token: str,
    expires_in: int,
    user_email: str,
) -> None:
    """Save token to ~/.amplifier/.auth-token with restricted permissions.

    Same format and path as daemon's save_cached_token().
    """
    AUTH_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)

    token_data = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at_unix": int(time.time()) + expires_in,
        "user_email": user_email,
    }

    # Write with restricted permissions (0600) like the daemon
    AUTH_TOKEN_PATH.write_text(json.dumps(token_data, indent=2))
    AUTH_TOKEN_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _exchange_code_for_token(
    code: str,
    redirect_uri: str,
    code_verifier: str,
) -> dict:
    """Exchange authorization code for tokens at the token endpoint."""
    data = urllib.parse.urlencode(
        {
            "client_id": CLIENT_ID,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Token exchange failed (HTTP {e.code}): {error_body}") from e


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that captures the OAuth2 callback."""

    # Class-level storage for the result
    auth_code: str | None = None
    error: str | None = None
    state_received: str | None = None

    def do_GET(self):  # noqa: N802
        """Handle the OAuth2 callback redirect."""
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/callback":
            if "error" in params:
                _CallbackHandler.error = params["error"][0]
                error_desc = params.get("error_description", [""])[0]
                self._respond(
                    f"<h2>Authentication Failed</h2><p>{_CallbackHandler.error}: {error_desc}</p>"
                    "<p>You can close this tab.</p>"
                )
            elif "code" in params:
                _CallbackHandler.auth_code = params["code"][0]
                _CallbackHandler.state_received = params.get("state", [None])[0]
                self._respond(
                    "<h2>Authentication Successful</h2>"
                    "<p>You can close this tab and return to the terminal.</p>"
                    "<script>window.close()</script>"
                )
            else:
                self._respond("<h2>Unexpected response</h2><p>No code or error received.</p>")
        else:
            self.send_response(404)
            self.end_headers()

    def _respond(self, html: str):
        body = f"<html><body style='font-family:system-ui;text-align:center;padding:40px'>{html}</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, format, *args):  # noqa: A002
        """Suppress default request logging."""
        pass


def _start_callback_server() -> tuple[http.server.HTTPServer, int]:
    """Start a local HTTP server for the OAuth2 callback.

    Tries ports 8080-8082 (same as daemon).
    """
    for port in CALLBACK_PORTS:
        try:
            server = http.server.HTTPServer(("127.0.0.1", port), _CallbackHandler)
            return server, port
        except OSError:
            continue
    raise RuntimeError(
        f"Could not bind to any callback port ({CALLBACK_PORTS}). "
        "Make sure ports 8080-8082 are free."
    )


def login(force: bool = False) -> str:
    """Run the OAuth2 login flow and save the token.

    Args:
        force: If True, always prompt for account selection even if
               a cached token exists. Uses prompt=select_account.

    Returns:
        The resolved user_id (e.g. "SC-dc174")

    Raises:
        RuntimeError: If the login flow fails
    """
    # Check for existing valid token unless forcing
    if not force and AUTH_TOKEN_PATH.exists():
        try:
            data = json.loads(AUTH_TOKEN_PATH.read_text())
            expires_at = data.get("expires_at_unix", 0)
            email = data.get("user_email", "")
            if email and expires_at > time.time() + 300:  # 5 min buffer
                user_id = _extract_user_id_from_email(email)
                print(f"Already authenticated as: {user_id} ({email})")
                print("Use --force to re-authenticate with a different account.")
                return user_id
        except (json.JSONDecodeError, OSError):
            pass  # Token file corrupt, proceed with login

    # Reset handler state
    _CallbackHandler.auth_code = None
    _CallbackHandler.error = None
    _CallbackHandler.state_received = None

    # Generate PKCE
    code_verifier, code_challenge = _generate_pkce()
    state = secrets.token_urlsafe(16)

    # Start callback server
    server, port = _start_callback_server()
    redirect_uri = f"http://localhost:{port}/callback"

    # Build authorize URL
    auth_params = urllib.parse.urlencode(
        {
            "client_id": CLIENT_ID,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": SCOPES,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "prompt": "select_account",
        }
    )
    auth_url = f"{AUTHORIZE_URL}?{auth_params}"

    print(f"Opening browser for authentication on port {port}...")
    print(f"If browser doesn't open, visit:\n{auth_url}\n")

    # Open browser (with WSL2 support)
    _open_browser(auth_url)

    # Wait for callback (with timeout)
    server.timeout = 120  # 2 minute timeout
    print("Waiting for authentication callback...")

    # Handle requests until we get a code or error
    deadline = time.time() + 120
    while _CallbackHandler.auth_code is None and _CallbackHandler.error is None:
        remaining = deadline - time.time()
        if remaining <= 0:
            server.server_close()
            raise RuntimeError("Authentication timed out (120s). Try again.")
        server.timeout = remaining
        server.handle_request()

    server.server_close()

    # Check for errors
    if _CallbackHandler.error:
        raise RuntimeError(f"Authentication failed: {_CallbackHandler.error}")

    if _CallbackHandler.auth_code is None:
        raise RuntimeError("No authorization code received")

    # Verify state
    if _CallbackHandler.state_received != state:
        raise RuntimeError("CSRF state mismatch - possible security issue. Try again.")

    print("Exchanging code for token...")

    # Exchange code for tokens
    token_response = _exchange_code_for_token(
        code=_CallbackHandler.auth_code,
        redirect_uri=redirect_uri,
        code_verifier=code_verifier,
    )

    access_token = token_response.get("access_token", "")
    refresh_token = token_response.get("refresh_token", "")
    expires_in = token_response.get("expires_in", 3600)

    if not access_token:
        raise RuntimeError(f"Token exchange failed: {token_response}")

    # Extract email from JWT claims
    email = _extract_email_from_jwt(access_token)
    if not email:
        raise RuntimeError(
            "Could not extract email from token. "
            "Check that openid/profile/email scopes are granted."
        )

    user_id = _extract_user_id_from_email(email)

    # Save token
    _save_token(access_token, refresh_token, expires_in, email)

    print(f"\nAuthenticated as: {user_id} ({email})")
    print(f"Token saved to: {AUTH_TOKEN_PATH}")

    return user_id


def refresh_token() -> str | None:
    """Refresh the access token using the stored refresh token.

    Same as daemon's refresh_token() in auth/mod.rs.

    Returns:
        The user_id if refresh succeeded, None if refresh failed
        (caller should re-run login()).
    """
    if not AUTH_TOKEN_PATH.exists():
        return None

    try:
        data = json.loads(AUTH_TOKEN_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    stored_refresh_token = data.get("refresh_token")
    if not stored_refresh_token:
        return None

    try:
        req_data = urllib.parse.urlencode(
            {
                "client_id": CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": stored_refresh_token,
                "scope": SCOPES,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            TOKEN_URL,
            data=req_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            token_response = json.loads(resp.read())

    except Exception:
        return None

    access_token = token_response.get("access_token", "")
    new_refresh_token = token_response.get("refresh_token", stored_refresh_token)
    expires_in = token_response.get("expires_in", 3600)

    if not access_token:
        return None

    email = _extract_email_from_jwt(access_token)
    if not email:
        return None

    user_id = _extract_user_id_from_email(email)
    _save_token(access_token, new_refresh_token, expires_in, email)

    return user_id


def show_current_user() -> None:
    """Display the currently authenticated user."""
    if not AUTH_TOKEN_PATH.exists():
        print("Not authenticated. Run with no arguments to log in.")
        return

    try:
        data = json.loads(AUTH_TOKEN_PATH.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"Token file corrupt: {e}")
        return

    email = data.get("user_email", "unknown")
    user_id = _extract_user_id_from_email(email) if email else "unknown"
    expires_at = data.get("expires_at_unix", 0)
    remaining = expires_at - time.time()

    print(f"User:    {user_id}")
    print(f"Email:   {email}")

    if remaining > 0:
        minutes = int(remaining // 60)
        print(f"Expires: in {minutes} minutes")
    else:
        print("Expires: EXPIRED")
        print("Run with --refresh to get a new token, or no args to re-login.")

    print(f"Token:   {AUTH_TOKEN_PATH}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Amplifier Session Storage - OAuth2 Login",
        epilog="Authenticates using the same OAuth2 flow as the sync daemon.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-authentication (select different account)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh the existing token without browser login",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current authentication status",
    )

    args = parser.parse_args()

    if args.status:
        show_current_user()
        return

    if args.refresh:
        print("Refreshing token...")
        user_id = refresh_token()
        if user_id:
            print(f"Token refreshed for: {user_id}")
        else:
            print("Refresh failed. Run without --refresh to re-authenticate.")
            sys.exit(1)
        return

    try:
        login(force=args.force)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)


if __name__ == "__main__":
    main()
