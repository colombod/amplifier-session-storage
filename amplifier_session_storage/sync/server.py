"""
Real-time sync server components.

Provides server-side handlers for SSE and WebSocket connections,
enabling real-time block streaming to connected clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..blocks.types import SessionBlock
from ..storage.base import BlockStorage

logger = logging.getLogger(__name__)


@dataclass
class ConnectedClient:
    """A connected sync client."""

    client_id: str
    user_id: str
    subscribed_sessions: set[str] = field(default_factory=set)
    connected_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_activity: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Queue for outgoing messages
    message_queue: asyncio.Queue[dict[str, Any]] = field(default_factory=lambda: asyncio.Queue())


class SyncHandler:
    """Handler for sync events from storage.

    Bridges storage operations to connected clients.
    Call notify_* methods when blocks are written to storage.

    Example:
        >>> handler = SyncHandler()
        >>> # When a block is written to storage:
        >>> await handler.notify_block_added(block)
        >>> # Connected clients receive the update
    """

    def __init__(self) -> None:
        """Initialize the sync handler."""
        self._clients: dict[str, ConnectedClient] = {}
        self._lock = asyncio.Lock()

    async def register_client(
        self,
        client_id: str,
        user_id: str,
        sessions: list[str] | None = None,
    ) -> ConnectedClient:
        """Register a new connected client.

        Args:
            client_id: Unique client identifier
            user_id: User ID for access control
            sessions: Initial session subscriptions

        Returns:
            The registered client
        """
        async with self._lock:
            client = ConnectedClient(
                client_id=client_id,
                user_id=user_id,
                subscribed_sessions=set(sessions) if sessions else set(),
            )
            self._clients[client_id] = client
            logger.info(f"Client registered: {client_id} (user={user_id})")
            return client

    async def unregister_client(self, client_id: str) -> None:
        """Unregister a client.

        Args:
            client_id: Client to unregister
        """
        async with self._lock:
            if client_id in self._clients:
                del self._clients[client_id]
                logger.info(f"Client unregistered: {client_id}")

    async def subscribe_client(
        self,
        client_id: str,
        session_id: str,
    ) -> None:
        """Subscribe a client to a session.

        Args:
            client_id: Client ID
            session_id: Session to subscribe to
        """
        async with self._lock:
            if client_id in self._clients:
                self._clients[client_id].subscribed_sessions.add(session_id)

    async def unsubscribe_client(
        self,
        client_id: str,
        session_id: str,
    ) -> None:
        """Unsubscribe a client from a session.

        Args:
            client_id: Client ID
            session_id: Session to unsubscribe from
        """
        async with self._lock:
            if client_id in self._clients:
                self._clients[client_id].subscribed_sessions.discard(session_id)

    async def notify_block_added(
        self,
        block: SessionBlock,
        exclude_client: str | None = None,
    ) -> None:
        """Notify clients about a new block.

        Args:
            block: The new block
            exclude_client: Optional client to exclude (e.g., the sender)
        """
        event = {
            "type": "block_added",
            "session_id": block.session_id,
            "block": block.to_dict(),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        await self._broadcast_to_session(
            block.session_id,
            block.user_id,
            event,
            exclude_client,
        )

    async def notify_session_deleted(
        self,
        session_id: str,
        user_id: str,
    ) -> None:
        """Notify clients about a deleted session.

        Args:
            session_id: The deleted session
            user_id: Owner of the session
        """
        event = {
            "type": "session_deleted",
            "session_id": session_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        await self._broadcast_to_session(session_id, user_id, event)

    async def notify_conflict(
        self,
        session_id: str,
        user_id: str,
        local_sequence: int,
        remote_sequence: int,
    ) -> None:
        """Notify clients about a sync conflict.

        Args:
            session_id: Session with conflict
            user_id: User owning the session
            local_sequence: Local sequence number
            remote_sequence: Remote sequence number
        """
        event = {
            "type": "conflict",
            "session_id": session_id,
            "local_sequence": local_sequence,
            "remote_sequence": remote_sequence,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        await self._broadcast_to_session(session_id, user_id, event)

    async def _broadcast_to_session(
        self,
        session_id: str,
        owner_user_id: str,
        event: dict[str, Any],
        exclude_client: str | None = None,
    ) -> None:
        """Broadcast event to all clients subscribed to a session.

        Args:
            session_id: Target session
            owner_user_id: Owner for access control
            event: Event data to send
            exclude_client: Optional client to exclude
        """
        async with self._lock:
            for client_id, client in self._clients.items():
                if exclude_client and client_id == exclude_client:
                    continue

                # Check subscription
                if session_id not in client.subscribed_sessions:
                    continue

                # Check access (only owner can see their sessions for now)
                # TODO: Add team/org visibility checks
                if client.user_id != owner_user_id:
                    continue

                # Queue the message
                await client.message_queue.put(event)

    def get_connected_clients(self) -> list[dict[str, Any]]:
        """Get list of connected clients.

        Returns:
            List of client info dictionaries
        """
        return [
            {
                "client_id": client.client_id,
                "user_id": client.user_id,
                "sessions": list(client.subscribed_sessions),
                "connected_at": client.connected_at,
            }
            for client in self._clients.values()
        ]


class SyncServer:
    """Server for real-time session sync.

    Provides endpoints for SSE and WebSocket connections.
    Integrates with storage to broadcast updates to clients.

    Example with aiohttp:
        >>> server = SyncServer(storage)
        >>> app = aiohttp.web.Application()
        >>> app.router.add_get("/sync/events", server.sse_endpoint)
        >>> app.router.add_get("/sync/ws", server.websocket_endpoint)

    Example with FastAPI:
        >>> server = SyncServer(storage)
        >>> @app.get("/sync/events")
        >>> async def sse_events(request: Request):
        >>>     return server.create_sse_response(request)
    """

    def __init__(
        self,
        storage: BlockStorage,
        auth_validator: Callable[[str], str | None] | None = None,
    ) -> None:
        """Initialize the sync server.

        Args:
            storage: Block storage backend
            auth_validator: Optional function to validate auth tokens.
                            Should return user_id or None if invalid.
        """
        self.storage = storage
        self.auth_validator = auth_validator
        self.handler = SyncHandler()

        # Hook into storage operations
        self._setup_storage_hooks()

    def _setup_storage_hooks(self) -> None:
        """Set up hooks to notify clients on storage operations."""
        # Wrap storage write_block to broadcast updates
        original_write = self.storage.write_block

        async def wrapped_write(block: SessionBlock) -> None:
            await original_write(block)
            await self.handler.notify_block_added(block)

        self.storage.write_block = wrapped_write  # type: ignore[method-assign]

    async def handle_sse_connection(
        self,
        user_id: str,
        sessions: list[str],
        send_event: Callable[[str, str], Any],
    ) -> None:
        """Handle an SSE connection.

        Args:
            user_id: Authenticated user ID
            sessions: Sessions to subscribe to
            send_event: Async function to send SSE events (event_type, data)
        """
        import uuid

        client_id = str(uuid.uuid4())
        client = await self.handler.register_client(client_id, user_id, sessions)

        try:
            # Send connected event
            await send_event("connected", json.dumps({"client_id": client_id}))

            # Stream events from queue
            while True:
                event = await client.message_queue.get()
                event_type = event.get("type", "message")
                await send_event(event_type, json.dumps(event))

        except asyncio.CancelledError:
            pass
        finally:
            await self.handler.unregister_client(client_id)

    async def handle_websocket_connection(
        self,
        user_id: str,
        receive: Callable[[], Any],
        send: Callable[[dict[str, Any]], Any],
    ) -> None:
        """Handle a WebSocket connection.

        Args:
            user_id: Authenticated user ID
            receive: Async function to receive messages
            send: Async function to send messages
        """
        import uuid

        client_id = str(uuid.uuid4())
        client = await self.handler.register_client(client_id, user_id)

        try:
            # Send connected event
            await send({"type": "connected", "client_id": client_id})

            # Create tasks for send and receive
            async def receive_loop() -> None:
                while True:
                    msg = await receive()
                    if msg is None:
                        break
                    await self._handle_ws_message(client_id, msg)

            async def send_loop() -> None:
                while True:
                    event = await client.message_queue.get()
                    await send(event)

            # Run both loops
            await asyncio.gather(receive_loop(), send_loop())

        except asyncio.CancelledError:
            pass
        finally:
            await self.handler.unregister_client(client_id)

    async def _handle_ws_message(
        self,
        client_id: str,
        message: dict[str, Any],
    ) -> None:
        """Handle incoming WebSocket message.

        Args:
            client_id: Client that sent the message
            message: Message data
        """
        msg_type = message.get("type")

        if msg_type == "subscribe":
            sessions = message.get("sessions", [])
            for session_id in sessions:
                await self.handler.subscribe_client(client_id, session_id)

        elif msg_type == "unsubscribe":
            sessions = message.get("sessions", [])
            for session_id in sessions:
                await self.handler.unsubscribe_client(client_id, session_id)

        elif msg_type == "ping":
            # Update last activity
            if client_id in self.handler._clients:
                self.handler._clients[client_id].last_activity = datetime.now(UTC).isoformat()

    def validate_auth(self, auth_header: str | None) -> str | None:
        """Validate authentication header.

        Args:
            auth_header: Authorization header value

        Returns:
            User ID if valid, None otherwise
        """
        if not auth_header:
            return None

        if not self.auth_validator:
            # No validator configured, extract user from Bearer token
            if auth_header.startswith("Bearer "):
                return auth_header[7:]  # Assume token is user_id
            return None

        return self.auth_validator(auth_header)
