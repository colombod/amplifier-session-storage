"""
Real-time sync client.

Provides client-side sync capabilities using Server-Sent Events (SSE)
or WebSocket connections for receiving live block updates.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from ..blocks.types import SessionBlock
from ..exceptions import StorageConnectionError
from ..storage.base import BlockStorage

logger = logging.getLogger(__name__)


class SyncEventType(Enum):
    """Types of sync events."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    BLOCK_ADDED = "block_added"
    BLOCK_UPDATED = "block_updated"
    SESSION_DELETED = "session_deleted"
    SYNC_ERROR = "sync_error"
    CONFLICT = "conflict"


@dataclass
class SyncEvent:
    """A sync event from the server."""

    event_type: SyncEventType
    session_id: str | None = None
    block: SessionBlock | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    error: str | None = None


class SyncClient:
    """Client for real-time session sync.

    Supports two modes:
    1. SSE (Server-Sent Events) - HTTP-based, works through proxies
    2. WebSocket - Lower latency, bidirectional

    The client:
    - Subscribes to session updates
    - Receives new blocks in real-time
    - Writes received blocks to local storage
    - Notifies callbacks on events

    Example:
        >>> client = SyncClient(
        ...     storage=local_storage,
        ...     sync_url="https://api.example.com/sync",
        ...     user_id="user-123",
        ... )
        >>> client.on_block_added = lambda block: print(f"New block: {block.block_id}")
        >>> await client.subscribe("session-abc")
        >>> # Client receives blocks in background
        >>> await client.unsubscribe("session-abc")
    """

    def __init__(
        self,
        storage: BlockStorage,
        sync_url: str,
        user_id: str,
        auth_token: str | None = None,
        use_websocket: bool = False,
        auto_reconnect: bool = True,
        reconnect_delay: float = 5.0,
    ) -> None:
        """Initialize the sync client.

        Args:
            storage: Local storage to write received blocks
            sync_url: URL of the sync server
            user_id: Current user ID
            auth_token: Optional auth token for the sync server
            use_websocket: Use WebSocket instead of SSE
            auto_reconnect: Automatically reconnect on disconnect
            reconnect_delay: Seconds to wait before reconnecting
        """
        self.storage = storage
        self.sync_url = sync_url.rstrip("/")
        self.user_id = user_id
        self.auth_token = auth_token
        self.use_websocket = use_websocket
        self.auto_reconnect = auto_reconnect
        self.reconnect_delay = reconnect_delay

        # Subscriptions
        self._subscribed_sessions: set[str] = set()
        self._running = False
        self._connection_task: asyncio.Task[None] | None = None

        # Callbacks
        self.on_connected: Callable[[], None] | None = None
        self.on_disconnected: Callable[[], None] | None = None
        self.on_block_added: Callable[[SessionBlock], None] | None = None
        self.on_conflict: Callable[[SyncEvent], None] | None = None
        self.on_error: Callable[[str], None] | None = None

    async def start(self) -> None:
        """Start the sync client."""
        if self._running:
            return

        self._running = True
        self._connection_task = asyncio.create_task(self._connection_loop())
        logger.info(f"Sync client started: {self.sync_url}")

    async def stop(self) -> None:
        """Stop the sync client."""
        self._running = False

        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
            self._connection_task = None

        logger.info("Sync client stopped")

    async def subscribe(self, session_id: str) -> None:
        """Subscribe to updates for a session.

        Args:
            session_id: Session ID to subscribe to
        """
        self._subscribed_sessions.add(session_id)
        logger.debug(f"Subscribed to session: {session_id}")

    async def unsubscribe(self, session_id: str) -> None:
        """Unsubscribe from a session.

        Args:
            session_id: Session ID to unsubscribe from
        """
        self._subscribed_sessions.discard(session_id)
        logger.debug(f"Unsubscribed from session: {session_id}")

    def is_connected(self) -> bool:
        """Check if the client is connected."""
        return self._running and self._connection_task is not None

    async def _connection_loop(self) -> None:
        """Main connection loop with auto-reconnect."""
        while self._running:
            try:
                if self.use_websocket:
                    await self._websocket_loop()
                else:
                    await self._sse_loop()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync connection error: {e}")
                if self.on_error:
                    self.on_error(str(e))

                if self.auto_reconnect and self._running:
                    logger.info(f"Reconnecting in {self.reconnect_delay}s...")
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    break

    async def _sse_loop(self) -> None:
        """SSE connection loop."""
        try:
            import aiohttp  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "aiohttp required for SSE sync. Install with: pip install aiohttp"
            ) from e

        headers = {"Accept": "text/event-stream"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        # Build subscription URL
        sessions_param = ",".join(self._subscribed_sessions)
        url = f"{self.sync_url}/events?user_id={self.user_id}&sessions={sessions_param}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    raise StorageConnectionError(f"SSE connection failed: {response.status}")

                if self.on_connected:
                    self.on_connected()

                async for event in self._parse_sse_stream(response.content):
                    await self._handle_event(event)

    async def _websocket_loop(self) -> None:
        """WebSocket connection loop."""
        try:
            import aiohttp  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "aiohttp required for WebSocket sync. Install with: pip install aiohttp"
            ) from e

        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        url = f"{self.sync_url}/ws?user_id={self.user_id}"

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url, headers=headers) as ws:
                if self.on_connected:
                    self.on_connected()

                # Send subscription message
                await ws.send_json(
                    {
                        "type": "subscribe",
                        "sessions": list(self._subscribed_sessions),
                    }
                )

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        event = self._parse_event(data)
                        await self._handle_event(event)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        raise StorageConnectionError(f"WebSocket error: {ws.exception()}")

    async def _parse_sse_stream(
        self,
        content: Any,
    ) -> AsyncIterator[SyncEvent]:
        """Parse SSE stream into events."""
        buffer = ""

        async for chunk in content.iter_any():
            buffer += chunk.decode("utf-8")

            while "\n\n" in buffer:
                event_str, buffer = buffer.split("\n\n", 1)
                event = self._parse_sse_event(event_str)
                if event:
                    yield event

    def _parse_sse_event(self, event_str: str) -> SyncEvent | None:
        """Parse a single SSE event."""
        event_type = None
        data = None

        for line in event_str.split("\n"):
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()

        if not data:
            return None

        try:
            parsed = json.loads(data)
            return self._parse_event(parsed, event_type)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse SSE data: {data}")
            return None

    def _parse_event(
        self,
        data: dict[str, Any],
        event_type_str: str | None = None,
    ) -> SyncEvent:
        """Parse event data into SyncEvent."""
        event_type_str = event_type_str or data.get("type", "block_added")

        try:
            event_type = SyncEventType(event_type_str)
        except ValueError:
            event_type = SyncEventType.BLOCK_ADDED

        block = None
        if "block" in data:
            block = SessionBlock.from_dict(data["block"])

        return SyncEvent(
            event_type=event_type,
            session_id=data.get("session_id"),
            block=block,
            data=data,
            error=data.get("error"),
        )

    async def _handle_event(self, event: SyncEvent) -> None:
        """Handle a received sync event."""
        # Filter by subscribed sessions
        if event.session_id and event.session_id not in self._subscribed_sessions:
            return

        if event.event_type == SyncEventType.BLOCK_ADDED:
            if event.block:
                # Write block to local storage
                await self.storage.write_block(event.block)

                if self.on_block_added:
                    self.on_block_added(event.block)

        elif event.event_type == SyncEventType.CONFLICT:
            if self.on_conflict:
                self.on_conflict(event)

        elif event.event_type == SyncEventType.SYNC_ERROR:
            if self.on_error:
                self.on_error(event.error or "Unknown sync error")

        elif event.event_type == SyncEventType.DISCONNECTED:
            if self.on_disconnected:
                self.on_disconnected()
