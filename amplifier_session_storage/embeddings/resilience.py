"""Resilience utilities for embedding API calls.

Provides retry with exponential backoff (especially for 429 rate limits)
and a circuit breaker to fail fast when the embedding service is down.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default batch size for embedding API calls.
# Keeps individual requests small to reduce blast radius on failure
# and stay within typical API rate-limit windows.
EMBED_BATCH_SIZE = 16


@dataclass
class RetryConfig:
    """Configuration for retry with exponential backoff."""

    max_retries: int = 5
    backoff_base: float = 1.0  # seconds
    backoff_max: float = 60.0  # cap
    backoff_multiplier: float = 2.0
    retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)
    retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )


class CircuitState(Enum):
    CLOSED = "closed"  # Normal — requests flow through
    OPEN = "open"  # Tripped — requests fail fast
    HALF_OPEN = "half_open"  # Probing — one request allowed to test recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for embedding API calls.

    Tracks consecutive failures. When the threshold is reached, the circuit
    opens and all calls fail fast for ``reset_timeout`` seconds. After that,
    a single probe request is allowed (half-open). If it succeeds the circuit
    closes; if it fails, the circuit re-opens.
    """

    failure_threshold: int = 5
    reset_timeout: float = 60.0  # seconds before half-open probe

    # Internal state (not constructor args)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False, repr=False)
    _failure_count: int = field(default=0, init=False, repr=False)
    _last_failure_time: float = field(default=0.0, init=False, repr=False)
    _total_trips: int = field(default=0, init=False, repr=False)

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.reset_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN — allowing probe request")
        return self._state

    def record_success(self) -> None:
        if self._state != CircuitState.CLOSED:
            logger.info("Circuit breaker CLOSED — embedding service recovered")
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            if self._state != CircuitState.OPEN:
                self._total_trips += 1
                logger.warning(
                    f"Circuit breaker OPEN — {self._failure_count} consecutive failures "
                    f"(trip #{self._total_trips}), "
                    f"will probe again in {self.reset_timeout}s"
                )
            self._state = CircuitState.OPEN

    def allow_request(self) -> bool:
        state = self.state  # triggers timeout check
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return True  # allow exactly one probe
        return False

    def stats(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "total_trips": self._total_trips,
        }


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open and requests are rejected."""


def _extract_status_code(exc: Exception) -> int | None:
    """Try to extract an HTTP status code from common SDK exceptions."""
    # Azure SDK: HttpResponseError, azure.core.exceptions
    status = getattr(exc, "status_code", None)
    if status is not None:
        return int(status)
    # OpenAI SDK: APIStatusError
    status = getattr(exc, "status", None)  # openai.APIStatusError
    if status is not None:
        return int(status)
    # Nested response object
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None) or getattr(response, "status", None)
        if code is not None:
            return int(code)
    return None


def _get_retry_after(exc: Exception) -> float | None:
    """Extract Retry-After header value from an exception if present."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after is None:
        return None
    try:
        return float(retry_after)
    except (ValueError, TypeError):
        return None


async def retry_with_backoff(
    fn: Callable[..., Coroutine[Any, Any, T]],
    *args: Any,
    config: RetryConfig | None = None,
    circuit: CircuitBreaker | None = None,
    context_msg: str = "",
    **kwargs: Any,
) -> T:
    """Execute an async function with retry, backoff, and circuit breaker.

    Args:
        fn: Async callable to execute
        *args: Positional args for fn
        config: Retry configuration (uses defaults if None)
        circuit: Circuit breaker instance (skipped if None)
        context_msg: Extra context for log messages (e.g. session id)
        **kwargs: Keyword args for fn

    Returns:
        Result of fn

    Raises:
        CircuitOpenError: If circuit breaker is open
        Exception: Last exception after all retries exhausted
    """
    cfg = config or RetryConfig()
    ctx = f" [{context_msg}]" if context_msg else ""

    for attempt in range(cfg.max_retries + 1):
        # Circuit breaker check
        if circuit and not circuit.allow_request():
            raise CircuitOpenError(f"Circuit breaker is OPEN — service unavailable{ctx}")

        try:
            result = await fn(*args, **kwargs)
        except Exception as exc:
            status_code = _extract_status_code(exc)
            is_retryable = isinstance(exc, cfg.retryable_exceptions) or (
                status_code is not None and status_code in cfg.retryable_status_codes
            )

            # Only count retryable failures toward the circuit breaker.
            # Non-retryable errors (auth, config, bad input) are permanent
            # and should not trip the breaker.
            if circuit and is_retryable:
                circuit.record_failure()

            if not is_retryable or attempt >= cfg.max_retries:
                logger.error(
                    "RETRY_EXHAUSTED: attempt=%d/%d status=%s retryable=%s%s: %s",
                    attempt + 1,
                    cfg.max_retries + 1,
                    status_code,
                    is_retryable,
                    ctx,
                    exc,
                )
                raise

            # Calculate backoff
            retry_after = _get_retry_after(exc)
            if retry_after is not None:
                delay = min(retry_after, cfg.backoff_max)
            else:
                delay = min(
                    cfg.backoff_base * (cfg.backoff_multiplier**attempt),
                    cfg.backoff_max,
                )

            # Log throttling (429) distinctly from other transient errors
            if status_code == 429:
                logger.warning(
                    "THROTTLED: 429 Too Many Requests, attempt=%d/%d, retry_after=%.1fs%s: %s",
                    attempt + 1,
                    cfg.max_retries + 1,
                    delay,
                    ctx,
                    exc,
                )
            else:
                logger.warning(
                    "RETRYING: attempt=%d/%d status=%s delay=%.1fs%s: %s",
                    attempt + 1,
                    cfg.max_retries + 1,
                    status_code,
                    delay,
                    ctx,
                    exc,
                )
            await asyncio.sleep(delay)
        else:
            if attempt > 0:
                # Succeeded after retries — log recovery so operators know
                # throttling happened but was handled
                logger.warning(
                    "RETRY_RECOVERED: succeeded on attempt %d/%d after %d retries%s",
                    attempt + 1,
                    cfg.max_retries + 1,
                    attempt,
                    ctx,
                )
            if circuit:
                circuit.record_success()
            return result

    # Unreachable, but satisfies type checker
    raise RuntimeError("retry_with_backoff exhausted without raising")  # pragma: no cover
