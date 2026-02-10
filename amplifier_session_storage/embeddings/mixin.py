"""Shared embedding logic for storage backends.

Extracted from DuckDB, Cosmos, and SQLite backends to eliminate duplication.
All three backends had functionally identical _embed_non_none() and
_generate_multi_vector_embeddings() methods.

Resilience features:
- Batch splitting (EMBED_BATCH_SIZE, default 16)
- Retry with exponential backoff (especially for 429 rate limits)
- Circuit breaker to fail fast when the embedding service is down
- Per-batch error isolation (one failed batch doesn't kill the rest)
"""

from __future__ import annotations

import logging
from typing import Any

from ..content_extraction import (
    EMBED_TOKEN_LIMIT,
    count_tokens,
    extract_all_embeddable_content,
    truncate_to_tokens,
)
from . import EmbeddingProvider
from .resilience import (
    EMBED_BATCH_SIZE,
    CircuitBreaker,
    CircuitOpenError,
    RetryConfig,
    retry_with_backoff,
)

logger = logging.getLogger(__name__)

# Module-level circuit breaker shared across all backends in this process.
# This is intentional — if the embedding service is down, ALL backends
# should fail fast rather than each discovering the outage independently.
_circuit_breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60.0)

# Default retry config for embedding calls
_retry_config = RetryConfig(
    max_retries=5,
    backoff_base=1.0,
    backoff_max=60.0,
    backoff_multiplier=2.0,
)


class EmbeddingMixin:
    """Mixin providing shared embedding generation logic for storage backends.

    Backends using this mixin must set ``self.embedding_provider: EmbeddingProvider | None``
    in their constructor.

    Resilience:
        - Splits large batches into chunks of EMBED_BATCH_SIZE (16)
        - Retries transient failures with exponential backoff (429, 5xx, network)
        - Circuit breaker fails fast when the service is down
        - Per-batch isolation: a failed batch returns None for those positions
    """

    embedding_provider: EmbeddingProvider | None

    async def _embed_non_none(
        self,
        texts: list[str | None],
        *,
        context_msg: str = "",
    ) -> list[list[float] | None]:
        """Generate embeddings only for non-None texts with resilience.

        Applies token-limit truncation as a safety net to prevent embedding API
        failures when text exceeds the model's context window (8192 tokens for
        text-embedding-3-large).

        Splits into batches of EMBED_BATCH_SIZE, retries transient failures,
        and isolates per-batch errors so a single API failure doesn't lose
        the entire set.

        Args:
            texts: List of text strings (some may be None)
            context_msg: Extra context for log messages (e.g. session_id)

        Returns:
            List of embeddings with None preserved at same indices.
            Positions whose batch failed will also be None.
        """
        if not self.embedding_provider:
            return [None] * len(texts)

        texts_to_embed: list[tuple[int, str]] = []
        for i, text in enumerate(texts):
            if text is not None:
                texts_to_embed.append((i, text))

        if not texts_to_embed:
            return [None] * len(texts)

        # Apply safety truncation for token limits
        safe_entries: list[tuple[int, str]] = []
        for idx, text in texts_to_embed:
            token_count = count_tokens(text)
            if token_count > EMBED_TOKEN_LIMIT:
                logger.warning(
                    f"Text exceeds embedding token limit "
                    f"({token_count} > {EMBED_TOKEN_LIMIT} tokens), "
                    f"truncating for embedding. "
                    f"First 100 chars: {text[:100]!r}"
                )
                text = truncate_to_tokens(text, EMBED_TOKEN_LIMIT)
            safe_entries.append((idx, text))

        # Split into batches of EMBED_BATCH_SIZE
        results: list[list[float] | None] = [None] * len(texts)
        total_embedded = 0
        total_failed = 0

        for batch_start in range(0, len(safe_entries), EMBED_BATCH_SIZE):
            batch = safe_entries[batch_start : batch_start + EMBED_BATCH_SIZE]
            batch_texts = [text for _, text in batch]
            batch_num = batch_start // EMBED_BATCH_SIZE + 1
            total_batches = (len(safe_entries) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

            try:
                embeddings = await retry_with_backoff(
                    self.embedding_provider.embed_batch,
                    batch_texts,
                    config=_retry_config,
                    circuit=_circuit_breaker,
                    context_msg=f"batch {batch_num}/{total_batches} "
                    f"({len(batch_texts)} texts){f' {context_msg}' if context_msg else ''}",
                )

                for (idx, _), embedding in zip(batch, embeddings, strict=True):
                    results[idx] = embedding
                total_embedded += len(batch)

            except CircuitOpenError:
                # Circuit is open — skip remaining batches entirely
                remaining = len(safe_entries) - batch_start
                logger.error(
                    f"Circuit breaker OPEN — skipping {remaining} remaining texts "
                    f"(batch {batch_num}/{total_batches})"
                    f"{f' {context_msg}' if context_msg else ''}"
                )
                total_failed += remaining
                break

            except Exception as exc:
                # This batch failed after all retries — log and continue
                total_failed += len(batch)
                logger.error(
                    f"Embedding batch {batch_num}/{total_batches} failed after retries "
                    f"({len(batch_texts)} texts lost)"
                    f"{f' {context_msg}' if context_msg else ''}: {exc}"
                )

        if total_failed > 0:
            logger.warning(
                f"Embedding completed with failures: "
                f"{total_embedded} succeeded, {total_failed} failed out of "
                f"{len(safe_entries)} texts"
                f"{f' {context_msg}' if context_msg else ''}"
            )
        elif total_embedded > 0:
            logger.info(
                f"Embedded {total_embedded} texts in "
                f"{(len(safe_entries) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE} batches"
                f"{f' {context_msg}' if context_msg else ''}"
            )

        return results

    async def _generate_multi_vector_embeddings(
        self, lines: list[dict[str, Any]], *, context_msg: str = ""
    ) -> dict[str, list[list[float] | None]]:
        """Generate embeddings for all content types in a batch.

        Extracts embeddable content from each message and generates embeddings
        for each content type (user_query, assistant_response, assistant_thinking,
        tool_output) separately.

        Returns empty dict if no embedding provider is configured.
        """
        if not self.embedding_provider:
            return {}

        user_queries: list[str | None] = []
        assistant_responses: list[str | None] = []
        assistant_thinking: list[str | None] = []
        tool_outputs: list[str | None] = []

        for line in lines:
            extracted = extract_all_embeddable_content(line)
            user_queries.append(extracted["user_query"])
            assistant_responses.append(extracted["assistant_response"])
            assistant_thinking.append(extracted["assistant_thinking"])
            tool_outputs.append(extracted["tool_output"])

        return {
            "user_query": await self._embed_non_none(
                user_queries, context_msg=f"user_query {context_msg}".strip()
            ),
            "assistant_response": await self._embed_non_none(
                assistant_responses, context_msg=f"assistant_response {context_msg}".strip()
            ),
            "assistant_thinking": await self._embed_non_none(
                assistant_thinking, context_msg=f"assistant_thinking {context_msg}".strip()
            ),
            "tool_output": await self._embed_non_none(
                tool_outputs, context_msg=f"tool_output {context_msg}".strip()
            ),
        }


def get_circuit_breaker_stats() -> dict[str, Any]:
    """Expose circuit breaker stats for monitoring/health checks."""
    return _circuit_breaker.stats()
