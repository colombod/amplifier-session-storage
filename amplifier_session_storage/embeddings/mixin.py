"""Shared embedding logic for storage backends.

Extracted from DuckDB, Cosmos, and SQLite backends to eliminate duplication.
All three backends had functionally identical _embed_non_none() and
_generate_multi_vector_embeddings() methods.
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

logger = logging.getLogger(__name__)


class EmbeddingMixin:
    """Mixin providing shared embedding generation logic for storage backends.

    Backends using this mixin must set ``self.embedding_provider: EmbeddingProvider | None``
    in their constructor.
    """

    embedding_provider: EmbeddingProvider | None

    async def _embed_non_none(self, texts: list[str | None]) -> list[list[float] | None]:
        """Generate embeddings only for non-None texts.

        Applies token-limit truncation as a safety net to prevent embedding API
        failures when text exceeds the model's context window (8192 tokens for
        text-embedding-3-large).

        Args:
            texts: List of text strings (some may be None)

        Returns:
            List of embeddings with None preserved at same indices
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
        safe_texts: list[str] = []
        for _, text in texts_to_embed:
            token_count = count_tokens(text)
            if token_count > EMBED_TOKEN_LIMIT:
                logger.warning(
                    f"Text exceeds embedding token limit "
                    f"({token_count} > {EMBED_TOKEN_LIMIT} tokens), "
                    f"truncating for embedding. "
                    f"First 100 chars: {text[:100]!r}"
                )
                text = truncate_to_tokens(text, EMBED_TOKEN_LIMIT)
            safe_texts.append(text)

        embeddings = await self.embedding_provider.embed_batch(safe_texts)

        results: list[list[float] | None] = [None] * len(texts)
        for (idx, _), embedding in zip(texts_to_embed, embeddings, strict=True):
            results[idx] = embedding

        return results

    async def _generate_multi_vector_embeddings(
        self, lines: list[dict[str, Any]]
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
            "user_query": await self._embed_non_none(user_queries),
            "assistant_response": await self._embed_non_none(assistant_responses),
            "assistant_thinking": await self._embed_non_none(assistant_thinking),
            "tool_output": await self._embed_non_none(tool_outputs),
        }
