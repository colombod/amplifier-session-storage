"""
LRU cache for embedding vectors with configurable size limits.

Provides hot cache for frequently used query embeddings to minimize API calls.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any


class EmbeddingCache:
    """
    LRU cache for embedding vectors with size control.

    Features:
    - Least Recently Used eviction policy
    - Configurable max entries
    - Content-based keying (hash of text)
    - Thread-safe for async usage
    """

    def __init__(self, max_entries: int = 1000):
        """
        Initialize embedding cache.

        Args:
            max_entries: Maximum number of embeddings to cache
        """
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")

        self.max_entries = max_entries
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    def _make_key(self, text: str, model_name: str) -> str:
        """
        Create cache key from text and model.

        Uses SHA256 hash to handle arbitrary text lengths and special characters.

        Args:
            text: Text to embed
            model_name: Model identifier

        Returns:
            Cache key string
        """
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, text: str, model_name: str) -> list[float] | None:
        """
        Get cached embedding if available.

        Args:
            text: Text to look up
            model_name: Model identifier

        Returns:
            Cached embedding vector or None if not found
        """
        key = self._make_key(text, model_name)

        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

        return None

    def put(self, text: str, model_name: str, embedding: list[float]) -> None:
        """
        Store embedding in cache.

        If cache is full, evicts least recently used entry.

        Args:
            text: Original text
            model_name: Model identifier
            embedding: Embedding vector to cache
        """
        key = self._make_key(text, model_name)

        # Remove if exists (will be re-added at end)
        if key in self._cache:
            del self._cache[key]

        # Add to cache (at end = most recent)
        self._cache[key] = embedding

        # Evict oldest if over limit
        if len(self._cache) > self.max_entries:
            # popitem(last=False) removes oldest (FIFO)
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()

    def size(self) -> int:
        """Get current number of cached embeddings."""
        return len(self._cache)

    def stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache metrics
        """
        return {
            "size": len(self._cache),
            "max_entries": self.max_entries,
            "utilization": len(self._cache) / self.max_entries if self.max_entries > 0 else 0.0,
        }
