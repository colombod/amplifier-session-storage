"""Tests for embedding cache functionality."""

import pytest

from amplifier_session_storage.embeddings.cache import EmbeddingCache


class TestEmbeddingCache:
    """Tests for LRU embedding cache."""

    def test_basic_get_put(self):
        """Basic cache operations work."""
        cache = EmbeddingCache(max_entries=10)

        # Put and get
        embedding = [0.1, 0.2, 0.3]
        cache.put("hello", "model-1", embedding)

        result = cache.get("hello", "model-1")
        assert result == embedding

    def test_cache_miss(self):
        """Cache miss returns None."""
        cache = EmbeddingCache(max_entries=10)

        result = cache.get("nonexistent", "model-1")
        assert result is None

    def test_different_models_separate_cache(self):
        """Same text with different models are cached separately."""
        cache = EmbeddingCache(max_entries=10)

        embedding1 = [0.1, 0.2, 0.3]
        embedding2 = [0.4, 0.5, 0.6]

        cache.put("hello", "model-1", embedding1)
        cache.put("hello", "model-2", embedding2)

        assert cache.get("hello", "model-1") == embedding1
        assert cache.get("hello", "model-2") == embedding2

    def test_lru_eviction(self):
        """Least recently used entries are evicted when full."""
        cache = EmbeddingCache(max_entries=3)

        # Fill cache
        cache.put("a", "model", [1.0])
        cache.put("b", "model", [2.0])
        cache.put("c", "model", [3.0])

        # All should be present
        assert cache.size() == 3

        # Add one more - should evict "a" (oldest)
        cache.put("d", "model", [4.0])

        assert cache.size() == 3
        assert cache.get("a", "model") is None  # Evicted
        assert cache.get("b", "model") == [2.0]
        assert cache.get("c", "model") == [3.0]
        assert cache.get("d", "model") == [4.0]

    def test_lru_update_on_access(self):
        """Accessing an entry moves it to most recent."""
        cache = EmbeddingCache(max_entries=3)

        cache.put("a", "model", [1.0])
        cache.put("b", "model", [2.0])
        cache.put("c", "model", [3.0])

        # Access "a" - moves it to most recent
        cache.get("a", "model")

        # Add "d" - should evict "b" (now oldest)
        cache.put("d", "model", [4.0])

        assert cache.get("a", "model") == [1.0]  # Still present
        assert cache.get("b", "model") is None  # Evicted
        assert cache.get("c", "model") == [3.0]
        assert cache.get("d", "model") == [4.0]

    def test_clear(self):
        """Clear removes all entries."""
        cache = EmbeddingCache(max_entries=10)

        cache.put("a", "model", [1.0])
        cache.put("b", "model", [2.0])

        cache.clear()

        assert cache.size() == 0
        assert cache.get("a", "model") is None
        assert cache.get("b", "model") is None

    def test_stats(self):
        """Stats return correct information."""
        cache = EmbeddingCache(max_entries=10)

        stats = cache.stats()
        assert stats["size"] == 0
        assert stats["max_entries"] == 10
        assert stats["utilization"] == 0.0

        cache.put("a", "model", [1.0])
        cache.put("b", "model", [2.0])

        stats = cache.stats()
        assert stats["size"] == 2
        assert stats["utilization"] == 0.2  # 2/10

    def test_invalid_max_entries(self):
        """Invalid max_entries raises ValueError."""
        with pytest.raises(ValueError, match="max_entries must be >= 1"):
            EmbeddingCache(max_entries=0)

        with pytest.raises(ValueError, match="max_entries must be >= 1"):
            EmbeddingCache(max_entries=-1)

    def test_content_hashing(self):
        """Different texts are cached separately."""
        cache = EmbeddingCache(max_entries=10)

        cache.put("hello", "model", [1.0])
        cache.put("world", "model", [2.0])

        assert cache.get("hello", "model") == [1.0]
        assert cache.get("world", "model") == [2.0]

    def test_overwrite_existing(self):
        """Putting same key updates the value."""
        cache = EmbeddingCache(max_entries=10)

        cache.put("hello", "model", [1.0])
        cache.put("hello", "model", [2.0])  # Overwrite

        assert cache.get("hello", "model") == [2.0]
        assert cache.size() == 1  # Still only one entry
