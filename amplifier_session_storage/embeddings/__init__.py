"""
Embedding provider abstraction and implementations.

Provides:
- Abstract EmbeddingProvider interface
- Azure OpenAI implementation
- OpenAI direct implementation
- LRU cache for hot query embeddings
- Resilience utilities (retry, circuit breaker, batch splitting)
"""

from .base import EmbeddingProvider
from .cache import EmbeddingCache
from .resilience import CircuitBreaker, CircuitOpenError, RetryConfig

__all__ = [
    "EmbeddingProvider",
    "EmbeddingCache",
    "CircuitBreaker",
    "CircuitOpenError",
    "RetryConfig",
]
