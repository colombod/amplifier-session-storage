"""
Embedding provider abstraction and implementations.

Provides:
- Abstract EmbeddingProvider interface
- Azure OpenAI implementation
- OpenAI direct implementation
- LRU cache for hot query embeddings
"""

from .base import EmbeddingProvider
from .cache import EmbeddingCache

__all__ = [
    "EmbeddingProvider",
    "EmbeddingCache",
]
