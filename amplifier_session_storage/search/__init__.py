"""
Search capabilities for session storage.

Provides:
- Full-text search (keyword-based)
- Semantic search (embedding-based)
- Hybrid search (combines both with MMR re-ranking)
- MMR re-ranking algorithm
"""

from .mmr import compute_mmr, cosine_similarity

__all__ = [
    "compute_mmr",
    "cosine_similarity",
]
