"""
Maximum Marginal Relevance (MMR) algorithm for diverse result selection.

Ported from C# implementation: https://github.com/AIGeekSquad/AIContext
Original paper: Carbonell and Goldstein (1998)

MMR Formula:
    MMR = λ × Sim(Di, Q) - (1-λ) × max(Sim(Di, Dj))

Where:
    Di: Candidate document/vector
    Q: Query vector
    Dj: Already selected documents
    λ: Relevance-diversity trade-off parameter

Lambda values:
    1.0: Pure relevance (select most similar to query, ignore diversity)
    0.7-0.9: Relevance-focused with some diversity
    0.5: Balanced (recommended default)
    0.1-0.3: Diversity-focused with some relevance
    0.0: Pure diversity (maximize variety, ignore query relevance)

Use cases:
    - RAG systems: Diverse context chunks (avoid redundancy)
    - Search results: Balance relevant and diverse results
    - Recommendation systems: Suggest variety not just top matches
    - Session search: Find related but non-redundant conversations
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cosine_similarity(a: NDArray, b: NDArray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Similarity score in [0, 1] where 1 is identical direction

    Note:
        Returns 0.0 for zero vectors to avoid division by zero
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))  # type: ignore


def compute_mmr(
    vectors: list[NDArray],
    query: NDArray,
    lambda_param: float = 0.5,
    top_k: int | None = None,
) -> list[tuple[int, NDArray]]:
    """
    Compute Maximum Marginal Relevance selection from vectors.

    Iteratively selects vectors that maximize a weighted combination of:
    1. Relevance: Similarity to query vector
    2. Diversity: Dissimilarity to already selected vectors

    Args:
        vectors: List of candidate vectors to select from
        query: Query vector representing search intent
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0)
                     Default 0.5 for balanced approach
        top_k: Number of results to select (None = all vectors)

    Returns:
        List of (index, vector) tuples in selection order.
        Index refers to position in original vectors list.

    Raises:
        ValueError: If lambda_param not in [0.0, 1.0] or dimension mismatch

    Time Complexity:
        O(n²k) where n = len(vectors), k = top_k

    Space Complexity:
        O(n) for similarity caching

    Examples:
        >>> import numpy as np
        >>> vectors = [
        ...     np.array([1.0, 0.0, 0.0]),
        ...     np.array([0.9, 0.1, 0.0]),  # Similar to first
        ...     np.array([0.0, 1.0, 0.0]),  # Orthogonal
        ... ]
        >>> query = np.array([1.0, 0.0, 0.0])
        >>> results = compute_mmr(vectors, query, lambda_param=0.7, top_k=2)
        >>> # With lambda=0.7, prefers relevance but includes diversity
        >>> # Likely selects index 0 (most relevant) and 2 (diverse)
    """
    # Validate parameters
    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError(f"lambda_param must be in [0.0, 1.0], got {lambda_param}")

    if query is None:
        raise ValueError("Query vector cannot be None")

    if not vectors:
        return []

    # Validate dimensions
    query_dims = len(query)
    for i, vec in enumerate(vectors):
        if len(vec) != query_dims:
            raise ValueError(
                f"Vector at index {i} has {len(vec)} dimensions, "
                f"but query has {query_dims} dimensions. "
                f"All vectors must have the same dimensionality."
            )

    # Handle edge cases
    k = min(top_k if top_k is not None else len(vectors), len(vectors))
    if k <= 0:
        return []
    if k >= len(vectors):
        # Return all vectors in original order
        return [(i, vec) for i, vec in enumerate(vectors)]

    # Precompute query similarities once (optimization from C# version)
    query_similarities = np.array([cosine_similarity(vec, query) for vec in vectors])

    # Execute MMR selection
    selected_indices = _execute_mmr_selection(
        vectors=vectors,
        query_similarities=query_similarities,
        lambda_param=lambda_param,
        k=k,
    )

    return [(i, vectors[i]) for i in selected_indices]


def _execute_mmr_selection(
    vectors: list[NDArray],
    query_similarities: NDArray,
    lambda_param: float,
    k: int,
) -> list[int]:
    """
    Execute the iterative MMR selection process.

    Args:
        vectors: List of candidate vectors
        query_similarities: Precomputed similarities to query
        lambda_param: Relevance-diversity trade-off
        k: Number of items to select

    Returns:
        List of selected indices in selection order
    """
    selected_indices: list[int] = []
    remaining_mask = np.ones(len(vectors), dtype=bool)

    for _ in range(k):
        best_idx = _find_best_mmr_candidate(
            vectors=vectors,
            query_similarities=query_similarities,
            selected_indices=selected_indices,
            remaining_mask=remaining_mask,
            lambda_param=lambda_param,
        )

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        remaining_mask[best_idx] = False

    return selected_indices


def _find_best_mmr_candidate(
    vectors: list[NDArray],
    query_similarities: NDArray,
    selected_indices: list[int],
    remaining_mask: NDArray,
    lambda_param: float,
) -> int:
    """
    Find the best MMR candidate from remaining vectors.

    Args:
        vectors: All candidate vectors
        query_similarities: Precomputed similarities to query
        selected_indices: Already selected vector indices
        remaining_mask: Boolean mask of remaining candidates
        lambda_param: Relevance-diversity trade-off

    Returns:
        Index of best candidate, or -1 if none available
    """
    best_idx = -1
    best_score = float("-inf")

    for i in range(len(vectors)):
        if not remaining_mask[i]:
            continue

        mmr_score = _calculate_mmr_score(
            vectors=vectors,
            query_similarities=query_similarities,
            selected_indices=selected_indices,
            candidate_idx=i,
            lambda_param=lambda_param,
        )

        if mmr_score > best_score:
            best_score = mmr_score
            best_idx = i

    return best_idx


def _calculate_mmr_score(
    vectors: list[NDArray],
    query_similarities: NDArray,
    selected_indices: list[int],
    candidate_idx: int,
    lambda_param: float,
) -> float:
    """
    Calculate MMR score for a candidate vector.

    MMR = λ × Sim(Di, Q) - (1-λ) × max(Sim(Di, Dj))

    Args:
        vectors: All vectors
        query_similarities: Precomputed query similarities
        selected_indices: Already selected indices
        candidate_idx: Index of candidate to score
        lambda_param: Relevance-diversity trade-off

    Returns:
        MMR score (higher is better)
    """
    # Relevance component: similarity to query
    relevance_score = lambda_param * query_similarities[candidate_idx]

    # Diversity component: dissimilarity to selected documents
    diversity_score = _calculate_diversity_score(
        vectors=vectors,
        selected_indices=selected_indices,
        candidate_idx=candidate_idx,
        lambda_param=lambda_param,
    )

    return relevance_score + diversity_score


def _calculate_diversity_score(
    vectors: list[NDArray],
    selected_indices: list[int],
    candidate_idx: int,
    lambda_param: float,
) -> float:
    """
    Calculate diversity score component for MMR.

    Diversity = (1-λ) × (1 - avg_similarity_to_selected)

    Args:
        vectors: All vectors
        selected_indices: Already selected indices
        candidate_idx: Index of candidate
        lambda_param: Relevance-diversity trade-off

    Returns:
        Diversity score contribution
    """
    if len(selected_indices) == 0:
        # First selection: only relevance matters, give full diversity bonus
        return 1.0 - lambda_param

    # Calculate average similarity to already selected documents
    similarities = [cosine_similarity(vectors[candidate_idx], vectors[j]) for j in selected_indices]
    avg_similarity = np.mean(similarities)

    # Higher average similarity = lower diversity score
    # We want to penalize candidates similar to already selected docs
    return (1.0 - lambda_param) * (1.0 - avg_similarity)
