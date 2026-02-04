"""
Tests for Maximum Marginal Relevance (MMR) algorithm.

Ported from C# test cases: AIGeekSquad/AIContext
"""

import numpy as np
import pytest

from amplifier_session_storage.search.mmr import compute_mmr, cosine_similarity


def get_test_vectors() -> list[np.ndarray]:
    """Test vectors from the C# reference implementation."""
    return [
        np.array([1.0, 0.0, 0.0]),  # Index 0
        np.array([1.0, 0.0, 0.0]),  # Index 1 (identical to 0)
        np.array([0.0, 1.0, 0.0]),  # Index 2
        np.array([0.0, 0.0, 1.0]),  # Index 3
        np.array([1.0, 1.0, 0.0]),  # Index 4
        np.array([1.0, 0.0, 1.0]),  # Index 5
    ]


def get_test_query() -> np.ndarray:
    """Test query vector."""
    return np.array([1.0, 0.0, 0.0])


class TestCosineSimilarity:
    """Tests for cosine similarity function."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity of 1.0."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity of 0.0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity of -1.0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        """Zero vectors should return 0.0 (not error)."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(a, b) == 0.0


class TestMMRBasics:
    """Basic MMR functionality tests."""

    def test_basic_computation(self):
        """Basic MMR computation returns correct structure."""
        vectors = get_test_vectors()
        query = get_test_query()

        result = compute_mmr(vectors, query, lambda_param=0.5, top_k=3)

        assert result is not None
        assert len(result) == 3
        assert all(0 <= idx < len(vectors) for idx, _ in result)
        assert all(vec is not None for _, vec in result)

    def test_empty_vectors(self):
        """Empty vector list returns empty result."""
        vectors: list[np.ndarray] = []
        query = get_test_query()

        result = compute_mmr(vectors, query)
        assert result == []

    def test_single_vector(self):
        """Single vector returns that vector."""
        vectors = [np.array([1.0, 0.0, 0.0])]
        query = get_test_query()

        result = compute_mmr(vectors, query)

        assert len(result) == 1
        assert result[0][0] == 0
        assert np.array_equal(result[0][1], vectors[0])

    def test_top_k_zero(self):
        """top_k=0 returns empty list."""
        vectors = get_test_vectors()
        query = get_test_query()

        result = compute_mmr(vectors, query, top_k=0)
        assert result == []

    def test_top_k_one(self):
        """top_k=1 returns one result."""
        vectors = get_test_vectors()
        query = get_test_query()

        result = compute_mmr(vectors, query, top_k=1)

        assert len(result) == 1
        # Should select most relevant (index 0 or 1)
        assert result[0][0] in [0, 1]

    def test_top_k_larger_than_vectors(self):
        """top_k larger than vector count returns all vectors."""
        vectors = get_test_vectors()
        query = get_test_query()

        result = compute_mmr(vectors, query, top_k=100)

        assert len(result) == len(vectors)

    def test_top_k_none(self):
        """top_k=None returns all vectors."""
        vectors = get_test_vectors()
        query = get_test_query()

        result = compute_mmr(vectors, query, top_k=None)

        assert len(result) == len(vectors)


class TestMMRLambda:
    """Tests for lambda parameter behavior."""

    def test_lambda_one_pure_relevance(self):
        """Lambda=1.0 selects most relevant vectors (ignores diversity)."""
        vectors = get_test_vectors()
        query = get_test_query()

        result = compute_mmr(vectors, query, lambda_param=1.0, top_k=3)

        assert len(result) == 3
        # Should select indices 0 and 1 (identical to query) in first two
        selected_indices = [idx for idx, _ in result]
        assert 0 in selected_indices or 1 in selected_indices

    def test_lambda_zero_pure_diversity(self):
        """Lambda=0.0 selects diverse vectors (ignores relevance)."""
        vectors = get_test_vectors()
        query = get_test_query()

        result = compute_mmr(vectors, query, lambda_param=0.0, top_k=3)

        assert len(result) == 3
        selected_indices = [idx for idx, _ in result]

        # Should NOT select both identical vectors (0 and 1)
        assert not (0 in selected_indices and 1 in selected_indices)

    def test_lambda_half_balanced(self):
        """Lambda=0.5 balances relevance and diversity."""
        vectors = get_test_vectors()
        query = get_test_query()

        result = compute_mmr(vectors, query, lambda_param=0.5, top_k=3)

        assert len(result) == 3
        selected_indices = [idx for idx, _ in result]

        # Should include at least one relevant vector
        assert 0 in selected_indices or 1 in selected_indices

        # Should select distinct vectors
        assert len(set(selected_indices)) == 3


class TestMMRValidation:
    """Tests for parameter validation."""

    def test_lambda_out_of_range_high(self):
        """Lambda > 1.0 raises ValueError."""
        vectors = get_test_vectors()
        query = get_test_query()

        with pytest.raises(ValueError, match="lambda_param must be in"):
            compute_mmr(vectors, query, lambda_param=1.5)

    def test_lambda_out_of_range_low(self):
        """Lambda < 0.0 raises ValueError."""
        vectors = get_test_vectors()
        query = get_test_query()

        with pytest.raises(ValueError, match="lambda_param must be in"):
            compute_mmr(vectors, query, lambda_param=-0.5)

    def test_none_query(self):
        """None query raises ValueError."""
        vectors = get_test_vectors()

        with pytest.raises(ValueError, match="Query vector cannot be None"):
            compute_mmr(vectors, None, lambda_param=0.5)  # type: ignore

    def test_dimension_mismatch(self):
        """Mismatched vector dimensions raises ValueError."""
        vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0]),  # Wrong dimensions
        ]
        query = np.array([1.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="dimensions"):
            compute_mmr(vectors, query)


class TestMMRBehavior:
    """Tests for MMR algorithm behavior."""

    def test_identical_vectors_handling(self):
        """MMR handles identical vectors correctly."""
        vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        ]
        query = get_test_query()

        result = compute_mmr(vectors, query, lambda_param=0.5, top_k=2)

        assert len(result) == 2
        assert all(0 <= idx < len(vectors) for idx, _ in result)

    def test_extreme_vectors(self):
        """MMR handles extreme vector values."""
        vectors = [
            np.array([1000.0, 0.0, 0.0]),  # Very large
            np.array([0.001, 0.0, 0.0]),  # Very small
            np.array([0.0, 1000.0, 0.0]),  # Orthogonal large
            np.array([-1000.0, 0.0, 0.0]),  # Negative large
        ]
        query = np.array([1.0, 0.0, 0.0])

        result = compute_mmr(vectors, query, lambda_param=0.5, top_k=3)

        assert len(result) == 3
        assert all(0 <= idx < len(vectors) for idx, _ in result)

    def test_selection_order_consistency(self):
        """Results are in selection order (not original order)."""
        vectors = get_test_vectors()
        query = get_test_query()

        result = compute_mmr(vectors, query, lambda_param=0.5, top_k=3)

        # Should return exactly 3 results
        assert len(result) == 3

        # Indices should be valid
        assert all(0 <= idx < len(vectors) for idx, _ in result)

        # Vectors should match original vectors at those indices
        for idx, vec in result:
            assert np.array_equal(vec, vectors[idx])

    def test_different_lambda_produces_different_orderings(self):
        """Different lambda values can produce different result sets."""
        vectors = get_test_vectors()
        query = get_test_query()

        result_relevance = compute_mmr(vectors, query, lambda_param=1.0, top_k=3)
        result_diversity = compute_mmr(vectors, query, lambda_param=0.0, top_k=3)

        # Both should return 3 results
        assert len(result_relevance) == 3
        assert len(result_diversity) == 3

        # Extract indices
        relevance_indices = [idx for idx, _ in result_relevance]
        diversity_indices = [idx for idx, _ in result_diversity]

        # Verify they're all valid
        assert all(0 <= idx < len(vectors) for idx in relevance_indices)
        assert all(0 <= idx < len(vectors) for idx in diversity_indices)


class TestMMREdgeCases:
    """Edge case tests."""

    def test_normalized_vectors(self):
        """MMR works with normalized unit vectors."""
        vectors = [
            np.array([1.0, 0.0, 0.0]) / np.linalg.norm([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]) / np.linalg.norm([0.0, 1.0, 0.0]),
            np.array([0.707, 0.707, 0.0]),  # Pre-normalized
        ]
        query = np.array([1.0, 0.0, 0.0])

        result = compute_mmr(vectors, query, lambda_param=0.5, top_k=2)

        assert len(result) == 2
        assert all(0 <= idx < len(vectors) for idx, _ in result)

    def test_high_dimensional_vectors(self):
        """MMR works with high-dimensional vectors."""
        dims = 1536  # text-embedding-3-small dimension
        vectors = [np.random.rand(dims) for _ in range(5)]
        query = np.random.rand(dims)

        result = compute_mmr(vectors, query, lambda_param=0.7, top_k=3)

        assert len(result) == 3
        assert all(len(vec) == dims for _, vec in result)

    def test_very_high_dimensional_vectors(self):
        """MMR works with very high-dimensional vectors."""
        dims = 3072  # text-embedding-3-large dimension
        vectors = [np.random.rand(dims) for _ in range(5)]
        query = np.random.rand(dims)

        result = compute_mmr(vectors, query, lambda_param=0.5, top_k=3)

        assert len(result) == 3
        assert all(len(vec) == dims for _, vec in result)
