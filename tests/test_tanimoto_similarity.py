"""Tests for Tanimoto similarity computation."""

import numpy as np

from chalcedon.tanimoto_similarity import TanimotoSimilarity, pairwise_tanimoto


def test_known_value() -> None:
    """Verify Tanimoto against hand-computed value."""
    fps = np.array([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=np.uint8)
    sim = pairwise_tanimoto(fps)
    np.testing.assert_almost_equal(sim[0, 1], 1 / 3)


def test_identical() -> None:
    """Identical fingerprints have similarity 1.0."""
    fps = np.array([[1, 0, 1, 0]] * 3, dtype=np.uint8)
    np.testing.assert_array_equal(pairwise_tanimoto(fps), np.ones((3, 3)))


def test_orthogonal() -> None:
    """Orthogonal fingerprints have zero pairwise similarity."""
    fps = np.eye(4, dtype=np.uint8)
    sim = pairwise_tanimoto(fps)
    np.testing.assert_array_equal(sim.diagonal(), np.ones(4))
    off = sim.copy()
    np.fill_diagonal(off, 0.0)
    np.testing.assert_array_equal(off, np.zeros((4, 4)))


def test_zero_fingerprint() -> None:
    """Zero fingerprints produce zero similarity."""
    fps = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    sim = pairwise_tanimoto(fps)
    assert sim[0, 1] == 0.0
    assert sim[1, 1] == 0.0


def test_symmetry() -> None:
    """Similarity matrix is symmetric."""
    rng = np.random.default_rng(42)
    fps = rng.integers(0, 2, size=(20, 64), dtype=np.uint8)
    sim = pairwise_tanimoto(fps)
    np.testing.assert_array_equal(sim, sim.T)


def test_chunk_matches_pairwise() -> None:
    """TanimotoSimilarity.chunk matches full pairwise matrix."""
    rng = np.random.default_rng(42)
    fps = rng.integers(0, 2, size=(50, 32), dtype=np.uint8)
    full = pairwise_tanimoto(fps)
    np.fill_diagonal(full, 0.0)

    sim = TanimotoSimilarity(fps)
    # test a chunk in the middle
    block = sim.chunk(10, 30)
    np.testing.assert_allclose(block, full[10:30], atol=1e-12)


def test_row_matches_pairwise() -> None:
    """TanimotoSimilarity.row matches full pairwise matrix."""
    rng = np.random.default_rng(42)
    fps = rng.integers(0, 2, size=(50, 32), dtype=np.uint8)
    full = pairwise_tanimoto(fps)
    np.fill_diagonal(full, 0.0)

    sim = TanimotoSimilarity(fps)
    for i in [0, 7, 25, 49]:
        np.testing.assert_allclose(sim.row(i), full[i], atol=1e-12)
