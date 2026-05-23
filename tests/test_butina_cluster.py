"""Tests for Butina clustering."""

import numpy as np
import pytest

from chalcedon.butina_cluster import butina_cluster

TWO_CLUSTER_FPS = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.uint8)


def test_two_clusters() -> None:
    """Two groups of similar fingerprints form two clusters."""
    labels = butina_cluster(TWO_CLUSTER_FPS, cutoff=0.5)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_no_neighbors() -> None:
    """Orthogonal fingerprints each form their own cluster."""
    fps = np.eye(5, dtype=np.uint8)
    assert len(np.unique(butina_cluster(fps, cutoff=0.5))) == 5


def test_all_identical() -> None:
    """Identical fingerprints form a single cluster."""
    fps = np.array([[1, 0, 1, 0]] * 5, dtype=np.uint8)
    assert len(np.unique(butina_cluster(fps, cutoff=0.5))) == 1


def test_single() -> None:
    """Single fingerprint gets cluster 0."""
    fps = np.array([[1, 0, 1]], dtype=np.uint8)
    assert butina_cluster(fps, cutoff=0.5).tolist() == [0]


@pytest.mark.parametrize("cutoff", [0.3, 0.5, 0.7, 0.9])
def test_chunk_sizes_agree(cutoff: float) -> None:
    """Different chunk sizes produce identical results."""
    rng = np.random.default_rng(42)
    fps = rng.integers(0, 2, size=(100, 64), dtype=np.uint8)
    reference = butina_cluster(fps, cutoff=cutoff, count_block_size=100)
    for cs in [10, 25, 50]:
        result = butina_cluster(fps, cutoff=cutoff, count_block_size=cs)
        np.testing.assert_array_equal(reference, result)


@pytest.mark.parametrize("cutoff", [0.3, 0.5, 0.7, 0.9])
def test_chunk_size_n_matches_default(cutoff: float) -> None:
    """count_block_size=n (single-pass) agrees with the default chunked path."""
    rng = np.random.default_rng(42)
    fps = rng.integers(0, 2, size=(100, 64), dtype=np.uint8)
    np.testing.assert_array_equal(
        butina_cluster(fps, cutoff=cutoff),
        butina_cluster(fps, cutoff=cutoff, count_block_size=len(fps)),
    )


def test_tiny_chunk_size() -> None:
    """Tiny chunk_size produces the same result as count_block_size=n."""
    rng = np.random.default_rng(7)
    fps = rng.integers(0, 2, size=(50, 32), dtype=np.uint8)
    reference = butina_cluster(fps, cutoff=0.5, count_block_size=len(fps))
    result = butina_cluster(fps, cutoff=0.5, count_block_size=3)
    np.testing.assert_array_equal(reference, result)


@pytest.mark.parametrize("cutoff", [0.3, 0.5, 0.7])
def test_float64_clusters(cutoff: float) -> None:
    """float64 mode returns valid cluster IDs covering every point."""
    rng = np.random.default_rng(11)
    fps = rng.integers(0, 2, size=(200, 64), dtype=np.uint8)
    labels = butina_cluster(fps, cutoff=cutoff, dtype="float64")
    assert labels.shape == (len(fps),)
    assert (labels >= 0).all()


def test_float64_matches_float32_at_standard_cutoff() -> None:
    """At cutoff=0.65 on random binary fingerprints float32 and float64 partition identically."""
    rng = np.random.default_rng(13)
    fps = rng.integers(0, 2, size=(300, 128), dtype=np.uint8)
    f32 = butina_cluster(fps, cutoff=0.65, dtype="float32")
    f64 = butina_cluster(fps, cutoff=0.65, dtype="float64")
    np.testing.assert_array_equal(f32, f64)
