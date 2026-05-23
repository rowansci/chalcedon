"""Tests for greedy cluster splitting."""

import numpy as np
import pytest

from chalcedon.greedy_cluster_split import greedy_cluster_split


def test_partitions_all_points() -> None:
    """Every point lands in exactly one split."""
    ids = np.array([0, 0, 1, 1, 2, 3, 3, 3])
    result = greedy_cluster_split(ids, {"train": 0.7, "val": 0.15, "test": 0.15})
    combined = np.sort(np.concatenate(list(result.values())))
    np.testing.assert_array_equal(combined, np.arange(len(ids)))


def test_whole_clusters_kept_together() -> None:
    """No cluster is split across multiple subsets."""
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 30, size=500)
    result = greedy_cluster_split(ids, {"train": 0.7, "val": 0.15, "test": 0.15})
    for indices in result.values():
        cluster_set = set(ids[indices].tolist())
        for cluster in cluster_set:
            cluster_members = set(np.flatnonzero(ids == cluster).tolist())
            assert cluster_members.issubset(set(indices.tolist()))


def test_largest_cluster_goes_to_largest_target() -> None:
    """Biggest cluster lands in the split with the biggest deficit (train)."""
    ids = np.array([0] * 100 + [1] * 5 + [2] * 5)
    result = greedy_cluster_split(ids, {"train": 0.7, "val": 0.15, "test": 0.15})
    assert set(result["train"].tolist()) >= set(range(100))


def test_hits_target_percentages_within_one_cluster() -> None:
    """Final split sizes are close to target fractions."""
    rng = np.random.default_rng(1)
    ids = rng.integers(0, 200, size=10_000)
    fractions = {"train": 0.7, "val": 0.15, "test": 0.15}
    result = greedy_cluster_split(ids, fractions)
    largest_cluster_size = np.bincount(ids).max()
    for name, target in fractions.items():
        actual = len(result[name]) / len(ids)
        assert abs(actual - target) <= largest_cluster_size / len(ids) + 1e-9


def test_empty_input() -> None:
    """Zero points produce empty arrays for every split."""
    ids = np.array([], dtype=np.int64)
    result = greedy_cluster_split(ids, {"train": 0.8, "test": 0.2})
    assert all(len(arr) == 0 for arr in result.values())
    assert set(result) == {"train", "test"}


def test_single_cluster_goes_to_first_largest_target() -> None:
    """One giant cluster goes entirely to the split with the biggest target."""
    ids = np.zeros(50, dtype=np.int64)
    result = greedy_cluster_split(ids, {"train": 0.6, "val": 0.2, "test": 0.2})
    assert len(result["train"]) == 50
    assert len(result["val"]) == 0
    assert len(result["test"]) == 0


def test_singletons_distribute_proportionally() -> None:
    """When every point is its own cluster, splits hit targets exactly."""
    ids = np.arange(1000)
    result = greedy_cluster_split(ids, {"train": 0.7, "val": 0.15, "test": 0.15})
    assert len(result["train"]) == 700
    assert len(result["val"]) == 150
    assert len(result["test"]) == 150


def test_deterministic() -> None:
    """Same input produces identical output across runs."""
    rng = np.random.default_rng(2)
    ids = rng.integers(0, 50, size=300)
    fractions = {"train": 0.6, "val": 0.2, "test": 0.2}
    first = greedy_cluster_split(ids, fractions)
    second = greedy_cluster_split(ids, fractions)
    for name in fractions:
        np.testing.assert_array_equal(first[name], second[name])


def test_supports_arbitrary_split_count() -> None:
    """Works for 2-way and 5-way splits alike."""
    ids = np.arange(100)
    five_way = greedy_cluster_split(ids, dict.fromkeys(["a", "b", "c", "d", "e"], 0.2))
    for name in ["a", "b", "c", "d", "e"]:
        assert len(five_way[name]) == 20


@pytest.mark.parametrize(
    ("fractions", "match"),
    [
        ({}, "non-empty"),
        ({"train": 1.0, "test": 0.0}, "positive"),
        ({"train": 0.5, "test": 0.4}, r"sum to 1\.0"),
    ],
)
def test_rejects_invalid_fractions(fractions: dict[str, float], match: str) -> None:
    """Fractions that violate the contract raise ValueError."""
    with pytest.raises(ValueError, match=match):
        greedy_cluster_split(np.array([0, 1]), fractions)
