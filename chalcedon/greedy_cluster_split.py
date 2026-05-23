"""Dataset splitting strategies for molecular datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def greedy_cluster_split(
    cluster_ids: NDArray[np.integer],
    fractions: dict[str, float],
) -> dict[str, NDArray[np.intp]]:
    """Split points into named groups while keeping each cluster intact.

    Walks clusters from largest to smallest and drops each one into whichever
    split is currently furthest below its target fraction. Because whole
    clusters stay together, points in different splits never share a cluster,
    which is what keeps the resulting train/val/test sets dissimilar.

    The underlying algorithm is Longest Processing Time (LPT) scheduling, from
    Graham, R. L. (1969), "Bounds on Multiprocessing Timing Anomalies", SIAM
    Journal on Applied Mathematics 17(2):416-429, doi:10.1137/0117039.

    Args:
        cluster_ids: cluster label per point, shape `(n,)`.
        fractions: mapping from split name to target fraction. Values must be
            positive and sum to 1.0. Iteration order breaks ties when multiple
            splits share the maximum deficit.

    Returns:
        Mapping from split name to ascending-sorted point indices.

    Raises:
        ValueError: if `fractions` is empty, contains non-positive values,
            or does not sum to 1.0 within 1e-6.

    Examples:
        >>> import numpy as np
        >>> ids = np.array([0, 0, 0, 1, 1, 2, 3])
        >>> result = greedy_cluster_split(ids, {"train": 0.6, "test": 0.4})
        >>> result["train"].tolist()
        [0, 1, 2, 5]
        >>> result["test"].tolist()
        [3, 4, 6]
    """
    if not fractions:
        raise ValueError("fractions must be non-empty")
    if any(value <= 0 for value in fractions.values()):
        raise ValueError("all target fractions must be positive")
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"target fractions must sum to 1.0, got {total}")

    point_count = len(cluster_ids)
    if point_count == 0:
        return {name: np.empty(0, dtype=np.intp) for name in fractions}

    _, inverse = np.unique(cluster_ids, return_inverse=True)
    inverse = inverse.ravel()
    sizes = np.bincount(inverse)
    order = np.argsort(-sizes, kind="stable")

    split_names = list(fractions)
    targets = np.array([fractions[name] for name in split_names], dtype=np.float64)
    counts = np.zeros(len(split_names), dtype=np.float64)
    members_by_split: dict[str, list[NDArray[np.intp]]] = {name: [] for name in split_names}

    for cluster_index in order:
        deficits = targets - counts / point_count
        chosen = int(np.argmax(deficits))
        members = np.flatnonzero(inverse == cluster_index)
        members_by_split[split_names[chosen]].append(members)
        counts[chosen] += sizes[cluster_index]

    return {
        name: np.sort(np.concatenate(arrays)) if arrays else np.empty(0, dtype=np.intp)
        for name, arrays in members_by_split.items()
    }
