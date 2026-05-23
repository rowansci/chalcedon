"""Butina clustering algorithm (Butina, JCICS 39 747-750, 1999)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from chalcedon.tanimoto_similarity import Precision, TanimotoSimilarity

if TYPE_CHECKING:
    from numpy.typing import NDArray

_DEFAULT_COUNT_BLOCK_SIZE = 2500
_DEFAULT_ASSIGN_BATCH_SIZE = 500
_DIAGONAL_SPLIT_MINIMUM = 512  # diagonal blocks below this size aren't split


def butina_cluster(
    fingerprints: NDArray[np.integer | np.floating],
    cutoff: float = 0.65,
    count_block_size: int = _DEFAULT_COUNT_BLOCK_SIZE,
    assign_batch_size: int = _DEFAULT_ASSIGN_BATCH_SIZE,
    dtype: Precision = "float32",
) -> NDArray[np.intp]:
    """Cluster fingerprints using the Butina algorithm.

    Count-sort-assign strategy: count neighbors via upper-triangle BLAS
    blocks, sort by count descending, greedily assign in batched sgemm
    passes. Produces clusters matching RDKit's reference implementation at
    typical cheminformatics cutoffs; at uncommon cutoffs a small fraction
    of boundary-pair decisions may differ due to float rounding.

    Args:
        fingerprints: non-negative fingerprint matrix of shape `(n, d)`.
            Binary, count, and positive float vectors are all supported.
        cutoff: Tanimoto distance cutoff; pairs with distance ≤ `cutoff`
            are neighbors.
        count_block_size: side length of the count-phase BLAS blocks.
            Workspace cost scales as O(count_block_size**2).
        assign_batch_size: number of centers per assign-phase sgemm call.
            Workspace cost scales as O(assign_batch_size * n).
        dtype: working precision. Pass `"float64"` for higher precision at
            ≈2x runtime and ≈2x memory.

    Returns:
        Cluster ID per point, shape `(n,)`. Cluster 0 is the largest.

    Examples:
        >>> import numpy as np
        >>> fingerprints = np.array([
        ...     [1, 1, 0, 0],
        ...     [1, 1, 1, 0],
        ...     [0, 0, 1, 1],
        ...     [0, 0, 0, 1],
        ... ], dtype=np.uint8)
        >>> butina_cluster(fingerprints, cutoff=0.5).tolist()
        [1, 1, 0, 0]
    """
    similarity = TanimotoSimilarity(fingerprints, dtype=dtype)
    fingerprint_count = similarity.fingerprint_count
    block_size = min(count_block_size, fingerprint_count)
    batch_size = min(assign_batch_size, fingerprint_count)

    # Shared workspace, reshaped per block.
    max_workspace = max(block_size * block_size, batch_size * fingerprint_count)
    dot_products_flat = np.empty(max_workspace, dtype=similarity.precision)
    unions_flat = np.empty(max_workspace, dtype=similarity.precision)
    boolean_flat = np.empty(max_workspace, dtype=bool)

    # Phase 1: count neighbors via upper-triangle blocks. Diagonal blocks split
    # recursively for ssyrk-equivalent FLOPs savings in pure numpy.
    neighbor_counts = np.zeros(fingerprint_count, dtype=np.intp)

    def count_block(row_start: int, row_end: int, column_start: int, column_end: int) -> None:
        """Add neighbor counts contributed by one upper-triangle block."""
        row_count = row_end - row_start
        column_count = column_end - column_start
        size = row_count * column_count
        neighbors = similarity._block_neighbors(
            row_start,
            row_end,
            column_start,
            column_end,
            cutoff,
            dot_products_flat[:size].reshape(row_count, column_count),
            unions_flat[:size].reshape(row_count, column_count),
            boolean_flat[:size].reshape(row_count, column_count),
        )
        neighbor_counts[row_start:row_end] += np.count_nonzero(neighbors, axis=1)
        if row_start != column_start:
            neighbor_counts[column_start:column_end] += np.count_nonzero(neighbors, axis=0)

    def count_diagonal(start: int, end: int) -> None:
        """Count neighbors in a diagonal block, recursively splitting if large enough."""
        if end - start >= _DIAGONAL_SPLIT_MINIMUM:
            middle = start + (end - start) // 2
            count_diagonal(start, middle)
            count_diagonal(middle, end)
            count_block(start, middle, middle, end)
        else:
            count_block(start, end, start, end)

    for row_start in tqdm(
        range(0, fingerprint_count, block_size), desc="step 1/2: counting neighbors", leave=False
    ):
        row_end = min(row_start + block_size, fingerprint_count)
        count_diagonal(row_start, row_end)
        for column_start in range(row_start + block_size, fingerprint_count, block_size):
            column_end = min(column_start + block_size, fingerprint_count)
            count_block(row_start, row_end, column_start, column_end)

    # Phase 2: sort by count desc, ties by higher index first.
    order = np.lexsort((-np.arange(fingerprint_count, dtype=np.intp), -neighbor_counts))

    # Phase 3: batched greedy assign against a shrinking compact unassigned list.
    cluster_id = np.full(fingerprint_count, -1, dtype=np.intp)
    unassigned = np.arange(fingerprint_count, dtype=np.intp)
    next_cluster_id = 0
    cursor = 0
    with tqdm(
        total=fingerprint_count, desc="step 2/2: assigning clusters", leave=False
    ) as progress_bar:
        while cursor < fingerprint_count:
            pending = order[cursor:]
            pending_positions = np.flatnonzero(cluster_id[pending] == -1)
            if len(pending_positions) == 0:
                progress_bar.update(fingerprint_count - cursor)
                break
            center_count = min(batch_size, len(pending_positions))
            step = int(pending_positions[center_count - 1]) + 1
            centers = pending[pending_positions[:center_count]]
            cursor += step
            progress_bar.update(step)

            unassigned_count = len(unassigned)
            size = center_count * unassigned_count
            is_neighbor = similarity._rows_neighbors_against(
                centers,
                unassigned,
                cutoff,
                dot_products_flat[:size].reshape(center_count, unassigned_count),
                unions_flat[:size].reshape(center_count, unassigned_count),
                boolean_flat[:size].reshape(center_count, unassigned_count),
            )
            # Avoids recomputing `cluster_id[unassigned] == -1` per center.
            still_unassigned = np.ones(unassigned_count, dtype=bool)
            for batch_index, center in enumerate(centers.tolist()):
                if cluster_id[center] != -1:
                    continue
                # Drop points already claimed by earlier centers in this batch.
                member_mask = is_neighbor[batch_index] & still_unassigned
                cluster_id[unassigned[member_mask]] = next_cluster_id
                still_unassigned[member_mask] = False
                next_cluster_id += 1

            unassigned = np.flatnonzero(cluster_id == -1)

    return cluster_id
