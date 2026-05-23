"""BLAS-based full-matrix Butina baseline for the benchmark.

Not part of the public Chalcedon API; lives in the benchmark suite only as
a comparison baseline. Mirrors RDKit's algorithmic structure (precompute the
full pairwise mask, then look up for counting and assignment) but uses BLAS
sgemm via Chalcedon's internal `TanimotoSimilarity._block_neighbors` instead
of RDKit's Python-loop `BulkTanimotoSimilarity`. Same partition output as
both `chalcedon.butina_cluster` and RDKit's `ClusterData` on every tested
input.

Reuses the per-block similarity kernel from Chalcedon's count phase (float32
throughout, fused neighbor-mask comparison, precomputed per-row norms, shared
workspace buffers). Wraps it with:

- Upper-triangle traversal with recursive diagonal splitting.
- A persistent symmetric n x n bool mask populated block-by-block.
- `np.count_nonzero` for neighbor counting (avoids int64 promotion of
  `mask.sum`).
- A lookup-based assign phase against the cached mask, with an in-place
  `unassigned` mask updated incrementally.

This is the cached-matrix alternative to Chalcedon's matrix-free design:
same BLAS dispatch, same numerical optimizations, different memory profile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from chalcedon.tanimoto_similarity import TanimotoSimilarity

if TYPE_CHECKING:
    from numpy.typing import NDArray

_BLOCK_SIZE = 2500
_DIAGONAL_SPLIT_MINIMUM = 512


def chalcedon_full_matrix(
    fingerprints: NDArray[np.integer | np.floating], cutoff: float = 0.65
) -> NDArray[np.intp]:
    """Cluster fingerprints using cached-matrix Butina via Chalcedon's similarity kernel.

    Args:
        fingerprints: non-negative fingerprint matrix of shape `(n, d)`.
        cutoff: Tanimoto distance cutoff; pairs with distance ≤ `cutoff`
            are neighbors.

    Returns:
        Cluster ID per point, shape `(n,)`. Cluster 0 is the first centroid
        chosen.
    """
    fingerprint_count = len(fingerprints)
    similarity = TanimotoSimilarity(fingerprints)
    block_size = min(_BLOCK_SIZE, fingerprint_count)
    workspace = block_size * block_size
    dot_products_buffer = np.empty(workspace, dtype=np.float32)
    unions_buffer = np.empty(workspace, dtype=np.float32)
    boolean_buffer = np.empty(workspace, dtype=bool)

    mask = np.zeros((fingerprint_count, fingerprint_count), dtype=bool)

    def write_block(row_start: int, row_end: int, column_start: int, column_end: int) -> None:
        """Populate the symmetric mask with neighbors for one upper-triangle block."""
        row_count = row_end - row_start
        column_count = column_end - column_start
        size = row_count * column_count
        neighbors = similarity._block_neighbors(
            row_start,
            row_end,
            column_start,
            column_end,
            cutoff,
            dot_products_buffer[:size].reshape(row_count, column_count),
            unions_buffer[:size].reshape(row_count, column_count),
            boolean_buffer[:size].reshape(row_count, column_count),
        )
        mask[row_start:row_end, column_start:column_end] = neighbors
        if column_start != row_start:
            mask[column_start:column_end, row_start:row_end] = neighbors.T

    def diagonal(start: int, end: int) -> None:
        """Populate a diagonal block of the mask, recursively splitting if large enough."""
        if end - start >= _DIAGONAL_SPLIT_MINIMUM:
            middle = start + (end - start) // 2
            diagonal(start, middle)
            diagonal(middle, end)
            write_block(start, middle, middle, end)
        else:
            write_block(start, end, start, end)

    # Upper-triangle traversal in row-major block order with diagonal splitting.
    for row_start in range(0, fingerprint_count, block_size):
        row_end = min(row_start + block_size, fingerprint_count)
        diagonal(row_start, row_end)
        for column_start in range(row_end, fingerprint_count, block_size):
            column_end = min(column_start + block_size, fingerprint_count)
            write_block(row_start, row_end, column_start, column_end)

    # _block_neighbors zeros the diagonal, so neighbor counts exclude self,
    # same convention as butina_cluster's count phase. The centroid claims
    # itself explicitly in the assign loop below.
    neighbor_counts = np.count_nonzero(mask, axis=1)
    order = np.lexsort((-np.arange(fingerprint_count, dtype=np.intp), -neighbor_counts))
    cluster_ids = np.full(fingerprint_count, -1, dtype=np.intp)
    unassigned = np.ones(fingerprint_count, dtype=bool)
    next_cluster_id = 0
    for candidate in order:
        if not unassigned[candidate]:
            continue
        members = np.flatnonzero(mask[candidate] & unassigned)
        cluster_ids[members] = next_cluster_id
        cluster_ids[candidate] = next_cluster_id  # mask diagonal is False; claim self explicitly
        unassigned[members] = False
        unassigned[candidate] = False
        next_cluster_id += 1
    return cluster_ids
