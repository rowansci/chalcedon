"""Tanimoto similarity computation for fingerprint arrays."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

Precision = Literal["float32", "float64"] | type[np.float32] | type[np.float64]
"""Working-precision spec: either the dtype name or the numpy scalar type."""


def pairwise_tanimoto(
    fingerprints: NDArray[Any],
    dtype: Precision = "float32",
) -> NDArray[np.floating]:
    """Compute pairwise Tanimoto similarity matrix.

    Uses BLAS-accelerated matrix multiplication for vectorized computation.
    Memory usage is O(n^2) for the output matrix.

    Args:
        fingerprints: non-negative fingerprint matrix of shape `(n, d)`.
            Binary, count, and positive float vectors are all supported.
        dtype: working precision. Pass `"float64"` for higher precision at
            ≈2x runtime and ≈2x memory.

    Returns:
        Symmetric similarity matrix of shape `(n, n)` with values in [0, 1].

    Examples:
        >>> import numpy as np
        >>> fingerprints = np.array([[1, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=np.uint8)
        >>> similarity = pairwise_tanimoto(fingerprints)
        >>> float(similarity[0, 1])
        0.3333333432674408
        >>> float(similarity[2, 0])
        0.0
    """
    fingerprints_float = np.asarray(fingerprints, dtype=dtype)
    fingerprint_count = fingerprints_float.shape[0]
    dot_products = np.empty((fingerprint_count, fingerprint_count), dtype=fingerprints_float.dtype)
    chunk_size = max(1, (2**31 - 1) // fingerprint_count)
    for start in range(0, fingerprint_count, chunk_size):
        dot_products[start : start + chunk_size] = (
            fingerprints_float[start : start + chunk_size] @ fingerprints_float.T
        )
    norms = dot_products.diagonal()
    unions = norms[:, None] + norms[None, :] - dot_products
    # In-place divide into `dot_products`; `where=unions > 0` leaves empty-vs-empty
    # cells untouched, and those cells already hold 0 (dot == 0 when norm == 0).
    np.divide(dot_products, unions, out=dot_products, where=unions > 0)
    return dot_products


@dataclass(slots=True, frozen=True)
class TanimotoSimilarity:
    """Row-at-a-time Tanimoto similarity for streaming algorithms.

    Precomputes per-fingerprint norms so each `row(i)` call needs only one
    matrix-vector product. Memory is O(n * d); no pairwise storage.

    Args:
        fingerprints: non-negative fingerprint matrix of shape `(n, d)`.
            Binary, count, and positive float vectors are all supported.
        dtype: working precision. Pass `"float64"` for higher precision at
            ≈2x runtime and ≈2x memory.

    Examples:
        >>> import numpy as np
        >>> fingerprints = np.array([[1, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=np.uint8)
        >>> similarity = TanimotoSimilarity(fingerprints)
        >>> similarity.fingerprint_count
        3
        >>> float(similarity.row(0)[1])
        0.3333333432674408
    """

    fingerprints: InitVar[NDArray[Any]]
    dtype: InitVar[Precision] = "float32"
    _fingerprints: NDArray[np.floating] = field(init=False)
    _norms: NDArray[np.floating] = field(init=False)

    def __post_init__(self, fingerprints: NDArray[Any], dtype: Precision) -> None:
        """Cast fingerprints to the requested dtype and precompute per-row norms."""
        cast_fingerprints = np.asarray(fingerprints, dtype=dtype)
        object.__setattr__(self, "_fingerprints", cast_fingerprints)
        object.__setattr__(
            self, "_norms", np.einsum("ij,ij->i", cast_fingerprints, cast_fingerprints)
        )

    @property
    def fingerprint_count(self) -> int:
        """Number of fingerprints."""
        return self._fingerprints.shape[0]

    @property
    def precision(self) -> np.dtype:
        """Working precision of the cached fingerprint matrix."""
        return self._fingerprints.dtype

    def row(self, index: int) -> NDArray[np.floating]:
        """Tanimoto similarity of point `index` to all points.

        Returns:
            Similarity array of shape `(n,)`. Self-similarity is 0.
        """
        dot_products = self._fingerprints @ self._fingerprints[index]
        unions = self._norms + self._norms[index] - dot_products
        zero = np.asarray(0.0, dtype=self._fingerprints.dtype)
        similarities = np.where(unions > 0, dot_products / unions, zero)
        similarities[index] = 0.0
        return similarities

    def chunk(self, start: int, end: int) -> NDArray[np.floating]:
        """Tanimoto similarity for rows `[start, end)` to all points.

        Returns:
            Similarity matrix of shape `(end - start, n)`. Self-similarities
            are 0.
        """
        dot_products = self._fingerprints[start:end] @ self._fingerprints.T
        unions = self._norms[start:end, None] + self._norms[None, :] - dot_products
        np.divide(dot_products, unions, out=dot_products, where=unions > 0)
        np.fill_diagonal(dot_products[:, start:end], 0.0)
        return dot_products

    def _fill_dot_products_and_unions(
        self,
        row_fingerprints: NDArray[np.floating],
        column_fingerprints: NDArray[np.floating],
        row_norms: NDArray[np.floating],
        column_norms: NDArray[np.floating],
        dot_products_buffer: NDArray[np.floating],
        unions_buffer: NDArray[np.floating],
    ) -> None:
        """Compute `row_fingerprints @ column_fingerprints.T` and the unions in-place."""
        np.matmul(row_fingerprints, column_fingerprints.T, out=dot_products_buffer)
        np.add(row_norms[:, None], column_norms[None, :], out=unions_buffer)
        unions_buffer -= dot_products_buffer

    def _rows_neighbors_against(
        self,
        indices: NDArray[np.intp],
        against: NDArray[np.intp],
        cutoff: float,
        dot_products_buffer: NDArray[np.floating],
        unions_buffer: NDArray[np.floating],
        boolean_buffer: NDArray[np.bool_],
    ) -> NDArray[np.bool_]:
        """Boolean matrix: distance <= cutoff for `rows[indices]` vs `rows[against]`.

        Internal kernel. All three buffers are written in-place; callers own
        them and reuse them across iterations to keep peak RSS bounded.
        Self-pairs are NOT masked: if `i == against[j]` the cell will be True.
        """
        self._fill_dot_products_and_unions(
            self._fingerprints[indices],
            self._fingerprints[against],
            self._norms[indices],
            self._norms[against],
            dot_products_buffer,
            unions_buffer,
        )
        # Fuse `(1 - dot/union) <= cutoff` as `dot >= (1-cutoff)*union`; scaling
        # `unions_buffer` in place avoids an extra workspace.
        unions_buffer *= np.asarray(1.0 - cutoff, dtype=unions_buffer.dtype)
        np.greater_equal(dot_products_buffer, unions_buffer, out=boolean_buffer)
        # Guard empty-vs-empty pairs (union == 0 ⇒ dot == 0, which would spuriously
        # satisfy the >=). Truthy-float AND avoids a `unions > 0` temp.
        np.logical_and(boolean_buffer, unions_buffer, out=boolean_buffer)
        return boolean_buffer

    def _block_neighbors(
        self,
        row_start: int,
        row_end: int,
        column_start: int,
        column_end: int,
        cutoff: float,
        dot_products_buffer: NDArray[np.floating],
        unions_buffer: NDArray[np.floating],
        boolean_buffer: NDArray[np.bool_],
    ) -> NDArray[np.bool_]:
        """Boolean matrix: Tanimoto distance <= cutoff for sub-block.

        Internal kernel. All three buffers are written in-place; callers own
        them and reuse them across iterations to keep peak RSS bounded.
        Row and column ranges must be either fully aligned (diagonal block,
        self-pairs are zeroed) or fully disjoint.
        """
        self._fill_dot_products_and_unions(
            self._fingerprints[row_start:row_end],
            self._fingerprints[column_start:column_end],
            self._norms[row_start:row_end],
            self._norms[column_start:column_end],
            dot_products_buffer,
            unions_buffer,
        )
        # Fuse `(1 - dot/union) <= cutoff` as `dot >= (1-cutoff)*union`; scaling
        # `unions_buffer` in place avoids an extra workspace.
        unions_buffer *= np.asarray(1.0 - cutoff, dtype=unions_buffer.dtype)
        np.greater_equal(dot_products_buffer, unions_buffer, out=boolean_buffer)
        # Guard empty-vs-empty pairs (union == 0 ⇒ dot == 0, which would spuriously
        # satisfy the >=). Truthy-float AND avoids a `unions > 0` temp.
        np.logical_and(boolean_buffer, unions_buffer, out=boolean_buffer)
        if row_start == column_start:
            np.fill_diagonal(boolean_buffer, False)
        return boolean_buffer
