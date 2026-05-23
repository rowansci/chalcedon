"""Train/val/test splitting via Butina clustering and LPT assignment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from chalcedon.butina_cluster import butina_cluster
from chalcedon.greedy_cluster_split import greedy_cluster_split

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from chalcedon.tanimoto_similarity import Precision

_MORGAN_RADIUS = 2
_MORGAN_N_BITS = 2048


def _morgan_fingerprints(smiles: list[str]) -> NDArray[np.uint8]:
    """Compute Morgan fingerprints (radius 2, 2048 bits) for SMILES strings.

    Args:
        smiles: SMILES strings to fingerprint.

    Returns:
        Fingerprint matrix of shape `(len(smiles), 2048)` with dtype `uint8`.

    Raises:
        ValueError: any SMILES string fails to parse.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=_MORGAN_RADIUS, fpSize=_MORGAN_N_BITS
    )
    fingerprints = np.zeros((len(smiles), _MORGAN_N_BITS), dtype=np.uint8)
    for index, smiles_string in enumerate(smiles):
        molecule = Chem.MolFromSmiles(smiles_string)
        if molecule is None:
            raise ValueError(f"failed to parse SMILES at index {index}: {smiles_string!r}")
        fingerprints[index] = generator.GetFingerprintAsNumPy(molecule)
    return fingerprints


def butina_split(
    smiles: list[str],
    fractions: dict[str, float],
    cutoff: float = 0.65,
    dtype: Precision = "float32",
) -> dict[str, list[str]]:
    """Split a SMILES dataset using Butina clustering and LPT packing.

    Generates Morgan fingerprints (radius 2, 2048 bits), clusters with Butina at
    the given Tanimoto distance cutoff, and assigns whole clusters across the
    requested splits via LPT (largest-first greedy assignment to the
    most-underfilled split). Whole clusters stay together, so points in different
    splits never share a cluster.

    For custom fingerprints (FCFP, atom pair, learned encoders, etc.), drop down
    to the primitives: call `butina_cluster` on your fingerprints, then pass the
    resulting cluster IDs to `greedy_cluster_split`.

    Args:
        smiles: SMILES strings to split.
        fractions: mapping from split name to target fraction. Values must be
            positive and sum to 1.0.
        cutoff: Tanimoto distance cutoff; pairs with distance ≤ `cutoff`
            are neighbors.
        dtype: working precision passed to `butina_cluster`. Pass `"float64"`
            for higher precision at ≈2x runtime and ≈2x memory.

    Returns:
        Mapping from split name to list of SMILES belonging to that split.
        Within each split, SMILES preserve their original input order.

    Examples:
        >>> result = butina_split(
        ...     ["CCO", "CCCO", "c1ccccc1", "c1ccc(C)cc1"],
        ...     {"train": 0.5, "test": 0.5},
        ...     cutoff=0.5,
        ... )
        >>> sorted(result["train"] + result["test"])
        ['CCCO', 'CCO', 'c1ccc(C)cc1', 'c1ccccc1']
    """
    fingerprints = _morgan_fingerprints(smiles)
    cluster_ids = butina_cluster(fingerprints, cutoff=cutoff, dtype=dtype)
    index_splits = greedy_cluster_split(cluster_ids, fractions)
    return {name: [smiles[i] for i in indices] for name, indices in index_splits.items()}
