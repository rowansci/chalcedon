"""Prepare SMILES subset files for benchmarking.

Samples from the GEOM drugs dataset and saves one SMILES file per size
alongside this script in benchmarks/data/. Run once before benchmarking.
Also prints atom count statistics per subset for inclusion in README.

Usage:
    uv run --group benchmark python benchmarks/scripts/create_subsets.py
        --smiles /path/to/drugs_SMILES.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from rdkit import Chem
from tqdm import tqdm

SIZES = [10, 100, 1_000, 10_000, 25_000, 50_000, 100_000]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 42


def atom_count_stats(smiles: list[str]) -> dict[str, int | float]:
    """Compute heavy-atom count statistics for SMILES strings.

    Args:
        smiles: SMILES strings to summarize.

    Returns:
        Mapping with keys `n`, `min`, `max`, `mean`, `median`, `std`.
    """
    counts = []
    for smiles_string in tqdm(smiles, desc="  Computing atom counts", leave=False):
        molecule = Chem.MolFromSmiles(smiles_string)
        if molecule is not None:
            counts.append(molecule.GetNumHeavyAtoms())
    counts_array = np.array(counts)
    return {
        "n": len(counts_array),
        "min": int(counts_array.min()),
        "max": int(counts_array.max()),
        "mean": float(counts_array.mean()),
        "median": float(np.median(counts_array)),
        "std": float(counts_array.std()),
    }


def main() -> None:
    """Sample, save SMILES subsets, and print atom count statistics."""
    parser = argparse.ArgumentParser(description="Prepare SMILES subsets for benchmarking")
    parser.add_argument("--smiles", type=Path, required=True, help="Path to source SMILES CSV")
    args = parser.parse_args()

    if not args.smiles.exists():
        raise FileNotFoundError(f"SMILES file not found: {args.smiles}")

    print(f"Loading SMILES from {args.smiles}...")
    with args.smiles.open() as smiles_file:
        reader = csv.DictReader(smiles_file)
        all_smiles = [row["SMILES"] for row in reader if row.get("SMILES")]
    print(f"Loaded {len(all_smiles):,} SMILES\n")

    rng = np.random.default_rng(SEED)
    max_size = max(SIZES)
    if len(all_smiles) < max_size:
        raise ValueError(f"Need at least {max_size:,} SMILES, only got {len(all_smiles):,}")

    sampled_indices = rng.choice(len(all_smiles), size=max_size, replace=False)
    sampled = [all_smiles[i] for i in sampled_indices]

    DATA_DIR.mkdir(exist_ok=True)
    for size in SIZES:
        output_path = DATA_DIR / f"smiles_{size}.txt"
        output_path.write_text("\n".join(sampled[:size]))
        print(f"Wrote {output_path} ({size:,} SMILES)")

    print("\nComputing atom count statistics per subset...")
    print("\n--- Paste into benchmarks/report.md ---\n")
    print(f"{'n':>8}  {'min':>5}  {'max':>5}  {'mean':>7}  {'median':>7}  {'std':>7}")
    print("-" * 50)
    for size in SIZES:
        stats = atom_count_stats(sampled[:size])
        print(
            f"{size:>8,}  {stats['min']:>5}  {stats['max']:>5}"
            f"  {stats['mean']:>7.1f}  {stats['median']:>7.1f}  {stats['std']:>7.1f}"
        )
    print("\n--- End paste ---")


if __name__ == "__main__":
    main()
