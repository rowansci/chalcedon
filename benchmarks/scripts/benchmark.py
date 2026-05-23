"""Benchmark Butina clustering: Chalcedon vs RDKit.

Each method runs in a fresh subprocess for accurate memory measurement.

Three modes:
  scaling  : sweep dataset sizes at fixed cutoff
  density  : sweep cutoffs at fixed n
  chunks   : sweep chunk_size for Chalcedon at fixed n / cutoff

Usage:
    uv run --group benchmark python benchmarks/scripts/benchmark.py scaling
    uv run --group benchmark python benchmarks/scripts/benchmark.py density
    uv run --group benchmark python benchmarks/scripts/benchmark.py chunks
    uv run --group benchmark python benchmarks/scripts/benchmark.py scaling --sizes 1000
    uv run --group benchmark python benchmarks/scripts/benchmark.py density --n 25000
    uv run --group benchmark python benchmarks/scripts/benchmark.py chunks --n 25000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from bblean import BitBirch
from chalcedon_full_matrix import chalcedon_full_matrix as _chalcedon_full_matrix
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina as RDKitButina
from tqdm import tqdm

import chalcedon

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048
CSV_FIELDS = ["machine", "n", "cutoff", "method", "wall_seconds", "peak_bytes", "n_clusters"]
CHUNKS_CSV_FIELDS = [
    "machine",
    "n",
    "cutoff",
    "method",
    "chunk_size",
    "wall_seconds",
    "peak_bytes",
    "n_clusters",
]


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


def _rdkit_butina(fingerprints: NDArray[np.uint8], cutoff: float = 0.65) -> NDArray[np.intp]:
    """RDKit's C++ Butina implementation (includes numpy->BitVect conversion)."""
    fingerprint_count, bit_count = fingerprints.shape
    rdkit_fingerprints = []
    for row in tqdm(fingerprints, desc="  RDKit convert", leave=False):
        bit_vector = DataStructs.ExplicitBitVect(bit_count)
        for bit in np.flatnonzero(row).tolist():
            bit_vector.SetBit(bit)
        rdkit_fingerprints.append(bit_vector)

    distances = []
    for i in tqdm(range(1, fingerprint_count), desc="  RDKit distances", leave=False):
        similarities = DataStructs.BulkTanimotoSimilarity(
            rdkit_fingerprints[i], rdkit_fingerprints[:i]
        )
        distances.extend(1.0 - similarity for similarity in similarities)

    clusters = RDKitButina.ClusterData(distances, fingerprint_count, cutoff, isDistData=True)
    labels = np.empty(fingerprint_count, dtype=np.intp)
    for cluster_id, cluster in enumerate(clusters):
        for member_index in cluster:
            labels[member_index] = cluster_id
    return labels


def _chalcedon_fp64(
    fingerprints: NDArray[np.integer | np.floating], cutoff: float = 0.65
) -> NDArray[np.intp]:
    """Chalcedon's chunked Butina with float64 working precision (dgemm)."""
    return chalcedon.butina_cluster(fingerprints, cutoff=cutoff, dtype="float64")


def _bblean(fingerprints: NDArray[np.uint8], cutoff: float = 0.65) -> NDArray[np.intp]:
    """BitBIRCH-Lean clustering for context.

    Not Butina; BitBIRCH is a BIRCH-style hierarchical algorithm with different
    output. Included as a different-algorithm reference. Translates Chalcedon's
    Tanimoto distance cutoff into the equivalent similarity threshold
    `1 - cutoff` for bblean's API. Requires bblean's C++ kernel for the
    published numbers; Python fallback runs ≈2x slower in our tests.
    """
    bb = BitBirch(threshold=1.0 - cutoff)
    bb.fit(fingerprints, input_is_packed=False)
    return np.asarray(bb.get_assignments(), dtype=np.intp)


METHODS: dict[str, Callable[..., NDArray[np.intp]]] = {
    "chalcedon": chalcedon.butina_cluster,
    "chalcedon_fp64": _chalcedon_fp64,
    "chalcedon_full_matrix": _chalcedon_full_matrix,
    "rdkit": _rdkit_butina,
    "bblean": _bblean,
}
METHOD_ORDER = [
    "chalcedon",
    "chalcedon_fp64",
    "chalcedon_full_matrix",
    "rdkit",
    "bblean",
]

# When --methods is not specified, the default sweep skips these methods above
# this n. An explicit --methods bypasses the cap.
_DEFAULT_EXPENSIVE_CAP = 25_000
_EXPENSIVE_METHODS = ("chalcedon_fp64", "chalcedon_full_matrix", "rdkit")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fingerprints(size: int) -> NDArray[np.uint8]:
    """Read the cached SMILES subset of given size and return its Morgan fingerprints."""
    path = DATA_DIR / f"smiles_{size}.txt"
    if not path.exists():
        raise FileNotFoundError(f"SMILES subset not found: {path}")
    smiles = path.read_text().splitlines()
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_N_BITS
    )
    fingerprints: list[NDArray[np.uint8]] = []
    for smiles_string in tqdm(smiles, desc=f"  fingerprints n={size:,}", leave=False):
        molecule = Chem.MolFromSmiles(smiles_string)
        if molecule is not None:
            fingerprints.append(generator.GetFingerprintAsNumPy(molecule))
    return np.array(fingerprints, dtype=np.uint8)


def _save_fingerprints_tmp(fingerprints: NDArray[np.uint8]) -> str:
    """Save fingerprints to a temp .npy file, return path."""
    file_descriptor, path = tempfile.mkstemp(suffix=".npy")
    os.close(file_descriptor)
    np.save(path, fingerprints)
    return path


def _run(
    method_name: str,
    fingerprints_path: str,
    cutoff: float,
    chunk_size: int | None = None,
) -> tuple[float, int, int]:
    """Run a single method in a fresh subprocess for isolated memory measurement.

    Returns:
        Tuple of `(wall_seconds, peak_bytes, n_clusters)`.
    """
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_run_single",
        "--fingerprints-path",
        fingerprints_path,
        "--method",
        method_name,
        "--cutoff",
        str(cutoff),
    ]
    if chunk_size is not None:
        command.extend(["--chunk-size", str(chunk_size)])
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Subprocess failed for {method_name} (rc={result.returncode})")
    data = json.loads(result.stdout.strip().splitlines()[-1])
    return data["wall_seconds"], data["peak_bytes"], data["n_clusters"]


def _format_memory(n_bytes: int) -> str:
    """Format a byte count as a human-readable string (B, KB, MB, GB, or TB)."""
    value = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _format_time(seconds: float) -> str:
    """Format a duration in seconds as µs, ms, or s depending on magnitude."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def _open_csv(path: Path) -> tuple[Any, Any]:
    """Open CSV for appending, write header if new."""
    RESULTS_DIR.mkdir(exist_ok=True)
    write_header = not path.exists()
    csv_file = path.open("a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(CSV_FIELDS)
    return csv_file, writer


def _load_completed(path: Path) -> set[tuple[str, int, float, str]]:
    """Return set of (machine, n, cutoff, method) tuples already present in `path`."""
    if not path.exists():
        return set()
    completed: set[tuple[str, int, float, str]] = set()
    with path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            completed.add((row["machine"], int(row["n"]), float(row["cutoff"]), row["method"]))
    return completed


# ---------------------------------------------------------------------------
# Scaling: sweep n at fixed cutoff
# ---------------------------------------------------------------------------

SCALING_SIZES = [100, 1_000, 10_000, 25_000, 50_000, 100_000]


def _bench_scaling(
    sizes: list[int],
    cutoff: float,
    methods: list[str] | None,
) -> None:
    """Run each method at every requested dataset size and append results to CSV.

    Skips combinations already present in `scaling.csv` so runs can resume.
    """
    machine = platform.node().split(".")[0].lower()
    output_path = RESULTS_DIR / "scaling.csv"
    completed = _load_completed(output_path)
    csv_file, writer = _open_csv(output_path)

    # Apply the default expensive-method cap only when --methods was not given.
    selected_methods = methods if methods is not None else METHOD_ORDER
    apply_cap = methods is None

    print(
        f"\nScaling benchmark  cutoff={cutoff}  methods={','.join(selected_methods)}"
        f"{f'  cap_expensive_above={_DEFAULT_EXPENSIVE_CAP:,}' if apply_cap else ''}\n"
    )
    header = f"{'method':<24}  {'n':>8}  {'time':>10}  {'peak mem':>10}  {'clusters':>10}"
    print(header)
    print("-" * len(header))

    for size in sizes:
        methods_to_run = [
            method_name
            for method_name in selected_methods
            if not (
                apply_cap and method_name in _EXPENSIVE_METHODS and size > _DEFAULT_EXPENSIVE_CAP
            )
            and (machine, size, cutoff, method_name) not in completed
        ]
        if not methods_to_run:
            print(f"  skip n={size:,} (all methods already in CSV)")
            continue

        fingerprints = _load_fingerprints(size)
        fingerprint_count = len(fingerprints)
        fingerprints_path = _save_fingerprints_tmp(fingerprints)
        del fingerprints

        for method_name in methods_to_run:
            elapsed, peak_bytes, n_clusters = _run(method_name, fingerprints_path, cutoff)
            print(
                f"{method_name:<24}  {fingerprint_count:>8,}"
                f"  {_format_time(elapsed):>10}  {_format_memory(peak_bytes):>10}"
                f"  {n_clusters:>10,}"
            )
            writer.writerow(
                [machine, fingerprint_count, cutoff, method_name, elapsed, peak_bytes, n_clusters]
            )
            csv_file.flush()
        print()
        os.unlink(fingerprints_path)

    csv_file.close()
    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Density: sweep cutoff at fixed n
# ---------------------------------------------------------------------------

CUTOFFS = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9]


def _bench_density(n: int, methods: list[str] | None) -> None:
    """Sweep cutoff thresholds at fixed dataset size and append results to CSV.

    Skips combinations already present in `density.csv` so runs can resume.
    """
    machine = platform.node().split(".")[0].lower()
    output_path = RESULTS_DIR / "density.csv"
    completed = _load_completed(output_path)
    selected_methods = methods if methods is not None else METHOD_ORDER

    pending = [
        (cutoff, method_name)
        for cutoff in CUTOFFS
        for method_name in selected_methods
        if (machine, n, cutoff, method_name) not in completed
    ]
    if not pending:
        print(f"\nDensity benchmark  n={n:,}: all combos already in CSV")
        return

    csv_file, writer = _open_csv(output_path)

    fingerprints = _load_fingerprints(n)
    fingerprint_count = len(fingerprints)
    fingerprints_path = _save_fingerprints_tmp(fingerprints)
    del fingerprints

    print(f"\nDensity benchmark  n={fingerprint_count:,}\n")
    header = f"{'method':<24}  {'cutoff':>8}  {'time':>10}  {'peak mem':>10}  {'clusters':>10}"
    print(header)
    print("-" * len(header))

    last_cutoff: float | None = None
    for cutoff, method_name in pending:
        if last_cutoff is not None and cutoff != last_cutoff:
            print()
        last_cutoff = cutoff
        elapsed, peak_bytes, n_clusters = _run(method_name, fingerprints_path, cutoff)
        print(
            f"{method_name:<24}  {cutoff:>8.2f}"
            f"  {_format_time(elapsed):>10}  {_format_memory(peak_bytes):>10}  {n_clusters:>10,}"
        )
        writer.writerow(
            [machine, fingerprint_count, cutoff, method_name, elapsed, peak_bytes, n_clusters]
        )
        csv_file.flush()
    print()

    os.unlink(fingerprints_path)
    csv_file.close()
    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Chunks: sweep chunk_size at fixed n / cutoff (Chalcedon only)
# ---------------------------------------------------------------------------

CHUNK_SIZES = [100, 250, 500, 1000, 2500, 5000, 10000]


def _load_completed_chunks(path: Path) -> set[tuple[str, int, float, str, int]]:
    """Return set of (machine, n, cutoff, method, chunk_size) tuples already in `path`."""
    if not path.exists():
        return set()
    completed: set[tuple[str, int, float, str, int]] = set()
    with path.open() as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            completed.add(
                (
                    row["machine"],
                    int(row["n"]),
                    float(row["cutoff"]),
                    row["method"],
                    int(row["chunk_size"]),
                )
            )
    return completed


def _bench_chunks(n: int, cutoff: float, chunk_sizes: list[int]) -> None:
    """Sweep `count_block_size` for Chalcedon at fixed n / cutoff and append to CSV.

    Skips combinations already present in `chunks.csv` so runs can resume.
    """
    machine = platform.node().split(".")[0].lower()
    output_path = RESULTS_DIR / "chunks.csv"
    RESULTS_DIR.mkdir(exist_ok=True)
    completed = _load_completed_chunks(output_path)

    method_name = "chalcedon"
    pending = [
        chunk_size
        for chunk_size in chunk_sizes
        if (machine, n, cutoff, method_name, chunk_size) not in completed
    ]
    if not pending:
        print(f"\nChunks benchmark  n={n:,} cutoff={cutoff}: all combos already in CSV")
        return

    write_header = not output_path.exists()
    csv_file = output_path.open("a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(CHUNKS_CSV_FIELDS)

    fingerprints = _load_fingerprints(n)
    fingerprint_count = len(fingerprints)
    fingerprints_path = _save_fingerprints_tmp(fingerprints)
    del fingerprints

    print(f"\nChunks benchmark  n={fingerprint_count:,}  cutoff={cutoff}  method={method_name}\n")
    header = f"{'chunk_size':>10}  {'time':>10}  {'peak mem':>10}  {'clusters':>10}"
    print(header)
    print("-" * len(header))

    for chunk_size in pending:
        elapsed, peak_bytes, n_clusters = _run(method_name, fingerprints_path, cutoff, chunk_size)
        print(
            f"{chunk_size:>10,}  {_format_time(elapsed):>10}"
            f"  {_format_memory(peak_bytes):>10}  {n_clusters:>10,}"
        )
        writer.writerow(
            [
                machine,
                fingerprint_count,
                cutoff,
                method_name,
                chunk_size,
                elapsed,
                peak_bytes,
                n_clusters,
            ]
        )
        csv_file.flush()
    print()

    os.unlink(fingerprints_path)
    csv_file.close()
    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------------


def _run_single(
    fingerprints_path: str,
    method_name: str,
    cutoff: float,
    chunk_size: int | None = None,
) -> None:
    """Load fingerprints, run one method, print JSON result to stdout."""
    fingerprints = np.load(fingerprints_path)
    method_function = METHODS[method_name]

    kwargs: dict[str, Any] = {"cutoff": cutoff}
    if chunk_size is not None and method_name == "chalcedon":
        kwargs["count_block_size"] = chunk_size

    start_time = time.perf_counter()
    labels = method_function(fingerprints, **kwargs)
    elapsed = time.perf_counter() - start_time

    resource_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = resource_usage * 1024 if sys.platform == "linux" else resource_usage

    print(
        json.dumps(
            {
                "wall_seconds": elapsed,
                "peak_bytes": peak_bytes,
                "n_clusters": int(labels.max() + 1),
            }
        )
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark Butina clustering")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    scaling_parser = subparsers.add_parser("scaling", help="Sweep dataset sizes")
    scaling_parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=SCALING_SIZES,
        metavar="N",
        help=f"Dataset sizes (default: {SCALING_SIZES})",
    )
    scaling_parser.add_argument("--cutoff", type=float, default=0.65)
    scaling_parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_ORDER,
        default=None,
        metavar="METHOD",
        help=(
            f"Methods to run (default: {' '.join(METHOD_ORDER)}, with "
            f"chalcedon_fp64, chalcedon_full_matrix, and rdkit auto-skipped above "
            f"n={_DEFAULT_EXPENSIVE_CAP:,}). "
            "Passing this flag bypasses the auto-skip cap."
        ),
    )

    density_parser = subparsers.add_parser("density", help="Sweep cutoff thresholds")
    density_parser.add_argument("--n", type=int, default=25000)
    density_parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_ORDER,
        default=None,
        metavar="METHOD",
        help=f"Methods to run (default: {' '.join(METHOD_ORDER)}).",
    )

    chunks_parser = subparsers.add_parser(
        "chunks", help="Sweep chunk_size for Chalcedon at fixed n/cutoff"
    )
    chunks_parser.add_argument("--n", type=int, default=25000)
    chunks_parser.add_argument("--cutoff", type=float, default=0.65)
    chunks_parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=CHUNK_SIZES,
        metavar="K",
        help=f"Chunk sizes to sweep (default: {CHUNK_SIZES})",
    )

    single_parser = subparsers.add_parser("_run_single")
    single_parser.add_argument("--fingerprints-path", required=True)
    single_parser.add_argument("--method", required=True)
    single_parser.add_argument("--cutoff", type=float, required=True)
    single_parser.add_argument("--chunk-size", type=int, default=None)

    args = parser.parse_args()
    if args.mode == "scaling":
        _bench_scaling(args.sizes, args.cutoff, args.methods)
    elif args.mode == "density":
        _bench_density(args.n, args.methods)
    elif args.mode == "chunks":
        _bench_chunks(args.n, args.cutoff, args.chunk_sizes)
    elif args.mode == "_run_single":
        _run_single(args.fingerprints_path, args.method, args.cutoff, args.chunk_size)


if __name__ == "__main__":
    main()
