"""Plot benchmark results from benchmark.py.

Usage:
    uv run --group benchmark python benchmarks/scripts/plot.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

METHOD_LABELS = {
    "chalcedon": "Chalcedon (chunked, float32)",
    "chalcedon_fp64": "Chalcedon (chunked, float64)",
    "chalcedon_full_matrix": "Chalcedon (full matrix, float32)",
    "rdkit": "RDKit",
    "bblean": "BitBIRCH-Lean",
}
COLORS = {
    "chalcedon": "#2196F3",
    "chalcedon_fp64": "#4CAF50",
    "chalcedon_full_matrix": "#FF9800",
    "rdkit": "#9C27B0",
    "bblean": "#757575",
}
# Methods drawn with a dashed line because they're a different algorithm,
# not a Butina implementation.
DASHED_METHODS = {"bblean"}

BUTINA_METHODS = ("chalcedon", "chalcedon_fp64", "chalcedon_full_matrix", "rdkit")
BITBIRCH_COMPARISON_METHODS = ("chalcedon", "bblean")


def _load_csv(path: Path) -> list[dict[str, Any]]:
    """Load benchmark CSV rows, parsing numeric fields and optional `chunk_size`."""
    rows = []
    with path.open() as csv_file:
        for row in csv.DictReader(csv_file):
            entry = {
                "n": int(row["n"]),
                "cutoff": float(row["cutoff"]),
                "method": row["method"],
                "wall_seconds": float(row["wall_seconds"]),
                "peak_bytes": float(row["peak_bytes"]),
                "n_clusters": int(row["n_clusters"]),
            }
            if "chunk_size" in row:
                entry["chunk_size"] = int(row["chunk_size"])
            rows.append(entry)
    return rows


def _median_by(rows: list[dict[str, Any]], key: str, value_field: str) -> dict[Any, float]:
    """Group rows by `key` and return the median of `value_field` per group."""
    grouped: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row[value_field])
    return {group_key: float(np.median(values)) for group_key, values in grouped.items()}


def _plot_series(
    axes: plt.Axes,
    x_values: list[float],
    y_values: list[float],
    method: str,
) -> None:
    """Plot one method's series with its assigned color and line style."""
    label = METHOD_LABELS.get(method, method)
    color = COLORS.get(method, "gray")
    linestyle = "--" if method in DASHED_METHODS else "-"
    axes.plot(
        x_values, y_values, marker="o", color=color, label=label, linewidth=2, linestyle=linestyle
    )


def plot_scaling(path: Path) -> None:
    """Wall time and peak memory vs n for the Butina methods (two panels)."""
    rows = _load_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Butina Scaling (log-log)", fontsize=13)

    for method in BUTINA_METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        time_by_size = _median_by(method_rows, "n", "wall_seconds")
        memory_by_size = _median_by(method_rows, "n", "peak_bytes")
        dataset_sizes = sorted(time_by_size)
        _plot_series(axes[0], dataset_sizes, [time_by_size[size] for size in dataset_sizes], method)
        _plot_series(
            axes[1],
            dataset_sizes,
            [memory_by_size[size] / 1024**3 for size in dataset_sizes],
            method,
        )

    for axis, title, ylabel in [
        (axes[0], "Time", "Wall time (s)"),
        (axes[1], "Memory", "Peak RSS (GB)"),
    ]:
        axis.set_title(title, fontsize=11)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Dataset size (n)")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=10)
        axis.grid(True, which="both", alpha=0.3)
        axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    output_path = RESULTS_DIR / "scaling.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close()


def plot_density(path: Path) -> None:
    """Time and memory vs cutoff for the Butina methods."""
    rows = _load_csv(path)
    dataset_size = rows[0]["n"] if rows else 0
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Butina Density Sensitivity (n={dataset_size:,})", fontsize=13)

    for method in BUTINA_METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        time_by_cutoff = _median_by(method_rows, "cutoff", "wall_seconds")
        memory_by_cutoff = _median_by(method_rows, "cutoff", "peak_bytes")
        cutoffs = sorted(time_by_cutoff)
        _plot_series(axes[0], cutoffs, [time_by_cutoff[cutoff] for cutoff in cutoffs], method)
        _plot_series(
            axes[1], cutoffs, [memory_by_cutoff[cutoff] / 1024**2 for cutoff in cutoffs], method
        )

    for axis, title, ylabel in [
        (axes[0], "Time", "Wall time (s)"),
        (axes[1], "Memory", "Peak RSS (MB)"),
    ]:
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Tanimoto distance cutoff")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=9)
        axis.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = RESULTS_DIR / "density.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close()


def plot_bitbirch_comparison(path: Path) -> None:
    """Plot Chalcedon (Butina) and BitBIRCH-Lean (hierarchical) on the same axes."""
    rows = _load_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Chalcedon vs BitBIRCH-Lean scaling (log-log)", fontsize=13)

    for method in BITBIRCH_COMPARISON_METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        time_by_size = _median_by(method_rows, "n", "wall_seconds")
        memory_by_size = _median_by(method_rows, "n", "peak_bytes")
        dataset_sizes = sorted(time_by_size)
        _plot_series(axes[0], dataset_sizes, [time_by_size[size] for size in dataset_sizes], method)
        _plot_series(
            axes[1],
            dataset_sizes,
            [memory_by_size[size] / 1024**3 for size in dataset_sizes],
            method,
        )

    for axis, title, ylabel in [
        (axes[0], "Time", "Wall time (s)"),
        (axes[1], "Memory", "Peak RSS (GB)"),
    ]:
        axis.set_title(title, fontsize=11)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Dataset size (n)")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=10)
        axis.grid(True, which="both", alpha=0.3)
        axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    output_path = RESULTS_DIR / "bitbirch_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close()


def plot_chunks(path: Path) -> None:
    """Time and memory vs chunk_size, one series per dataset size n."""
    rows = _load_csv(path)
    dataset_sizes = sorted({row["n"] for row in rows})
    color_map = plt.get_cmap("viridis")
    colors = {
        size: color_map(i / max(len(dataset_sizes) - 1, 1)) for i, size in enumerate(dataset_sizes)
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Butina Chunk Size Sensitivity (cutoff = 0.65, log-log)", fontsize=13)

    for size in dataset_sizes:
        rows_for_size = [row for row in rows if row["n"] == size]
        time_by_chunk = _median_by(rows_for_size, "chunk_size", "wall_seconds")
        memory_by_chunk = _median_by(rows_for_size, "chunk_size", "peak_bytes")
        chunk_sizes = sorted(time_by_chunk)
        label = f"n = {size:,}"
        axes[0].plot(
            chunk_sizes,
            [time_by_chunk[chunk_size] for chunk_size in chunk_sizes],
            marker="o",
            color=colors[size],
            label=label,
            linewidth=2,
        )
        axes[1].plot(
            chunk_sizes,
            [memory_by_chunk[chunk_size] / 1024**2 for chunk_size in chunk_sizes],
            marker="o",
            color=colors[size],
            label=label,
            linewidth=2,
        )

    for axis, title, ylabel in [
        (axes[0], "Time", "Wall time (s)"),
        (axes[1], "Memory", "Peak RSS (MB)"),
    ]:
        axis.set_title(title, fontsize=11)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("chunk_size")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=10)
        axis.grid(True, which="both", alpha=0.3)
        axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    output_path = RESULTS_DIR / "chunks.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close()


def main() -> None:
    """Plot benchmark results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scaling_csv = RESULTS_DIR / "scaling.csv"
    if scaling_csv.exists():
        plot_scaling(scaling_csv)
        plot_bitbirch_comparison(scaling_csv)
    density_csv = RESULTS_DIR / "density.csv"
    if density_csv.exists():
        plot_density(density_csv)
    chunks_csv = RESULTS_DIR / "chunks.csv"
    if chunks_csv.exists():
        plot_chunks(chunks_csv)


if __name__ == "__main__":
    main()
