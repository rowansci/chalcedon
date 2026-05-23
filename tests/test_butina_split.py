"""Tests for butina_split."""

import pytest

from chalcedon.butina_split import butina_split

ALCOHOLS = ["CCO", "CCCO", "CCCCO", "CCCCCO"]
AROMATICS = ["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "CCCc1ccccc1"]
ALL_SMILES = ALCOHOLS + AROMATICS


def test_partitions_all_smiles() -> None:
    """Every input SMILES appears in exactly one split."""
    result = butina_split(ALL_SMILES, {"train": 0.75, "test": 0.25}, cutoff=0.6)
    combined = sorted(result["train"] + result["test"])
    assert combined == sorted(ALL_SMILES)


def test_distinct_chemotypes_separate() -> None:
    """Alcohols and aromatics fall into different splits at 50/50."""
    result = butina_split(ALL_SMILES, {"train": 0.5, "test": 0.5}, cutoff=0.5)
    train_set = set(result["train"])
    test_set = set(result["test"])
    assert (train_set == set(ALCOHOLS) and test_set == set(AROMATICS)) or (
        train_set == set(AROMATICS) and test_set == set(ALCOHOLS)
    )


def test_returns_lists_of_strings() -> None:
    """Values are plain Python lists of SMILES strings."""
    result = butina_split(ALL_SMILES, {"train": 0.5, "test": 0.5}, cutoff=0.5)
    for items in result.values():
        assert isinstance(items, list)
        assert all(isinstance(s, str) for s in items)


def test_invalid_smiles_raises() -> None:
    """Unparseable SMILES raise ValueError with index."""
    with pytest.raises(ValueError, match="index 1"):
        butina_split(["CCO", "not_a_smiles", "CCN"], {"train": 0.5, "test": 0.5})


def test_three_way_split() -> None:
    """Three-way split assigns every point."""
    smiles: list[str] = [f"C{'C' * i}O" for i in range(20)]
    result = butina_split(smiles, {"train": 0.6, "val": 0.2, "test": 0.2}, cutoff=0.1)
    assert len(result["train"]) + len(result["val"]) + len(result["test"]) == 20
