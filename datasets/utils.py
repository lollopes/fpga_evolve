"""
datasets/utils.py — shared utilities for NMNIST and SHD dataset_init modules.

Provides:
    parse_task          — validate task string, return structured task info
    build_class_index   — map class label → list of global dataset indices
    subsample_indices   — balanced subsampling with label remapping
    LabelRemapDataset   — thin Dataset wrapper that remaps labels for binary tasks
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Task parsing
# ---------------------------------------------------------------------------

def parse_task(task_str: str, n_classes: int) -> dict:
    """
    Parse and validate a task string. Returns a dict with keys:
        mode     : "full" | "binary_pair" | "vs_all"
        class_a  : int or None
        class_b  : int or None
        out_dim  : int

    Valid formats
    -------------
    "full"      → all n_classes classes, out_dim = n_classes
    "A_vs_B"    → binary, class A (label 0) vs class B (label 1)
    "A_vs_all"  → binary, class A (label 1) vs all others (label 0)

    Raises ValueError for anything else.
    """
    if task_str == "full":
        return {"mode": "full", "class_a": None, "class_b": None, "out_dim": n_classes}

    if task_str.endswith("_vs_all"):
        prefix = task_str[: -len("_vs_all")]
        try:
            a = int(prefix)
        except ValueError:
            raise ValueError(
                f"Invalid task {task_str!r}: expected an integer before '_vs_all'."
            )
        _check_range(a, n_classes, "class_a", task_str)
        return {"mode": "vs_all", "class_a": a, "class_b": None, "out_dim": 2}

    if "_vs_" in task_str:
        parts = task_str.split("_vs_")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid task {task_str!r}: expected exactly one '_vs_' separator."
            )
        try:
            a, b = int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(
                f"Invalid task {task_str!r}: both sides of '_vs_' must be integers."
            )
        _check_range(a, n_classes, "class_a", task_str)
        _check_range(b, n_classes, "class_b", task_str)
        if a == b:
            raise ValueError(
                f"Invalid task {task_str!r}: class_a and class_b must differ."
            )
        return {"mode": "binary_pair", "class_a": a, "class_b": b, "out_dim": 2}

    raise ValueError(
        f"Unknown task {task_str!r}. Valid formats: "
        f"'full', 'A_vs_B' (e.g. '3_vs_7'), 'A_vs_all' (e.g. '3_vs_all')."
    )


def _check_range(value: int, n_classes: int, name: str, task_str: str) -> None:
    if not (0 <= value < n_classes):
        raise ValueError(
            f"Invalid task {task_str!r}: {name}={value} out of range "
            f"[0, {n_classes - 1}]."
        )


# ---------------------------------------------------------------------------
# Class index builder
# ---------------------------------------------------------------------------

def build_class_index(dataset) -> dict[int, list[int]]:
    """
    Return {class_label: [global_idx, ...]} for a tonic dataset.

    Tries .targets first (some tonic versions), then .data (list of
    (path, label) tuples — works for NMNIST and SHD), raises if neither works.
    """
    if hasattr(dataset, "targets") and dataset.targets is not None:
        idx: dict = {}
        for i, label in enumerate(dataset.targets):
            idx.setdefault(int(label), []).append(i)
        return idx

    if hasattr(dataset, "data"):
        try:
            idx = {}
            for i, item in enumerate(dataset.data):
                label = int(item[1])
                idx.setdefault(label, []).append(i)
            return idx
        except (IndexError, TypeError, KeyError):
            pass

    raise RuntimeError(
        "Cannot build class index: dataset exposes neither .targets nor a "
        "usable .data attribute. Check your tonic version."
    )


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

def subsample_indices(
    class_idx: dict[int, list[int]],
    task: dict,
    n_total: Optional[int],
    seed: int,
) -> list[tuple[int, int]]:
    """
    Return a shuffled list of (dataset_idx, remapped_label) pairs.

    n_total=None means use all available samples (balanced across classes
    for "full" and "vs_all"; min-capped for "binary_pair").
    """
    rng = np.random.default_rng(seed)
    mode = task["mode"]
    pairs: list[tuple[int, int]] = []

    if mode == "full":
        classes = sorted(class_idx.keys())
        n_classes = len(classes)
        per_class = (n_total // n_classes) if n_total is not None else None
        for c in classes:
            idxs = rng.permutation(class_idx[c])
            if per_class is not None:
                idxs = idxs[:per_class]
            pairs.extend((int(i), c) for i in idxs)

    elif mode == "binary_pair":
        a, b = task["class_a"], task["class_b"]
        idxs_a = rng.permutation(class_idx[a])
        idxs_b = rng.permutation(class_idx[b])
        if n_total is not None:
            half = n_total // 2
            idxs_a = idxs_a[:half]
            idxs_b = idxs_b[:half]
        else:
            cap = min(len(idxs_a), len(idxs_b))
            idxs_a = idxs_a[:cap]
            idxs_b = idxs_b[:cap]
        pairs.extend((int(i), 0) for i in idxs_a)
        pairs.extend((int(i), 1) for i in idxs_b)

    elif mode == "vs_all":
        a = task["class_a"]
        other_classes = sorted(c for c in class_idx if c != a)
        idxs_a = rng.permutation(class_idx[a])

        if n_total is not None:
            half = n_total // 2
            idxs_a = idxs_a[:half]
            # distribute the other half evenly across remaining classes
            per_other = max(1, half // len(other_classes))
            other_pairs: list[tuple[int, int]] = []
            for c in other_classes:
                sub = rng.permutation(class_idx[c])[:per_other]
                other_pairs.extend((int(i), 0) for i in sub)
            # trim to exactly half if rounding gave slightly more
            rng.shuffle(other_pairs)
            other_pairs = other_pairs[:half]
        else:
            # use all of class_a; match with equal total from others (balanced)
            n_a = len(idxs_a)
            per_other = max(1, n_a // len(other_classes))
            other_pairs = []
            for c in other_classes:
                sub = rng.permutation(class_idx[c])[:per_other]
                other_pairs.extend((int(i), 0) for i in sub)

        pairs.extend((int(i), 1) for i in idxs_a)
        pairs.extend(other_pairs)

    # Shuffle the combined list
    order = rng.permutation(len(pairs))
    return [pairs[i] for i in order]


# ---------------------------------------------------------------------------
# Label-remapping dataset wrapper
# ---------------------------------------------------------------------------

class LabelRemapDataset(Dataset):
    """
    Wraps a tonic dataset and serves samples at arbitrary indices with
    overridden labels.

    Parameters
    ----------
    base_dataset  : the underlying tonic dataset (untransformed or transformed)
    index_label_pairs : list of (original_idx, remapped_label) tuples
    """

    def __init__(self, base_dataset, index_label_pairs: list[tuple[int, int]]):
        self._base = base_dataset
        self._pairs = index_label_pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, i: int):
        orig_idx, label = self._pairs[i]
        x, _ = self._base[orig_idx]   # discard original label
        return x, label
