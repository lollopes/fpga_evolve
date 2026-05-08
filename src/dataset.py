"""
dataset.py — dataset loaders for N-MNIST and SHD experiments.

N-MNIST uses tonic.datasets.NMNIST with first_saccade_only=True (default) for
event loading and tonic.transforms.ToFrame for temporal binning. Spatial
pooling and binarization are applied as a numpy step inside the tonic Compose
pipeline.

SHD uses tonic.datasets.SHD and restricts class selection to the English digit
words ("zero"..."nine"), ignoring the German labels in the 20-class dataset.

Tasks:
    "7_vs_rest"  binary: class 7 vs equal sample from others
    "0_vs_1"     binary: class 0 vs class 1
    "0_vs_8"     binary: class 0 vs class 8
    "10class"    10-class: all digits 0–9
"""

from __future__ import annotations
import pathlib
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import tonic
import tonic.transforms as T_tonic


SENSOR_SIZE = tonic.datasets.NMNIST.sensor_size   # (34, 34, 2)  — (x, y, polarity)
SHD_SENSOR_SIZE = tonic.datasets.SHD.sensor_size  # (700, 1, 1)  — cochlea channels

_DEFAULT_DATA_ROOT = str(pathlib.Path(__file__).resolve().parent.parent / "datasets" / "NMNIST" / "data")
_DEFAULT_SHD_DATA_ROOT = str(pathlib.Path(__file__).resolve().parent.parent / "datasets" / "SHD" / "data")
_SHD_ENGLISH_DIGITS = (
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
)


# ---------------------------------------------------------------------------
# Transform pipeline
# ---------------------------------------------------------------------------

def _make_transform(n_time_bins: int, grid_size: Optional[int]):
    """
    Build the tonic-compatible transform pipeline.

    ToFrame bins events into n_time_bins equal windows  → [T, 2, H, W]
    _pool_binarize merges polarities, pools, binarizes  → [T, grid_size²]

    If grid_size is None the full 34×34 spatial resolution is kept
    (→ 1156 features per step).
    """

    def _pool_binarize(frames: np.ndarray) -> np.ndarray:
        # frames: [T, 2, H, W]  (int counts from ToFrame)
        spatial = frames.sum(axis=1)          # merge polarities → [T, H, W]
        T, H, W = spatial.shape

        if grid_size is not None and grid_size < H:
            gh = gw = grid_size
            step_h = H // gh
            step_w = W // gw
            crop_h = step_h * gh
            crop_w = step_w * gw
            pooled = (
                spatial[:, :crop_h, :crop_w]
                .reshape(T, gh, step_h, gw, step_w)
                .sum(axis=(2, 4))              # [T, gh, gw]
            )
        else:
            pooled = spatial                  # no spatial downsampling

        return (pooled > 0).astype(np.uint8).reshape(T, -1)   # [T, F]

    return T_tonic.Compose([
        T_tonic.ToFrame(sensor_size=SENSOR_SIZE, n_time_bins=n_time_bins),
        _pool_binarize,
    ])


def _make_shd_transform(n_time_bins: int):
    """
    Build the SHD transform pipeline.

    ToFrame bins events into n_time_bins equal windows  → [T, 1, 700]
    flatten/binarize keeps the 700 cochlea channels      → [T, 700]
    """

    def _flatten_binarize(frames: np.ndarray) -> np.ndarray:
        return (frames > 0).astype(np.uint8).reshape(frames.shape[0], -1)

    return T_tonic.Compose([
        T_tonic.ToFrame(sensor_size=SHD_SENSOR_SIZE, n_time_bins=n_time_bins),
        _flatten_binarize,
    ])


# ---------------------------------------------------------------------------
# Class index builder — no event I/O
# ---------------------------------------------------------------------------

def _build_class_index(dataset, data_root: str, train: bool) -> dict:
    """
    Return dict mapping class_int → list of global dataset indices.

    Tries common tonic attributes first (.targets, .data); falls back to a
    directory scan that relies on tonic iterating classes in sorted order.
    """
    # tonic >= 1.x sometimes exposes .targets
    if hasattr(dataset, "targets") and dataset.targets is not None:
        idx: dict = {}
        for i, label in enumerate(dataset.targets):
            idx.setdefault(int(label), []).append(i)
        return idx

    # tonic often stores (file_path, label) pairs in .data
    if hasattr(dataset, "data"):
        try:
            idx = {}
            for i, item in enumerate(dataset.data):
                label = int(item[1])
                idx.setdefault(label, []).append(i)
            return idx
        except (IndexError, TypeError, KeyError):
            pass

    # Fallback: reconstruct from directory structure.
    # Relies on tonic iterating class dirs in sorted order — safe for NMNIST
    # whose class dirs are single digits "0"–"9".
    split = "Train" if train else "Test"
    base  = pathlib.Path(data_root) / "NMNIST" / split
    global_i = 0
    idx = {}
    for c_str in sorted(d.name for d in base.iterdir() if d.is_dir()):
        try:
            c = int(c_str)
        except ValueError:
            continue
        n = len(sorted((base / c_str).glob("*.bin")))
        idx[c] = list(range(global_i, global_i + n))
        global_i += n
    return idx


def _build_shd_class_index(dataset) -> dict[int, list[int]]:
    """
    Return dict mapping SHD class_int → list of global dataset indices.

    tonic's SHD dataset does not populate .targets/.data in this environment,
    so read labels directly from the backing HDF5 file.
    """
    file_path = pathlib.Path(dataset.location_on_system) / dataset.data_filename
    with h5py.File(file_path, "r") as f:
        labels = np.asarray(f["labels"], dtype=np.int64)

    idx: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        idx.setdefault(int(label), []).append(i)
    return idx


def _shd_english_digit_ids(dataset) -> list[int]:
    """Return SHD class ids for English digit words in 0..9 order."""
    if not hasattr(dataset, "classes") or dataset.classes is None:
        raise RuntimeError("SHD dataset does not expose class names via .classes")

    class_names = [
        name.decode() if isinstance(name, (bytes, np.bytes_)) else str(name)
        for name in dataset.classes
    ]
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    missing = [name for name in _SHD_ENGLISH_DIGITS if name not in name_to_idx]
    if missing:
        raise RuntimeError(
            f"SHD English digit classes missing from dataset metadata: {missing}"
        )
    return [name_to_idx[name] for name in _SHD_ENGLISH_DIGITS]


# ---------------------------------------------------------------------------
# Per-split loader
# ---------------------------------------------------------------------------

def _load_split(
    data_root: str,
    train: bool,
    first_saccade_only: bool,
    transform,
    task: str,
    n_per_class: Optional[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one split (train or test), subsampled per class, with transforms."""

    dataset   = tonic.datasets.NMNIST(
        save_to=data_root,
        train=train,
        first_saccade_only=first_saccade_only,
        transform=transform,
    )
    class_idx = _build_class_index(dataset, data_root, train)

    NEG_CLASSES = [0, 1, 2, 3, 4, 5, 6, 8, 9]

    if task == "7_vs_rest":
        pos = rng.permutation(class_idx[7])[:n_per_class].tolist()
        n_neg_each = (
            max(1, (len(pos) + len(NEG_CLASSES) - 1) // len(NEG_CLASSES))
            if n_per_class else None
        )
        neg: list = []
        for c in NEG_CLASSES:
            idxs = rng.permutation(class_idx.get(c, []))
            neg.extend(idxs[:n_neg_each].tolist() if n_neg_each else idxs.tolist())
        neg_arr = rng.permutation(neg)[:len(pos)].tolist()
        chosen  = [(i, 1) for i in pos] + [(i, 0) for i in neg_arr]

    elif task == "0_vs_1":
        i0 = rng.permutation(class_idx[0])[:n_per_class].tolist()
        i1 = rng.permutation(class_idx[1])[:n_per_class].tolist()
        chosen = [(i, 0) for i in i0] + [(i, 1) for i in i1]

    elif task == "0_vs_8":
        i0 = rng.permutation(class_idx[0])[:n_per_class].tolist()
        i8 = rng.permutation(class_idx[8])[:n_per_class].tolist()
        chosen = [(i, 0) for i in i0] + [(i, 1) for i in i8]

    else:   # 10class
        chosen = []
        for c in range(10):
            idxs = rng.permutation(class_idx[c])[:n_per_class].tolist()
            chosen.extend((i, c) for i in idxs)

    # Shuffle
    chosen = [chosen[i] for i in rng.permutation(len(chosen))]

    # Load selected samples — transform is applied inside dataset.__getitem__
    X, y = [], []
    for global_idx, label in chosen:
        x, _ = dataset[global_idx]
        X.append(x)
        y.append(label)

    return np.stack(X), np.array(y, dtype=np.int64)


def _load_shd_split(
    data_root: str,
    train: bool,
    transform,
    task: str,
    n_per_class: Optional[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one SHD split, restricted to the English spoken digit classes."""

    dataset = tonic.datasets.SHD(
        save_to=data_root,
        train=train,
        transform=transform,
    )
    class_idx = _build_shd_class_index(dataset)
    english_ids = _shd_english_digit_ids(dataset)

    if task == "7_vs_rest":
        pos_class = english_ids[7]
        pos = rng.permutation(class_idx[pos_class])[:n_per_class].tolist()
        neg_classes = [english_ids[d] for d in range(10) if d != 7]
        n_neg_each = (
            max(1, (len(pos) + len(neg_classes) - 1) // len(neg_classes))
            if n_per_class else None
        )
        neg: list[int] = []
        for c in neg_classes:
            idxs = rng.permutation(class_idx[c])
            neg.extend(idxs[:n_neg_each].tolist() if n_neg_each else idxs.tolist())
        neg_arr = rng.permutation(neg)[:len(pos)].tolist()
        chosen = [(i, 1) for i in pos] + [(i, 0) for i in neg_arr]

    elif task == "0_vs_1":
        c0, c1 = english_ids[0], english_ids[1]
        i0 = rng.permutation(class_idx[c0])[:n_per_class].tolist()
        i1 = rng.permutation(class_idx[c1])[:n_per_class].tolist()
        chosen = [(i, 0) for i in i0] + [(i, 1) for i in i1]

    elif task == "0_vs_8":
        c0, c8 = english_ids[0], english_ids[8]
        i0 = rng.permutation(class_idx[c0])[:n_per_class].tolist()
        i8 = rng.permutation(class_idx[c8])[:n_per_class].tolist()
        chosen = [(i, 0) for i in i0] + [(i, 1) for i in i8]

    else:   # 10class
        chosen = []
        for digit, class_id in enumerate(english_ids):
            idxs = rng.permutation(class_idx[class_id])[:n_per_class].tolist()
            chosen.extend((i, digit) for i in idxs)

    chosen = [chosen[i] for i in rng.permutation(len(chosen))]

    X, y = [], []
    for global_idx, label in chosen:
        x, _ = dataset[global_idx]
        X.append(x)
        y.append(label)

    return np.stack(X), np.array(y, dtype=np.int64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def n_classes_for_task(task: str) -> int:
    """Return the number of output classes for a given task string."""
    if task in ("7_vs_rest", "0_vs_1", "0_vs_8"):
        return 2
    if task == "10class":
        return 10
    raise ValueError(f"Unknown task {task!r}. Choose: 7_vs_rest, 0_vs_1, 0_vs_8, 10class.")


def load_nmnist(
    data_root: str = _DEFAULT_DATA_ROOT,
    task: str = "7_vs_rest",
    n_time_bins: int = 10,
    grid_size: Optional[int] = 16,
    n_train_per_class: Optional[int] = 200,
    n_val_per_class: Optional[int] = 100,
    seed: int = 42,
    device: Optional[torch.device] = None,
    first_saccade_only: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load N-MNIST spike sequences via tonic as torch tensors.

    Parameters
    ----------
    data_root           : directory where tonic will find (or download) NMNIST
    task                : "7_vs_rest" | "0_vs_1" | "0_vs_8" | "10class"
    n_time_bins         : fixed temporal bins per sample (T axis)
    grid_size           : spatial downsampling side length (Column expects 16 → 256
                          inputs/step). Pass None to keep full 34×34 resolution.
    n_train_per_class   : samples per class from Train split (None = all)
    n_val_per_class     : samples per class from Test split  (None = all)
    seed                : RNG seed for reproducible subsampling
    device              : torch device for output tensors (default cpu)
    first_saccade_only  : if True, use only the first of the three N-MNIST saccades
                          (recommended: cleaner temporal structure, consistent with
                          most SNN benchmarks)

    Returns
    -------
    X_train : [N_train, T, F] long   F = grid_size² (or 34*34 if grid_size=None)
    y_train : [N_train] long
    X_val   : [N_val,   T, F] long
    y_val   : [N_val]   long
    """
    dev = device or torch.device("cpu")
    rng = np.random.default_rng(seed)
    n_classes_for_task(task)   # validate early

    transform = _make_transform(n_time_bins, grid_size)
    grid_str  = f"{grid_size}×{grid_size}" if grid_size else "34×34 (no pool)"

    print(
        f"Loading N-MNIST [{task}]  T={n_time_bins}  grid={grid_str}  "
        f"first_saccade_only={first_saccade_only}"
    )

    X_tr, y_tr = _load_split(
        data_root, True,  first_saccade_only, transform, task, n_train_per_class, rng
    )
    X_va, y_va = _load_split(
        data_root, False, first_saccade_only, transform, task, n_val_per_class,   rng
    )

    print(f"  train: {X_tr.shape}  val: {X_va.shape}")

    return (
        torch.from_numpy(X_tr).long().to(dev),
        torch.from_numpy(y_tr).long().to(dev),
        torch.from_numpy(X_va).long().to(dev),
        torch.from_numpy(y_va).long().to(dev),
    )


def load_shd(
    data_root: str = _DEFAULT_SHD_DATA_ROOT,
    task: str = "7_vs_rest",
    n_time_bins: int = 10,
    n_train_per_class: Optional[int] = 200,
    n_val_per_class: Optional[int] = 100,
    seed: int = 42,
    device: Optional[torch.device] = None,
    include_test_split: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load SHD English spoken-digit spike sequences via tonic as torch tensors.

    Parameters
    ----------
    data_root           : directory where tonic will find (or download) SHD
    task                : "7_vs_rest" | "0_vs_1" | "0_vs_8" | "10class"
                          over the English spoken digits only
    n_time_bins         : fixed temporal bins per sample (T axis)
    n_train_per_class   : samples per English class from Train split (None = all)
    n_val_per_class     : samples per English class from Test split  (None = all)
    seed                : RNG seed for reproducible subsampling
    device              : torch device for output tensors (default cpu)
    include_test_split  : if False, skip loading/downloading SHD test split and
                          return empty validation tensors

    Returns
    -------
    X_train : [N_train, T, 700] long
    y_train : [N_train] long
    X_val   : [N_val,   T, 700] long  (or empty if include_test_split=False)
    y_val   : [N_val]   long
    """
    dev = device or torch.device("cpu")
    rng = np.random.default_rng(seed)
    n_classes_for_task(task)   # validate early

    transform = _make_shd_transform(n_time_bins)

    print(f"Loading SHD English digits [{task}]  T={n_time_bins}  input=1x700")

    X_tr, y_tr = _load_shd_split(
        data_root, True, transform, task, n_train_per_class, rng
    )

    if include_test_split:
        X_va, y_va = _load_shd_split(
            data_root, False, transform, task, n_val_per_class, rng
        )
    else:
        X_va = np.empty((0, n_time_bins, SHD_SENSOR_SIZE[0]), dtype=np.uint8)
        y_va = np.empty((0,), dtype=np.int64)

    print(f"  train: {X_tr.shape}  val: {X_va.shape}")

    return (
        torch.from_numpy(X_tr).long().to(dev),
        torch.from_numpy(y_tr).long().to(dev),
        torch.from_numpy(X_va).long().to(dev),
        torch.from_numpy(y_va).long().to(dev),
    )
