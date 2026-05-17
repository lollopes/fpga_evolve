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

_DEFAULT_DATA_ROOT = "/home/lorenzo/Desktop/PhD/Projects/Mine/Various/sigprop_snn/data/"
_DEFAULT_SHD_DATA_ROOT = str(pathlib.Path(__file__).resolve().parent.parent / "datasets" / "SHD" / "data")
_SHD_ENGLISH_DIGITS = (
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
)


# ---------------------------------------------------------------------------
# Transform pipeline
# ---------------------------------------------------------------------------

def _make_transform(time_window: int):
    """
    Build the tonic-compatible transform pipeline.

    ToFrame bins events into fixed-duration windows of `time_window` µs → [T, 2, 34, 34]
    T varies per sample depending on recording duration.
    _binarize_flatten keeps both polarities and binarizes              → [T, 34*34*2]
    """

    def _binarize_flatten(frames: np.ndarray) -> np.ndarray:
        # frames: [T, 2, 34, 34]  (int counts from ToFrame)
        return (frames > 0).astype(np.uint8).reshape(frames.shape[0], -1)  # [T, 2312]

    return T_tonic.Compose([
        T_tonic.ToFrame(sensor_size=SENSOR_SIZE, time_window=time_window),
        _binarize_flatten,
    ])


def _make_shd_transform(time_window: int):
    """
    Build the SHD transform pipeline.

    ToFrame bins events into fixed-duration windows of `time_window` µs → [T, 1, 700]
    T varies per sample depending on recording duration.
    flatten/binarize keeps the 700 cochlea channels                      → [T, 700]
    """

    def _flatten_binarize(frames: np.ndarray) -> np.ndarray:
        return (frames > 0).astype(np.uint8).reshape(frames.shape[0], -1)

    return T_tonic.Compose([
        T_tonic.ToFrame(sensor_size=SHD_SENSOR_SIZE, time_window=time_window),
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

    chosen = _choose_nmnist_examples(class_idx, task, n_per_class, rng)
    return _load_nmnist_samples(dataset, chosen)


def _load_selected_samples(dataset, chosen: list[tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Load selected samples from a tonic dataset, assuming chosen=(global_idx, label)."""
    X, y = [], []
    for global_idx, label in chosen:
        x, _ = dataset[global_idx]
        X.append(x)
        y.append(label)

    return np.stack(X), np.array(y, dtype=np.int64)


def _load_nmnist_samples(dataset, chosen: list[tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load N-MNIST samples and zero-pad to the maximum T in this set.

    N-MNIST recordings have variable duration when using time_window framing,
    so ToFrame produces a different number of steps per sample. All samples are
    padded with zeros to the longest sequence in the set.
    """
    X, y = [], []
    for global_idx, label in chosen:
        x, _ = dataset[global_idx]   # x: [T_i, 2312]
        X.append(x)
        y.append(label)

    T_max = max(x.shape[0] for x in X)
    F     = X[0].shape[1]
    X_pad = np.zeros((len(X), T_max, F), dtype=X[0].dtype)
    for i, x in enumerate(X):
        X_pad[i, :x.shape[0]] = x

    return X_pad, np.array(y, dtype=np.int64)


def _load_shd_samples(dataset, chosen: list[tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load SHD samples and zero-pad to the maximum T in this set.

    SHD recordings have variable duration, so ToFrame with time_window produces
    a different number of steps per sample. All samples are padded with zeros to
    the longest sequence in the set; zero frames produce no spikes in the network.
    """
    X, y = [], []
    for global_idx, label in chosen:
        x, _ = dataset[global_idx]   # x: [T_i, 700]
        X.append(x)
        y.append(label)

    T_max = max(x.shape[0] for x in X)
    F     = X[0].shape[1]
    X_pad = np.zeros((len(X), T_max, F), dtype=X[0].dtype)
    for i, x in enumerate(X):
        X_pad[i, :x.shape[0]] = x

    return X_pad, np.array(y, dtype=np.int64)


def _shuffle_chosen(
    chosen: list[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Shuffle a chosen index/label list with the provided RNG."""
    return [chosen[i] for i in rng.permutation(len(chosen))]


def _resolve_split_counts(
    available: int,
    n_train: Optional[int],
    n_val: Optional[int],
    split_name: str,
) -> tuple[int, int]:
    """
    Resolve disjoint train/val sample counts from a single class pool.

    If one of n_train/n_val is None, the remainder of the available pool is used
    for that split. If both are None, the split would be ambiguous and is
    rejected.
    """
    if n_train is None and n_val is None:
        raise ValueError(
            f"{split_name}: at least one of n_train_per_class or n_val_per_class must be set"
        )

    if n_train is None:
        if n_val is None or n_val > available:
            raise ValueError(f"{split_name}: requested val count exceeds {available} available samples")
        return available - n_val, n_val

    if n_val is None:
        if n_train > available:
            raise ValueError(f"{split_name}: requested train count exceeds {available} available samples")
        return n_train, available - n_train

    if n_train + n_val > available:
        raise ValueError(
            f"{split_name}: requested {n_train + n_val} train/val samples but only {available} are available"
        )
    return n_train, n_val


def _choose_nmnist_examples(
    class_idx: dict[int, list[int]],
    task: str,
    n_per_class: Optional[int],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Choose labeled N-MNIST examples for one split without loading event data."""

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

    return _shuffle_chosen(chosen, rng)


def _choose_nmnist_train_val_examples(
    class_idx: dict[int, list[int]],
    task: str,
    n_train_per_class: Optional[int],
    n_val_per_class: Optional[int],
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Choose disjoint labeled N-MNIST train/val examples from the official train split."""
    train_chosen: list[tuple[int, int]] = []
    val_chosen: list[tuple[int, int]] = []

    if task == "7_vs_rest":
        neg_classes = [0, 1, 2, 3, 4, 5, 6, 8, 9]
        pos_pool = rng.permutation(class_idx[7]).tolist()
        n_train_pos, n_val_pos = _resolve_split_counts(
            len(pos_pool),
            n_train_per_class,
            n_val_per_class,
            "N-MNIST class 7",
        )

        train_pos = pos_pool[:n_train_pos]
        val_pos = pos_pool[n_train_pos : n_train_pos + n_val_pos]
        train_chosen.extend((i, 1) for i in train_pos)
        val_chosen.extend((i, 1) for i in val_pos)

        n_neg_each_train = (n_train_pos + len(neg_classes) - 1) // len(neg_classes) if n_train_pos else 0
        n_neg_each_val = (n_val_pos + len(neg_classes) - 1) // len(neg_classes) if n_val_pos else 0

        train_neg: list[int] = []
        val_neg: list[int] = []
        for c in neg_classes:
            idxs = rng.permutation(class_idx.get(c, [])).tolist()
            need = n_neg_each_train + n_neg_each_val
            if need > len(idxs):
                raise ValueError(
                    f"N-MNIST class {c}: requested {need} negatives for train/val but only {len(idxs)} are available"
                )
            train_neg.extend(idxs[:n_neg_each_train])
            val_neg.extend(idxs[n_neg_each_train : n_neg_each_train + n_neg_each_val])

        train_neg = rng.permutation(train_neg)[:n_train_pos].tolist()
        val_neg = rng.permutation(val_neg)[:n_val_pos].tolist()
        train_chosen.extend((i, 0) for i in train_neg)
        val_chosen.extend((i, 0) for i in val_neg)

    elif task == "0_vs_1":
        for src_class, label in ((0, 0), (1, 1)):
            idxs = rng.permutation(class_idx[src_class]).tolist()
            n_train, n_val = _resolve_split_counts(
                len(idxs),
                n_train_per_class,
                n_val_per_class,
                f"N-MNIST class {src_class}",
            )
            train_chosen.extend((i, label) for i in idxs[:n_train])
            val_chosen.extend((i, label) for i in idxs[n_train : n_train + n_val])

    elif task == "0_vs_8":
        for src_class, label in ((0, 0), (8, 1)):
            idxs = rng.permutation(class_idx[src_class]).tolist()
            n_train, n_val = _resolve_split_counts(
                len(idxs),
                n_train_per_class,
                n_val_per_class,
                f"N-MNIST class {src_class}",
            )
            train_chosen.extend((i, label) for i in idxs[:n_train])
            val_chosen.extend((i, label) for i in idxs[n_train : n_train + n_val])

    else:   # 10class
        for c in range(10):
            idxs = rng.permutation(class_idx[c]).tolist()
            n_train, n_val = _resolve_split_counts(
                len(idxs),
                n_train_per_class,
                n_val_per_class,
                f"N-MNIST class {c}",
            )
            train_chosen.extend((i, c) for i in idxs[:n_train])
            val_chosen.extend((i, c) for i in idxs[n_train : n_train + n_val])

    return _shuffle_chosen(train_chosen, rng), _shuffle_chosen(val_chosen, rng)


def _choose_shd_examples(
    class_idx: dict[int, list[int]],
    english_ids: list[int],
    task: str,
    n_per_class: Optional[int],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Choose labeled SHD examples for one split without loading event data."""

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
        chosen: list[tuple[int, int]] = [(i, 1) for i in pos] + [(i, 0) for i in neg_arr]

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

    return _shuffle_chosen(chosen, rng)


def _choose_shd_train_val_examples(
    class_idx: dict[int, list[int]],
    english_ids: list[int],
    task: str,
    n_train_per_class: Optional[int],
    n_val_per_class: Optional[int],
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Choose disjoint labeled SHD train/val examples from the official train split."""
    train_chosen: list[tuple[int, int]] = []
    val_chosen: list[tuple[int, int]] = []

    if task == "7_vs_rest":
        pos_class = english_ids[7]
        neg_classes = [english_ids[d] for d in range(10) if d != 7]

        pos_pool = rng.permutation(class_idx[pos_class]).tolist()
        n_train_pos, n_val_pos = _resolve_split_counts(
            len(pos_pool), n_train_per_class, n_val_per_class, "SHD English digit 7"
        )
        train_chosen.extend((i, 1) for i in pos_pool[:n_train_pos])
        val_chosen.extend((i, 1) for i in pos_pool[n_train_pos:n_train_pos + n_val_pos])

        n_neg_each_train = (n_train_pos + len(neg_classes) - 1) // len(neg_classes) if n_train_pos else 0
        n_neg_each_val   = (n_val_pos   + len(neg_classes) - 1) // len(neg_classes) if n_val_pos   else 0

        train_neg: list[int] = []
        val_neg:   list[int] = []
        for c in neg_classes:
            idxs = rng.permutation(class_idx.get(c, [])).tolist()
            need = n_neg_each_train + n_neg_each_val
            if need > len(idxs):
                raise ValueError(
                    f"SHD class {c}: requested {need} negatives for train/val "
                    f"but only {len(idxs)} available"
                )
            train_neg.extend(idxs[:n_neg_each_train])
            val_neg.extend(idxs[n_neg_each_train:n_neg_each_train + n_neg_each_val])

        train_neg = rng.permutation(train_neg)[:n_train_pos].tolist()
        val_neg   = rng.permutation(val_neg)[:n_val_pos].tolist()
        train_chosen.extend((i, 0) for i in train_neg)
        val_chosen.extend((i, 0) for i in val_neg)

    elif task == "0_vs_1":
        for src_digit, label in ((0, 0), (1, 1)):
            idxs = rng.permutation(class_idx[english_ids[src_digit]]).tolist()
            n_train, n_val = _resolve_split_counts(
                len(idxs), n_train_per_class, n_val_per_class,
                f"SHD English digit {src_digit}"
            )
            train_chosen.extend((i, label) for i in idxs[:n_train])
            val_chosen.extend((i, label) for i in idxs[n_train:n_train + n_val])

    elif task == "0_vs_8":
        for src_digit, label in ((0, 0), (8, 1)):
            idxs = rng.permutation(class_idx[english_ids[src_digit]]).tolist()
            n_train, n_val = _resolve_split_counts(
                len(idxs), n_train_per_class, n_val_per_class,
                f"SHD English digit {src_digit}"
            )
            train_chosen.extend((i, label) for i in idxs[:n_train])
            val_chosen.extend((i, label) for i in idxs[n_train:n_train + n_val])

    else:   # 10class
        for digit, class_id in enumerate(english_ids):
            idxs = rng.permutation(class_idx[class_id]).tolist()
            n_train, n_val = _resolve_split_counts(
                len(idxs), n_train_per_class, n_val_per_class,
                f"SHD English digit {digit}"
            )
            train_chosen.extend((i, digit) for i in idxs[:n_train])
            val_chosen.extend((i, digit) for i in idxs[n_train:n_train + n_val])

    return _shuffle_chosen(train_chosen, rng), _shuffle_chosen(val_chosen, rng)


def _load_shd_split(
    data_root: str,
    train: bool,
    transform,
    task: str,
    n_per_class: Optional[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one SHD split, restricted to the English spoken digit classes."""

    dataset     = tonic.datasets.SHD(save_to=data_root, train=train, transform=transform)
    class_idx   = _build_shd_class_index(dataset)
    english_ids = _shd_english_digit_ids(dataset)
    chosen      = _choose_shd_examples(class_idx, english_ids, task, n_per_class, rng)

    return _load_shd_samples(dataset, chosen)


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
    time_window: int = 1000,
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
    time_window         : frame duration in µs (ToFrame time_window parameter).
                          Both polarities are kept; samples are zero-padded to
                          the longest sequence in each split. F = 34*34*2 = 2312.
    n_train_per_class   : samples per class from Train split (None = all)
    n_val_per_class     : samples per class from Test split  (None = all)
    seed                : RNG seed for reproducible subsampling
    device              : torch device for output tensors (default cpu)
    first_saccade_only  : if True, use only the first of the three N-MNIST saccades

    Returns
    -------
    X_train : [N_train, T_train, 2312] long
    y_train : [N_train] long
    X_val   : [N_val,   T_val,   2312] long
    y_val   : [N_val]   long
    """
    dev = device or torch.device("cpu")
    rng = np.random.default_rng(seed)
    n_classes_for_task(task)   # validate early

    transform = _make_transform(time_window)

    print(
        f"Loading N-MNIST [{task}]  time_window={time_window}µs  input=34×34×2  "
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


def load_nmnist_train_val_test(
    data_root: str = _DEFAULT_DATA_ROOT,
    task: str = "7_vs_rest",
    time_window: int = 1000,
    n_train_per_class: Optional[int] = 200,
    n_val_per_class: Optional[int] = 100,
    n_test_per_class: Optional[int] = None,
    seed: int = 42,
    device: Optional[torch.device] = None,
    first_saccade_only: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load N-MNIST as a true train/val/test split.

    Train and validation are disjoint subsets drawn from the official N-MNIST
    training split. The final test set is drawn from the official N-MNIST test
    split and should be used only once after model selection.

    Both polarities are kept (F = 34*34*2 = 2312). Each split is zero-padded
    to the longest sequence in that split.
    """
    dev = device or torch.device("cpu")
    rng = np.random.default_rng(seed)
    n_classes_for_task(task)   # validate early

    transform = _make_transform(time_window)

    print(
        f"Loading N-MNIST [{task}]  time_window={time_window}µs  input=34×34×2  "
        f"first_saccade_only={first_saccade_only}"
    )

    train_dataset = tonic.datasets.NMNIST(
        save_to=data_root,
        train=True,
        first_saccade_only=first_saccade_only,
        transform=transform,
    )
    train_class_idx = _build_class_index(train_dataset, data_root, True)
    train_chosen, val_chosen = _choose_nmnist_train_val_examples(
        train_class_idx,
        task,
        n_train_per_class,
        n_val_per_class,
        rng,
    )

    test_dataset = tonic.datasets.NMNIST(
        save_to=data_root,
        train=False,
        first_saccade_only=first_saccade_only,
        transform=transform,
    )
    test_class_idx = _build_class_index(test_dataset, data_root, False)
    test_chosen = _choose_nmnist_examples(test_class_idx, task, n_test_per_class, rng)

    X_tr, y_tr = _load_nmnist_samples(train_dataset, train_chosen)
    X_va, y_va = _load_nmnist_samples(train_dataset, val_chosen)
    X_te, y_te = _load_nmnist_samples(test_dataset,  test_chosen)

    print(f"  train: {X_tr.shape}  val: {X_va.shape}  test: {X_te.shape}")

    return (
        torch.from_numpy(X_tr).long().to(dev),
        torch.from_numpy(y_tr).long().to(dev),
        torch.from_numpy(X_va).long().to(dev),
        torch.from_numpy(y_va).long().to(dev),
        torch.from_numpy(X_te).long().to(dev),
        torch.from_numpy(y_te).long().to(dev),
    )


def load_shd(
    data_root: str = _DEFAULT_SHD_DATA_ROOT,
    task: str = "7_vs_rest",
    time_window: int = 10000,
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
    time_window         : frame duration in µs (ToFrame time_window parameter).
                          Samples have variable T; each split is zero-padded to
                          the longest sequence in that split.
    n_train_per_class   : samples per English class from Train split (None = all)
    n_val_per_class     : samples per English class from Test split  (None = all)
    seed                : RNG seed for reproducible subsampling
    device              : torch device for output tensors (default cpu)
    include_test_split  : if False, skip loading/downloading SHD test split and
                          return empty validation tensors

    Returns
    -------
    X_train : [N_train, T_train, 700] long
    y_train : [N_train] long
    X_val   : [N_val,   T_val,   700] long  (or empty if include_test_split=False)
    y_val   : [N_val]   long
    """
    dev = device or torch.device("cpu")
    rng = np.random.default_rng(seed)
    n_classes_for_task(task)   # validate early

    transform = _make_shd_transform(time_window)

    print(f"Loading SHD English digits [{task}]  time_window={time_window}µs  input=700")

    X_tr, y_tr = _load_shd_split(
        data_root, True, transform, task, n_train_per_class, rng
    )

    if include_test_split:
        X_va, y_va = _load_shd_split(
            data_root, False, transform, task, n_val_per_class, rng
        )
    else:
        X_va = np.empty((0, 0, SHD_SENSOR_SIZE[0]), dtype=np.uint8)
        y_va = np.empty((0,), dtype=np.int64)

    print(f"  train: {X_tr.shape}  val: {X_va.shape}")

    return (
        torch.from_numpy(X_tr).long().to(dev),
        torch.from_numpy(y_tr).long().to(dev),
        torch.from_numpy(X_va).long().to(dev),
        torch.from_numpy(y_va).long().to(dev),
    )


def _or_pool_channels(X: np.ndarray, n_channels: int) -> np.ndarray:
    """OR-pool adjacent frequency channels: [N, T, 700] → [N, T, n_channels].

    Groups of ceil(700/n_channels) adjacent channels are collapsed with OR.
    Any channel that fires within a group produces a 1 in the pooled channel.
    The last group may be smaller if 700 is not divisible by n_channels.
    """
    N, T, F = X.shape
    group = F / n_channels           # may be non-integer
    out = np.zeros((N, T, n_channels), dtype=X.dtype)
    for i in range(n_channels):
        start = int(round(i * group))
        end   = int(round((i + 1) * group))
        out[:, :, i] = X[:, :, start:end].any(axis=-1)
    return out


def load_shd_train_val_test(
    data_root: str = _DEFAULT_SHD_DATA_ROOT,
    task: str = "7_vs_rest",
    time_window: int = 10000,
    n_train_per_class: Optional[int] = 200,
    n_val_per_class: Optional[int] = 50,
    n_test_per_class: Optional[int] = None,
    n_channels: Optional[int] = None,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load SHD English spoken-digit sequences as a true train/val/test split.

    Train and validation are disjoint subsets drawn from the official SHD
    training split. The final test set is drawn from the official SHD test
    split and should be used only once after model selection.

    Parameters
    ----------
    data_root           : directory where tonic will find (or download) SHD
    task                : "7_vs_rest" | "0_vs_1" | "0_vs_8" | "10class"
    time_window         : frame duration in µs (ToFrame time_window parameter).
                          Samples have variable T; each split is zero-padded to
                          the longest sequence in that split.
    n_train_per_class   : samples per English class from train split for training
    n_val_per_class     : samples per English class from train split for validation
                          (disjoint from training samples)
    n_test_per_class    : samples per English class from test split  (None = all)
    n_channels          : if set, OR-pool the 700 cochlear channels down to this
                          many channels before returning.  Adjacent channels are
                          grouped with OR (a group fires if any channel fires).
                          None (default) returns all 700 channels unchanged.
                          Recommended values: 50–100 (see benchmarks in codebase).
    seed                : RNG seed for reproducible subsampling
    device              : torch device for output tensors (default cpu)

    Returns
    -------
    X_train : [N_train, T_train, F] long   F=700 or n_channels if set
    y_train : [N_train] long
    X_val   : [N_val,   T_val,   F] long
    y_val   : [N_val]   long
    X_test  : [N_test,  T_test,  F] long
    y_test  : [N_test]  long
    """
    dev = device or torch.device("cpu")
    rng = np.random.default_rng(seed)
    n_classes_for_task(task)   # validate early

    transform = _make_shd_transform(time_window)
    f_str = f"{n_channels} (OR-pooled from 700)" if n_channels else "700"
    print(f"Loading SHD English digits [{task}]  time_window={time_window}µs  input={f_str}")

    train_dataset   = tonic.datasets.SHD(save_to=data_root, train=True,  transform=transform)
    train_class_idx = _build_shd_class_index(train_dataset)
    english_ids     = _shd_english_digit_ids(train_dataset)

    train_chosen, val_chosen = _choose_shd_train_val_examples(
        train_class_idx, english_ids, task, n_train_per_class, n_val_per_class, rng
    )

    test_dataset     = tonic.datasets.SHD(save_to=data_root, train=False, transform=transform)
    test_class_idx   = _build_shd_class_index(test_dataset)
    test_english_ids = _shd_english_digit_ids(test_dataset)
    test_chosen = _choose_shd_examples(
        test_class_idx, test_english_ids, task, n_test_per_class, rng
    )

    X_tr, y_tr = _load_shd_samples(train_dataset, train_chosen)
    X_va, y_va = _load_shd_samples(train_dataset, val_chosen)
    X_te, y_te = _load_shd_samples(test_dataset,  test_chosen)

    if n_channels is not None and n_channels < X_tr.shape[-1]:
        X_tr = _or_pool_channels(X_tr, n_channels)
        X_va = _or_pool_channels(X_va, n_channels)
        X_te = _or_pool_channels(X_te, n_channels)

    print(f"  train: {X_tr.shape}  val: {X_va.shape}  test: {X_te.shape}")

    return (
        torch.from_numpy(X_tr).long().to(dev),
        torch.from_numpy(y_tr).long().to(dev),
        torch.from_numpy(X_va).long().to(dev),
        torch.from_numpy(y_va).long().to(dev),
        torch.from_numpy(X_te).long().to(dev),
        torch.from_numpy(y_te).long().to(dev),
    )
