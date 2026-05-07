from __future__ import annotations
from typing import Optional, Tuple
import os

import torch
from torch.utils.data import DataLoader
import tonic
import tonic.transforms as T

from datasets.utils import parse_task, build_class_index, subsample_indices, LabelRemapDataset


SENSOR_SIZE = tonic.datasets.SHD.sensor_size   # (700, 1, 1) — 700 cochlea channels
N_CLASSES   = 20
IN_DIM      = 700


def _default_data_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


def dataset_init(
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    task: str = "full",
    batch_size: int = 128,
    num_workers: int = 8,
    save_to: Optional[str] = None,
    seed: int = 42,
    time_window: int = 10000,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Load Spiking Heidelberg Digits (SHD) as PyTorch DataLoaders.

    Parameters
    ----------
    n_train    : total training samples to use (None = all)
    n_test     : total test samples to use (None = all)
    task       : "full"      — 20-class classification
                 "A_vs_B"   — binary, e.g. "5_vs_12"
                 "A_vs_all" — binary, e.g. "5_vs_all"
    batch_size : DataLoader batch size
    num_workers: DataLoader worker processes
    save_to    : directory where tonic stores/finds the dataset
    seed       : RNG seed for reproducible subsampling
    time_window: temporal bin width in microseconds (ToFrame)

    Returns
    -------
    train_loader, test_loader, out_dim, in_dim
    """
    save_to = _default_data_root() if save_to is None else os.path.abspath(os.path.expanduser(save_to))
    os.makedirs(save_to, exist_ok=True)

    task_info = parse_task(task, N_CLASSES)
    out_dim   = task_info["out_dim"]

    transform = T.Compose([
        T.ToFrame(sensor_size=SENSOR_SIZE, time_window=time_window),
        lambda x: x.reshape(x.shape[0], -1),
    ])

    train_ds_raw = tonic.datasets.SHD(save_to=save_to, train=True,  transform=transform)
    test_ds_raw  = tonic.datasets.SHD(save_to=save_to, train=False, transform=transform)

    train_class_idx = build_class_index(train_ds_raw)
    test_class_idx  = build_class_index(test_ds_raw)

    train_pairs = subsample_indices(train_class_idx, task_info, n_train, seed)
    test_pairs  = subsample_indices(test_class_idx,  task_info, n_test,  seed + 1)

    train_dataset = LabelRemapDataset(train_ds_raw, train_pairs)
    test_dataset  = LabelRemapDataset(test_ds_raw,  test_pairs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=True),
        num_workers=num_workers,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=True),
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader, out_dim, IN_DIM
