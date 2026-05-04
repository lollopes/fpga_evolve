"""
tasks.py — Supervised Boolean classification tasks.

Each dataset function returns (X, y):
  X : np.ndarray of shape (n_samples, n_inputs), dtype uint8, values in {0,1}
  y : np.ndarray of shape (n_samples,),          dtype uint8, values in {0,1}

evaluate_binary_task runs the network on every sample and measures accuracy.
"""

from __future__ import annotations
import pathlib
from typing import Optional, Tuple

import numpy as np

from .config import EvoConfig
from .network import EvoBooleanNetwork


# ---------------------------------------------------------------------------
# N-MNIST helpers (5-byte binary format)
# ---------------------------------------------------------------------------

def _read_nmnist_bin(path) -> np.ndarray:
    """Parse one N-MNIST .bin file into a structured event array."""
    with open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    n = len(raw) // 5
    raw = raw[: n * 5].reshape(n, 5)
    dtype = np.dtype([("x", "<i8"), ("y", "<i8"), ("t", "<i8"), ("p", "<i8")])
    events = np.zeros(n, dtype=dtype)
    events["x"] = raw[:, 0]
    events["y"] = raw[:, 1]
    events["p"] = (raw[:, 2] >> 7) & 1
    events["t"] = (
        ((raw[:, 2] & 0x7F).astype(np.int64) << 16)
        | (raw[:, 3].astype(np.int64) << 8)
        | raw[:, 4]
    )
    return events


def _pool_spatial(spatial: np.ndarray, grid_size: int) -> np.ndarray:
    """Pool a (H, W) count frame to (grid_size, grid_size) and binarize."""
    H = spatial.shape[0]
    crop = (H // grid_size) * grid_size
    step = crop // grid_size
    pooled = (
        spatial[:crop, :crop]
        .reshape(grid_size, step, grid_size, step)
        .sum(axis=(1, 3))
    )
    return (pooled > 0).astype(np.uint8).flatten()


def _events_to_binary(events, frame_t, grid_size: int, sensor_size) -> np.ndarray:
    """Convert event stream to a flat binary vector of length grid_size**2.

    All values are guaranteed to be in {0, 1}.
    """
    frame = frame_t(events)              # (1, 2, H, W)
    spatial = frame.sum(axis=(0, 1))     # (H, W) — event count per pixel
    return _pool_spatial(spatial, grid_size)


def _events_to_sequence(events, frame_t, grid_size: int, n_time_bins: int) -> np.ndarray:
    """Convert event stream to a binary spike sequence of shape (n_time_bins, grid_size**2).

    Each time bin is pooled and binarized independently.
    All values are guaranteed to be in {0, 1}.
    """
    frame = frame_t(events)   # (T, 2, H, W)
    seq = np.zeros((n_time_bins, grid_size * grid_size), dtype=np.uint8)
    for t in range(n_time_bins):
        spatial = frame[t].sum(axis=0)   # (H, W)
        seq[t] = _pool_spatial(spatial, grid_size)
    return seq


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

def xor_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """2-input XOR truth table (4 samples)."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    y = np.array([0, 1, 1, 0], dtype=np.uint8)
    return X, y


def and_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """2-input AND truth table (4 samples)."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    y = np.array([0, 0, 0, 1], dtype=np.uint8)
    return X, y


def or_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """2-input OR truth table (4 samples)."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    y = np.array([0, 1, 1, 1], dtype=np.uint8)
    return X, y


def nmnist_dataset(
    data_root: str,
    train: bool = True,
    grid_size: int = 8,
    class_a: int = 0,
    class_b: int = 1,
    n_samples_per_class: int = 100,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load N-MNIST and convert event streams to binary feature vectors.

    Each sample is projected onto a grid_size × grid_size spatial grid.
    A cell is 1 if any event (either polarity) fell in that region, else 0.
    All input values are guaranteed to be in {0, 1}.

    Two-class split: samples from class_a get label 0,
                     samples from class_b get label 1.

    Parameters
    ----------
    data_root : str
        Path whose 'NMNIST/Train/{class}/*.bin' subtree holds the dataset.
    train : bool
        Load the Train split (True) or Test split (False).
    grid_size : int
        Side length of the spatial grid (n_inputs = grid_size**2).
    class_a, class_b : int
        Which NMNIST digit classes to use (0-9). class_a → label 0, class_b → label 1.
    n_samples_per_class : int
        How many samples to draw from each class.
    seed : int | None
        RNG seed for reproducible shuffling.

    Returns
    -------
    X : np.ndarray of shape (2*n_samples_per_class, grid_size**2), dtype uint8, values in {0,1}
    y : np.ndarray of shape (2*n_samples_per_class,), dtype uint8, values in {0,1}
    """
    try:
        import tonic
        import tonic.transforms as T
    except ImportError:
        raise ImportError("tonic is required: pip install tonic")

    sensor_size = tonic.datasets.NMNIST.sensor_size  # (34, 34, 2)
    frame_t = T.ToFrame(sensor_size=sensor_size, n_time_bins=1)
    rng = np.random.default_rng(seed)

    split = "Train" if train else "Test"
    split_dir = pathlib.Path(data_root) / "NMNIST" / split

    def _load_class(class_id: int) -> np.ndarray:
        files = sorted((split_dir / str(class_id)).glob("*.bin"))
        idxs = rng.permutation(len(files))[:n_samples_per_class]
        samples = []
        for i in idxs:
            events = _read_nmnist_bin(files[i])
            samples.append(_events_to_binary(events, frame_t, grid_size, sensor_size))
        return np.array(samples, dtype=np.uint8)

    Xa = _load_class(class_a)
    Xb = _load_class(class_b)
    ya = np.zeros(len(Xa), dtype=np.uint8)
    yb = np.ones(len(Xb), dtype=np.uint8)

    X = np.vstack([Xa, Xb])
    y = np.concatenate([ya, yb])
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def nmnist_sequence_dataset(
    data_root: str,
    train: bool = True,
    grid_size: int = 8,
    n_time_bins: int = 16,
    class_a: int = 0,
    class_b: int = 1,
    n_samples_per_class: int = 100,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load N-MNIST as binary spike sequences for time-varying input to the network.

    Each sample becomes a (n_time_bins, grid_size**2) binary array where
    sequence[t] is the spatial activity frame for time bin t.
    All values are guaranteed to be in {0, 1}.

    Designed for use with EvoBooleanNetwork.run_sequence(), where one time bin
    is fed as external_inputs at each clock cycle.

    Parameters
    ----------
    data_root : str
        Path whose 'NMNIST/Train/{class}/*.bin' subtree holds the dataset.
    train : bool
        Load the Train split (True) or Test split (False).
    grid_size : int
        Side length of the spatial grid (n_inputs = grid_size**2 per step).
    n_time_bins : int
        Number of temporal bins. Should match EvoConfig.n_steps so each
        network clock cycle sees one input frame.
    class_a, class_b : int
        Which NMNIST digit classes to use (0-9). class_a → label 0, class_b → label 1.
    n_samples_per_class : int
        How many samples to draw from each class.
    seed : int | None
        RNG seed for reproducible shuffling.

    Returns
    -------
    X : np.ndarray of shape (2*n_samples_per_class, n_time_bins, grid_size**2),
        dtype uint8, values in {0,1}
    y : np.ndarray of shape (2*n_samples_per_class,), dtype uint8, values in {0,1}
    """
    try:
        import tonic
        import tonic.transforms as T
    except ImportError:
        raise ImportError("tonic is required: pip install tonic")

    sensor_size = tonic.datasets.NMNIST.sensor_size  # (34, 34, 2)
    frame_t = T.ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins)
    rng = np.random.default_rng(seed)

    split = "Train" if train else "Test"
    split_dir = pathlib.Path(data_root) / "NMNIST" / split

    def _load_class(class_id: int) -> np.ndarray:
        files = sorted((split_dir / str(class_id)).glob("*.bin"))
        idxs = rng.permutation(len(files))[:n_samples_per_class]
        samples = []
        for i in idxs:
            events = _read_nmnist_bin(files[i])
            samples.append(_events_to_sequence(events, frame_t, grid_size, n_time_bins))
        return np.array(samples, dtype=np.uint8)

    Xa = _load_class(class_a)
    Xb = _load_class(class_b)
    ya = np.zeros(len(Xa), dtype=np.uint8)
    yb = np.ones(len(Xb), dtype=np.uint8)

    X = np.vstack([Xa, Xb])
    y = np.concatenate([ya, yb])
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def pattern_dataset(n_inputs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full truth table for n_inputs bits (2^n_inputs samples).

    Label is 1 if the majority of input bits are 1 (majority vote), else 0.
    For even n_inputs, ties are labeled 0.
    """
    n_samples = 2 ** n_inputs
    X = np.zeros((n_samples, n_inputs), dtype=np.uint8)
    for i in range(n_samples):
        for bit in range(n_inputs):
            X[i, bit] = (i >> bit) & 1
    y = (X.sum(axis=1) > n_inputs / 2).astype(np.uint8)
    return X, y


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_binary_task(
    network: EvoBooleanNetwork,
    genome: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    output_node: int = 0,
    n_steps: Optional[int] = None,
    mode: str = "final",
) -> Tuple[float, np.ndarray]:
    """
    Evaluate a genome on a binary classification task.

    For each sample in X:
      1. Run the network from a zero initial state.
      2. Apply external_inputs = X[i].
      3. Read a single output node (default: config.output_nodes[0]).
      4. Compare to target y[i].

    Parameters
    ----------
    network : EvoBooleanNetwork
    genome : np.ndarray
    X : np.ndarray of shape (n_samples, n_inputs)
    y : np.ndarray of shape (n_samples,)
    output_node : int
        Index into network.config.output_nodes to use for prediction.
        E.g., output_node=0 reads config.output_nodes[0].
    n_steps : int | None
        Overrides config.n_steps if provided.
    mode : str
        "final"    — use the state at the last timestep.
        "activity" — threshold on cumulative activity (>= n_steps/2 -> 1).

    Returns
    -------
    accuracy : float in [0, 1]
    predictions : np.ndarray of shape (n_samples,), dtype uint8
    """
    config = network.config
    n_samples = X.shape[0]
    steps = n_steps if n_steps is not None else config.n_steps
    out_idx = config.output_nodes[output_node]

    predictions = np.zeros(n_samples, dtype=np.uint8)

    for i in range(n_samples):
        traj = network.run(
            genome,
            external_inputs=X[i],
            n_steps=steps,
            initial_state=None,
            return_trajectory=True,
        )

        if mode == "final":
            pred = int(traj[-1, out_idx])
        elif mode == "activity":
            # Winner-takes-all: sum spikes over time per output node, pick argmax
            activities = np.array([traj[:, idx].sum() for idx in config.output_nodes])
            pred = int(np.argmax(activities))
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        predictions[i] = pred

    accuracy = float(np.mean(predictions == y))
    return accuracy, predictions


def evaluate_sequence_task(
    network: EvoBooleanNetwork,
    genome: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    output_node: int = 0,
    mode: str = "final",
) -> Tuple[float, np.ndarray]:
    """
    Evaluate a genome on a binary classification task with time-varying inputs.

    For each sample in X:
      1. Run the network from a zero initial state using run_sequence().
      2. Feed X[i, t] as external_inputs at clock cycle t.
      3. Read output node(s) and compare to target y[i].

    Parameters
    ----------
    network : EvoBooleanNetwork
    genome : np.ndarray
    X : np.ndarray of shape (n_samples, T, n_inputs), values in {0,1}
    y : np.ndarray of shape (n_samples,)
    output_node : int
        Used only in "final" mode: index into config.output_nodes to read.
    mode : str
        "final"    — predict from the state of output_nodes[output_node] at t=T.
        "activity" — winner-takes-all: sum spikes across all timesteps for each
                     output node in config.output_nodes; predict argmax.
                     Requires len(config.output_nodes) >= n_classes.

    Returns
    -------
    accuracy : float in [0, 1]
    predictions : np.ndarray of shape (n_samples,), dtype uint8
    """
    config = network.config
    n_samples, T = X.shape[0], X.shape[1]
    out_idx = config.output_nodes[output_node]

    predictions = np.zeros(n_samples, dtype=np.uint8)

    for i in range(n_samples):
        traj = network.run_sequence(
            genome,
            input_sequence=X[i],
            initial_state=None,
            return_trajectory=True,
        )

        if mode == "final":
            pred = int(traj[-1, out_idx])
        elif mode == "activity":
            # Sum spikes over time for each output node, pick the most active one
            activities = np.array([traj[:, idx].sum() for idx in config.output_nodes])
            pred = int(np.argmax(activities))
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        predictions[i] = pred

    accuracy = float(np.mean(predictions == y))
    return accuracy, predictions
