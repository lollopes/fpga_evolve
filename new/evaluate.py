"""
evaluate.py — Evaluate a batch of Boolean circuits on MNIST classification.

Pipeline (per batch of S samples):
  1. MNIST images normalized to [0,1] then thresholded → 784 bits per sample.
  2. The 784-bit vector is clamped at the circuit's external inputs and held
     constant for n_steps clock cycles. (Requires circuit.n_inputs == 784.)
  3. At every clock cycle, n_classes designated nodes are read as the one-hot
     prediction; their bits are compared to the target one-hot.
  4. Per-circuit loss = mean over (sample, timestep, class) of squared error.
     Range: [0, 1] — interpretable as the fraction of (s, t, c) cells where
     the output bit disagrees with the target bit.

Lower loss is better; an evolution loop pushes this down.

Batch axes
----------
  B  = circuit.batch_size  (vectorized inside circuit.step)
  S  = number of samples in the eval batch  (Python loop here)
  T  = n_steps  (Python loop here, inside circuit.step)

The S loop is intentional: each iteration is one circuit.step call which is
already vectorized over B. If the S loop becomes a bottleneck, lift state to
(B, S, N), broadcast circuit params, and replace the loop with a single
vectorized step.
"""

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

from circuit import Circuit


# ---------------------------------------------------------------------------
# Data loading — pure stdlib + numpy, no tensorflow / keras
# ---------------------------------------------------------------------------

_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

# Reliable Google-hosted mirror of the original Yann LeCun dataset.
_MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def _ensure_mnist_files(cache_dir: Path) -> dict[str, Path]:
    """Download MNIST IDX gz files into cache_dir if not already there."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, fname in _MNIST_FILES.items():
        local = cache_dir / fname
        if not local.exists():
            url = _MNIST_BASE_URL + fname
            print(f"  downloading {fname} ...")
            with urllib.request.urlopen(url) as resp, open(local, "wb") as f:
                f.write(resp.read())
        paths[key] = local
    return paths


def _read_idx(path: Path) -> np.ndarray:
    """Parse one gzipped IDX file (images or labels) into a numpy array.

    IDX format:
      4-byte big-endian magic, 4-byte big-endian count, optional dim headers,
      then raw uint8 data.
    """
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic == 0x00000803:
            rows, cols = struct.unpack(">II", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(count, rows, cols)
        elif magic == 0x00000801:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f"Unknown IDX magic 0x{magic:08x} in {path}")
    return data


def load_mnist_binary(
    n_train: int, n_test: int, threshold: float, cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST from a local IDX cache (downloading on first use) and binarize.

    Pixels are normalized to [0, 1] then thresholded against `threshold` →
    uint8 bits in {0, 1}.

    Parameters
    ----------
    n_train, n_test : int — number of samples to keep from the train/test splits
    threshold       : float in [0, 1]
    cache_dir       : pathlib.Path where the four .gz files are stored
                      (created if missing)

    Returns
    -------
    X_train : (n_train, 784) uint8
    y_train : (n_train,)     uint8 in [0, 10)
    X_test  : (n_test, 784)  uint8
    y_test  : (n_test,)      uint8 in [0, 10)
    """
    paths = _ensure_mnist_files(cache_dir)
    X_tr_raw = _read_idx(paths["train_images"])     # (60000, 28, 28) uint8
    y_tr_raw = _read_idx(paths["train_labels"])     # (60000,)
    X_te_raw = _read_idx(paths["test_images"])      # (10000, 28, 28)
    y_te_raw = _read_idx(paths["test_labels"])      # (10000,)

    X_tr = (X_tr_raw[:n_train].astype(np.float32) / 255.0 > threshold).astype(np.uint8)
    X_te = (X_te_raw[:n_test].astype(np.float32) / 255.0 > threshold).astype(np.uint8)
    X_tr = X_tr.reshape(X_tr.shape[0], -1)          # (n_train, 784)
    X_te = X_te.reshape(X_te.shape[0], -1)          # (n_test, 784)

    return (
        X_tr,
        y_tr_raw[:n_train].astype(np.uint8),
        X_te,
        y_te_raw[:n_test].astype(np.uint8),
    )


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Encode integer labels as one-hot uint8 vectors.

    Parameters
    ----------
    y : (S,) int — class indices in [0, n_classes)
    n_classes : int

    Returns
    -------
    (S, n_classes) uint8
    """
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")
    if int(y.max(initial=0)) >= n_classes or int(y.min(initial=0)) < 0:
        raise ValueError(
            f"y has labels outside [0, {n_classes})"
        )
    out = np.zeros((y.shape[0], n_classes), dtype=np.uint8)
    out[np.arange(y.shape[0]), y] = 1
    return out


# ---------------------------------------------------------------------------
# Forward pass: trajectory for one input batch
# ---------------------------------------------------------------------------

def run_static(
    circuit: Circuit, ext_inputs: np.ndarray, n_steps: int,
) -> np.ndarray:
    """Run the batch of circuits with constant external inputs and record the trajectory.

    Parameters
    ----------
    circuit : Circuit (batch_size = B)
    ext_inputs : (B, n_inputs) uint8 — held constant for every clock cycle
    n_steps : int

    Returns
    -------
    trajectory : (n_steps + 1, B, n_nodes) uint8
        trajectory[0] is the zero initial state; trajectory[t] is the state
        after t clock cycles.
    """
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")
    B, N = circuit.batch_size, circuit.n_nodes
    trajectory = np.zeros((n_steps + 1, B, N), dtype=np.uint8)
    trajectory[0] = circuit.zero_state()
    for t in range(n_steps):
        trajectory[t + 1] = circuit.step(trajectory[t], ext_inputs)
    return trajectory


# ---------------------------------------------------------------------------
# Per-circuit loss + accuracy on a batch of (X, y) samples
# ---------------------------------------------------------------------------

def evaluate(
    circuit: Circuit,
    X: np.ndarray,
    y: np.ndarray,
    output_node_ids: np.ndarray,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-pass each (X, y) sample through B circuits and return (loss, accuracy).

    Computed in a single vectorized sweep over n_steps clock cycles:

      loss[b]     = mean over (s, t, c) of (state[b,s,t,output_ids[c]] XOR y_oh[s,c])
                    For binary outputs vs binary one-hot targets, XOR equals
                    squared error. Range [0, 1].
      accuracy[b] = fraction of samples where argmax_c (Σ_t state[b,s,t,output_ids[c]])
                    equals y[s]. Range [0, 1].

    Argmax breaks ties toward the lowest class index, so an all-zero output
    always predicts class 0 and accuracy collapses to "fraction of samples
    that are class 0".

    Parameters
    ----------
    circuit : Circuit (batch_size = B)
    X : (S, n_inputs) uint8 — binary inputs, one row per sample
    y : (S,) integer — class labels in [0, n_classes)
    output_node_ids : (n_classes,) int — which nodes are read out
    n_steps : int — clock cycles per sample (T >= 1)

    Returns
    -------
    loss : (B,) float32
    accuracy : (B,) float32
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    B = circuit.batch_size
    S = int(X.shape[0])
    K = circuit.n_inputs
    N = circuit.n_nodes

    if X.ndim != 2 or X.shape[1] != K:
        raise ValueError(f"X must have shape (S, {K}), got {X.shape}")
    if y.ndim != 1 or y.shape[0] != S:
        raise ValueError(f"y must have shape ({S},), got {y.shape}")

    out_ids = np.asarray(output_node_ids, dtype=np.int64)
    n_classes = int(out_ids.shape[0])
    if out_ids.size and (int(out_ids.max()) >= N or int(out_ids.min()) < 0):
        raise ValueError(
            f"output_node_ids out of range [0, {N}); got "
            f"min={int(out_ids.min())} max={int(out_ids.max())}"
        )
    y_int = np.asarray(y, dtype=np.int64)
    if int(y_int.max(initial=0)) >= n_classes or int(y_int.min(initial=0)) < 0:
        raise ValueError(
            f"y has labels outside [0, {n_classes}); "
            f"min={int(y_int.min())}, max={int(y_int.max())}"
        )

    # One-hot encoding for the loss term.
    y_oh = np.zeros((S, n_classes), dtype=np.uint8)
    y_oh[np.arange(S), y_int] = 1
    y_oh_b = y_oh[None, :, :]                                     # (1, S, n_classes)

    state = np.zeros((B, S, N), dtype=np.uint8)
    sse = np.zeros((B, S), dtype=np.float32)
    counts = np.zeros((B, S, n_classes), dtype=np.int32)
    ext = np.asarray(X, dtype=np.uint8)                            # (S, K)

    for _t in range(n_steps):
        state = circuit.step(state, ext)                           # (B, S, N)
        out = state[:, :, out_ids]                                 # (B, S, n_classes), uint8
        sse += (out ^ y_oh_b).sum(axis=-1, dtype=np.float32)       # (B, S)
        counts += out                                              # (B, S, n_classes)

    loss = (sse.sum(axis=1) / float(S * n_steps * n_classes)).astype(np.float32)
    preds = counts.argmax(axis=-1)                                 # (B, S)
    accuracy = (preds == y_int[None, :]).mean(axis=-1).astype(np.float32)

    return loss, accuracy


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from circuit import genome_bits

    BATCH_SIZE = 16
    N_INPUTS = 28 * 28
    N_NODES = 64
    N_CLASSES = 10
    N_STEPS = 8
    N_TRAIN = 200
    N_TEST = 100
    THRESHOLD = 0.5
    SEED = 0
    OUTPUT_NODES = np.arange(N_NODES - N_CLASSES, N_NODES, dtype=np.int64)
    CACHE_DIR = Path(__file__).resolve().parent.parent / "datasets" / "mnist"

    rng = np.random.default_rng(SEED)
    n_bits = genome_bits(N_NODES, N_INPUTS)
    genome = rng.integers(0, 2, size=(BATCH_SIZE, n_bits), dtype=np.uint8)
    circuit = Circuit(genome, N_NODES, N_INPUTS)
    print(circuit)

    X_tr, y_tr, X_te, y_te = load_mnist_binary(N_TRAIN, N_TEST, THRESHOLD, CACHE_DIR)
    print(f"X_train: {X_tr.shape}  y_train: {y_tr.shape}  (active pixel fraction={X_tr.mean():.3f})")

    loss, accuracy = evaluate(circuit, X_tr, y_tr, OUTPUT_NODES, N_STEPS)
    best_idx = int(np.argmin(loss))
    print(f"L2 loss per circuit:    {np.round(loss, 4).tolist()}")
    print(f"Accuracy per circuit:   {[f'{a:.1%}' for a in accuracy.tolist()]}")
    print(
        f"best loss : idx={best_idx}  "
        f"loss={float(loss.min()):.4f}  acc={float(accuracy[best_idx]):.1%}\n"
        f"max acc   : idx={int(np.argmax(accuracy))}  "
        f"acc={float(accuracy.max()):.1%}\n"
        f"random baseline (10 classes, balanced): ~10%"
    )
