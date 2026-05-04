"""
solve_nmnist.py — Evolve a Boolean network to classify Neuromorphic MNIST
                  using time-varying spike inputs.

Each N-MNIST sample is a DVS event stream. We divide time into n_time_bins
bins and the sensor plane into a grid_size × grid_size spatial grid, producing
a binary spike sequence of shape (n_time_bins, grid_size²).

At each clock cycle t the network receives the t-th frame as external_inputs,
so the Boolean network integrates temporal spike information across its
recurrent dynamics — a genuine neuromorphic inference loop.

Default task: digit 0 vs digit 1, 16 time steps, 8×8 spatial grid (64 inputs).

Run from the project root:
    python examples/solve_nmnist.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
from evo_boolean_network import (
    EvoConfig,
    EvoBooleanNetwork,
    nmnist_sequence_dataset,
    evaluate_sequence_task,
    evolve_binary_classifier,
    EvolutionViewer,
)

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
DATA_ROOT   = str(pathlib.Path(__file__).parent.parent / "datasets")
GRID_SIZE   = 8    # spatial grid → 8×8 = 64 binary inputs per step
N_TIME_BINS = 16   # temporal bins == network steps
CLASS_A     = 0    # digit class → label 0
CLASS_B     = 1    # digit class → label 1
N_TRAIN     = 150  # samples per class for training
N_TEST      = 50   # samples per class for test
SEED        = 42

# ------------------------------------------------------------------
# Load sequences
# ------------------------------------------------------------------
print("Loading N-MNIST spike sequences ...")
X_train, y_train = nmnist_sequence_dataset(
    data_root=DATA_ROOT,
    train=True,
    grid_size=GRID_SIZE,
    n_time_bins=N_TIME_BINS,
    class_a=CLASS_A,
    class_b=CLASS_B,
    n_samples_per_class=N_TRAIN,
    seed=SEED,
)
X_test, y_test = nmnist_sequence_dataset(
    data_root=DATA_ROOT,
    train=False,
    grid_size=GRID_SIZE,
    n_time_bins=N_TIME_BINS,
    class_a=CLASS_A,
    class_b=CLASS_B,
    n_samples_per_class=N_TEST,
    seed=SEED + 1,
)

n_inputs = GRID_SIZE * GRID_SIZE
print(f"  Train : {X_train.shape[0]} samples, shape {X_train.shape}  (samples × time × inputs)")
print(f"  Test  : {X_test.shape[0]} samples")
print(f"  Input value range: [{X_train.min()}, {X_train.max()}]  (must be 0/1)")
print(f"  Class balance (train): {(y_train==0).sum()} vs {(y_train==1).sum()}")
print()

# ------------------------------------------------------------------
# Network config
# n_steps must match N_TIME_BINS so each clock cycle sees one input frame
# ------------------------------------------------------------------
config = EvoConfig(
    n_nodes=128,
    n_inputs=n_inputs,
    n_steps=N_TIME_BINS,
    output_nodes=[0, 1],   # node 0 → class A, node 1 → class B
    seed=SEED,
)

print("=" * 60)
print(f"N-MNIST sequence evolution  (digit {CLASS_A} vs digit {CLASS_B})")
print("=" * 60)
print(f"  Spatial grid : {GRID_SIZE}×{GRID_SIZE} = {n_inputs} inputs/step")
print(f"  Time bins    : {N_TIME_BINS}  (one frame per clock cycle)")
print(f"  Network      : {config.n_nodes} nodes, {config.n_steps} steps")
print(f"  Genome       : {config.total_genome_bits} bits")
print()

# ------------------------------------------------------------------
# Live viewer — updates every generation
# ------------------------------------------------------------------
viewer = EvolutionViewer(config, port=8050, auto_port=True)

# ------------------------------------------------------------------
# Evolve  (X_train is 3-D → evolve_binary_classifier routes to
#          evaluate_sequence_task automatically via X.ndim == 3)
# ------------------------------------------------------------------
result = evolve_binary_classifier(
    config=config,
    X=X_train,
    y=y_train,
    pop_size=80,
    n_jobs=-1,
    n_generations=1000,
    mutation_rate=0.01,
    elite_size=8,
    output_node=0,
    mode="activity",   # WTA: argmax of per-node spike counts
    seed=SEED,
    verbose=True,
    print_every=10,
    snapshot_callback=viewer.update,
    snapshot_every=1,
    eval_verbose=True,
    eval_progress_every=20,
    report_pred_balance=True,
    pred_balance_every=5,
)

# ------------------------------------------------------------------
# Evaluate on held-out test set
# ------------------------------------------------------------------
print()
print("=" * 60)
print("Test-set evaluation")
print("=" * 60)

network = EvoBooleanNetwork(config)

train_acc, _ = evaluate_sequence_task(
    network, result.best_genome, X_train, y_train, mode="activity"
)
test_acc, test_preds = evaluate_sequence_task(
    network, result.best_genome, X_test, y_test, mode="activity"
)

print(f"  Train accuracy : {train_acc:.2%}")
print(f"  Test  accuracy : {test_acc:.2%}")
print()

# ------------------------------------------------------------------
# Show a few test samples as ASCII spike rasters
# ------------------------------------------------------------------
def show_raster(seq, grid_size):
    """Print a (T, grid_size²) binary sequence as a spatial ASCII grid per time step."""
    for t, frame in enumerate(seq):
        row = "".join("█" if frame[c] else "·" for c in range(grid_size * grid_size))
        print(f"    t={t:02d}  {row}")

label_name = {0: str(CLASS_A), 1: str(CLASS_B)}
n_show = min(3, len(X_test))
print(f"Spike rasters for first {n_show} test samples:")
for i in range(n_show):
    true_lbl = label_name[int(y_test[i])]
    pred_lbl = label_name[int(test_preds[i])]
    ok = "OK" if y_test[i] == test_preds[i] else "FAIL"
    print(f"\n  Sample {i} | true={true_lbl}  pred={pred_lbl}  {ok}")
    show_raster(X_test[i], GRID_SIZE)

viewer.wait()
