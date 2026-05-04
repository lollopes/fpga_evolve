"""
watch_nmnist.py — Evolve a Boolean network on N-MNIST with live animated windows.

Two windows open immediately and update every `SNAPSHOT_EVERY` generations:

  "Fitness"       — best + mean accuracy over time.
  "Best network"  — connectivity matrix (which node wires to which) and
                    a histogram of LUT function types (AND, XOR, …).

Run from the project root:
    python examples/watch_nmnist.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
from evo_boolean_network import (
    EvoConfig,
    evolve_binary_classifier,
    nmnist_sequence_dataset,
    EvolutionViewer,
)

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
DATA_ROOT      = str(pathlib.Path(__file__).parent.parent / "datasets")
GRID_SIZE      = 8
N_TIME_BINS    = 16
CLASS_A        = 0
CLASS_B        = 1
N_TRAIN        = 150
SEED           = 42
SNAPSHOT_EVERY = 5    # redraw windows every N generations

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
print("Loading N-MNIST ...")
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
print(f"  {X_train.shape[0]} training samples  shape={X_train.shape}")

# ------------------------------------------------------------------
# Network config
# ------------------------------------------------------------------
config = EvoConfig(
    n_nodes=128,
    n_inputs=GRID_SIZE * GRID_SIZE,
    n_steps=N_TIME_BINS,
    output_nodes=[0, 1],
    seed=SEED,
)
print(f"  {config.n_nodes} nodes  {config.n_inputs} inputs  genome={config.total_genome_bits} bits\n")

# ------------------------------------------------------------------
# Open live viewer
# ------------------------------------------------------------------
viewer = EvolutionViewer(config)

# ------------------------------------------------------------------
# Evolve
# ------------------------------------------------------------------
result = evolve_binary_classifier(
    config=config,
    X=X_train,
    y=y_train,
    pop_size=200,
    n_generations=2000,
    mutation_rate=0.01,
    elite_size=10,
    output_node=0,
    mode="activity",
    seed=SEED,
    verbose=True,
    print_every=50,
    snapshot_callback=viewer.update,
    snapshot_every=SNAPSHOT_EVERY,
)

print(f"\nBest training accuracy: {result.best_accuracy:.2%}")

# Keep windows open until the user closes them
viewer.wait()
