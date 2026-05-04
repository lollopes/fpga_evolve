"""
plot_spike_raster.py — Visualise node activity as a spike raster.

Two plots are produced:

  1. Single raster — all nodes, one input vector, 8 timesteps.
  2. Grid raster   — output nodes only, one subplot per XOR input combination.

Run from the project root:
    python examples/plot_spike_raster.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
from evo_boolean_network import (
    EvoConfig,
    EvoBooleanNetwork,
    random_genome,
    xor_dataset,
    evolve_binary_classifier,
    plot_spike_raster,
    plot_raster_grid,
)

# ------------------------------------------------------------------
# 1. Single raster: random genome, one input, show all nodes
# ------------------------------------------------------------------
config = EvoConfig(n_nodes=16, n_inputs=4, n_steps=8,
                   output_nodes=[12, 13, 14, 15], seed=42)

rng = np.random.default_rng(config.seed)
genome = random_genome(config, rng)
network = EvoBooleanNetwork(config)

traj = network.run(genome, external_inputs=np.array([1, 0, 1, 1], dtype=np.uint8))

plot_spike_raster(
    traj,
    nodes=list(range(config.n_nodes)),       # all 16 nodes
    title="Random genome — all nodes — input [1,0,1,1]",
    show=True,
)

# ------------------------------------------------------------------
# 2. Grid raster: evolved XOR genome, one subplot per input combo
#    showing only the output node (node 15)
# ------------------------------------------------------------------
xor_config = EvoConfig(n_nodes=16, n_inputs=2, n_steps=8,
                       output_nodes=[15], seed=7)

X, y = xor_dataset()

result = evolve_binary_classifier(
    config=xor_config, X=X, y=y,
    pop_size=100, n_generations=500, mutation_rate=0.02,
    elite_size=5, seed=xor_config.seed, verbose=False,
)

xor_network = EvoBooleanNetwork(xor_config)

# Collect one trajectory per XOR input sample
trajectories = []
input_labels = []
for i, (inp, label) in enumerate(zip(X, y)):
    traj = xor_network.run(result.best_genome, external_inputs=inp)
    trajectories.append(traj)
    input_labels.append(
        f"Input [{inp[0]}, {inp[1]}]  →  target={label}"
    )

plot_raster_grid(
    trajectories,
    input_labels=input_labels,
    nodes=xor_config.output_nodes,          # only node 15
    node_labels=["node 15 (output)"],
    suptitle="Evolved XOR genome — output node raster per input",
    show=True,
)
