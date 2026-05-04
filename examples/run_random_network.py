"""
run_random_network.py — Demo: random genome on a 16-node, 4-input network.

Run from the project root:
    python examples/run_random_network.py
"""

import sys
import pathlib

# Allow running from the project root without installing the package
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
from evo_boolean_network import EvoConfig, EvoBooleanNetwork, random_genome

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
config = EvoConfig(
    n_nodes=16,
    n_inputs=4,
    n_steps=8,
    output_nodes=[12, 13, 14, 15],  # last 4 nodes as outputs
    seed=42,
)

print("=" * 60)
print("Network configuration")
print("=" * 60)
print(f"  n_nodes            : {config.n_nodes}")
print(f"  n_inputs           : {config.n_inputs}")
print(f"  n_steps            : {config.n_steps}")
print(f"  n_candidates/node  : {config.n_candidates_per_node}")
print(f"  sel_bits           : {config.sel_bits}")
print(f"  node_genome_bits   : {config.node_genome_bits}")
print(f"  total_genome_bits  : {config.total_genome_bits}")
print(f"  output_nodes       : {config.output_nodes}")

# ------------------------------------------------------------------
# Random genome
# ------------------------------------------------------------------
rng = np.random.default_rng(config.seed)
genome = random_genome(config, rng)
print(f"\nGenome (first 32 bits): {genome[:32]}")
print(f"Genome length: {len(genome)} bits")

# ------------------------------------------------------------------
# Input vector
# ------------------------------------------------------------------
external_inputs = np.array([1, 0, 1, 1], dtype=np.uint8)
print(f"\nExternal inputs: {external_inputs}")

# ------------------------------------------------------------------
# Run the network
# ------------------------------------------------------------------
network = EvoBooleanNetwork(config)
trajectory = network.run(
    genome,
    external_inputs=external_inputs,
    n_steps=config.n_steps,
    initial_state=None,
    return_trajectory=True,
)

print("\n" + "=" * 60)
print("Trajectory  (rows=timestep, cols=node index)")
print("=" * 60)
print(f"{'t':>4} | {'state':}")
for t, row in enumerate(trajectory):
    label = "  (initial)" if t == 0 else ""
    print(f"{t:>4} | {''.join(str(b) for b in row)}{label}")

# ------------------------------------------------------------------
# Read outputs
# ------------------------------------------------------------------
final_outputs = network.read_outputs(trajectory, mode="final")
activity = network.read_outputs(trajectory, mode="activity")

print(f"\nOutput nodes {config.output_nodes}:")
print(f"  Final state  : {final_outputs}")
print(f"  Activity sum : {activity}")
