"""
inspect_genome.py — Print the decoded configuration of every node.

Run from the project root:
    python examples/inspect_genome.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
from evo_boolean_network import EvoConfig, random_genome, describe_genome

# ------------------------------------------------------------------
# Small network for readability
# ------------------------------------------------------------------
config = EvoConfig(
    n_nodes=6,
    n_inputs=2,
    n_steps=4,
    seed=123,
)

genome = random_genome(config, np.random.default_rng(config.seed))

print("=" * 70)
print("Genome inspection")
print("=" * 70)
print(f"  n_nodes          : {config.n_nodes}")
print(f"  n_inputs         : {config.n_inputs}")
print(f"  n_candidates     : {config.n_candidates_per_node}")
print(f"  sel_bits         : {config.sel_bits}")
print(f"  node_genome_bits : {config.node_genome_bits}")
print(f"  total_bits       : {config.total_genome_bits}")
print()
print("Candidate vector layout for a given node:")
print("  positions 0..n_inputs-1 -> external inputs")
print("  positions n_inputs..    -> other node states (node_id skipped)")
print()

nodes = describe_genome(genome, config)

header = f"{'Node':>5} | {'selA':>5} {'selB':>5} | {'LUT bits':>8} | {'function':>8} | src_A -> src_B"
print(header)
print("-" * len(header))

for n in nodes:
    lut_str = "".join(str(b) for b in n["lut_bits"])

    # Human-readable source labels
    def src_label(src: int) -> str:
        if src < 0:
            return f"ext[{-src-1}]"
        return f"node[{src}]"

    print(
        f"{n['node_id']:>5} | "
        f"{n['sel_a_eff']:>5} {n['sel_b_eff']:>5} | "
        f"{lut_str:>8} | "
        f"{n['lut_name']:>8} | "
        f"{src_label(n['src_a'])} -> {src_label(n['src_b'])}"
    )

print()
print("LUT bit indexing: lut_bits[index] where index = (A << 1) | B")
print("  index 0 -> A=0, B=0")
print("  index 1 -> A=0, B=1")
print("  index 2 -> A=1, B=0")
print("  index 3 -> A=1, B=1")
