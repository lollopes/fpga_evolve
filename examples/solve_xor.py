"""
solve_xor.py — Evolve a Boolean network to compute XOR.

Run from the project root:
    python examples/solve_xor.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
from evo_boolean_network import (
    EvoConfig,
    EvoBooleanNetwork,
    xor_dataset,
    evaluate_binary_task,
    evolve_binary_classifier,
)

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
config = EvoConfig(
    n_nodes=2,
    n_inputs=2,
    n_steps=8,
    output_nodes=[0],  # single output: last node
    seed=7,
)

X, y = xor_dataset()

print("=" * 60)
print("XOR evolution demo")
print("=" * 60)
print(f"  Network: {config.n_nodes} nodes, {config.n_inputs} inputs, {config.n_steps} steps")
print(f"  Genome bits: {config.total_genome_bits}")
print(f"  Dataset:\n    X={X.tolist()}\n    y={y.tolist()}")
print()

# ------------------------------------------------------------------
# Evolve
# ------------------------------------------------------------------
result = evolve_binary_classifier(
    config=config,
    X=X,
    y=y,
    pop_size=100,
    n_generations=500,
    mutation_rate=0.02,
    elite_size=5,
    output_node=0,
    seed=config.seed,
    verbose=True,
    print_every=50,
)

# ------------------------------------------------------------------
# Evaluate best genome
# ------------------------------------------------------------------
print()
print("=" * 60)
print("Best genome evaluation on XOR truth table")
print("=" * 60)

network = EvoBooleanNetwork(config)
acc, preds = evaluate_binary_task(
    network, result.best_genome, X, y, output_node=0
)

print(f"Accuracy: {acc:.2%}")
print()
print(f"{'A':>3} {'B':>3} | {'target':>6} {'pred':>6} {'ok':>3}")
print("-" * 30)
for (a, b), target, pred in zip(X, y, preds):
    ok = "OK" if target == pred else "FAIL"
    print(f"{a:>3} {b:>3} | {target:>6} {pred:>6} {ok:>3}")
