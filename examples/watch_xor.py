"""
watch_xor.py — Animated evolution viewer on the XOR task (no dataset needed).

Good for quickly verifying the live windows work before running the full
N-MNIST experiment.

Run from the project root:
    python examples/watch_xor.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from evo_boolean_network import (
    EvoConfig,
    evolve_binary_classifier,
    xor_dataset,
    EvolutionViewer,
)

config = EvoConfig(n_nodes=8, n_inputs=2, n_steps=8, output_nodes=[0], seed=7)
X, y = xor_dataset()

viewer = EvolutionViewer(config)

result = evolve_binary_classifier(
    config=config,
    X=X,
    y=y,
    pop_size=100,
    n_generations=500,
    mutation_rate=0.02,
    elite_size=5,
    output_node=0,
    seed=7,
    verbose=True,
    print_every=50,
    snapshot_callback=viewer.update,
    snapshot_every=5,
)

print(f"\nBest accuracy: {result.best_accuracy:.2%}")
viewer.wait()
