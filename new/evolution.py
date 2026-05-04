"""
evolution.py — Truncation-selection evolution for Boolean circuits.

One generation:
  1. Decode the current `(B, L)` genome batch into a Circuit.
  2. Evaluate L2 classification loss per circuit.
  3. Sort by loss (ascending — lower is better).
  4. Survivors  = top half of the population.
  5. Children   = bit-flip mutations of the survivors.
  6. Next pop   = survivors ++ children   (length B).

No crossover, no fancy strategies. Survivors are carried over verbatim —
elitism by construction, so best-ever fitness is non-increasing.

Everything is batched: a population of B circuits == one Circuit object with
batch_size = B; mutation operates on the (B, L) genome tensor.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np

from circuit import Circuit, genome_bits
from evaluate import evaluate, load_mnist_binary


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def mutate(
    genomes: np.ndarray, mutation_rate: float, rng: np.random.Generator,
) -> np.ndarray:
    """Independent bit-flip mutation per bit.

    Parameters
    ----------
    genomes : (B, L) uint8 — one row per genome
    mutation_rate : float in [0, 1] — per-bit flip probability
    rng : np.random.Generator

    Returns
    -------
    mutated : (B, L) uint8 — new array; original is not modified
    """
    if not (0.0 <= mutation_rate <= 1.0):
        raise ValueError(f"mutation_rate must be in [0, 1], got {mutation_rate}")
    flip_mask = rng.random(size=genomes.shape) < mutation_rate
    return np.where(flip_mask, 1 - genomes, genomes).astype(np.uint8)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def evolve(
    pop_size: int,
    n_generations: int,
    n_nodes: int,
    n_inputs: int,
    X: np.ndarray,
    y: np.ndarray,
    output_node_ids: np.ndarray,
    n_steps: int,
    mutation_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Run truncation-selection evolution for `n_generations` generations.

    Parameters
    ----------
    pop_size : int — number of circuits in the population (must be even)
    n_generations : int
    n_nodes, n_inputs : int — network dimensions
    X : (S, n_inputs) uint8 — binary inputs (training batch)
    y : (S,) integer — class labels in [0, n_classes)
    output_node_ids : (n_classes,) int — which nodes are the readout
    n_steps : int — clock cycles per sample
    mutation_rate : float — per-bit flip probability for children
    seed : int

    Returns
    -------
    final_genomes : (pop_size, L) uint8 — sorted by loss (best→worst)
    final_losses  : (pop_size,) float32 — aligned with final_genomes
    final_accs    : (pop_size,) float32 — aligned with final_genomes
    history       : list[dict] — per-generation metrics (loss + accuracy)
    """
    if pop_size < 2 or pop_size % 2 != 0:
        raise ValueError(f"pop_size must be an even integer >= 2, got {pop_size}")
    if n_generations < 1:
        raise ValueError(f"n_generations must be >= 1, got {n_generations}")

    rng = np.random.default_rng(seed)
    L = genome_bits(n_nodes, n_inputs)
    half = pop_size // 2

    genomes = rng.integers(0, 2, size=(pop_size, L), dtype=np.uint8)
    history: list[dict] = []

    for gen in range(n_generations):
        circuit = Circuit(genomes, n_nodes, n_inputs)
        losses, accs = evaluate(circuit, X, y, output_node_ids, n_steps)

        # Selection: rank by accuracy DESCENDING, break ties by loss ASCENDING.
        # np.lexsort takes the LAST key as primary, earlier keys as tiebreakers.
        order = np.lexsort((losses, -accs))
        genomes = genomes[order]                       # best accuracy → worst
        losses = losses[order]
        accs = accs[order]

        history.append({
            "gen": gen,
            "best_acc": float(accs[0]),
            "mean_acc": float(accs.mean()),
            "loss_at_best_acc": float(losses[0]),
            "mean_loss": float(losses.mean()),
            "min_loss": float(losses.min()),
        })
        print(
            f"gen {gen:4d} | acc best={float(accs[0]):.1%} mean={float(accs.mean()):.1%} "
            f"| loss={losses[0]:.4f} mean={losses.mean():.4f}"
        )

        # Build next population: survivors + mutated copies of survivors.
        survivors = genomes[:half]
        children = mutate(survivors, mutation_rate, rng)
        genomes = np.concatenate([survivors, children], axis=0)

    # Final evaluation so the returned arrays are aligned and sorted by acc.
    final_circuit = Circuit(genomes, n_nodes, n_inputs)
    final_losses, final_accs = evaluate(
        final_circuit, X, y, output_node_ids, n_steps,
    )
    order = np.lexsort((final_losses, -final_accs))
    return genomes[order], final_losses[order], final_accs[order], history


# ---------------------------------------------------------------------------
# Demo: evolve on MNIST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    POP_SIZE = 200
    N_GENERATIONS = 500
    N_NODES = 36
    N_INPUTS = 28 * 28
    N_CLASSES = 10
    N_STEPS = 8
    MUTATION_RATE = 0.01
    N_TRAIN = 64           
    THRESHOLD = 0.5
    SEED = 42
    OUTPUT_NODES = np.arange(N_NODES - N_CLASSES, N_NODES, dtype=np.int64)
    CACHE_DIR = Path(__file__).resolve().parent.parent / "datasets" / "mnist"

    print(
        f"Population: {POP_SIZE} | Generations: {N_GENERATIONS} | "
        f"N={N_NODES} K={N_INPUTS} T={N_STEPS} | "
        f"mutation={MUTATION_RATE} | train_size={N_TRAIN}"
    )

    X_tr, y_tr, _, _ = load_mnist_binary(N_TRAIN, 1, THRESHOLD, CACHE_DIR)
    print(f"Loaded MNIST: X={X_tr.shape}  active fraction={X_tr.mean():.3f}")

    # Majority-class baseline: always predict the most common label in the train set.
    class_counts = np.bincount(y_tr, minlength=N_CLASSES)
    majority_acc = float(class_counts.max() / class_counts.sum())
    print(f"Majority-class baseline (always predict most common): {majority_acc:.1%}\n")

    final_genomes, final_losses, final_accs, history = evolve(
        pop_size=POP_SIZE,
        n_generations=N_GENERATIONS,
        n_nodes=N_NODES,
        n_inputs=N_INPUTS,
        X=X_tr,
        y=y_tr,
        output_node_ids=OUTPUT_NODES,
        n_steps=N_STEPS,
        mutation_rate=MUTATION_RATE,
        seed=SEED,
    )

    g0 = history[0]
    print(
        f"\n=== Done ===\n"
        f"  initial : acc best={g0['best_acc']:.1%}  mean={g0['mean_acc']:.1%}  "
        f"| loss={g0['loss_at_best_acc']:.4f}\n"
        f"  final   : acc best={float(final_accs[0]):.1%}  "
        f"mean={float(final_accs.mean()):.1%}  "
        f"| loss={float(final_losses[0]):.4f}\n"
        f"  improvement on best acc : "
        f"{(float(final_accs[0]) - g0['best_acc']) * 100:+.1f} pp\n"
        f"  baselines : majority-class={majority_acc:.1%}  random=10.0%  perfect=100.0%"
    )

    # Visualize the best circuit of the final population, running it on a single
    # MNIST sample so the activity traces are real (not all-zero state).
    from evaluate import run_static
    from visualize import write_html

    best_circuit = Circuit(final_genomes[0:1], N_NODES, N_INPUTS)   # keep batch dim
    sample_idx = 0
    ext_inputs = np.broadcast_to(X_tr[sample_idx], (1, N_INPUTS))    # (1, K)
    trajectory = run_static(best_circuit, ext_inputs, N_STEPS)       # (T+1, 1, N)

    viz_state = trajectory[:, 0, :]                                  # (T+1, N)
    viz_ext = np.broadcast_to(X_tr[sample_idx], (N_STEPS + 1, N_INPUTS))  # (T+1, K)

    output_path = Path(__file__).resolve().parent / "best_circuit.html"
    write_html(
        best_circuit, output_path, 0, viz_state, viz_ext,
        title=f"Best evolved circuit  |  digit={int(y_tr[sample_idx])}  "
              f"|  loss={float(final_losses[0]):.4f}",
    )
    print(f"Saved {output_path}")

    # Save the best genome + structural metadata so the live server
    # (server.py) can reload it without re-running evolution.
    checkpoint_path = Path(__file__).resolve().parent / "best_circuit.npz"
    np.savez(
        checkpoint_path,
        genome=final_genomes[0],
        n_nodes=np.array(N_NODES, dtype=np.int64),
        n_inputs=np.array(N_INPUTS, dtype=np.int64),
        n_steps=np.array(N_STEPS, dtype=np.int64),
        output_node_ids=OUTPUT_NODES,
        train_loss=np.array(float(final_losses[0]), dtype=np.float32),
        train_acc=np.array(float(final_accs[0]), dtype=np.float32),
    )
    print(f"Saved {checkpoint_path}")

