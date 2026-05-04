"""
evolution.py — Simple (1+λ) / (μ+λ) evolutionary algorithm.

Strategy:
  1. Initialize a random population of genomes.
  2. Evaluate all on the task (fitness = accuracy).
  3. Keep the top `elite_size` genomes unchanged.
  4. Fill the rest of the next generation by copying and mutating elites.
  5. Repeat for `n_generations`.

No crossover — mutations only. This keeps the search simple and the code
readable. The focus is on verifying that evolution works at all; more
sophisticated strategies can be layered on top.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

SnapshotCallback = Callable[[int, np.ndarray, np.ndarray], None]
"""Signature: (generation: int, best_genome: ndarray, fitnesses: ndarray) -> None"""

from .config import EvoConfig
from .genome import random_genome
from .network import EvoBooleanNetwork
from .tasks import evaluate_binary_task, evaluate_sequence_task

# Worker globals used by ProcessPoolExecutor initializer.
_WORKER_NETWORK: Optional[EvoBooleanNetwork] = None
_WORKER_X: Optional[np.ndarray] = None
_WORKER_Y: Optional[np.ndarray] = None
_WORKER_OUTPUT_NODE: int = 0
_WORKER_N_STEPS: Optional[int] = None
_WORKER_MODE: str = "final"
_WORKER_USE_SEQUENCE: bool = False


def _resolve_n_jobs(n_jobs: int) -> int:
    """
    Resolve n_jobs into a positive worker count.

    Conventions:
      -1  -> use most available cores (cpu_count - 1, minimum 1)
      < -1 -> cpu_count + 1 + n_jobs
       1+ -> exact worker count
    """
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0. Use 1 for serial or -1 for most cores.")

    cpu = os.cpu_count() or 1
    if n_jobs == -1:
        return max(1, cpu - 1)
    if n_jobs < -1:
        return max(1, cpu + 1 + n_jobs)
    return max(1, n_jobs)


def _init_eval_worker(
    config: EvoConfig,
    X: np.ndarray,
    y: np.ndarray,
    output_node: int,
    n_steps: Optional[int],
    mode: str,
) -> None:
    """Initialize one worker process with read-only task state."""
    global _WORKER_NETWORK, _WORKER_X, _WORKER_Y
    global _WORKER_OUTPUT_NODE, _WORKER_N_STEPS, _WORKER_MODE, _WORKER_USE_SEQUENCE

    _WORKER_NETWORK = EvoBooleanNetwork(config)
    _WORKER_X = X
    _WORKER_Y = y
    _WORKER_OUTPUT_NODE = output_node
    _WORKER_N_STEPS = n_steps
    _WORKER_MODE = mode
    _WORKER_USE_SEQUENCE = (X.ndim == 3)


def _evaluate_genome_worker(genome: np.ndarray) -> float:
    """Evaluate one genome inside a worker process and return accuracy."""
    if _WORKER_NETWORK is None or _WORKER_X is None or _WORKER_Y is None:
        raise RuntimeError("Parallel worker not initialized.")

    if _WORKER_USE_SEQUENCE:
        acc, _ = evaluate_sequence_task(
            _WORKER_NETWORK,
            genome,
            _WORKER_X,
            _WORKER_Y,
            output_node=_WORKER_OUTPUT_NODE,
            mode=_WORKER_MODE,
        )
    else:
        acc, _ = evaluate_binary_task(
            _WORKER_NETWORK,
            genome,
            _WORKER_X,
            _WORKER_Y,
            output_node=_WORKER_OUTPUT_NODE,
            n_steps=_WORKER_N_STEPS,
            mode=_WORKER_MODE,
        )
    return float(acc)


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def mutate_genome(
    genome: np.ndarray,
    mutation_rate: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Bit-flip mutation: each bit flips independently with probability mutation_rate.

    Returns a new genome array (original is not modified).
    """
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.random(len(genome)) < mutation_rate
    mutated = genome.copy()
    mutated[mask] ^= 1  # XOR flip
    return mutated


# ---------------------------------------------------------------------------
# Population evaluation
# ---------------------------------------------------------------------------

def evaluate_population(
    population: list[np.ndarray],
    network: EvoBooleanNetwork,
    X: np.ndarray,
    y: np.ndarray,
    output_node: int = 0,
    n_steps: Optional[int] = None,
    mode: str = "final",
    n_jobs: int = 1,
    executor: Optional[ProcessPoolExecutor] = None,
    eval_verbose: bool = False,
    eval_progress_every: int = 0,
    eval_progress_prefix: str = "",
) -> np.ndarray:
    """
    Evaluate every genome in the population and return an accuracy array.

    Parameters
    ----------
    n_jobs : int
        Worker process count. Use 1 for serial, -1 for most available cores.
    executor : ProcessPoolExecutor | None
        Optional pre-created pool. Intended for internal reuse across generations.
    eval_verbose : bool
        If True, print progress while genomes are being evaluated.
    eval_progress_every : int
        Progress print interval in completed genomes. If <= 0, uses ~10 updates.
    eval_progress_prefix : str
        Optional text prefix for progress lines (for example "Gen 0003").

    Returns
    -------
    np.ndarray of shape (pop_size,) with accuracy values in [0, 1].
    """
    total = len(population)
    if total == 0:
        return np.zeros(0, dtype=np.float64)

    if eval_progress_every <= 0:
        eval_progress_every = max(1, total // 10)

    def _print_progress(done: int, elapsed_s: float) -> None:
        if not eval_verbose:
            return
        if done != total and done % eval_progress_every != 0:
            return
        pct = 100.0 * done / total
        prefix = f"{eval_progress_prefix} " if eval_progress_prefix else ""
        print(f"{prefix}eval {done:4d}/{total:4d} ({pct:5.1f}%)  elapsed={elapsed_s:6.1f}s")

    n_workers = _resolve_n_jobs(n_jobs)
    if n_workers == 1 or total <= 1:
        t0 = time.perf_counter()
        fitnesses = np.zeros(total, dtype=np.float64)
        use_sequence = X.ndim == 3
        for i, genome in enumerate(population):
            if use_sequence:
                acc, _ = evaluate_sequence_task(
                    network, genome, X, y, output_node=output_node, mode=mode,
                )
            else:
                acc, _ = evaluate_binary_task(
                    network, genome, X, y, output_node=output_node,
                    n_steps=n_steps, mode=mode,
                )
            fitnesses[i] = acc
            _print_progress(i + 1, time.perf_counter() - t0)
        return fitnesses

    # Parallel path: one worker process evaluates each genome independently.
    def _parallel_eval(pool: ProcessPoolExecutor) -> np.ndarray:
        t0 = time.perf_counter()
        accs = np.zeros(total, dtype=np.float64)
        futures = {
            pool.submit(_evaluate_genome_worker, genome): idx
            for idx, genome in enumerate(population)
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            accs[idx] = float(fut.result())
            done += 1
            _print_progress(done, time.perf_counter() - t0)
        return accs

    if executor is None:
        with ProcessPoolExecutor(
            max_workers=min(n_workers, total),
            initializer=_init_eval_worker,
            initargs=(network.config, X, y, output_node, n_steps, mode),
        ) as pool:
            return _parallel_eval(pool)

    return _parallel_eval(executor)


def _evaluate_with_predictions(
    network: EvoBooleanNetwork,
    genome: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    output_node: int = 0,
    n_steps: Optional[int] = None,
    mode: str = "final",
) -> tuple[float, np.ndarray]:
    """Evaluate one genome and return (accuracy, predictions)."""
    if X.ndim == 3:
        return evaluate_sequence_task(
            network,
            genome,
            X,
            y,
            output_node=output_node,
            mode=mode,
        )
    return evaluate_binary_task(
        network,
        genome,
        X,
        y,
        output_node=output_node,
        n_steps=n_steps,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Main evolutionary loop
# ---------------------------------------------------------------------------

@dataclass
class EvolutionResult:
    best_genome: np.ndarray
    best_accuracy: float
    history: list[float] = field(default_factory=list)   # best accuracy per generation


def evolve_binary_classifier(
    config: EvoConfig,
    X: np.ndarray,
    y: np.ndarray,
    pop_size: int = 50,
    n_generations: int = 200,
    mutation_rate: float = 0.02,
    elite_size: int = 5,
    output_node: int = 0,
    n_steps: Optional[int] = None,
    mode: str = "final",
    seed: Optional[int] = None,
    verbose: bool = True,
    print_every: int = 10,
    snapshot_callback: Optional[SnapshotCallback] = None,
    snapshot_every: int = 10,
    n_jobs: int = -1,
    eval_verbose: bool = False,
    eval_progress_every: int = 0,
    report_pred_balance: bool = False,
    pred_balance_every: int = 0,
) -> EvolutionResult:
    """
    Evolve a Boolean network genome to solve a binary classification task.

    Parameters
    ----------
    config : EvoConfig
        Network structure.
    X : np.ndarray of shape (n_samples, n_inputs) or (n_samples, T, n_inputs)
    y : np.ndarray of shape (n_samples,)
    pop_size : int
        Number of genomes in the population.
    n_generations : int
        Number of evolutionary generations.
    mutation_rate : float
        Probability of flipping each bit during mutation.
    elite_size : int
        Number of top genomes preserved unchanged each generation.
    output_node : int
        Which output node to read for predictions.
    n_steps : int | None
        Timesteps per forward pass. Defaults to config.n_steps.
    mode : str
        "final" or "activity" — how to read the output.
    seed : int | None
        RNG seed for reproducibility.
    verbose : bool
        Print progress every `print_every` generations.
    print_every : int
        Logging interval (generations).
    snapshot_callback : SnapshotCallback | None
        If provided, called every `snapshot_every` generations (and on the final
        generation) with (gen, best_genome, fitnesses). Use this to drive live
        visualisation without modifying the evolution loop.
    snapshot_every : int
        How often to invoke snapshot_callback (in generations).
    n_jobs : int
        Number of worker processes for population fitness evaluation.
        Use 1 for serial, -1 for most available cores, or a positive integer.
    eval_verbose : bool
        If True, print population-evaluation progress inside each generation.
    eval_progress_every : int
        Population progress print interval in completed genomes. If <= 0, uses
        about 10 updates per generation.
    report_pred_balance : bool
        If True, print class-0/class-1 prediction fractions for the generation-best
        genome at intervals controlled by pred_balance_every.
    pred_balance_every : int
        Interval (in generations) to print prediction balance. If <= 0, uses
        print_every.

    Returns
    -------
    EvolutionResult with best_genome, best_accuracy, and history.
    """
    rng = np.random.default_rng(seed if seed is not None else config.seed)
    network = EvoBooleanNetwork(config)

    # --- Initialize random population ---
    population = [random_genome(config, rng) for _ in range(pop_size)]

    result = EvolutionResult(
        best_genome=population[0].copy(),
        best_accuracy=0.0,
    )

    n_workers = _resolve_n_jobs(n_jobs)
    if n_workers > 1 and pop_size > 1:
        with ProcessPoolExecutor(
            max_workers=min(n_workers, pop_size),
            initializer=_init_eval_worker,
            initargs=(config, X, y, output_node, n_steps, mode),
        ) as executor:
            for gen in range(n_generations):
                fitnesses = evaluate_population(
                    population, network, X, y,
                    output_node=output_node, n_steps=n_steps, mode=mode,
                    n_jobs=n_workers, executor=executor,
                    eval_verbose=eval_verbose,
                    eval_progress_every=eval_progress_every,
                    eval_progress_prefix=f"Gen {gen:4d}",
                )

                # Sort descending by fitness
                ranked = np.argsort(fitnesses)[::-1]
                best_acc = fitnesses[ranked[0]]
                result.history.append(best_acc)

                if best_acc > result.best_accuracy:
                    result.best_accuracy = best_acc
                    result.best_genome = population[ranked[0]].copy()

                if verbose and (gen % print_every == 0 or gen == n_generations - 1):
                    print(
                        f"Gen {gen:4d} | best={best_acc:.4f} | "
                        f"mean={fitnesses.mean():.4f} | "
                        f"all-time-best={result.best_accuracy:.4f}"
                    )

                balance_every = pred_balance_every if pred_balance_every > 0 else print_every
                if report_pred_balance and (gen % balance_every == 0 or gen == n_generations - 1):
                    bal_acc, bal_preds = _evaluate_with_predictions(
                        network,
                        population[ranked[0]],
                        X,
                        y,
                        output_node=output_node,
                        n_steps=n_steps,
                        mode=mode,
                    )
                    frac0 = float(np.mean(bal_preds == 0))
                    frac1 = float(np.mean(bal_preds == 1))
                    print(
                        f"Gen {gen:4d} | pred-balance (gen-best): "
                        f"class0={frac0:.3f} class1={frac1:.3f} | acc={bal_acc:.4f}"
                    )

                if snapshot_callback is not None and (
                    gen % snapshot_every == 0 or gen == n_generations - 1
                ):
                    snapshot_callback(gen, population[ranked[0]], fitnesses)

                # Early stopping if perfect
                if result.best_accuracy >= 1.0:
                    if verbose:
                        print(f"Perfect accuracy reached at generation {gen}.")
                    break

                # --- Reproduce ---
                elites = [population[ranked[i]].copy() for i in range(min(elite_size, pop_size))]

                next_pop: list[np.ndarray] = elites[:]
                while len(next_pop) < pop_size:
                    parent_idx = rng.integers(0, len(elites))
                    child = mutate_genome(elites[parent_idx], mutation_rate, rng)
                    next_pop.append(child)

                population = next_pop
    else:
        for gen in range(n_generations):
            fitnesses = evaluate_population(
                population, network, X, y,
                output_node=output_node, n_steps=n_steps, mode=mode,
                n_jobs=1,
                eval_verbose=eval_verbose,
                eval_progress_every=eval_progress_every,
                eval_progress_prefix=f"Gen {gen:4d}",
            )

            # Sort descending by fitness
            ranked = np.argsort(fitnesses)[::-1]
            best_acc = fitnesses[ranked[0]]
            result.history.append(best_acc)

            if best_acc > result.best_accuracy:
                result.best_accuracy = best_acc
                result.best_genome = population[ranked[0]].copy()

            if verbose and (gen % print_every == 0 or gen == n_generations - 1):
                print(
                    f"Gen {gen:4d} | best={best_acc:.4f} | "
                    f"mean={fitnesses.mean():.4f} | "
                    f"all-time-best={result.best_accuracy:.4f}"
                )

            balance_every = pred_balance_every if pred_balance_every > 0 else print_every
            if report_pred_balance and (gen % balance_every == 0 or gen == n_generations - 1):
                bal_acc, bal_preds = _evaluate_with_predictions(
                    network,
                    population[ranked[0]],
                    X,
                    y,
                    output_node=output_node,
                    n_steps=n_steps,
                    mode=mode,
                )
                frac0 = float(np.mean(bal_preds == 0))
                frac1 = float(np.mean(bal_preds == 1))
                print(
                    f"Gen {gen:4d} | pred-balance (gen-best): "
                    f"class0={frac0:.3f} class1={frac1:.3f} | acc={bal_acc:.4f}"
                )

            if snapshot_callback is not None and (
                gen % snapshot_every == 0 or gen == n_generations - 1
            ):
                snapshot_callback(gen, population[ranked[0]], fitnesses)

            # Early stopping if perfect
            if result.best_accuracy >= 1.0:
                if verbose:
                    print(f"Perfect accuracy reached at generation {gen}.")
                break

            # --- Reproduce ---
            elites = [population[ranked[i]].copy() for i in range(min(elite_size, pop_size))]

            next_pop: list[np.ndarray] = elites[:]
            while len(next_pop) < pop_size:
                parent_idx = rng.integers(0, len(elites))
                child = mutate_genome(elites[parent_idx], mutation_rate, rng)
                next_pop.append(child)

            population = next_pop

    return result
