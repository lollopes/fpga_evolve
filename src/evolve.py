"""
evolve.py — Template genome evolution for the 3-area Boolean cortical hierarchy.

Each genome is a tuple (g1, g2, g3) of three CPU long tensors shaped [12, 20]:
    [:, 0:4]  MUX selectors (SEL) — valid ranges differ per node group
    [:, 4:20] LUT truth tables    — binary {0, 1}

Node groups and their SEL upper bounds (exclusive):
    nodes 0–5  (E_feat)  : sel < 19   (base 13 + prev_E 3 + prev_I 3)
    nodes 6–8  (E_out)   : sel < 25   (above + E05 output 6)
    nodes 9–11 (I_nodes) : sel < 28   (above + E68_raw 3)
"""

from __future__ import annotations
import time
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .brain import Brain
from .readout import Readout
from .area import Area


_SEL_MAX = 16  # all node groups use uniform SEL range [0, 15]


# ---------------------------------------------------------------------------
# Genome helpers
# ---------------------------------------------------------------------------

def random_genome(
    seed: int = 0,
    area_grids: list = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return a random genome (g1, g2, g3) with valid SEL ranges.

    Each gi is a CPU long tensor of shape [12, 20].
    Uses Brain.random to guarantee SEL values stay in range per node group.
    """
    brain = Brain.random(B=1, seed=seed, device=torch.device("cpu"), area_grids=area_grids)
    return tuple(g.squeeze(0) for g in brain.to_genomes())


def _mutate_single(
    g: torch.Tensor,
    lut_rate: float,
    sel_rate: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """
    Mutate one [12, 20] genome tensor in-place and return it.

    LUT bits ([:, 4:]) are flipped with probability lut_rate.
    SEL values ([:, :4]) are resampled within valid range with probability sel_rate.
    """
    g = g.clone()

    # LUT mutation — bit flip
    lut = g[:, 4:]                                           # [12, 16]
    mask = torch.from_numpy(
        rng.random(lut.shape).astype(np.float32) < lut_rate
    )
    lut[mask] = 1 - lut[mask]
    g[:, 4:] = lut

    # SEL mutation — uniform resample in [0, 15] for all node groups
    sel = g[:, :4]                                           # [12, 4]
    mask_s = torch.from_numpy(
        rng.random(sel.shape).astype(np.float32) < sel_rate
    )
    new_vals = torch.from_numpy(
        rng.integers(0, _SEL_MAX, sel.shape).astype(np.int64)
    )
    sel[mask_s] = new_vals[mask_s]
    g[:, :4] = sel

    return g


def mutate(
    genome: tuple,
    lut_rate: float,
    sel_rate: float,
    rng: np.random.Generator,
) -> tuple:
    """Apply independent mutations to each area genome (1 or more areas)."""
    return tuple(_mutate_single(g, lut_rate, sel_rate, rng) for g in genome)


# ---------------------------------------------------------------------------
# Penalty computation
# ---------------------------------------------------------------------------

def _penalties_from_trajectory(traj: torch.Tensor) -> Tuple[float, float, float]:
    """
    Compute three collapse/instability penalties from an Area-3 trajectory.

    Parameters
    ----------
    traj : [N, T, 144] long — Area-3 excitatory node outputs

    Returns
    -------
    silent_frac   : fraction of samples with zero total spikes across all T steps
    active_frac   : fraction of samples with mean firing rate > 0.90
    instability   : mean temporal std of per-step mean activity (0 = perfectly flat)
    """
    f = traj.float()

    total_spikes = f.sum(dim=(1, 2))                          # [N]
    silent_frac = (total_spikes == 0).float().mean().item()

    mean_rate = f.mean(dim=(1, 2))                            # [N]
    active_frac = (mean_rate > 0.90).float().mean().item()

    per_step = f.mean(dim=2)                                  # [N, T]
    instability = per_step.std(dim=1).mean().item()

    return silent_frac, active_frac, instability


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

def genome_fitness(
    genome: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    X_train: torch.Tensor,          # [N, T, input_size] long, CPU
    y_train: torch.Tensor,          # [N] long
    X_val: torch.Tensor,            # [M, T, input_size] long, CPU
    y_val: torch.Tensor,            # [M] long
    n_classes: int,
    device: torch.device,
    readout_epochs: int = 50,
    n_penalty_samples: int = 200,
    penalty_weights: Tuple[float, float, float] = (0.3, 0.3, 0.05),
    readout_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    in_features: int = 144,
    n_columns: int = None,
    nodes_per_column: int = 9,
    area_grids: list = None,
) -> Tuple[float, dict]:
    """
    Evaluate one genome: train a linear readout, measure val accuracy + penalties.

    Parameters
    ----------
    genome           : (g1, g2, g3) each [12, 20] CPU long
    X_train / y_train: training data on CPU
    X_val   / y_val  : validation data on CPU
    n_classes        : number of output classes
    device           : torch device for brain and readout
    readout_epochs   : gradient steps on the linear readout
    n_penalty_samples: samples used to estimate penalties (subset of train)
    penalty_weights  : (w_silent, w_active, w_instability)
    readout_lr       : Adam learning rate for the linear readout

    Returns
    -------
    fitness : float  (val_acc minus weighted penalties)
    metrics : dict   (val_acc, train_acc, silent_frac, active_frac,
                      instability, fitness)
    """
    brain = Brain.from_genomes(
        *[g.unsqueeze(0) for g in genome],
        device=device, area_grids=area_grids,
    )

    # ---- Penalties on a small training subset (cheap) --------------------
    n_pen = min(n_penalty_samples, X_train.shape[0])
    with torch.no_grad():
        traj_pen = Readout.compute_trajectory(brain, X_train[:n_pen])

    silent_frac, active_frac, instability = _penalties_from_trajectory(traj_pen)

    # Short-circuit: completely silent genome — skip readout training.
    if silent_frac > 0.95:
        fitness = -1.0
        return fitness, {
            "val_acc": 0.0, "train_acc": 0.0,
            "silent_frac": silent_frac, "active_frac": active_frac,
            "instability": instability, "fitness": fitness,
        }

    # ---- Train linear readout on full training set -----------------------
    readout = Readout(
        n_classes=n_classes,
        in_features=in_features,
        n_columns=n_columns,
        nodes_per_column=nodes_per_column,
    )
    readout.fit(
        brain, X_train, y_train,
        n_epochs=readout_epochs,
        lr=readout_lr,
        weight_decay=weight_decay,
        verbose=False,
    )

    with torch.no_grad():
        val_acc   = readout.accuracy(brain, X_val,   y_val)
        train_acc = readout.accuracy(brain, X_train, y_train)

    w_s, w_a, w_i = penalty_weights
    fitness = val_acc - w_s * silent_frac - w_a * active_frac - w_i * instability

    return fitness, {
        "val_acc":      val_acc,
        "train_acc":    train_acc,
        "silent_frac":  silent_frac,
        "active_frac":  active_frac,
        "instability":  instability,
        "fitness":      fitness,
    }


# ---------------------------------------------------------------------------
# Area activity helper (for logging)
# ---------------------------------------------------------------------------

@torch.no_grad()
def area_activities(
    brain: Brain,
    frames: torch.Tensor,       # [N, T, 256] long
    batch_size: int = 64,
) -> dict:
    """
    Compute mean column activity per area over a spike sequence.

    Runs the brain step-by-step so all three area states are observable.

    Returns
    -------
    dict with keys "area1", "area2", "area3" — mean fraction of active
    columns across all samples and time steps.
    """
    dev = brain.area1.sel.device
    N, T, _ = frames.shape
    area_names = ["area1"] + (["area2"] if brain.area2 is not None else []) + (["area3"] if brain.area3 is not None else [])
    totals  = {k: 0.0 for k in area_names}
    n_steps = 0

    for start in range(0, N, batch_size):
        batch = frames[start : start + batch_size].to(dev)
        bs = batch.shape[0]
        brain_bs = brain.expand(bs)
        brain_bs.reset_state()

        for t in range(T):
            brain_bs.forward(batch[:, t])
            for k in area_names:
                totals[k] += getattr(brain_bs, k).activity().float().mean().item()
            n_steps += 1

    return {k: v / n_steps for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Evolution loop
# ---------------------------------------------------------------------------

def evolve(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_classes: int,
    pop_size: int = 20,
    n_generations: int = 50,
    elite_size: int = 4,
    lut_mut_rate: float = 0.005,
    sel_mut_rate: float = 0.01,
    readout_epochs: int = 50,
    n_penalty_samples: int = 200,
    penalty_weights: Tuple[float, float, float] = (0.3, 0.3, 0.05),
    readout_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    seed: int = 42,
    log_every: int = 5,
    verbose: bool = True,
    in_features: int = 144,
    n_columns: int = None,
    nodes_per_column: int = 9,
    area_grids: list = None,
) -> Tuple[tuple, list]:
    """
    Evolve three area templates using (μ + λ) elitist selection.

    Each generation:
      1. Evaluate all pop_size genomes (train readout, measure val accuracy).
      2. Keep the top elite_size genomes unchanged.
      3. Fill the rest by mutating randomly chosen elites.

    Parameters
    ----------
    X_train / y_train   : training data on CPU
    X_val   / y_val     : validation data on CPU
    n_classes           : output classes (2 or 10)
    pop_size            : population size
    n_generations       : number of generations to run
    elite_size          : number of top genomes carried forward each generation
    lut_mut_rate        : per-bit flip probability for LUT entries
    sel_mut_rate        : per-entry resample probability for SEL values
    readout_epochs      : gradient steps per genome evaluation
    n_penalty_samples   : training samples used to estimate penalties
    penalty_weights     : (w_silent, w_active, w_instability)
    readout_lr          : Adam lr for readout
    device              : torch device (default cpu)
    seed                : base RNG seed
    log_every           : print summary every this many generations
    verbose             : print per-generation summary

    Returns
    -------
    best_genome : (g1, g2, g3) tensors of the highest-fitness genome found
    history     : list of dicts, one per generation, with keys:
                  generation, best_fitness, mean_fitness, best_val_acc,
                  best_silent_frac, best_active_frac, best_instability,
                  elapsed_s
    """
    dev = device or torch.device("cpu")
    rng = np.random.default_rng(seed)

    if elite_size >= pop_size:
        raise ValueError(f"elite_size ({elite_size}) must be < pop_size ({pop_size})")

    # ---- Initialise population -------------------------------------------
    population = [random_genome(seed=seed + i, area_grids=area_grids) for i in range(pop_size)]
    history: list = []
    best_genome = population[0]
    best_fitness = -float("inf")

    if verbose:
        print(
            f"\nEvolving  pop={pop_size}  gens={n_generations}  "
            f"elite={elite_size}  lut_rate={lut_mut_rate}  sel_rate={sel_mut_rate}"
        )
        print(
            f"           readout_epochs={readout_epochs}  "
            f"penalty_weights={penalty_weights}  device={dev}\n"
        )

    for gen in range(n_generations):
        t0 = time.time()

        # ---- Evaluate every genome in the population ---------------------
        fitnesses: List[float] = []
        metrics_list: List[dict] = []

        for genome in population:
            f, m = genome_fitness(
                genome, X_train, y_train, X_val, y_val,
                n_classes=n_classes,
                device=dev,
                readout_epochs=readout_epochs,
                n_penalty_samples=n_penalty_samples,
                penalty_weights=penalty_weights,
                readout_lr=readout_lr,
                weight_decay=weight_decay,
                in_features=in_features,
                n_columns=n_columns,
                nodes_per_column=nodes_per_column,
                area_grids=area_grids,
            )
            fitnesses.append(f)
            metrics_list.append(m)

        # ---- Track best --------------------------------------------------
        best_idx = int(np.argmax(fitnesses))
        if fitnesses[best_idx] > best_fitness:
            best_fitness = fitnesses[best_idx]
            best_genome = population[best_idx]

        # ---- Logging -----------------------------------------------------
        elapsed = time.time() - t0
        best_m = metrics_list[best_idx]
        record = {
            "generation":       gen,
            "best_fitness":     fitnesses[best_idx],
            "mean_fitness":     float(np.mean(fitnesses)),
            "best_val_acc":     best_m["val_acc"],
            "best_train_acc":   best_m["train_acc"],
            "best_silent_frac": best_m["silent_frac"],
            "best_active_frac": best_m["active_frac"],
            "best_instability": best_m["instability"],
            "elapsed_s":        elapsed,
        }
        history.append(record)

        if verbose and (gen % log_every == 0 or gen == n_generations - 1):
            print(
                f"  gen {gen:4d}  "
                f"best_fit={record['best_fitness']:.4f}  "
                f"mean_fit={record['mean_fitness']:.4f}  "
                f"val_acc={record['best_val_acc']:.3f}  "
                f"silent={record['best_silent_frac']:.2f}  "
                f"active={record['best_active_frac']:.2f}  "
                f"t={elapsed:.1f}s"
            )

        # ---- Selection: keep elites, mutate the rest ---------------------
        sorted_idx = np.argsort(fitnesses)[::-1]
        elites = [population[i] for i in sorted_idx[:elite_size]]

        new_population = list(elites)
        while len(new_population) < pop_size:
            parent = elites[int(rng.integers(elite_size))]
            child  = mutate(parent, lut_mut_rate, sel_mut_rate, rng)
            new_population.append(child)

        population = new_population

    if verbose:
        print(f"\nEvolution complete.  best_fitness={best_fitness:.4f}\n")

    return best_genome, history


# ---------------------------------------------------------------------------
# WTA fitness — no linear readout, spike counts as logits
# ---------------------------------------------------------------------------

def _wta_logits(
    traj: torch.Tensor,          # [N, T, n_columns*nodes]
    assignment: torch.Tensor,    # [n_columns] long
    n_classes: int,
    n_columns: int,
    nodes_per_column: int,
) -> torch.Tensor:
    """Sum spike counts per class group → [N, n_classes] float logits."""
    N, T, _ = traj.shape
    counts = traj.reshape(N, T, n_columns, nodes_per_column).sum(dim=(1, 3)).float()
    logits = torch.zeros(N, n_classes)
    for k in range(n_classes):
        mask = assignment == k
        if mask.any():
            logits[:, k] = counts[:, mask].sum(dim=1)
    return logits


def genome_fitness_wta(
    genome: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_classes: int,
    device: torch.device,
    n_columns: int = 16,
    nodes_per_column: int = 9,
    ce_weight: float = 0.1,
    area_grids: list = None,
) -> Tuple[float, dict]:
    """
    Evaluate one genome using WTA spike-count logits — no linear readout.

    Fitness
    -------
    Column→class assignment is derived from training data (most-responsive
    class per column). Val logits = sum of spike counts per class group.
    Fitness = val_acc - ce_weight * (val_CE / log(n_classes))

    The CE term is normalised by log(n_classes) so it stays in [0, 1],
    matching the scale of val_acc.
    """
    brain = Brain.from_genomes(
        *[g.unsqueeze(0) for g in genome],
        device=device, area_grids=area_grids,
    )

    with torch.no_grad():
        traj_tr = Readout.compute_trajectory(brain, X_train)   # [N, T, F]
        traj_va = Readout.compute_trajectory(brain, X_val)

    # Column spike counts [N, n_columns]
    def col_counts(traj):
        N, T, _ = traj.shape
        return traj.reshape(N, T, n_columns, nodes_per_column).sum(dim=(1, 3)).float()

    counts_tr = col_counts(traj_tr)
    counts_va = col_counts(traj_va)

    # Assignment: each column → class it fires most for on training data
    class_mean = torch.zeros(n_columns, n_classes)
    for k in range(n_classes):
        mask = y_train.cpu() == k
        if mask.sum() > 0:
            class_mean[:, k] = counts_tr[mask].mean(dim=0)
    assignment = class_mean.argmax(dim=1)   # [n_columns]

    logits_tr = _wta_logits(traj_tr, assignment, n_classes, n_columns, nodes_per_column)
    logits_va = _wta_logits(traj_va, assignment, n_classes, n_columns, nodes_per_column)

    criterion = nn.CrossEntropyLoss()
    val_ce    = criterion(logits_va, y_val.cpu()).item()
    train_ce  = criterion(logits_tr, y_train.cpu()).item()

    val_acc   = (logits_va.argmax(1) == y_val.cpu()).float().mean().item()
    train_acc = (logits_tr.argmax(1) == y_train.cpu()).float().mean().item()

    import math
    ce_norm   = val_ce / math.log(n_classes)        # normalise to [0, ~1]
    fitness   = val_acc - ce_weight * ce_norm

    return fitness, {
        "val_acc":   val_acc,
        "train_acc": train_acc,
        "val_ce":    val_ce,
        "train_ce":  train_ce,
        "fitness":   fitness,
        "assignment": assignment,
    }


def evolve_wta(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_classes: int,
    pop_size: int = 20,
    n_generations: int = 50,
    elite_size: int = 4,
    lut_mut_rate: float = 0.005,
    sel_mut_rate: float = 0.01,
    ce_weight: float = 0.1,
    n_columns: int = 16,
    nodes_per_column: int = 9,
    device: Optional[torch.device] = None,
    seed: int = 42,
    log_every: int = 5,
    verbose: bool = True,
    area_grids: list = None,
) -> Tuple[tuple, list]:
    """
    Evolve using WTA spike-count logits — no linear readout anywhere.

    Same (μ+λ) elitist strategy as evolve(), but genome_fitness_wta
    replaces genome_fitness. Fitness = val_acc - ce_weight * CE_normalised.
    """
    dev = device or torch.device("cpu")
    rng = np.random.default_rng(seed)

    if elite_size >= pop_size:
        raise ValueError(f"elite_size ({elite_size}) must be < pop_size ({pop_size})")

    population = [random_genome(seed=seed + i, area_grids=area_grids) for i in range(pop_size)]
    history: list = []
    best_genome   = population[0]
    best_fitness  = -float("inf")

    if verbose:
        print(
            f"\nEvolving (WTA)  pop={pop_size}  gens={n_generations}  "
            f"elite={elite_size}  lut_rate={lut_mut_rate}  sel_rate={sel_mut_rate}  "
            f"ce_weight={ce_weight}"
        )

    for gen in range(n_generations):
        t0 = time.time()
        fitnesses, metrics_list = [], []

        for genome in population:
            f, m = genome_fitness_wta(
                genome, X_train, y_train, X_val, y_val,
                n_classes=n_classes,
                device=dev,
                n_columns=n_columns,
                nodes_per_column=nodes_per_column,
                ce_weight=ce_weight,
                area_grids=area_grids,
            )
            fitnesses.append(f)
            metrics_list.append(m)

        best_idx = int(np.argmax(fitnesses))
        if fitnesses[best_idx] > best_fitness:
            best_fitness = fitnesses[best_idx]
            best_genome  = population[best_idx]

        elapsed = time.time() - t0
        best_m  = metrics_list[best_idx]
        record  = {
            "generation":    gen,
            "best_fitness":  fitnesses[best_idx],
            "mean_fitness":  float(np.mean(fitnesses)),
            "best_val_acc":  best_m["val_acc"],
            "best_train_acc":best_m["train_acc"],
            "best_val_ce":   best_m["val_ce"],
            "elapsed_s":     elapsed,
        }
        history.append(record)

        if verbose and (gen % log_every == 0 or gen == n_generations - 1):
            print(
                f"  gen {gen:4d}  "
                f"best_fit={record['best_fitness']:.4f}  "
                f"val_acc={record['best_val_acc']:.3f}  "
                f"val_ce={record['best_val_ce']:.3f}  "
                f"t={elapsed:.1f}s"
            )

        sorted_idx  = np.argsort(fitnesses)[::-1]
        elites      = [population[i] for i in sorted_idx[:elite_size]]
        new_pop     = list(elites)
        while len(new_pop) < pop_size:
            parent  = elites[int(rng.integers(elite_size))]
            new_pop.append(mutate(parent, lut_mut_rate, sel_mut_rate, rng))
        population = new_pop

    if verbose:
        print(f"\nEvolution complete.  best_fitness={best_fitness:.4f}\n")

    return best_genome, history
