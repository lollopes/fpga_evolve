from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple
import math
import random
import time
import numpy as np
import torch

from .genome import Genome, InnovationRegistry, genome_distance, crossover
from .network import evaluate_genome


@dataclass
class NEATConfig:
    pop_size: int = 50
    generations: int = 50
    elite_per_species: int = 1
    elite_min_size: int = 5
    tournament_k: int = 3

    k: int = 3
    initial_connections_per_output: int = 3
    initial_e_nodes: int = 0

    lut_bit_rate: float = 0.02
    p_add_connection: float = 0.25
    p_add_node: float = 0.05
    p_toggle_connection: float = 0.02
    p_crossover: float = 0.75
    p_new_node_is_inhibitory: float = 0.2
    p_mutate_node_type: float = 0.01
    allow_output_feedback: bool = False

    compatibility_threshold: float = 1.5
    c_disjoint: float = 1.0
    c_lut: float = 0.4
    c_type: float = 0.2

    batch_size: int = 64

    max_stale: int = 15
    fitness_sample: int = 0  # 0 = full X_train; >0 = random subsample per generation

    seed: int = 42
    log_every: int = 1


@dataclass
class Species:
    representative: Genome
    members: List[Genome]
    best_fitness: float = -1e9
    stale: int = 0


def _make_initial_population(
    input_size: int,
    n_outputs: int,
    config: NEATConfig,
    registry: InnovationRegistry,
    rng: random.Random,
) -> List[Genome]:
    template = Genome.minimal(
        input_size=input_size,
        n_outputs=n_outputs,
        k=config.k,
        registry=registry,
        rng=rng,
        initial_connections_per_output=config.initial_connections_per_output,
        initial_e_nodes=config.initial_e_nodes,
    )
    pop = []
    for _ in range(config.pop_size):
        g = template.clone()
        g.mutate_luts(rng, config.lut_bit_rate)
        if rng.random() < 0.5:
            g.add_connection_mutation(registry, rng, allow_output_feedback=config.allow_output_feedback)
        pop.append(g)
    return pop


def _speciate(population: List[Genome], species: List[Species], config: NEATConfig) -> List[Species]:
    for s in species:
        s.members = []

    for g in population:
        placed = False
        for s in species:
            d = genome_distance(g, s.representative, config.c_disjoint, config.c_lut, config.c_type)
            if d < config.compatibility_threshold:
                s.members.append(g)
                placed = True
                break
        if not placed:
            species.append(Species(representative=g.clone(), members=[g]))

    new_species = []
    for s in species:
        if not s.members:
            continue
        s.representative = random.choice(s.members).clone()
        new_species.append(s)
    return new_species


def _tournament_select(members: List[Genome], rng: random.Random, k: int) -> Genome:
    k = min(k, len(members))
    competitors = rng.sample(members, k)
    competitors.sort(key=lambda g: g.fitness if g.fitness is not None else -1e9, reverse=True)
    return competitors[0]


def _mutate_child(child: Genome, registry: InnovationRegistry, rng: random.Random, config: NEATConfig) -> None:
    child.mutate(
        registry=registry,
        rng=rng,
        lut_bit_rate=config.lut_bit_rate,
        p_add_connection=config.p_add_connection,
        p_add_node=config.p_add_node,
        p_toggle_connection=config.p_toggle_connection,
        p_new_node_is_inhibitory=config.p_new_node_is_inhibitory,
        p_mutate_node_type=config.p_mutate_node_type,
        allow_output_feedback=config.allow_output_feedback,
    )


def _reproduce(
    species: List[Species],
    config: NEATConfig,
    registry: InnovationRegistry,
    rng: random.Random,
) -> List[Genome]:
    adjusted_scores = []
    for s in species:
        s.members.sort(key=lambda g: g.fitness if g.fitness is not None else -1e9, reverse=True)
        best = s.members[0].fitness if s.members[0].fitness is not None else -1e9
        if best > s.best_fitness:
            s.best_fitness = best
            s.stale = 0
        else:
            s.stale += 1
        raw = [max(0.0, (g.fitness or 0.0) + 1.0) for g in s.members]
        adjusted_scores.append(sum(raw) / max(1, len(s.members)))

    global_best_fitness = max(s.best_fitness for s in species)
    champion = next(s for s in species if s.best_fitness >= global_best_fitness)
    live = [
        (s, sc)
        for s, sc in zip(species, adjusted_scores)
        if s.stale < config.max_stale or s is champion
    ]
    if live:
        species, adjusted_scores = zip(*live)  # type: ignore[assignment]
        species = list(species)
        adjusted_scores = list(adjusted_scores)

    score_sum = sum(adjusted_scores)
    if score_sum <= 0:
        quotas = [max(1, config.pop_size // max(1, len(species))) for _ in species]
    else:
        quotas = [max(1, int(round(config.pop_size * sc / score_sum))) for sc in adjusted_scores]

    while sum(quotas) > config.pop_size:
        quotas[int(np.argmax(quotas))] -= 1
    while sum(quotas) < config.pop_size:
        quotas[int(np.argmax(adjusted_scores))] += 1

    new_pop: List[Genome] = []

    for s, quota in zip(species, quotas):
        if quota <= 0 or not s.members:
            continue

        if len(s.members) >= config.elite_min_size:
            elites = s.members[: min(config.elite_per_species, len(s.members), quota)]
            for e in elites:
                new_pop.append(e.clone())
        else:
            elites = []

        remaining = quota - len(elites)
        while remaining > 0 and len(new_pop) < config.pop_size:
            remaining -= 1
            p1 = _tournament_select(s.members, rng, config.tournament_k)
            if rng.random() < config.p_crossover and len(s.members) > 1:
                p2 = _tournament_select(s.members, rng, config.tournament_k)
                if (p2.fitness or -1e9) > (p1.fitness or -1e9):
                    p1, p2 = p2, p1
                child = crossover(p1, p2, rng)
            else:
                child = p1.clone()
            _mutate_child(child, registry, rng, config)
            new_pop.append(child)

    all_members = sorted(
        [g for s in species for g in s.members],
        key=lambda g: g.fitness if g.fitness is not None else -1e9,
        reverse=True,
    )
    while len(new_pop) < config.pop_size:
        child = _tournament_select(all_members, rng, config.tournament_k).clone()
        _mutate_child(child, registry, rng, config)
        new_pop.append(child)

    return new_pop[: config.pop_size]


def evolve(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_outputs: int,
    config: NEATConfig,
    device: torch.device | None = None,
) -> Tuple[Genome, List[dict], InnovationRegistry]:
    """
    Evolve a Typed E/I/O Boolean NEAT population on the given dataset.

    X tensors: [N, T, F].
    Returns the best genome (chosen by val accuracy), full generation history,
    and the shared innovation registry.
    """
    rng = random.Random(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    dev = device or torch.device("cpu")

    input_size = int(X_train.shape[-1])
    registry = InnovationRegistry()
    population = _make_initial_population(input_size, n_outputs, config, registry, rng)
    species: List[Species] = []
    history: List[dict] = []
    best_train: Genome | None = None
    best_val: Genome | None = None
    best_val_acc = -math.inf
    best_val_fitness = -math.inf

    n_train = X_train.shape[0]

    for gen in range(config.generations):
        t0 = time.time()

        if config.fitness_sample > 0 and config.fitness_sample < n_train:
            idx = torch.tensor(rng.sample(range(n_train), k=config.fitness_sample))
            X_fit, y_fit = X_train[idx], y_train[idx]
        else:
            X_fit, y_fit = X_train, y_train

        for g in population:
            f, m = evaluate_genome(
                g, X_fit, y_fit,
                device=dev,
                batch_size=config.batch_size,
            )
            g.fitness = f
            g.metrics = m

        population.sort(key=lambda g: g.fitness if g.fitness is not None else -1e9, reverse=True)
        if best_train is None or (population[0].fitness or -1e9) > (best_train.fitness or -1e9):
            best_train = population[0].clone()

        species = _speciate(population, species, config)

        val_f, val_m = evaluate_genome(
            population[0], X_val, y_val,
            device=dev,
            batch_size=config.batch_size,
        )
        val_acc = float(val_m.get("acc", 0.0))
        if (
            best_val is None
            or val_acc > best_val_acc
            or (math.isclose(val_acc, best_val_acc) and val_f > best_val_fitness)
        ):
            best_val = population[0].clone()
            best_val.fitness = val_f
            best_val.metrics = {**best_val.metrics, "val_acc": val_acc, "val_fitness": float(val_f)}
            best_val_acc = val_acc
            best_val_fitness = val_f

        m0 = population[0].metrics
        rec = {
            "generation": gen,
            "best_fitness": float(population[0].fitness or 0.0),
            "best_ce": float(m0.get("ce", 0.0)),
            "best_train_acc": float(m0.get("acc", 0.0)),
            "best_val_acc": val_acc,
            "mean_fitness": float(np.mean([g.fitness or 0.0 for g in population])),
            "n_species": len(species),
            "best_connections": int(m0.get("enabled_data_conns", 0) + m0.get("enabled_inh_conns", 0)),
            "best_hidden_E": int(m0.get("enabled_e", 0)),
            "best_hidden_I": int(m0.get("enabled_i", 0)),
            "best_silent_frac": float(m0.get("silent_frac", 0.0)),
            "elapsed_s": time.time() - t0,
        }
        history.append(rec)

        if config.log_every and gen % config.log_every == 0:
            print(
                f"gen {gen:4d} | "
                f"train={rec['best_train_acc']:.1%}  val={rec['best_val_acc']:.1%} "
                f"ce={rec['best_ce']:.3f} | "
                f"species={rec['n_species']}  "
                f"E={rec['best_hidden_E']}  I={rec['best_hidden_I']}  "
                f"conn={rec['best_connections']}  "
                f"silent={rec['best_silent_frac']:.2f} | "
                f"{rec['elapsed_s']:.1f}s"
            )

        if gen < config.generations - 1:
            population = _reproduce(species, config, registry, rng)

    selected = best_val if best_val is not None else best_train
    assert selected is not None
    return selected, history, registry
