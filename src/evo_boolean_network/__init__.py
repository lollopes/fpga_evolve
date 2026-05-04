"""
evo_boolean_network — Evolvable synchronous Boolean network simulator.

Public API re-exports for convenience.
"""

from typing import TYPE_CHECKING

from .config import EvoConfig
from .genome import (
    random_genome,
    bits_to_int,
    int_to_bits,
    decode_node_genome,
    encode_node_genome,
    describe_genome,
)
from .node import compute_node_output, build_candidate_vector
from .network import EvoBooleanNetwork
from .tasks import (
    xor_dataset,
    and_dataset,
    or_dataset,
    pattern_dataset,
    nmnist_dataset,
    nmnist_sequence_dataset,
    evaluate_binary_task,
    evaluate_sequence_task,
)
from .evolution import mutate_genome, evaluate_population, evolve_binary_classifier, SnapshotCallback
from .viz import plot_spike_raster, plot_raster_grid

if TYPE_CHECKING:
    from .viz_evolution import EvolutionViewer

__all__ = [
    "EvoConfig",
    "random_genome",
    "bits_to_int",
    "int_to_bits",
    "decode_node_genome",
    "encode_node_genome",
    "describe_genome",
    "compute_node_output",
    "build_candidate_vector",
    "EvoBooleanNetwork",
    "xor_dataset",
    "and_dataset",
    "or_dataset",
    "pattern_dataset",
    "nmnist_dataset",
    "nmnist_sequence_dataset",
    "evaluate_binary_task",
    "evaluate_sequence_task",
    "mutate_genome",
    "evaluate_population",
    "evolve_binary_classifier",
    "SnapshotCallback",
    "plot_spike_raster",
    "plot_raster_grid",
    "EvolutionViewer",
]


def __getattr__(name: str):
    if name == "EvolutionViewer":
        from .viz_evolution import EvolutionViewer
        return EvolutionViewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
