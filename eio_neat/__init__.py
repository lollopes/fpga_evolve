"""
Typed E/I/O Boolean NEAT.

Each evolved node is a single typed Boolean LUT-register unit:
  E  — excitatory processing node
  I  — inhibitory gating node
  O  — output/readout node (one per class)

Node update rule (synchronous, all connections delayed one timestep):
  raw        = LUT(data_inputs)
  inhibition = OR(state[t-1][s] for all incoming I sources)
  state[t]   = raw & ~inhibition

Class logit for class c = spike count of O_c over the full sequence.
"""

from .genome import Genome, NodeGene, ConnectionGene, InnovationRegistry
from .network import EIONetwork, evaluate_genome
from .neat import NEATConfig, evolve

__all__ = [
    "Genome",
    "NodeGene",
    "ConnectionGene",
    "InnovationRegistry",
    "EIONetwork",
    "evaluate_genome",
    "NEATConfig",
    "evolve",
]
