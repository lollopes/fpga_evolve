"""
config.py — Network configuration dataclass.

Defines EvoConfig, which holds all structural parameters of the Boolean network.
This drives genome encoding, node connectivity, and simulation behavior.

FPGA mapping:
  These constants determine the bit-widths of all configuration registers.
  Changing n_nodes or n_inputs changes the register map — treat as compile-time params.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvoConfig:
    """
    Structural configuration for the evolvable synchronous Boolean network.

    Parameters
    ----------
    n_nodes : int
        Number of internal Boolean nodes in the network (must be >= 2).
    n_inputs : int
        Number of external input bits fed into the network (default 0).
    n_steps : int
        Default number of clock cycles (timesteps) per forward pass.
    output_nodes : list[int] | None
        Node indices used as network outputs. If None, uses the last node.
    seed : int | None
        Global RNG seed for reproducibility.

    Derived attributes (computed automatically)
    -------------------------------------------
    n_candidates_per_node : int
        Total selectable inputs for any node = n_inputs + (n_nodes - 1).
        A node can see all external inputs and all other node states, but NOT itself.
    sel_bits : int
        Number of bits needed to address any candidate input = ceil(log2(n_candidates)).
    node_genome_bits : int
        Bits used by a single node's genome = 2 * sel_bits + 4 (4 bits for the LUT).
    total_genome_bits : int
        Total bits in a full network genome = n_nodes * node_genome_bits.
    """

    n_nodes: int
    n_inputs: int = 0
    n_steps: int = 8
    output_nodes: Optional[list[int]] = None
    seed: Optional[int] = None

    # Derived fields — populated in __post_init__
    n_candidates_per_node: int = field(init=False)
    sel_bits: int = field(init=False)
    node_genome_bits: int = field(init=False)
    total_genome_bits: int = field(init=False)

    def __post_init__(self) -> None:
        if self.n_nodes < 2:
            raise ValueError(f"n_nodes must be >= 2, got {self.n_nodes}")

        # Each node sees every other node (n_nodes - 1) plus all external inputs
        self.n_candidates_per_node = self.n_inputs + self.n_nodes - 1

        if self.n_candidates_per_node < 1:
            raise ValueError(
                f"n_candidates_per_node must be >= 1, got {self.n_candidates_per_node}. "
                "Increase n_nodes or n_inputs."
            )

        # Bits to represent any index in [0, n_candidates_per_node)
        self.sel_bits = math.ceil(math.log2(max(self.n_candidates_per_node, 2)))

        # Per-node genome layout:  [sel_a (sel_bits)] [sel_b (sel_bits)] [lut_init (4)]
        self.node_genome_bits = 2 * self.sel_bits + 4

        self.total_genome_bits = self.n_nodes * self.node_genome_bits

        # Default output: last node only
        if self.output_nodes is None:
            self.output_nodes = [self.n_nodes - 1]

        # Validate output node indices
        for idx in self.output_nodes:
            if not (0 <= idx < self.n_nodes):
                raise ValueError(
                    f"output_nodes contains invalid index {idx} "
                    f"(n_nodes={self.n_nodes})"
                )

    def candidate_index(self, node_id: int, candidate_pos: int) -> int:
        """
        Map a candidate position back to a concrete signal source.

        The candidate vector for node `node_id` is ordered as:
            [ext_input_0, ..., ext_input_{n_inputs-1},
             node_0_state, ..., node_{node_id-1}_state,
             node_{node_id+1}_state, ..., node_{n_nodes-1}_state]

        Returns the actual node index (n_inputs offset removed) or a negative
        value (-1 - ext_idx) for external inputs — useful for debugging.
        """
        if candidate_pos < self.n_inputs:
            return -(candidate_pos + 1)  # external input, encoded as negative
        internal_pos = candidate_pos - self.n_inputs
        # Skip over the node itself in the sorted node list
        if internal_pos >= node_id:
            internal_pos += 1
        return internal_pos
