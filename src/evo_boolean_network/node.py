"""
node.py — Single-node computation.

Each node implements:
  1. Two input-selection muxes (MUX A and MUX B).
  2. One 2-input LUT producing one output bit.

The candidate input vector for node i is:
    [ext_input_0, ..., ext_input_{n_inputs-1},
     node_0_state, ..., node_{i-1}_state,
     node_{i+1}_state, ..., node_{n_nodes-1}_state]

Node i is excluded from its own candidates — it cannot read itself.
This is a key connectivity rule that mirrors FPGA hard-wiring constraints.

FPGA mapping:
  MUX A / MUX B -> combinational mux with sel_bits-wide select
  LUT           -> 4-entry SRAM lookup table (like a Xilinx/Intel 2-LUT)
  compute_node_output -> pure combinational logic (no registers)
"""

from __future__ import annotations

import numpy as np

from .config import EvoConfig


def build_candidate_vector(
    state: np.ndarray,
    external_inputs: np.ndarray,
    node_id: int,
) -> np.ndarray:
    """
    Build the ordered candidate input vector for a given node.

    Layout:
        [external_inputs | node states except node_id]

    Parameters
    ----------
    state : np.ndarray of shape (n_nodes,)
        Current node states (all nodes, including node_id which will be skipped).
    external_inputs : np.ndarray of shape (n_inputs,)
        External input bits.
    node_id : int
        The node for which we are building candidates (excluded from its own list).

    Returns
    -------
    np.ndarray of shape (n_inputs + n_nodes - 1,)
    """
    # Build internal candidates: all nodes except node_id
    internal = np.concatenate([state[:node_id], state[node_id + 1 :]])
    return np.concatenate([external_inputs, internal])


def compute_node_output(
    state: np.ndarray,
    external_inputs: np.ndarray,
    config: EvoConfig,
    node_id: int,
    sel_a: int,
    sel_b: int,
    lut_bits: np.ndarray,
) -> int:
    """
    Compute the output bit of a single node given the current network state.

    The computation is purely combinational — no state is modified here.

    Parameters
    ----------
    state : np.ndarray of shape (n_nodes,)
        Current node states (snapshot from time t — NOT partially updated).
    external_inputs : np.ndarray of shape (n_inputs,)
        External input bits for this timestep.
    config : EvoConfig
        Network configuration.
    node_id : int
        Which node to evaluate.
    sel_a : int
        Raw selector value for MUX A (from genome). Applied modulo n_candidates.
    sel_b : int
        Raw selector value for MUX B (from genome). Applied modulo n_candidates.
    lut_bits : np.ndarray of shape (4,)
        LUT truth table. Index = (A << 1) | B.

    Returns
    -------
    int : 0 or 1 — the node output bit.
    """
    candidates = build_candidate_vector(state, external_inputs, node_id)
    n_cand = len(candidates)

    # Modulo wrapping makes every genome bit-pattern valid — no genome is "illegal"
    idx_a = sel_a % n_cand
    idx_b = sel_b % n_cand

    A = int(candidates[idx_a])
    B = int(candidates[idx_b])

    # LUT address: A is the high bit, B is the low bit
    lut_index = (A << 1) | B
    return int(lut_bits[lut_index])
