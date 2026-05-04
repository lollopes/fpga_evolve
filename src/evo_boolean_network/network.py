"""
network.py — Synchronous Boolean network simulator.

EvoBooleanNetwork wraps the per-node computation into a clocked network.

The synchronous update rule mirrors FPGA clocked hardware:

    current_state  = state(t)           # latched values
    next_state     = f(current_state)   # pure combinational
    state          = next_state         # rising clock edge

This means ALL nodes read the OLD state and ALL writes happen simultaneously.
Sequential evaluation (node-by-node with partial updates) would give different
results and does NOT match FPGA behavior — it is explicitly avoided here.

FPGA mapping:
  self.state  -> flip-flop register array (one FF per node)
  step()      -> one clock cycle
  run()       -> multiple clock cycles
"""

from __future__ import annotations
from typing import Optional, Union

import numpy as np

from .config import EvoConfig
from .genome import decode_node_genome
from .node import compute_node_output


class EvoBooleanNetwork:
    """
    Synchronous evolvable Boolean network.

    All nodes are evaluated in parallel from a frozen state snapshot,
    then the snapshot is replaced atomically — exactly like a synchronous
    register file on an FPGA.
    """

    def __init__(self, config: EvoConfig) -> None:
        self.config = config
        # State vector: one bit per node, initialized to zero
        self.state: np.ndarray = np.zeros(config.n_nodes, dtype=np.uint8)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self, initial_state: Optional[np.ndarray] = None) -> None:
        """
        Reset the network state.

        Parameters
        ----------
        initial_state : array-like of shape (n_nodes,) | None
            If None, resets to all-zeros. Otherwise copies the provided state.
        """
        if initial_state is None:
            self.state = np.zeros(self.config.n_nodes, dtype=np.uint8)
        else:
            arr = np.asarray(initial_state, dtype=np.uint8)
            if arr.shape != (self.config.n_nodes,):
                raise ValueError(
                    f"initial_state must have shape ({self.config.n_nodes},), "
                    f"got {arr.shape}"
                )
            self.state = arr.copy()

    # ------------------------------------------------------------------
    # Single synchronous step
    # ------------------------------------------------------------------

    def step(
        self,
        genome: np.ndarray,
        external_inputs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Advance the network by one clock cycle.

        All nodes read self.state (time t), compute their next output,
        and only THEN is self.state updated to next_state (time t+1).

        Parameters
        ----------
        genome : np.ndarray
            Flat bit array of length config.total_genome_bits.
        external_inputs : array-like of shape (n_inputs,) | None
            External input bits. Must be provided if config.n_inputs > 0.

        Returns
        -------
        np.ndarray of shape (n_nodes,) — the new state after this step.
        """
        config = self.config

        if external_inputs is None:
            ext = np.zeros(config.n_inputs, dtype=np.uint8)
        else:
            ext = np.asarray(external_inputs, dtype=np.uint8)
            if ext.shape != (config.n_inputs,):
                raise ValueError(
                    f"external_inputs must have shape ({config.n_inputs},), "
                    f"got {ext.shape}"
                )

        # ---------------------------------------------------------------
        # SYNCHRONOUS CRITICAL SECTION
        # Freeze current state — this is the "latch" in clocked hardware.
        # All node computations read from `frozen_state`, not self.state.
        # ---------------------------------------------------------------
        frozen_state = self.state.copy()   # <-- snapshot at time t

        next_state = np.zeros(config.n_nodes, dtype=np.uint8)
        for node_id in range(config.n_nodes):
            sel_a, sel_b, lut_bits = decode_node_genome(genome, config, node_id)
            next_state[node_id] = compute_node_output(
                frozen_state, ext, config, node_id, sel_a, sel_b, lut_bits
            )

        # Atomic state update — equivalent to the rising clock edge
        self.state = next_state
        return self.state.copy()

    # ------------------------------------------------------------------
    # Multi-step run
    # ------------------------------------------------------------------

    def run(
        self,
        genome: np.ndarray,
        external_inputs: Optional[np.ndarray] = None,
        n_steps: Optional[int] = None,
        initial_state: Optional[np.ndarray] = None,
        return_trajectory: bool = True,
    ) -> np.ndarray:
        """
        Run the network for multiple timesteps.

        Parameters
        ----------
        genome : np.ndarray
            Network genome.
        external_inputs : array-like | None
            External input bits (held constant across all steps).
        n_steps : int | None
            Number of steps. Defaults to config.n_steps.
        initial_state : array-like | None
            Starting state. Defaults to all-zeros.
        return_trajectory : bool
            If True, return shape (n_steps+1, n_nodes) including the initial state.
            If False, return only the final state of shape (n_nodes,).

        Returns
        -------
        np.ndarray — trajectory or final state.
        """
        self.reset_state(initial_state)
        steps = n_steps if n_steps is not None else self.config.n_steps

        if return_trajectory:
            # Row 0 is the initial state before any step
            trajectory = np.zeros((steps + 1, self.config.n_nodes), dtype=np.uint8)
            trajectory[0] = self.state.copy()
            for t in range(steps):
                trajectory[t + 1] = self.step(genome, external_inputs)
            return trajectory
        else:
            for _ in range(steps):
                self.step(genome, external_inputs)
            return self.state.copy()

    # ------------------------------------------------------------------
    # Multi-step run with time-varying inputs
    # ------------------------------------------------------------------

    def run_sequence(
        self,
        genome: np.ndarray,
        input_sequence: np.ndarray,
        initial_state: Optional[np.ndarray] = None,
        return_trajectory: bool = True,
    ) -> np.ndarray:
        """
        Run the network with a different input vector at each clock cycle.

        Parameters
        ----------
        genome : np.ndarray
            Network genome.
        input_sequence : np.ndarray of shape (T, n_inputs), dtype uint8, values in {0,1}
            input_sequence[t] is fed as external_inputs at step t.
            T is used as the number of steps (config.n_steps is ignored).
        initial_state : array-like | None
            Starting state. Defaults to all-zeros.
        return_trajectory : bool
            If True, return shape (T+1, n_nodes) including the initial state.
            If False, return only the final state of shape (n_nodes,).

        Returns
        -------
        np.ndarray — trajectory or final state.
        """
        T = input_sequence.shape[0]
        self.reset_state(initial_state)

        if return_trajectory:
            trajectory = np.zeros((T + 1, self.config.n_nodes), dtype=np.uint8)
            trajectory[0] = self.state.copy()
            for t in range(T):
                trajectory[t + 1] = self.step(genome, input_sequence[t])
            return trajectory
        else:
            for t in range(T):
                self.step(genome, input_sequence[t])
            return self.state.copy()

    # ------------------------------------------------------------------
    # Output readout
    # ------------------------------------------------------------------

    def read_outputs(
        self,
        trajectory_or_state: np.ndarray,
        mode: str = "final",
    ) -> np.ndarray:
        """
        Extract values from selected output nodes.

        Parameters
        ----------
        trajectory_or_state : np.ndarray
            Either a trajectory of shape (T, n_nodes) or a state of shape (n_nodes,).
        mode : str
            "final"    — return output node values at the last timestep.
            "activity" — return sum of output node values across all timesteps
                         (useful as a continuous fitness signal).

        Returns
        -------
        np.ndarray of shape (n_output_nodes,).
        """
        out_nodes = self.config.output_nodes
        arr = np.asarray(trajectory_or_state)

        if arr.ndim == 1:
            # Single state vector
            return arr[out_nodes]

        # Trajectory: shape (T, n_nodes)
        if mode == "final":
            return arr[-1, out_nodes]
        elif mode == "activity":
            return arr[:, out_nodes].sum(axis=0)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'final' or 'activity'.")
