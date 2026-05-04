"""
circuit.py — Initialize a batch of Boolean circuits from a genome tensor.

A circuit is a fixed network of N 2-input LUT nodes plus K external inputs.
Each node selects two signals from a candidate pool (external inputs + every
OTHER node's state, never its own) and applies a 4-bit truth table.

The genome is **always batched** — its first dimension is the batch:

    genome.shape == (batch_size, total_genome_bits)

Each row encodes one independent circuit (same n_nodes / n_inputs structure).
A "single" circuit is just batch_size == 1.

Per-row layout (flat uint8 bits, little-endian within each field — bits[0]
is the LSB):

    per_node = [ sel_a (sel_bits) | sel_b (sel_bits) | lut (4) ]
    row      = [ node_0 | node_1 | ... | node_{N-1} ]

Selector values are wrapped modulo n_candidates, so every bitstring decodes
to a valid configuration — there are no illegal genomes.

This module is initialization only: it parses the genome into per-node arrays
ready for downstream simulation. The whole batch is decoded vectorized in one
shot — no Python loop over batch index.
"""

import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np


def selector_bits(n_candidates: int) -> int:
    """Number of bits needed to address one candidate signal (always >= 1)."""
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
    return max(1, math.ceil(math.log2(max(n_candidates, 2))))


def genome_bits(n_nodes: int, n_inputs: int) -> int:
    """Total bits in a genome for a network of (n_nodes, n_inputs)."""
    if n_nodes < 2:
        raise ValueError(f"n_nodes must be >= 2, got {n_nodes}")
    if n_inputs < 0:
        raise ValueError(f"n_inputs must be >= 0, got {n_inputs}")
    n_cand = n_inputs + n_nodes - 1
    return n_nodes * (2 * selector_bits(n_cand) + 4)


class Circuit:
    """Batch of static Boolean circuits decoded once from a genome tensor.

    Attributes set after construction:
      batch_size, n_nodes, n_inputs, n_candidates : int
      sel_bits, node_genome_bits, total_genome_bits : int
      sel_a : np.ndarray, shape (batch_size, n_nodes), dtype int32
          Mux-A index per node, already taken modulo n_candidates.
      sel_b : np.ndarray, shape (batch_size, n_nodes), dtype int32
      luts  : np.ndarray, shape (batch_size, n_nodes, 4), dtype uint8
          Truth table per node. Address as luts[b, i, (A << 1) | B].
    """

    def __init__(self, genome: np.ndarray, n_nodes: int, n_inputs: int) -> None:

        self.n_nodes: int = n_nodes
        self.n_inputs: int = n_inputs
        self.n_candidates: int = n_inputs + n_nodes - 1
        self.sel_bits: int = selector_bits(self.n_candidates)
        self.node_genome_bits: int = 2 * self.sel_bits + 4
        self.total_genome_bits: int = n_nodes * self.node_genome_bits

        g = np.asarray(genome, dtype=np.uint8)
        if g.ndim != 2:
            raise ValueError(
                f"genome must be 2-D (batch_size, total_genome_bits), got shape {g.shape}"
            )
        if g.shape[1] != self.total_genome_bits:
            raise ValueError(
                f"genome must have shape (B, {self.total_genome_bits}), got {g.shape}"
            )
        if g.size and int(g.max()) > 1:
            raise ValueError("genome must contain only 0s and 1s")

        self.batch_size: int = int(g.shape[0])

        # (B, n_nodes, node_genome_bits)
        blocks = g.reshape(self.batch_size, n_nodes, self.node_genome_bits)
        sb = self.sel_bits

        sel_a_bits = blocks[:, :, 0:sb]                       # (B, N, sb)
        sel_b_bits = blocks[:, :, sb : 2 * sb]                # (B, N, sb)
        lut_bits   = blocks[:, :, 2 * sb : 2 * sb + 4]        # (B, N, 4)

        # Little-endian per row of bits: weight bit i by 2**i, sum the field axis.
        powers = (1 << np.arange(sb, dtype=np.uint64))         # (sb,)
        raw_a = (sel_a_bits.astype(np.uint64) * powers).sum(axis=2)   # (B, N)
        raw_b = (sel_b_bits.astype(np.uint64) * powers).sum(axis=2)   # (B, N)

        self.sel_a: np.ndarray = (raw_a % self.n_candidates).astype(np.int32)
        self.sel_b: np.ndarray = (raw_b % self.n_candidates).astype(np.int32)
        self.luts: np.ndarray = lut_bits.astype(np.uint8)

        # Lookup table: (N, n_candidates) → index into a "combined" vector
        # combined = [ext (K) | state (N)] of length (K + N).
        # For node i and candidate position p in [0, n_candidates):
        #   q = p                 if p <  K + i      (ext OR state[<i])
        #   q = p + 1             if p >= K + i      (state[>i] — the +1 skips self)
        # This is exactly equivalent to the old "internal-with-self-removed"
        # candidate vector but lets step() avoid a fat concatenate.
        i_grid, p_grid = np.indices((n_nodes, self.n_candidates))
        self._cand_to_combined: np.ndarray = (
            p_grid + (p_grid >= self.n_inputs + i_grid)
        ).astype(np.int32)
        self._node_idx: np.ndarray = np.arange(n_nodes, dtype=np.int32)

    def zero_state(self) -> np.ndarray:
        """Return an all-zero state of shape (batch_size, n_nodes)."""
        return np.zeros((self.batch_size, self.n_nodes), dtype=np.uint8)

    def step(self, state: np.ndarray, ext_inputs: np.ndarray) -> np.ndarray:
        """One synchronous clock cycle for the whole batch (and, optionally, samples).

        All circuits read the FROZEN `state` (time t) and produce `next_state`
        (time t+1) — no in-place updates, mirroring a clocked register file.

        Two shape modes are supported:

          (B, N)    state and (B, K)            ext_inputs   →   (B, N) next_state
          (B, S, N) state and (S, K) or (B,S,K) ext_inputs   →   (B, S, N) next_state

        The (B, S, N) mode is what evaluation uses to march S samples in
        lockstep through the same B circuits. The (B, N) mode is just the
        S = 1 case with axis 1 squeezed away.
        """
        B = self.batch_size
        N = self.n_nodes
        K = self.n_inputs

        s = np.asarray(state, dtype=np.uint8)
        if s.ndim == 2:
            if s.shape != (B, N):
                raise ValueError(f"state must have shape ({B}, {N}), got {s.shape}")
            s = s[:, None, :]                                       # (B, 1, N)
            squeeze_out = True
        elif s.ndim == 3:
            if s.shape[0] != B or s.shape[2] != N:
                raise ValueError(
                    f"state must have shape ({B}, S, {N}), got {s.shape}"
                )
            squeeze_out = False
        else:
            raise ValueError(f"state must be 2-D or 3-D, got shape {s.shape}")
        S = s.shape[1]

        # Ext shape is interpreted via the STATE shape (not via matching against
        # B vs S), because B and S can collide. The state's dimensionality is
        # the unambiguous signal of which mode we're in.
        e = np.asarray(ext_inputs, dtype=np.uint8)
        if squeeze_out:
            # 2-D state (B, N) mode → ext_inputs must be (B, K).
            if e.shape != (B, K):
                raise ValueError(
                    f"ext_inputs must have shape ({B}, {K}) for 2-D state; got {e.shape}"
                )
            e = e[:, None, :]                                       # (B, 1, K)
        else:
            # 3-D state (B, S, N) mode → ext_inputs must be (S, K) or (B, S, K).
            if e.ndim == 2:
                if e.shape != (S, K):
                    raise ValueError(
                        f"ext_inputs must have shape ({S}, {K}) for 3-D state; "
                        f"got {e.shape}"
                    )
                e = e[None, :, :]                                   # (1, S, K)
            elif e.ndim == 3:
                if e.shape != (B, S, K):
                    raise ValueError(
                        f"ext_inputs must have shape ({B}, {S}, {K}); got {e.shape}"
                    )
            else:
                raise ValueError(f"ext_inputs must be 2-D or 3-D, got shape {e.shape}")

        # Combined "wire bus" — one row per (circuit, sample), holding both
        # external inputs and current node states. Only B*S*(K+N) elements.
        e_bs = np.broadcast_to(e, (B, S, K))                        # zero-copy
        combined = np.concatenate([e_bs, s], axis=-1)               # (B, S, K+N)

        # Resolve each (b, i)'s sel_a/sel_b into a position in `combined` via
        # the precomputed lookup table. Done once per step, not per (s, t, b).
        a_combined = self._cand_to_combined[self._node_idx, self.sel_a]   # (B, N)
        b_combined = self._cand_to_combined[self._node_idx, self.sel_b]   # (B, N)
        a_idx_bs = np.broadcast_to(a_combined[:, None, :], (B, S, N))
        b_idx_bs = np.broadcast_to(b_combined[:, None, :], (B, S, N))

        a_bits = np.take_along_axis(combined, a_idx_bs, axis=-1)    # (B, S, N)
        b_bits = np.take_along_axis(combined, b_idx_bs, axis=-1)    # (B, S, N)

        # LUT lookup per (b, s, i): index = (A << 1) | B in {0,1,2,3}.
        lut_idx = ((a_bits << 1) | b_bits)[..., None]               # (B, S, N, 1)
        luts_bs = np.broadcast_to(self.luts[:, None, :, :], (B, S, N, 4))
        next_state = np.take_along_axis(luts_bs, lut_idx, axis=-1).squeeze(-1)
        # next_state is uint8 (luts is uint8) — no astype needed.

        if squeeze_out:
            return next_state[:, 0, :]
        return next_state

    def __repr__(self) -> str:
        return (
            f"Circuit(batch_size={self.batch_size}, n_nodes={self.n_nodes}, "
            f"n_inputs={self.n_inputs}, n_candidates={self.n_candidates}, "
            f"sel_bits={self.sel_bits}, total_genome_bits={self.total_genome_bits})"
        )


if __name__ == "__main__":
    import webbrowser
    from visualize import write_html

    BATCH_SIZE = 1
    N_NODES = 64
    N_INPUTS = 8
    SEED = 0
    BATCH_ID = 0
    OUTPUT_PATH = Path(__file__).resolve().parent / "circuit.html"
    OPEN_BROWSER = True

    rng = np.random.default_rng(SEED)
    n_bits = genome_bits(N_NODES, N_INPUTS)
    genome = rng.integers(0, 2, size=(BATCH_SIZE, n_bits), dtype=np.uint8)

    circuit = Circuit(genome, N_NODES, N_INPUTS)
    print(circuit)
    
    # --- run the whole batch for N_STEPS clock cycles, recording trajectory ---
    N_STEPS = 32
    ext_inputs = rng.integers(0, 2, size=(BATCH_SIZE, N_INPUTS), dtype=np.uint8)

    # state_traj: (T+1, B, N) — pre-allocate, fill in place.
    state_traj = np.zeros((N_STEPS + 1, BATCH_SIZE, N_NODES), dtype=np.uint8)
    state_traj[0] = circuit.zero_state()
    for t in range(N_STEPS):
        state_traj[t + 1] = circuit.step(state_traj[t], ext_inputs)

    # ext_traj for the visualizer: same external inputs broadcast across time.
    ext_traj = np.broadcast_to(
        ext_inputs[None, :, :], (N_STEPS + 1, BATCH_SIZE, N_INPUTS),
    )

    print(
        f"  ran {N_STEPS} steps: trajectory shape={state_traj.shape}, "
        f"final mean activity per circuit="
        f"{state_traj[-1].mean(axis=1).round(3).tolist()}"
    )

    # Slice for the chosen batch_id: (T+1, n_nodes) and (T+1, n_inputs)
    viz_state = state_traj[:, BATCH_ID, :]
    viz_ext = ext_traj[:, BATCH_ID, :]

    path = write_html(
        circuit, OUTPUT_PATH, BATCH_ID, viz_state, viz_ext,
        title=f"Random circuit (seed={SEED}, batch_id={BATCH_ID})",
    )
    print(f"Saved {path}")

    if OPEN_BROWSER:
        webbrowser.open(path.as_uri())