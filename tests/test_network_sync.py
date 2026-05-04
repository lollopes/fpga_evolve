"""
test_network_sync.py — Verify synchronous update semantics.

The critical test: construct a 2-node network where sequential evaluation
would produce a DIFFERENT result than synchronous evaluation, and confirm
the network produces the synchronous result.

  Node 0 computes:  COPY_B of node 1
  Node 1 computes:  COPY_A of node 0

  Initial state: [0, 1]

  Synchronous (correct):
    t+1: node0 reads node1(t)=1, node1 reads node0(t)=0
    -> next_state = [1, 0]

  Sequential (wrong — evaluates node0 first, then node1 reads the already-updated node0):
    node0 gets node1=1 -> node0_new = 1
    node1 gets node0_new=1 (!) -> node1_new = 1
    -> state = [1, 1]   # <-- wrong

The synchronous simulator must yield [1, 0], not [1, 1].
"""

import numpy as np
import pytest

from evo_boolean_network.config import EvoConfig
from evo_boolean_network.genome import encode_node_genome
from evo_boolean_network.network import EvoBooleanNetwork


def _make_swap_network() -> tuple[EvoBooleanNetwork, np.ndarray, EvoConfig]:
    """
    Build a 2-node, 0-input network where nodes swap values each step.

      Node 0: reads node 1 (sel_a and sel_b both point to node1, LUT=COPY_A)
      Node 1: reads node 0 (sel_a and sel_b both point to node0, LUT=COPY_A)

    Candidate vector for node 0: [node1]  (only other node, no external inputs)
    Candidate vector for node 1: [node0]

    COPY_A lut = [0, 0, 1, 1]  (output = A = candidates[sel_a])
    """
    config = EvoConfig(n_nodes=2, n_inputs=0, n_steps=1)
    # n_candidates = 0 + 2 - 1 = 1 -> sel_bits = 1 (ceil(log2(2)) = 1)
    # sel_a=0, sel_b=0 -> both select candidate[0]
    # For node 0: candidate[0] = node1
    # For node 1: candidate[0] = node0
    COPY_A = np.array([0, 0, 1, 1], dtype=np.uint8)

    genome = np.zeros(config.total_genome_bits, dtype=np.uint8)
    encode_node_genome(config, 0, sel_a=0, sel_b=0, lut_bits=COPY_A, genome=genome)
    encode_node_genome(config, 1, sel_a=0, sel_b=0, lut_bits=COPY_A, genome=genome)

    network = EvoBooleanNetwork(config)
    return network, genome, config


class TestSynchronousUpdate:

    def test_swap_one_step(self):
        """After one step [0,1] should become [1,0] — a swap."""
        network, genome, config = _make_swap_network()
        initial = np.array([0, 1], dtype=np.uint8)
        traj = network.run(genome, n_steps=1, initial_state=initial, return_trajectory=True)
        assert list(traj[0]) == [0, 1], "Initial state wrong"
        assert list(traj[1]) == [1, 0], (
            "Synchronous swap failed — got sequential result instead"
        )

    def test_swap_two_steps(self):
        """Two swaps should restore the original state."""
        network, genome, config = _make_swap_network()
        initial = np.array([0, 1], dtype=np.uint8)
        traj = network.run(genome, n_steps=2, initial_state=initial, return_trajectory=True)
        assert list(traj[0]) == [0, 1]
        assert list(traj[1]) == [1, 0]
        assert list(traj[2]) == [0, 1]

    def test_sequential_would_differ(self):
        """
        Demonstrate that sequential evaluation gives [1,1] while synchronous gives [1,0].
        This is the reference case proving we must not update in-place during a step.
        """
        network, genome, config = _make_swap_network()
        initial = np.array([0, 1], dtype=np.uint8)

        # Correct synchronous result
        traj = network.run(genome, n_steps=1, initial_state=initial, return_trajectory=True)
        sync_result = list(traj[1])

        # Simulate what sequential (wrong) evaluation would produce:
        # node0 reads node1(t=0)=1 -> node0_new=1
        # node1 reads already-updated node0=1 -> node1_new=1
        sequential_wrong = [1, 1]

        assert sync_result == [1, 0], f"Expected [1,0], got {sync_result}"
        assert sync_result != sequential_wrong, "Sync and sequential should differ"

    def test_trajectory_shape(self):
        network, genome, config = _make_swap_network()
        traj = network.run(genome, n_steps=5, initial_state=np.zeros(2, dtype=np.uint8))
        assert traj.shape == (6, 2)  # n_steps+1 rows, n_nodes cols

    def test_no_return_trajectory(self):
        network, genome, config = _make_swap_network()
        final = network.run(
            genome, n_steps=2, initial_state=np.array([0, 1], dtype=np.uint8),
            return_trajectory=False
        )
        assert final.shape == (2,)
        assert list(final) == [0, 1]  # two swaps -> back to original

    def test_reset_state_zeros(self):
        network, genome, config = _make_swap_network()
        network.state = np.array([1, 1], dtype=np.uint8)
        network.reset_state()
        assert list(network.state) == [0, 0]

    def test_reset_state_custom(self):
        network, genome, config = _make_swap_network()
        network.reset_state(np.array([1, 0], dtype=np.uint8))
        assert list(network.state) == [1, 0]

    def test_read_outputs_final(self):
        network, genome, config = _make_swap_network()
        traj = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.uint8)
        # output_nodes defaults to [n_nodes-1] = [1]
        out = network.read_outputs(traj, mode="final")
        assert list(out) == [1]  # last row, node 1

    def test_read_outputs_activity(self):
        network, genome, config = _make_swap_network()
        traj = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.uint8)
        out = network.read_outputs(traj, mode="activity")
        # node 1 is active in rows 0 and 2 -> sum = 2
        assert list(out) == [2]
