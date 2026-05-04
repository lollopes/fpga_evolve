"""
test_node.py — Tests for per-node computation and candidate vector construction.
"""

import numpy as np
import pytest

from evo_boolean_network.config import EvoConfig
from evo_boolean_network.node import build_candidate_vector, compute_node_output


class TestCandidateVector:
    def test_node_not_in_its_own_candidates(self):
        """A node must never see its own state as a candidate input."""
        config = EvoConfig(n_nodes=4, n_inputs=0)
        state = np.array([10, 20, 30, 40], dtype=np.uint8)
        ext = np.array([], dtype=np.uint8)

        for nid in range(config.n_nodes):
            candidates = build_candidate_vector(state, ext, nid)
            # The node's own value must not appear in candidates
            # (we check that node_id's state value is absent and length is correct)
            assert len(candidates) == config.n_candidates_per_node
            assert state[nid] not in candidates or (
                # edge case: coincidental value collision — check by position
                # by verifying candidates == state without index nid
                list(candidates) == list(np.concatenate([state[:nid], state[nid+1:]]))
            )

    def test_candidate_order_external_first(self):
        """External inputs come before internal node states."""
        config = EvoConfig(n_nodes=3, n_inputs=2)
        state = np.array([5, 6, 7], dtype=np.uint8)
        ext = np.array([11, 22], dtype=np.uint8)

        # For node 0: candidates = [ext0, ext1, node1, node2]
        cands = build_candidate_vector(state, ext, node_id=0)
        assert cands[0] == 11
        assert cands[1] == 22
        assert cands[2] == 6   # node 1
        assert cands[3] == 7   # node 2

    def test_candidate_skips_correct_node(self):
        """For each node_id, verify the skipped entry is the node itself."""
        config = EvoConfig(n_nodes=5, n_inputs=0)
        state = np.arange(5, dtype=np.uint8)
        ext = np.array([], dtype=np.uint8)

        for nid in range(5):
            cands = build_candidate_vector(state, ext, nid)
            expected = np.concatenate([state[:nid], state[nid+1:]])
            np.testing.assert_array_equal(cands, expected)

    def test_candidate_length(self):
        config = EvoConfig(n_nodes=6, n_inputs=3)
        state = np.zeros(6, dtype=np.uint8)
        ext = np.zeros(3, dtype=np.uint8)
        for nid in range(6):
            cands = build_candidate_vector(state, ext, nid)
            assert len(cands) == config.n_candidates_per_node


class TestComputeNodeOutput:
    """Test LUT behavior for canonical Boolean functions."""

    def _make_single_node_lut(self, lut_bits):
        """Return a compute function configured with given lut_bits."""
        config = EvoConfig(n_nodes=2, n_inputs=2)
        # sel_a=0 picks ext[0]=A, sel_b=1 picks ext[1]=B
        state = np.zeros(2, dtype=np.uint8)
        lut = np.array(lut_bits, dtype=np.uint8)

        def fn(A, B):
            ext = np.array([A, B], dtype=np.uint8)
            return compute_node_output(state, ext, config, 0, 0, 1, lut)

        return fn

    def test_lut_and(self):
        # AND: 0&0=0, 0&1=0, 1&0=0, 1&1=1 -> lut = [0,0,0,1]
        fn = self._make_single_node_lut([0, 0, 0, 1])
        assert fn(0, 0) == 0
        assert fn(0, 1) == 0
        assert fn(1, 0) == 0
        assert fn(1, 1) == 1

    def test_lut_or(self):
        # OR: [0,1,1,1]
        fn = self._make_single_node_lut([0, 1, 1, 1])
        assert fn(0, 0) == 0
        assert fn(0, 1) == 1
        assert fn(1, 0) == 1
        assert fn(1, 1) == 1

    def test_lut_xor(self):
        # XOR: [0,1,1,0]
        fn = self._make_single_node_lut([0, 1, 1, 0])
        assert fn(0, 0) == 0
        assert fn(0, 1) == 1
        assert fn(1, 0) == 1
        assert fn(1, 1) == 0

    def test_lut_copy_a(self):
        # COPY_A: output = A regardless of B -> lut = [0,0,1,1]
        fn = self._make_single_node_lut([0, 0, 1, 1])
        assert fn(0, 0) == 0
        assert fn(0, 1) == 0
        assert fn(1, 0) == 1
        assert fn(1, 1) == 1

    def test_lut_not_a(self):
        # NOT_A: [1,1,0,0]
        fn = self._make_single_node_lut([1, 1, 0, 0])
        assert fn(0, 0) == 1
        assert fn(0, 1) == 1
        assert fn(1, 0) == 0
        assert fn(1, 1) == 0

    def test_sel_modulo_wrapping(self):
        """Selectors that exceed n_candidates must wrap silently."""
        config = EvoConfig(n_nodes=2, n_inputs=1)
        # n_candidates = 1 + 2 - 1 = 2 (ext[0], node[1] for node 0)
        state = np.array([0, 1], dtype=np.uint8)
        ext = np.array([0], dtype=np.uint8)
        lut = np.array([0, 0, 0, 1], dtype=np.uint8)  # AND

        # sel_a=0 -> ext[0]=0, sel_b=100 -> 100 % 2 = 0 -> also ext[0]=0
        # AND(0, 0) = 0
        out = compute_node_output(state, ext, config, 0, 0, 100, lut)
        assert out == 0
