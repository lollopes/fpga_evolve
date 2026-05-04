"""
test_tasks.py — Tests for dataset generators and task evaluation.
"""

import numpy as np
import pytest

from evo_boolean_network.config import EvoConfig
from evo_boolean_network.genome import encode_node_genome
from evo_boolean_network.network import EvoBooleanNetwork
from evo_boolean_network.tasks import (
    xor_dataset,
    and_dataset,
    or_dataset,
    pattern_dataset,
    evaluate_binary_task,
)


class TestDatasets:

    def test_xor_dataset_values(self):
        X, y = xor_dataset()
        assert X.shape == (4, 2)
        assert y.shape == (4,)
        # Truth table: 0^0=0, 0^1=1, 1^0=1, 1^1=0
        expected = [0, 1, 1, 0]
        assert list(y) == expected

    def test_xor_dataset_dtypes(self):
        X, y = xor_dataset()
        assert X.dtype == np.uint8
        assert y.dtype == np.uint8

    def test_xor_all_input_combos(self):
        X, y = xor_dataset()
        input_set = {tuple(row) for row in X}
        assert input_set == {(0, 0), (0, 1), (1, 0), (1, 1)}

    def test_and_dataset_values(self):
        X, y = and_dataset()
        assert list(y) == [0, 0, 0, 1]

    def test_or_dataset_values(self):
        X, y = or_dataset()
        assert list(y) == [0, 1, 1, 1]

    def test_pattern_dataset_shape(self):
        for n in [1, 2, 3, 4]:
            X, y = pattern_dataset(n)
            assert X.shape == (2**n, n)
            assert y.shape == (2**n,)

    def test_pattern_dataset_majority(self):
        X, y = pattern_dataset(3)
        for row, label in zip(X, y):
            majority = int(row.sum() > 1.5)
            assert label == majority


class TestEvaluateBinaryTask:

    def _make_xor_network(self):
        """
        Build a 2-node, 2-input network where node 0 directly computes XOR.

        Node 0: XOR of ext[0] and ext[1] — sel_a=0, sel_b=1, lut=XOR=[0,1,1,0]
        Node 1: anything (not used as output)

        output_nodes=[0] so we read node 0 directly after 1 step.

        Synchronous semantics: after 1 step node0 = XOR(ext0, ext1).
        Reading node1 after 1 step would give its XOR of OLD node0 state (=0),
        not the new one — that is correct synchronous behavior, not a bug.
        """
        config = EvoConfig(n_nodes=2, n_inputs=2, n_steps=1, output_nodes=[0])
        # Candidate vector for node 0: [ext0, ext1, node1]
        # sel_a=0 -> ext0, sel_b=1 -> ext1
        XOR = np.array([0, 1, 1, 0], dtype=np.uint8)
        CONST_ZERO = np.array([0, 0, 0, 0], dtype=np.uint8)

        genome = np.zeros(config.total_genome_bits, dtype=np.uint8)
        encode_node_genome(config, 0, sel_a=0, sel_b=1, lut_bits=XOR, genome=genome)
        encode_node_genome(config, 1, sel_a=0, sel_b=0, lut_bits=CONST_ZERO, genome=genome)

        network = EvoBooleanNetwork(config)
        return network, genome, config

    def test_perfect_xor(self):
        """A hand-crafted XOR genome should achieve 100% accuracy."""
        network, genome, config = self._make_xor_network()
        X, y = xor_dataset()
        acc, preds = evaluate_binary_task(network, genome, X, y, output_node=0, n_steps=1)
        assert acc == 1.0, f"Expected 1.0, got {acc} with predictions {preds}"
        np.testing.assert_array_equal(preds, y)

    def test_return_types(self):
        network, genome, config = self._make_xor_network()
        X, y = xor_dataset()
        acc, preds = evaluate_binary_task(network, genome, X, y, output_node=0, n_steps=1)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0
        assert preds.shape == y.shape
        assert preds.dtype == np.uint8

    def test_predictions_are_binary(self):
        network, genome, config = self._make_xor_network()
        X, y = xor_dataset()
        _, preds = evaluate_binary_task(network, genome, X, y, output_node=0, n_steps=1)
        assert set(preds.tolist()).issubset({0, 1})

    def test_const_zero_genome_accuracy(self):
        """A genome that outputs constant 0 achieves 0.5 on balanced XOR."""
        config = EvoConfig(n_nodes=2, n_inputs=2, n_steps=1, output_nodes=[0])
        CONST_ZERO = np.array([0, 0, 0, 0], dtype=np.uint8)
        genome = np.zeros(config.total_genome_bits, dtype=np.uint8)
        encode_node_genome(config, 0, sel_a=0, sel_b=0, lut_bits=CONST_ZERO, genome=genome)
        encode_node_genome(config, 1, sel_a=0, sel_b=0, lut_bits=CONST_ZERO, genome=genome)

        network = EvoBooleanNetwork(config)
        X, y = xor_dataset()
        acc, preds = evaluate_binary_task(network, genome, X, y, output_node=0, n_steps=1)
        # XOR has 2 zeros and 2 ones -> constant-0 gets 2/4 = 0.5
        assert acc == 0.5
        assert list(preds) == [0, 0, 0, 0]
