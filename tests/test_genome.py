"""
test_genome.py — Tests for genome encoding / decoding utilities.
"""

import numpy as np
import pytest

from evo_boolean_network.config import EvoConfig
from evo_boolean_network.genome import (
    bits_to_int,
    int_to_bits,
    decode_node_genome,
    encode_node_genome,
    random_genome,
)


# ---------------------------------------------------------------------------
# bits_to_int / int_to_bits round-trip
# ---------------------------------------------------------------------------

class TestBitConversions:
    def test_roundtrip_zero(self):
        for n in [1, 4, 8]:
            bits = int_to_bits(0, n)
            assert bits_to_int(bits) == 0

    def test_roundtrip_one(self):
        bits = int_to_bits(1, 4)
        assert bits_to_int(bits) == 1
        assert bits[0] == 1  # LSB first (little-endian)
        assert bits[1] == 0

    def test_roundtrip_five(self):
        # 5 = 0b0101 -> little-endian [1,0,1,0]
        bits = int_to_bits(5, 4)
        assert list(bits) == [1, 0, 1, 0]
        assert bits_to_int(bits) == 5

    def test_roundtrip_max(self):
        for n_bits in [1, 3, 5, 8]:
            max_val = (1 << n_bits) - 1
            bits = int_to_bits(max_val, n_bits)
            assert bits_to_int(bits) == max_val

    def test_roundtrip_random(self):
        rng = np.random.default_rng(0)
        for n_bits in [2, 4, 6, 8]:
            for _ in range(50):
                val = int(rng.integers(0, 1 << n_bits))
                bits = int_to_bits(val, n_bits)
                assert len(bits) == n_bits
                assert bits_to_int(bits) == val

    def test_int_to_bits_raises_on_overflow(self):
        with pytest.raises(ValueError):
            int_to_bits(8, 3)  # max is 7

    def test_int_to_bits_raises_on_negative(self):
        with pytest.raises(ValueError):
            int_to_bits(-1, 4)

    def test_bits_dtype(self):
        bits = int_to_bits(3, 4)
        assert bits.dtype == np.uint8

    def test_bits_values_are_binary(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            val = int(rng.integers(0, 256))
            bits = int_to_bits(val, 8)
            assert set(bits.tolist()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Genome decode / encode
# ---------------------------------------------------------------------------

class TestDecodeNodeGenome:
    def setup_method(self):
        self.config = EvoConfig(n_nodes=4, n_inputs=2, n_steps=4)

    def test_decoded_sizes(self):
        genome = random_genome(self.config, np.random.default_rng(42))
        for nid in range(self.config.n_nodes):
            sel_a, sel_b, lut_bits = decode_node_genome(genome, self.config, nid)
            assert isinstance(sel_a, int)
            assert isinstance(sel_b, int)
            assert lut_bits.shape == (4,)

    def test_lut_bits_are_binary(self):
        genome = random_genome(self.config, np.random.default_rng(7))
        for nid in range(self.config.n_nodes):
            _, _, lut_bits = decode_node_genome(genome, self.config, nid)
            assert set(lut_bits.tolist()).issubset({0, 1})

    def test_encode_decode_roundtrip(self):
        # Manually encode known values and verify decode matches
        config = self.config
        genome = np.zeros(config.total_genome_bits, dtype=np.uint8)
        for nid in range(config.n_nodes):
            encode_node_genome(config, nid, sel_a=1, sel_b=2,
                               lut_bits=np.array([0, 1, 1, 0]), genome=genome)

        for nid in range(config.n_nodes):
            sel_a, sel_b, lut_bits = decode_node_genome(genome, config, nid)
            assert sel_a == 1
            assert sel_b == 2
            assert list(lut_bits) == [0, 1, 1, 0]

    def test_different_nodes_have_independent_sections(self):
        config = self.config
        genome = np.zeros(config.total_genome_bits, dtype=np.uint8)
        encode_node_genome(config, 0, sel_a=0, sel_b=0,
                           lut_bits=np.array([1, 1, 1, 1]), genome=genome)
        encode_node_genome(config, 1, sel_a=1, sel_b=1,
                           lut_bits=np.array([0, 0, 0, 0]), genome=genome)

        _, _, lut0 = decode_node_genome(genome, config, 0)
        _, _, lut1 = decode_node_genome(genome, config, 1)
        assert list(lut0) == [1, 1, 1, 1]
        assert list(lut1) == [0, 0, 0, 0]

    def test_random_genome_length(self):
        config = EvoConfig(n_nodes=8, n_inputs=3)
        genome = random_genome(config, np.random.default_rng(0))
        assert len(genome) == config.total_genome_bits

    def test_random_genome_is_binary(self):
        config = EvoConfig(n_nodes=8, n_inputs=3)
        genome = random_genome(config, np.random.default_rng(0))
        assert set(genome.tolist()).issubset({0, 1})
