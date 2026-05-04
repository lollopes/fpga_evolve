"""
genome.py — Genome encoding and decoding for the Boolean network.

A genome is a flat numpy array of uint8 bits (0 or 1).
It is partitioned into one section per node, and each section is further
split into three fields:

    [ sel_a (sel_bits) | sel_b (sel_bits) | lut_init (4 bits) ]

All multi-bit integers use little-endian bit ordering:
    bits[0] is the LSB, bits[-1] is the MSB.

FPGA mapping:
  The genome is directly equivalent to the FPGA configuration memory.
  Each node's section maps to one configuration register block.
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np

from .config import EvoConfig


# ---------------------------------------------------------------------------
# Bit / integer conversion utilities
# ---------------------------------------------------------------------------

def bits_to_int(bits: np.ndarray) -> int:
    """
    Convert a little-endian bit array to an unsigned integer.

    bits[0] is the LSB (weight 2^0), bits[-1] is the MSB (weight 2^(n-1)).

    Examples
    --------
    >>> bits_to_int(np.array([1, 0, 1, 0]))  # 0b0101 = 5
    5
    """
    bits = np.asarray(bits, dtype=np.uint8)
    powers = (1 << np.arange(len(bits), dtype=np.uint64))
    return int(np.sum(bits.astype(np.uint64) * powers))


def int_to_bits(value: int, n_bits: int) -> np.ndarray:
    """
    Convert an unsigned integer to a little-endian bit array of length n_bits.

    bits[0] is the LSB. Raises ValueError if value does not fit in n_bits bits.

    Examples
    --------
    >>> int_to_bits(5, 4)  # 5 = 0b0101 -> [1, 0, 1, 0]
    array([1, 0, 1, 0], dtype=uint8)
    """
    if value < 0:
        raise ValueError(f"value must be non-negative, got {value}")
    if n_bits < 1:
        raise ValueError(f"n_bits must be >= 1, got {n_bits}")
    max_val = (1 << n_bits) - 1
    if value > max_val:
        raise ValueError(f"value {value} does not fit in {n_bits} bits (max {max_val})")
    bits = np.zeros(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        bits[i] = (value >> i) & 1
    return bits


# ---------------------------------------------------------------------------
# Genome creation
# ---------------------------------------------------------------------------

def random_genome(config: EvoConfig, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generate a uniformly random genome as a flat uint8 bit array.

    Parameters
    ----------
    config : EvoConfig
        Network configuration (determines total_genome_bits).
    rng : np.random.Generator | None
        Random number generator. If None, uses config.seed or an arbitrary seed.

    Returns
    -------
    np.ndarray of shape (total_genome_bits,), dtype uint8, values in {0, 1}.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)
    return rng.integers(0, 2, size=config.total_genome_bits, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Per-node genome decode
# ---------------------------------------------------------------------------

def node_genome_slice(config: EvoConfig, node_id: int) -> slice:
    """Return the slice of the flat genome array belonging to node_id."""
    start = node_id * config.node_genome_bits
    end = start + config.node_genome_bits
    return slice(start, end)


def decode_node_genome(
    genome: np.ndarray,
    config: EvoConfig,
    node_id: int,
) -> Tuple[int, int, np.ndarray]:
    """
    Extract the configuration fields for a single node from the flat genome.

    Genome layout for node i (little-endian within each field):
        [ sel_a[0..sel_bits-1] | sel_b[0..sel_bits-1] | lut[0..3] ]

    Parameters
    ----------
    genome : np.ndarray
        Flat bit array of length config.total_genome_bits.
    config : EvoConfig
        Network configuration.
    node_id : int
        Index of the node to decode (0-indexed).

    Returns
    -------
    sel_a : int
        Raw selector value for MUX A (use modulo n_candidates for actual index).
    sel_b : int
        Raw selector value for MUX B.
    lut_bits : np.ndarray of shape (4,), dtype uint8
        Truth table of the 2-input LUT. lut_bits[i] is the output for input index i,
        where index = (A << 1) | B.
    """
    s = node_genome_slice(config, node_id)
    node_bits = genome[s]

    sb = config.sel_bits

    sel_a_bits = node_bits[0:sb]
    sel_b_bits = node_bits[sb : 2 * sb]
    lut_bits = node_bits[2 * sb : 2 * sb + 4]

    sel_a = bits_to_int(sel_a_bits)
    sel_b = bits_to_int(sel_b_bits)

    return sel_a, sel_b, lut_bits.copy()


def encode_node_genome(
    config: EvoConfig,
    node_id: int,
    sel_a: int,
    sel_b: int,
    lut_bits: np.ndarray,
    genome: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Write a node's config fields back into a genome array (in-place if provided).

    Useful for constructing hand-crafted genomes in tests and examples.

    Returns the modified (or newly created) genome array.
    """
    if genome is None:
        genome = np.zeros(config.total_genome_bits, dtype=np.uint8)

    s = node_genome_slice(config, node_id)
    node_bits = genome[s]

    sb = config.sel_bits
    node_bits[0:sb] = int_to_bits(sel_a, sb)
    node_bits[sb : 2 * sb] = int_to_bits(sel_b, sb)
    node_bits[2 * sb : 2 * sb + 4] = np.asarray(lut_bits, dtype=np.uint8)

    return genome


def describe_genome(genome: np.ndarray, config: EvoConfig) -> list[dict]:
    """
    Decode all nodes and return a list of human-readable dicts.

    Useful for the inspect_genome example and debugging.
    """
    descriptions = []
    lut_names = {
        (0, 0, 0, 0): "CONST_0",
        (1, 1, 1, 1): "CONST_1",
        (0, 0, 1, 1): "COPY_A",    # output = A  (index = (A<<1)|B, so bit2 and bit3 set)
        (0, 1, 0, 1): "COPY_B",    # output = B
        (1, 1, 0, 0): "NOT_A",
        (1, 0, 1, 0): "NOT_B",
        (0, 0, 0, 1): "AND",
        (1, 1, 1, 0): "NAND",
        (0, 1, 1, 1): "OR",
        (1, 0, 0, 0): "NOR",
        (0, 1, 1, 0): "XOR",
        (1, 0, 0, 1): "XNOR",
    }
    for nid in range(config.n_nodes):
        sel_a, sel_b, lut_bits = decode_node_genome(genome, config, nid)
        actual_a = sel_a % config.n_candidates_per_node
        actual_b = sel_b % config.n_candidates_per_node
        src_a = config.candidate_index(nid, actual_a)
        src_b = config.candidate_index(nid, actual_b)
        lut_key = tuple(int(b) for b in lut_bits)
        lut_name = lut_names.get(lut_key, "CUSTOM")
        descriptions.append(
            {
                "node_id": nid,
                "sel_a_raw": sel_a,
                "sel_b_raw": sel_b,
                "sel_a_eff": actual_a,
                "sel_b_eff": actual_b,
                "src_a": src_a,
                "src_b": src_b,
                "lut_bits": lut_bits,
                "lut_name": lut_name,
            }
        )
    return descriptions
