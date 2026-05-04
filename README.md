# evo_boolean_network

A Python simulator for an **evolvable synchronous Boolean network** — a programmable grid of configurable digital nodes designed as a software model before FPGA implementation.

The key idea: a network of simple logic cells, each controlled by a **genome** (a flat bitstring), that can be evolved via a genetic algorithm to solve Boolean classification tasks.

---

## Table of contents

1. [Motivation](#motivation)
2. [What a node looks like](#what-a-node-looks-like)
3. [Network connectivity](#network-connectivity)
4. [The genome](#the-genome)
5. [Why the update must be synchronous](#why-the-update-must-be-synchronous)
6. [FPGA mapping](#fpga-mapping)
7. [Project structure](#project-structure)
8. [Installation](#installation)
9. [Quick start](#quick-start)
10. [Examples](#examples)
11. [Running tests](#running-tests)
12. [API reference](#api-reference)

---

## Motivation

FPGAs are arrays of configurable logic cells connected by a programmable routing fabric. This project models a simplified version of that idea in Python:

- Each **node** is a 2-input LUT (look-up table) with configurable input routing.
- The **genome** is a bitstring that encodes which inputs each node reads and what Boolean function it computes — analogous to an FPGA bitstream.
- An **evolutionary algorithm** searches the genome space to find configurations that solve a target task.

The Python model is designed so that every concept maps directly to synthesisable FPGA hardware. You can prototype the logic and the evolution here, then translate node-by-node to SystemVerilog.

---

## What a node looks like

```
 candidates[]
 ┌─────────────────────────────────────────────┐
 │  ext_in[0] ... ext_in[k-1]  node[0] ... (all except self) ...  │
 └──────────────────────┬──────────────────────┘
                        │
              ┌─────────┴─────────┐
              │ MUX A   │ MUX B   │   ← sel_a, sel_b from genome
              └────┬────┴────┬────┘
                   │ A       │ B
              ┌────┴─────────┴────┐
              │   2-input LUT     │   ← 4-bit truth table from genome
              └─────────┬─────────┘
                        │ output bit
```

Each node has:

| Component | Role | Genome bits |
|-----------|------|-------------|
| **MUX A** | Selects the first LUT input from the candidate pool | `sel_bits` bits |
| **MUX B** | Selects the second LUT input from the candidate pool | `sel_bits` bits |
| **2-input LUT** | Computes one output bit from A and B | 4 bits |

The LUT truth table is indexed as `lut[( A << 1 ) | B]`:

| index | A | B | example: XOR |
|-------|---|---|--------------|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 1 | 1 |
| 2 | 1 | 0 | 1 |
| 3 | 1 | 1 | 0 |

Any 2-input Boolean function (AND, OR, XOR, NAND, COPY, NOT, …) is expressible by setting the 4 LUT bits appropriately.

---

## Network connectivity

For a network with `N` nodes and `K` external inputs, the **candidate vector** available to node `i` is:

```
candidates = [ext_in[0], ..., ext_in[K-1],
              node[0].state, ..., node[i-1].state,
              node[i+1].state, ..., node[N-1].state]
```

Key rules:
- **A node cannot read itself.** Node `i` is always excluded from its own candidate list.
- Every node can read every other node and all external inputs — full connectivity except for self-loops.
- Both MUX A and MUX B select independently from the same candidate list, so a node can combine any two signals in the network.

The number of candidates per node is:

```
n_candidates = K + N - 1
sel_bits     = ceil(log2(n_candidates))
```

---

## The genome

The genome is a flat `numpy` array of bits (`uint8`, values 0 or 1). It is partitioned into `N` equal sections, one per node:

```
genome = [ node_0_bits | node_1_bits | ... | node_{N-1}_bits ]
```

Each node section is laid out as:

```
node_i_bits = [ sel_a (sel_bits) | sel_b (sel_bits) | lut_init (4 bits) ]
```

All multi-bit integers use **little-endian** bit ordering: `bits[0]` is the LSB.

Selector values are wrapped **modulo n_candidates**, so every possible bitstring produces a valid (if not necessarily useful) configuration — there are no illegal genomes.

### Example: 4 nodes, 2 external inputs

```
n_candidates = 2 + 4 - 1 = 5
sel_bits     = ceil(log2(5)) = 3
node_genome_bits = 3 + 3 + 4 = 10
total_genome_bits = 4 * 10 = 40
```

---

## Why the update must be synchronous

In clocked digital hardware, all flip-flops sample their inputs simultaneously on the rising clock edge. The new values are only visible to other nodes on the **next** clock cycle.

This simulator enforces the same rule:

```python
frozen_state = self.state.copy()      # latch: snapshot of state(t)
next_state = compute_all(frozen_state) # combinational: all nodes read frozen
self.state = next_state               # clock edge: atomic update
```

**Why this matters:** if you updated nodes one at a time (node 0, then node 1 reads the already-updated node 0), you would get a different result. That would model a ripple-propagation circuit, not a synchronous register file. The tests include a concrete case where the two strategies give different answers, and verify the simulator gives the correct synchronous result.

---

## FPGA mapping

| Python | FPGA / SystemVerilog |
|--------|----------------------|
| `self.state` (numpy array) | Flip-flop register array (one FF per node) |
| Genome bitstring | Configuration memory (SRAM block or bitstream) |
| `compute_node_output()` | Combinational logic: mux tree + LUT |
| `network.step()` | One rising clock edge |
| `network.run(n_steps=T)` | T clock cycles |
| `config.sel_bits` | Width of MUX select bus |
| `config.node_genome_bits` | Width of per-node config register |
| `config.total_genome_bits` | Total config memory depth |

Each Python node translates directly to a small SystemVerilog module:

```systemverilog
module node #(parameter SEL_BITS=3, N_CAND=5) (
    input  logic [N_CAND-1:0] candidates,
    input  logic [SEL_BITS-1:0] sel_a, sel_b,
    input  logic [3:0] lut_init,
    output logic out
);
    logic a = candidates[sel_a];
    logic b = candidates[sel_b];
    assign out = lut_init[(a << 1) | b];
endmodule
```

---

## Project structure

```
evo_boolean_network/
├── pyproject.toml
├── README.md
├── src/
│   └── evo_boolean_network/
│       ├── __init__.py       — Public API re-exports
│       ├── config.py         — EvoConfig dataclass: network dimensions + derived constants
│       ├── genome.py         — Genome creation, bit conversion, encode/decode per node
│       ├── node.py           — Single-node combinational computation (pure function)
│       ├── network.py        — EvoBooleanNetwork: synchronous multi-step simulator
│       ├── tasks.py          — Boolean datasets (XOR, AND, OR, majority) + evaluator
│       └── evolution.py      — Simple (μ+λ) evolutionary algorithm
├── examples/
│   ├── run_random_network.py — Forward pass with a random genome, print trajectory
│   ├── solve_xor.py          — Evolve to solve XOR, print accuracy per generation
│   └── inspect_genome.py     — Decode and print every node's configuration
└── tests/
    ├── test_genome.py         — Bit conversion round-trips, encode/decode correctness
    ├── test_node.py           — LUT truth tables, candidate vector construction
    ├── test_network_sync.py   — Synchronous update vs. sequential evaluation
    └── test_tasks.py          — Dataset correctness, evaluate_binary_task
```

---

## Installation

**With pip (recommended):**

```bash
cd evo_boolean_network
pip install -e ".[dev]"   # installs numpy + pytest in editable mode
pip install -e ".[viz]"   # optional: live Dash evolution viewer dependencies
```

**Without installing** (examples handle `sys.path` automatically):

```bash
pip install numpy pytest dash dash-cytoscape plotly networkx
```

---

## Quick start

```python
import numpy as np
from evo_boolean_network import EvoConfig, EvoBooleanNetwork, random_genome

# 1. Define the network structure
config = EvoConfig(
    n_nodes=16,
    n_inputs=4,
    n_steps=8,
    output_nodes=[15],
    seed=42,
)

# 2. Generate a random genome
genome = random_genome(config)

# 3. Run the network
network = EvoBooleanNetwork(config)
trajectory = network.run(genome, external_inputs=np.array([1, 0, 1, 1], dtype=np.uint8))
# trajectory.shape == (n_steps+1, n_nodes)

# 4. Read the output node
output = network.read_outputs(trajectory, mode="final")
print(output)   # e.g. [1]
```

### Evolving a genome to solve XOR

```python
from evo_boolean_network import EvoConfig, xor_dataset, evolve_binary_classifier

config = EvoConfig(n_nodes=16, n_inputs=2, n_steps=8, output_nodes=[15], seed=7)
X, y = xor_dataset()

result = evolve_binary_classifier(
    config=config,
    X=X,
    y=y,
    pop_size=100,
    n_generations=500,
    mutation_rate=0.02,
    n_jobs=-1,  # use most CPU cores for population fitness evaluation
)

print(f"Best accuracy: {result.best_accuracy:.2%}")
```

---

## Examples

Run from the `evo_boolean_network/` directory:

```bash
# Random genome — prints step-by-step trajectory for a 16-node network
python examples/run_random_network.py

# Evolutionary XOR solver — prints accuracy per generation and final predictions
python examples/solve_xor.py

# Genome inspector — prints each node's sel_a, sel_b, LUT bits, and inferred function name
python examples/inspect_genome.py
```

---

## Running tests

```bash
cd evo_boolean_network
pytest          # run all 45 tests
pytest -v       # verbose output
pytest tests/test_network_sync.py -v   # just the synchrony tests
```

The test suite covers:
- `bits_to_int` / `int_to_bits` round-trips for random values.
- Genome encode/decode consistency across all nodes.
- No node appears in its own candidate vector.
- Synchronous update gives a different result from sequential update (and the simulator gives the correct one).
- LUT AND, OR, XOR, COPY, NOT behavior verified by truth table.
- XOR dataset has correct labels.
- `evaluate_binary_task` returns valid accuracy and binary predictions.

---

## API reference

### `EvoConfig`
```python
EvoConfig(n_nodes, n_inputs=0, n_steps=8, output_nodes=None, seed=None)
```
Holds all structural parameters. Derived constants (`sel_bits`, `node_genome_bits`, `total_genome_bits`, `n_candidates_per_node`) are computed automatically.

### `random_genome(config, rng=None) -> np.ndarray`
Returns a uniformly random flat bit array of length `config.total_genome_bits`.

### `decode_node_genome(genome, config, node_id) -> (sel_a, sel_b, lut_bits)`
Extracts the three fields for one node from the flat genome.

### `EvoBooleanNetwork(config)`
- `.reset_state(initial_state=None)` — reset to zeros or a given state.
- `.step(genome, external_inputs=None)` — one synchronous clock cycle.
- `.run(genome, external_inputs=None, n_steps=None, initial_state=None, return_trajectory=True)` — multi-step run, returns trajectory `(T+1, N)` or final state `(N,)`.
- `.read_outputs(trajectory_or_state, mode="final"|"activity")` — extract selected output node values.

### `evolve_binary_classifier(config, X, y, ...) -> EvolutionResult`
Runs a simple (μ+λ) evolutionary loop. Returns `EvolutionResult` with `.best_genome`, `.best_accuracy`, and `.history` (best accuracy per generation).

Use `n_jobs` to control population-level parallelism:
- `n_jobs=1` → serial evaluation
- `n_jobs=-1` → use most available CPU cores (`cpu_count - 1`)
- `n_jobs>1` → exact number of worker processes
