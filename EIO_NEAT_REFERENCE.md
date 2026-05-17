# EIO NEAT — Comprehensive Reference

---

## Table of Contents

1. [Node Architecture](#1-node-architecture)
   - [1a. Baseline Node (LUT-Register)](#1a-baseline-node-lut-register)
   - [1b. Membrane Node (LIF-Compiled LUT)](#1b-membrane-node-lif-compiled-lut)
2. [Network Topology](#2-network-topology)
3. [How NEAT Works](#3-how-neat-works)
4. [Evolution Parameters](#4-evolution-parameters)
5. [Global Readout Layer (src/)](#5-global-readout-layer-src)

---

## 1. Node Architecture

Two node variants are supported, selected by the `membrane_bits` config value.

---

### 1a. Baseline Node (LUT-Register)

Every non-input node is a **LUT-register unit** — a synchronous Boolean cell with 1 bit of internal state:

1. Collects up to `k` binary data inputs (one per LUT input port)
2. Evaluates a `k`-input Boolean function stored in a Look-Up Table (LUT)
3. Holds the resulting **1-bit** spike state until the next timestep

The LUT stores `2^k` bits — every possible `k`-bit input combination maps to one output bit. Evolution directly flips individual LUT bits.

```
  k data inputs
  ─────────────
  x[0] ─┐
  x[1] ─┤─► [ LUT  ]──► ┌───┐
  ...    │   [2^k×1]     │ D ├──► state[t]
  x[k-1]┘               └───┘
```

**Update rule:**
```
addr    = binary_to_int(data_inputs[0..k-1])
state[t] = LUT[addr]
```

**Internal state:** 1 bit per node. **LUT size:** `2^k × 1 bit`.

---

### 1b. Membrane Node (LIF-Compiled LUT)

Enabled by setting `membrane_bits = β > 0` (H nodes) or `output_membrane_bits > 0` (O nodes).

Each H node carries a **β-bit membrane register** alongside its spike output. The key insight (from LogicNets): a LIF neuron with `k` spike inputs and a β-bit membrane state is a `(k+β) → (1+β)` LUT.

**Crucially, the LUT is not directly evolved.** It is **compiled** deterministically from three evolved parameters + connection weights:

| H node parameter | Range | Role |
|---|---|---|
| `threshold` | `[1, 2^β−1]` | membrane value at which node fires |
| `leak_shift` | `[1, β]` | leak rate: `v -= v >> leak_shift` each step |
| `reset_mode` | `zero / subtract` | hard reset vs subtract-threshold on spike |
| `weight` (per connection) | `[w_min, w_max]` | signed integer contribution of each input port |

At compile time, `compile_lif_luts_for_node()` sweeps all `2^(k+β)` address combinations, applies the LIF dynamics, and writes the result into `lut_spike` and `lut_mem`.

```
  k data inputs   β membrane bits (previous step)
  ─────────────   ──────────────────────────────
  x[0]  ─┐         m[0]  ─┐
  x[1]  ─┤         m[1]  ─┤
  ...    ─┤──────── ...   ─┤──► addr (k+β bits)
  x[k-1]─┘         m[β-1]─┘      │
                                  ├──► [ LUT_spike ]──► ┌───┐
                                  │    [2^(k+β)×1 ]     │ D ├──► spike[t]
                                  │                     └───┘
                                  └──► [ LUT_mem   ]──► ┌───┐
                                       [2^(k+β)×β ]     │ D ├──► mem[t]
                                                         └───┘
```

**Update rule:**
```
inputs   = [data_inputs[0..k-1], mem[t-1][0..β-1]]   (k+β bits)
addr     = binary_to_int(inputs)
spike[t] = LUT_spike[addr]
mem[t]   = LUT_mem[addr]
```

All nodes update synchronously. Recurrence is implicit — H nodes can read the previous spike state of other H nodes.

**Internal state:** `(1 + β)` bits per node. **LUT size:** `2^(k+β)` entries, split into two compiled tables.

---

### Node Types

#### Input node
Passthrough — injects the raw spike value `x_t[f]` into the network at each timestep. No LUT, no state. There are `input_size` input nodes (one per feature dimension).

#### H — Hidden node
- A LIF-compiled LUT-register unit with `k` data input ports.
- Receives connections from input nodes and other H nodes only.
- Has three evolved LIF parameters (`threshold`, `leak_shift`, `reset_mode`) plus per-connection integer weights.
- LUT is recompiled after every mutation to these parameters.

#### O — Output node
- One per class. Fixed at genome initialisation; never added or removed by mutation.
- **Pure weighted spike counter** — no LIF dynamics, no threshold, no leak, no bias.
- Receives connections from H nodes only (no direct input→O connections).
- Has only one evolved attribute per connection: the integer `weight`.
- The LUT is compiled with `threshold = max_v + 1` (never fires) and `leak_shift = β` (no decay), making it a perfect accumulator.
- Final membrane value after T timesteps = total weighted input spike count = class logit.

---

### Connection Genes

| Field | Type | Description |
|---|---|---|
| `src` | node id | source node |
| `dst` | node id | destination node |
| `dst_port` | 0..k-1 | which LUT input port to feed |
| `weight` | int, `[w_min, w_max]` | signed integer weight (used at LUT compile time) |
| `enabled` | bool | whether this connection is active |

Each destination node has exactly `k` input ports. If a port has no incoming connection, it contributes 0 current. Only one connection per (dst, dst_port) slot — duplicates are resolved by `repair_duplicate_slots`.

---

### Genome Representation

A genome contains:
- **`nodes`**: `Dict[node_id → NodeGene]` — kind (`input/H/O`), enabled flag, compiled `lut` + `lut_mem` arrays, `lif` params (H nodes only; `None` for O and input nodes)
- **`connections`**: `Dict[innovation_number → ConnectionGene]` — src, dst, dst_port, weight, enabled
- **`k`**: fixed LUT arity shared by all nodes
- **`membrane_bits`** (β): membrane bits for H nodes
- **`output_membrane_bits`**: membrane bits for O nodes (defaults to `membrane_bits` if 0)
- **`output_ids`**: ordered list of O node IDs (position = class index)

The **innovation number** is a global counter assigned the first time a `(src, dst, port)` triple appears. It enables crossover alignment across genomes with different topologies.

---

## 2. Network Topology

The architecture enforces a **strict two-layer flow**:

```
Input nodes ──→ H nodes (LIF, k ports, temporal processing)
                    └──→ O nodes (pure counters, k ports, readout)
```

**Rules enforced in both mutation and runtime:**
- `input → H`: allowed
- `H → H`: allowed (recurrent)
- `H → O`: allowed
- `input → O`: **forbidden**

This ensures O nodes only receive processed features from H nodes, never raw inputs directly. The separation of concerns is clean: H nodes handle all temporal feature extraction; O nodes are a purely linear weighted readout over H spike trains.

---

## 3. How NEAT Works

NEAT (NeuroEvolution of Augmenting Topologies) simultaneously optimises topology and parameters. It solves three problems:

1. **Competing conventions** — two genomes solving the same problem may use different structures. Crossover between them can produce broken offspring. Innovation numbers align genes across genomes.
2. **Premature topology convergence** — new structure initially hurts fitness. Speciation protects new innovations by competing within similar groups.
3. **Minimal structure bias** — networks start small and grow only when useful.

---

### Phase 1 — Initialisation

A **minimal template genome** is created:
- `input_size` input nodes + `n_outputs` O nodes.
- `initial_h_nodes` H nodes are added, each wired with `k` random input→H connections and one H→O connection (round-robin across outputs).
- No direct input→O connections.

The population is created by cloning this template and mutating each clone.

---

### Phase 2 — Fitness Evaluation

Every genome is run on the training data (or a subsample of size `fitness_sample`). Fitness combines cross-entropy with a complexity penalty:

```
fitness = -mean_CE(logits / T, y) - complexity_coef × n_active_H
```

- `logits[b, c]` = final O node membrane value for class `c`, sample `b` (total weighted spike count over T steps)
- Dividing by `T` normalises to mean weighted input rate per step before softmax
- `n_active_H` = number of enabled H nodes
- `complexity_coef` penalises network growth — each additional H node must earn its place by improving CE more than the penalty cost

This directly maps to the FPGA objective: fewer H nodes = fewer LUT registers on chip.

---

### Phase 3 — Speciation

Genomes are grouped into species based on genome distance:

```
d(a, b) = c_disjoint × (|innovations(a) Δ innovations(b)| / max(|a|, |b|))
        + c_lut      × mean_LIF_distance(shared_H_nodes)
```

- **Disjoint term**: fraction of connection genes not shared (structural divergence)
- **LIF distance term**: normalised distance between `threshold`, `leak_shift`, `reset_mode`, and connection weights across shared nodes

Each genome is compared to the representative of each existing species. If none is close enough, a new species is formed.

---

### Phase 4 — Reproduction

**Species quotas:** each species receives offspring proportional to its mean adjusted fitness. Stale species (`max_stale` generations without improvement) are killed, except the global champion species.

**Elite preservation:** top `elite_per_species` genomes from species with ≥ `elite_min_size` members are copied unchanged.

**Child production:**
1. Select parent `p1` via tournament selection (`tournament_k` competitors).
2. With probability `p_crossover`, select `p2` and perform crossover. Otherwise clone `p1`.
3. Mutate the child.

**Crossover:**
- Connections in both parents: gene chosen randomly from either (75% chance disabled if either parent has it disabled)
- Connections only in the fitter parent: inherited
- Connections only in the weaker parent: discarded
- H node LIF params mixed field-by-field (`threshold`, `leak_shift`, `reset_mode` each independently chosen from one parent)
- LUT tables are recompiled after crossover — never mixed directly

---

### Phase 5 — Mutation

Each child undergoes the following mutations independently:

| Mutation | Trigger | Effect |
|---|---|---|
| **LIF param perturbation** | Each H node param with prob `p_mutate_lif_param` | `threshold`, `leak_shift` ± 1; `reset_mode` flip with prob `p_mutate_lif_param × 0.2` |
| **Connection weight perturbation** | Each enabled connection with prob `p_mutate_conn_weight` | weight ± 1, clamped to `[weight_min, weight_max]` |
| **Add connection** | With prob `p_add_connection` | Picks a valid (src, dst, free_port) respecting topology rules; wires a new connection |
| **Add node** | With prob `p_add_node` | Splits an existing H→H or input→H connection, inserts a new H node with random LIF params |
| **Toggle connection** | With prob `p_toggle_connection` | Enables or disables a random connection |

After any structural change, LUTs are **recompiled** from current LIF params + weights.

**Add node (split):** disables connection `src → dst (port p)`, creates new H node `n`, wires `src → n (port 0)` and `n → dst (port p)`. The new H node inherits a fresh set of random LIF params.

---

### Phase 6 — Val Tracking and Termination

Each generation, the best genome by training fitness is evaluated on the validation set. The genome with the highest **validation accuracy** across all generations is returned as the final result (ties broken by val fitness).

---

## 4. Evolution Parameters

### Population and Time

| Parameter | Default | Effect |
|---|---|---|
| `pop_size` | 50 | Genomes per generation. Larger = more exploration, proportionally slower. |
| `generations` | 50 | Number of evolutionary cycles. |
| `seed` | 42 | RNG seed for reproducibility. |
| `log_every` | 1 | Print progress every N generations. |

---

### Network Architecture

| Parameter | Default | Effect |
|---|---|---|
| `k` | 3 | LUT arity — data inputs per node. LUT size = `2^(k+β)`. Typical range: 2–6. |
| `membrane_bits` | 0 | β for H nodes. 0 = baseline 1-bit mode; >0 = LIF-compiled mode. |
| `output_membrane_bits` | 0 | β for O nodes. 0 = use `membrane_bits`. |
| `initial_h_nodes` | 0 | H nodes in the template genome. 0 = start from scratch; >0 = richer scaffold. |
| `allow_output_feedback` | False | If True, O nodes can be sources in new connections. |

---

### Mutation Rates

| Parameter | Default | Effect |
|---|---|---|
| `p_mutate_lif_param` | 0.1 | Per-parameter probability of ±1 perturbation for H node LIF params. |
| `p_mutate_conn_weight` | 0.1 | Per-connection probability of ±1 weight perturbation. |
| `weight_min` / `weight_max` | -3 / 3 | Integer weight range. |
| `p_add_connection` | 0.25 | Probability of adding a new connection per child. |
| `p_add_node` | 0.05 | Probability of splitting a connection and inserting a new H node. |
| `p_toggle_connection` | 0.02 | Probability of enabling/disabling a random connection. |
| `p_crossover` | 0.75 | Probability of crossover vs cloning. |
| `lut_bit_rate` | 0.02 | Used only in legacy baseline mode (`membrane_bits=0`): per-bit LUT flip rate. |

---

### Speciation

| Parameter | Default | Effect |
|---|---|---|
| `compatibility_threshold` | 1.5 | Maximum genome distance for same species. Lower = more species, more niching. |
| `c_disjoint` | 1.0 | Weight of structural divergence in genome distance. |
| `c_lut` | 0.4 | Weight of LIF/weight content divergence in genome distance. |

**Tuning:** too many species → raise `compatibility_threshold`. Diversity collapses quickly → lower it.

---

### Selection and Elitism

| Parameter | Default | Effect |
|---|---|---|
| `elite_per_species` | 1 | Top genomes copied unchanged per species per generation. |
| `elite_min_size` | 5 | Minimum species size to qualify for elitism. |
| `tournament_k` | 3 | Tournament size. Higher = stronger selection pressure. |
| `max_stale` | 15 | Generations without improvement before species is killed. |

---

### Training and Fitness

| Parameter | Default | Effect |
|---|---|---|
| `batch_size` | 64 | Mini-batch size for `evaluate_genome`. Affects speed only. |
| `fitness_sample` | 0 | Random training subsample size per generation. 0 = full training set. |
| `val_fitness_alpha` | 0.0 | Blends validation fitness: `(1-α)*train + α*val`. 0 = train CE only. |
| `complexity_coef` | 0.0 | Penalty per active H node: `fitness -= coef * n_active_H`. Prevents bloat and favours minimal FPGA footprint. |
| `eval_test_each_gen` | False | Evaluate best genome on test set every generation (for monitoring only). |

---

## 5. Global Readout Layer (`src/`)

`src/global_readout.py` implements a **co-evolved Boolean readout** for experiments where the network is structured as a spatial lattice (pyramid, per-column, etc.) rather than a free-topology NEAT graph.

### Motivation

In lattice-based experiments, the hidden layer produces a `H×W` binary feature map at each timestep. The readout must aggregate this into `n_classes` class logits. A fixed pooling (e.g. winner-takes-all column assignment) can overfit and introduces a circular dependency. Instead, the readout is **co-evolved** with the lattice: each of the `C` output nodes is itself a `k_ro`-input LUT-register that can freely select which `k_ro` positions in the feature map to read.

### Readout Genome

```
sel : [pop_size, C, k_ro]      — which H×W positions each output reads (long indices)
lut : [pop_size, C, 2^k_ro]    — the Boolean function evaluated at those positions (uint8)
```

### Forward Pass (`eval_readout`)

All `pop_size` readout genomes are evaluated on all `N` samples in a **single vectorised GPU pass**:

1. Reshape the lattice trajectory from `[N, B, T, F]` to `[B, N*T, F]` for a single gather call.
2. Use `readout_sel` to gather the `k_ro` selected feature values per (genome, sample, timestep, class).
3. Compute the `k_ro`-bit LUT address for each combination.
4. Look up the LUT to get the spike for each (genome, sample, timestep, class).
5. Sum over T → spike counts `[N, B, C]`, argmax over C → predicted class `[N, B]`.
6. Compare against labels → accuracy `[B]`, one value per genome.

### Mutation (`mutate_readout`)

- **SEL mutation** (`sel_rate`): each feature-index entry is independently resampled uniformly over `[0, H*W)`.
- **LUT mutation** (`lut_rate`): each LUT bit is independently flipped.

Both are fully vectorised tensor operations.
