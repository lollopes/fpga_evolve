# EIO NEAT — Comprehensive Reference

---

## Table of Contents

1. [Node Architecture — The EIO Building Block](#1-node-architecture--the-eio-building-block)
   - [1a. Baseline Node (LUT-Register)](#1a-baseline-node-lut-register)
   - [1b. Membrane Node (LIF-Equivalent)](#1b-membrane-node-lif-equivalent)
2. [How NEAT Works](#2-how-neat-works)
3. [Evolution Parameters](#3-evolution-parameters)
4. [Global Readout Layer (src/)](#4-global-readout-layer-src)

---

## 1. Node Architecture — The EIO Building Block

Two node variants are supported. Both share the same NEAT topology (E/I/O types, data and inhibitory connections); they differ only in how much internal state each node carries.

---

### 1a. Baseline Node (LUT-Register)

Every non-input node is a **LUT-register unit** — a synchronous Boolean cell with a single bit of internal state:

1. Collects up to `k` binary data inputs (one per LUT input port)
2. Optionally receives inhibitory signals from I-type nodes
3. Evaluates a `k`-input Boolean function stored in a Look-Up Table (LUT)
4. Suppresses its output if any inhibitory source fired in the previous timestep
5. Holds the resulting **1-bit** state until the next timestep

The LUT stores `2^k` bits — every possible `k`-bit input combination maps to one output bit. Any Boolean function of `k` inputs is representable. Evolution discovers which function is useful by flipping individual bits.

```
  k data inputs
  ─────────────
  x[0] ─┐
  x[1] ─┤─► [ LUT  ]──► raw ──► AND NOT inh ──► ┌───┐
  ...    │   [2^k×1]                               │ D ├──► state[t]
  x[k-1]┘                                         └─┬─┘
                                                     │
  inhibitory sources                                 └──► (fanout to network)
  ──────────────────
  s[0] ─┐
  s[1] ─┴─► OR ──► inh
```

**Update rule at each timestep `t`:**

```
data_inputs[p] = x_t[src]          if src is an input node
               = state[t-1][src]   if src is a hidden/output node

raw        = LUT[ addr(data_inputs) ]
inhibition = OR( state[t-1][s]  for all incoming I-sources s )
state[t]   = raw AND NOT inhibition
```

All nodes update synchronously. Recurrence is implicit — any node can read the previous state of any other node via a data connection.

**Internal state:** 1 bit per node.  
**LUT size:** `2^k` entries × 1 bit.

---

### 1b. Membrane Node (LIF-Equivalent)

Enabled by setting `membrane_bits = β > 0` in the config. Each node now carries a **β-bit membrane register** alongside its spike output, making it a fully learned leaky-integrate-and-fire equivalent.

The key insight (from LogicNets): a neuron with X input bits and Y output bits is exactly a `X→Y` truth table. A LIF neuron with `k` spike inputs and a β-bit internal membrane state is therefore a `(k+β) → (1+β)` LUT — learned entirely by evolution, no hand-coded dynamics.

```
  k data inputs   β membrane bits (previous step)
  ─────────────   ──────────────────────────────
  x[0]  ─┐         m[0]  ─┐
  x[1]  ─┤         m[1]  ─┤
  ...    ├──────── ...    ─┤──► addr  (k+β bits)
  x[k-1]─┘         m[β-1]─┘      │
                                  ├──► [ LUT_spike ]──► AND NOT inh ──► ┌───┐
                                  │    [2^(k+β)×1 ]                      │ D ├──► spike[t]
                                  │                                      └───┘
                                  └──► [ LUT_mem   ]──────────────────► ┌───┐
                                       [2^(k+β)×β ]                     │ D ├──► mem[t]
                                                                         └─┬─┘
                                                                           │
                                                                           └──► (fed back as m next step)
```

**Update rule at each timestep `t`:**

```
inputs = [ data_inputs[0..k-1],  mem[t-1][0..β-1] ]   (k+β bits total)
addr   = binary_to_int( inputs )

raw    = LUT_spike[ addr ]          (1 bit)
mem[t] = LUT_mem  [ addr ]          (β-bit integer, range 0 … 2^β−1)

inhibition = OR( state[t-1][s]  for all incoming I-sources s )
state[t]   = raw AND NOT inhibition
```

#### How integration actually works

The β-bit membrane is a small integer that the node writes to itself every timestep. It is private scratch space — not shared with other nodes, not a sum of inputs.

**Concrete example (β=2, k=2 → 4-bit address):**

```
t=0  inputs=[x0=1, x1=0 | m=00]  addr=8   LUT_mem→01(=1)  LUT_spike→0   mem:0→1
t=1  inputs=[x0=1, x1=1 | m=01]  addr=13  LUT_mem→10(=2)  LUT_spike→0   mem:1→2
t=2  inputs=[x0=1, x1=0 | m=10]  addr=10  LUT_mem→11(=3)  LUT_spike→0   mem:2→3
t=3  inputs=[x0=0, x1=0 | m=11]  addr=3   LUT_mem→00(=0)  LUT_spike→1   mem:3→0  ← SPIKE+RESET
```

Here `LUT_mem` happened to learn "accumulate; spike and reset at 3" — a counter. But it is just a lookup table. Evolution can equally discover:

- **Counter / integrate-and-fire**: membrane increments on active input, resets on spike
- **Leaky integrator**: membrane grows with input, decays by 1 each step without input
- **Burst detector**: spikes only when membrane has been elevated for 2+ consecutive steps
- **Pure relay**: `LUT_mem` always outputs 0 → reduces to baseline 1-bit behaviour

The membrane bits are not the inputs summed. They are whatever β-bit value `LUT_mem[addr]` returns for the current combined input+membrane address. The dynamics — integration, leak, threshold, reset — are not hand-coded anywhere; they emerge entirely from what the evolution writes into the LUT tables.

The β-bit membrane are fed back into the same node's LUT address the next timestep. They are **not** shared between nodes — each node integrates its own private state.

**Internal state:** `(1 + β)` bits per node — 1 spike bit + β membrane bits.  
**LUT size:** `2^(k+β)` entries × `(1+β)` bits, split into two tables:
- `LUT_spike`: `2^(k+β) × 1 bit`  
- `LUT_mem`:   `2^(k+β) × β bits`

**Comparison:**

| | Baseline | Membrane (β bits) |
|---|---|---|
| Internal state | 1 bit | 1 + β bits |
| LUT inputs | k | k + β |
| LUT entries | 2^k | 2^(k+β) |
| Temporal memory | 1 timestep (1 bit) | richer — β-bit accumulator |
| Dynamics | hardcoded Boolean | fully learned |
| Genome fields | `lut` | `lut` + `lut_mem` |

**Config key:** set `membrane_bits = 4` (or any β > 0) in `config.json`. The genome and network build the correct LUT sizes automatically.

---

### Node Types

#### Input node
- Passthrough — directly injects the raw spike value from the input feature vector `x_t[f]` into the network at each timestep.
- Has no LUT, no inhibitory inputs, no own state.
- There are `input_size` input nodes (one per feature dimension of the encoded event frame).

#### E — Excitatory node
- A full LUT-register unit with `k` data input ports.
- Fires (outputs 1) when `LUT(data_inputs) = 1` AND no inhibitory source is active.
- Acts as a processing element: can implement feature detectors, coincidence detectors, or any Boolean function.
- Its output can fan out to any other E, I, or O node via data connections, or be an inhibitory source for other nodes (though an E node connected inhibitorily is not standard — only I nodes are used as inhibitory sources in the mutation logic).

#### I — Inhibitory node
- Structurally identical to an E node (has a LUT, k data ports, can itself be inhibited).
- Its outgoing connections are **always inhibitory** (`dst_port = -1`): when it fires, it suppresses all downstream targets via OR-gating.
- Acts as a gating mechanism — it can learn to silence a downstream node under specific input conditions.
- When `add_node_mutation` inserts a new I node, it automatically wires its outgoing connection as inhibitory.

#### O — Output node
- One per class (fixed at genome initialisation, never added/removed by mutation).
- Has a LUT and `k` data input ports like E nodes.
- Spike counts accumulated over all T timesteps form the raw class logits: `logit[c] = Σ_t state_t[O_c]`.
- Prediction = `argmax(logits)`. A sample is **silent** (wrong by definition) if all output counts are zero.

---

### Connection Types

| `dst_port` | Type | Effect |
|---|---|---|
| `≥ 0` | Data connection | Routes a source's output into port `dst_port` of the destination LUT. Only one source per port. |
| `= -1` | Inhibitory connection | Source (must be an I node) suppresses the destination via OR. Multiple inhibitory sources are OR-reduced. |

Each destination LUT has exactly `k` input ports (0 to k-1). If a port has no incoming data connection, it reads as 0.

---

### Genome Representation

A genome contains:
- **`nodes`**: `Dict[node_id → NodeGene]` — type, enabled flag, `lut` (spike table), `lut_mem` (membrane table, `None` in baseline), optional class label (for O nodes)
- **`connections`**: `Dict[innovation_number → ConnectionGene]` — src, dst, dst_port, enabled flag
- **`k`**: the fixed LUT arity shared by all nodes in this genome
- **`membrane_bits`** (`β`): 0 for baseline, >0 for membrane variant — controls LUT size and whether `lut_mem` is present
- **`output_ids`**: ordered list of O node IDs (position = class index)

The **innovation number** on each connection is a global counter assigned the first time a particular `(src, dst, port)` triple appears in the population. It serves as a historical marker for NEAT crossover alignment.

---

## 2. How NEAT Works

NEAT (NeuroEvolution of Augmenting Topologies) is an evolutionary algorithm that simultaneously optimises both the **weights** (here: LUT bits) and the **topology** (node types and connections) of a neural network. It solves three key problems:

1. **Competing conventions** — two networks solving the same problem may use different topologies. Crossover between them can produce broken offspring. NEAT resolves this with historical markings (innovation numbers).
2. **Premature topology convergence** — adding new structure initially hurts fitness. NEAT protects innovation by grouping similar genomes into species and competing within species.
3. **Minimal structure bias** — networks start small and grow only when it helps.

---

### Phase 1 — Initialisation

A **minimal template genome** is created:
- `input_size` input nodes + `n_outputs` output nodes (each O node wired to zero or more random input nodes).
- If `initial_e_nodes > 0`, a small number of E nodes are added and connected between inputs and outputs.
- `initial_connections_per_output` random input→output data connections are drawn per output node.

The full population is created by cloning this template and lightly perturbing each clone's LUT bits via `mutate_luts`.

---

### Phase 2 — Fitness Evaluation

Every genome is converted to an `EIONetwork` and run on the training data (or a random subsample of size `fitness_sample` if set). Fitness is the **negative mean cross-entropy loss**:

```
fitness = -mean_CE(output_counts, y)
```

where `output_counts[b, c]` is the spike count of output node `c` over all T timesteps for sample `b`. Cross-entropy is applied directly to raw spike counts (treating them as logits, with softmax applied internally). Higher fitness = lower CE = better discrimination.

Silent samples (all output counts zero) contribute `log(n_classes)` to CE — a "maximally uncertain" penalty that is weaker than a confident wrong prediction, which correctly reflects that a silent network is bad but not as bad as a confident wrong one.

---

### Phase 3 — Speciation

Genomes are grouped into **species** so that structurally similar individuals compete against each other, protecting innovation from being immediately discarded.

Two genomes `a` and `b` are in the same species if their **genome distance** is below `compatibility_threshold`:

```
d(a, b) = c_disjoint × (|innovations(a) Δ innovations(b)| / max(|a|, |b|))
        + c_lut      × mean_LUT_Hamming(shared_nodes)
        + c_type     × mean_type_mismatch(shared_E/I_nodes)
```

- **Disjoint term**: fraction of connection genes not shared between the two genomes (structural divergence).
- **LUT Hamming term**: average fraction of LUT bits that differ across shared nodes (functional divergence).
- **Type term**: fraction of shared hidden nodes where one is E and the other is I (role divergence).

Each genome is compared to the representative of each existing species in order. If no species is close enough, a new one is formed.

---

### Phase 4 — Reproduction

**Species fitness and quotas:**

Each species receives a quota of offspring proportional to its mean adjusted fitness. Stale species (no improvement in `max_stale` generations) are killed, except for the current global champion species which is always kept alive.

**Elite preservation:**

The top `elite_per_species` genomes from each species with at least `elite_min_size` members are copied unchanged into the next generation.

**Child production (per species, filling its quota):**

1. Select parent `p1` via **tournament selection** (pick `tournament_k` random members, keep the best).
2. With probability `p_crossover` and if the species has more than one member, select a second parent `p2` the same way and perform **crossover**. Otherwise clone `p1`.
3. Apply **mutation** to the child.

**Crossover:**
- For each innovation number in the union of both parents' connections, if it exists in both parents a gene is chosen randomly from either parent (with 75% chance of being disabled if either parent has it disabled). Genes only in the fitter parent are inherited; genes only in the less fit parent are discarded.
- Node genes for all referenced nodes are inherited, with LUT bits mixed bit-by-bit from whichever parent each node came from.

---

### Phase 5 — Mutation

Each child undergoes the following mutations independently:

| Mutation | Trigger |
|---|---|
| **LUT_spike bit flip** | Each bit in each node's `lut` flips independently with probability `lut_bit_rate / 2^(k+β)` |
| **LUT_mem bit flip** | When `membrane_bits > 0`: each individual bit within each `lut_mem` integer entry is flipped with the same per-bit probability |
| **Add connection** | With probability `p_add_connection` |
| **Add node** | With probability `p_add_node` |
| **Toggle connection** | With probability `p_toggle_connection` |
| **Flip node type (E↔I)** | Each hidden node independently with probability `p_mutate_node_type` |

**Add connection:** picks a random source (any enabled node, optionally including O nodes if `allow_output_feedback`) and a random target (any active node), finds a free LUT port on the target, and wires them. If the source is an I node, wires an inhibitory connection instead.

**Add node (node split):** disables a randomly chosen enabled data connection `src → dst (port p)`, inserts a new hidden node `n` (type E with probability `1 - p_new_node_is_inhibitory`, else I), and wires `src → n (port 0)` and then `n → dst (port p)` (or inhibitory if n is I). This is the classic NEAT split operation — it starts with a pass-through (LUT initialised to OR) so fitness is not immediately hurt.

**Toggle connection:** randomly enables or disables a connection. After toggling, `repair_duplicate_slots` resolves any port conflicts.

---

### Phase 6 — Val Tracking and Termination

At each generation, the current best genome (by training fitness) is evaluated on the validation set. The genome with the highest **validation accuracy** across all generations is returned as the final result (ties broken by val fitness). This prevents overfitting to the training sample and ensures the returned network generalises.

---

## 3. Evolution Parameters

### Population and Time

| Parameter | Default | Effect |
|---|---|---|
| `pop_size` | 50 | Total number of genomes per generation. Larger populations explore more of the search space per generation but are proportionally slower to evaluate. |
| `generations` | 50 | Number of evolutionary cycles. Total compute ∝ `pop_size × generations`. |
| `seed` | 42 | RNG seed for full reproducibility. |
| `log_every` | 1 | Print a progress line every N generations. Set to 0 to silence. |

---

### Network Architecture

| Parameter | Default | Effect |
|---|---|---|
| `k` | 3 | LUT arity — number of data inputs per node. LUT size = `2^k` bits. Higher k means each node can implement more complex functions but the LUT mutation space grows exponentially. Typical range: 2–6. |
| `initial_connections_per_output` | 3 | Number of random input→output data connections drawn per O node at initialisation. Determines the starting richness of the input representation seen by each readout node. |
| `initial_e_nodes` | 0 | Number of E nodes added to the template genome before population creation. Starting with hidden nodes gives evolution a richer initial scaffold; starting with 0 forces structure to grow entirely from scratch (minimal topology bias). |
| `allow_output_feedback` | False | If True, O nodes can be sources for new connections (recurrent feedback from outputs to hidden/output nodes). Enables more expressive temporal dynamics at the cost of potential instability. Useful for multi-class tasks. |

---

### Mutation Rates

| Parameter | Default | Effect |
|---|---|---|
| `lut_bit_rate` | 0.02 | Expected fraction of LUT bits flipped per node per generation. Per-bit probability = `lut_bit_rate / 2^k`. This is the primary fine-tuning operator — it changes what Boolean function a node computes without changing topology. Too high → random walk; too low → stagnation. |
| `p_add_connection` | 0.25 | Probability of attempting to add a new data or inhibitory connection per child. Controls the rate at which genomes grow new wiring. High values increase complexity quickly; low values keep networks sparse. |
| `p_add_node` | 0.05 | Probability of splitting a data connection by inserting a new hidden node. This is the only way to increase network depth. Lower than `p_add_connection` because topology growth is harder to undo and takes longer to integrate into the population. |
| `p_toggle_connection` | 0.02 | Probability of enabling or disabling a random connection. Provides a form of structural annealing — a disabled connection retains its innovation number and can be re-enabled, allowing topological features to re-emerge through crossover. |
| `p_new_node_is_inhibitory` | 0.2 | When a new hidden node is added via `add_node_mutation`, this is the probability it becomes an I node rather than an E node. Low values keep the network mostly excitatory; higher values allow more gating structure to evolve. |
| `p_mutate_node_type` | 0.01 | Per-node probability of flipping type E↔I. Acts independently of `add_node_mutation`. Allows existing nodes to change role without restructuring connectivity. |
| `p_crossover` | 0.75 | Probability that a child is produced by crossover between two parents rather than cloning a single parent. High values promote recombination of structural innovations across the species. |

---

### Speciation

| Parameter | Default | Effect |
|---|---|---|
| `compatibility_threshold` | 1.5 | Maximum genome distance for two genomes to be in the same species. Lower values → more, smaller species (finer-grained niching, more protection for novelty). Higher values → fewer, larger species (faster convergence, less diversity). This is the most sensitive speciation hyperparameter. |
| `c_disjoint` | 1.0 | Weight of the structural divergence term in genome distance. Higher values make connection topology the dominant criterion for speciation. |
| `c_lut` | 0.4 | Weight of the LUT Hamming distance term. Higher values cause networks with similar topology but different LUT functions to be placed in different species. |
| `c_type` | 0.2 | Weight of the E/I type mismatch term. Relatively low — type differences alone are rarely enough to drive speciation. |

**How to tune:** if you see too many species (fragmented population) → raise `compatibility_threshold`. If you see 1–2 species dominating quickly and diversity collapses → lower it.

---

### Selection and Elitism

| Parameter | Default | Effect |
|---|---|---|
| `elite_per_species` | 1 | Number of top genomes copied unchanged into the next generation per species. Prevents the best solution in a species from being lost through crossover/mutation. |
| `elite_min_size` | 5 | A species must have at least this many members to qualify for elitism. Prevents tiny species from monopolising the elite quota. |
| `tournament_k` | 3 | Number of competitors drawn for tournament selection. Higher values → stronger selection pressure (best genome more likely to be chosen). Lower values → more uniform sampling (more exploration). |
| `max_stale` | 15 | Number of generations a species can go without improving its best fitness before it is killed. The global champion species is immune. Prevents stagnant species from wasting reproductive quota. |

---

### Training and Fitness

| Parameter | Default | Effect |
|---|---|---|
| `batch_size` | 64 | Mini-batch size for `evaluate_genome`. Affects only memory usage and speed, not the fitness value (the full training set is always evaluated). |
| `fitness_sample` | 0 | If `> 0`, each generation uses a random subsample of this many training samples to compute fitness instead of the full training set. Introduces noise (like stochastic gradient descent) which can help escape local optima, at the cost of a noisier fitness signal. `0` = always use full training set. |

---

## 4. Global Readout Layer (`src/`)

`src/global_readout.py` implements a **co-evolved Boolean readout** for experiments where the network is structured as a spatial lattice (pyramid, per-column, etc.) rather than a free-topology NEAT graph.

### Motivation

In lattice-based experiments, the hidden layer produces a `H×W` binary feature map at each timestep. The readout must aggregate this into `n_classes` class logits. A fixed pooling (e.g. winner-takes-all column assignment) can overfit and introduces a circular dependency. Instead, the readout is **co-evolved** with the lattice: each of the `C` output nodes is itself a `k_ro`-input LUT-register that can freely select which `k_ro` positions in the feature map to read.

### Readout Genome

```
sel : [pop_size, C, k_ro]      — which H×W positions each output reads (long indices)
lut : [pop_size, C, 2^k_ro]    — the Boolean function evaluated at those positions (uint8)
```

- `sel[b, c, j]` = the feature-map position that input slot `j` of class-c's LUT reads in genome `b`.
- `lut[b, c, addr]` = the output bit for LUT address `addr` in class-c of genome `b`.

### Forward Pass (`eval_readout`)

All `pop_size` readout genomes are evaluated on all `N` samples in a **single vectorised GPU pass** — no Python loops over genomes, samples, or timesteps:

1. Reshape the lattice trajectory from `[N, B, T, F]` to `[B, N*T, F]` for a single gather call.
2. Use `readout_sel` to gather the `k_ro` selected feature values per (genome, sample, timestep, class).
3. Compute the `k_ro`-bit LUT address for each combination.
4. Look up the LUT to get the spike for each (genome, sample, timestep, class).
5. Sum over T → spike counts `[N, B, C]`, argmax over C → predicted class `[N, B]`.
6. Compare against labels → accuracy `[B]`, one value per genome.

### Mutation (`mutate_readout`)

Two independent operators applied each generation:
- **SEL mutation** (`sel_rate`): each feature-index entry is independently resampled uniformly over `[0, H*W)`.
- **LUT mutation** (`lut_rate`): each LUT bit is independently flipped.

Both operate as fully vectorised tensor operations — no Python loops over the population.

### Key Advantage

Because the readout is co-evolved and fitness is honest classification accuracy (not a re-scored assignment), there is no circular overfitting. The readout learns to pick the most class-discriminative positions in the feature map, guided directly by CE loss on the actual labels.
