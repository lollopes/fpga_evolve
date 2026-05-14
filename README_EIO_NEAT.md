# Typed E/I/O Boolean NEAT for N-MNIST

NEAT-style evolution where each evolved node is a single **typed Boolean LUT-register unit**.

## Node types

| Type | Role |
|------|------|
| `input` | Exposes input feature `x_i[t]` directly. No LUT, no register. |
| `E` | Excitatory processing node. Has one k-input LUT and one registered state bit. |
| `I` | Inhibitory gating node. Same structure as E; its outgoing connections act as Boolean inhibition on downstream targets. |
| `O` | Output/readout node. One per class. Has a LUT and register. Class logit = spike count over the full sequence. |

## Hardware model

Every non-input node is a **LUT-register cell**:

```
data_inputs[0..k-1]  ← registered outputs of source nodes (or x_t[i] for input sources)
raw        = LUT(data_inputs)
inhibition = OR(state[t-1][s]  for all incoming I sources)
state[t]   = raw & ~inhibition
```

Key properties:
- **One LUT per node** — `k` data inputs, truth table of size 2^k.
- **One register per node** — the output state is a single bit.
- **All recurrent connections are delayed one timestep** — no combinational loops.
- **Inhibition is a Boolean gate** — `state[t] = raw & ~inhibition`, making it friendly for direct hardware implementation.

## Connection semantics

Two types of connections, distinguished by `dst_port`:

| `dst_port` | Meaning |
|------------|---------|
| `0 .. k-1` | **Data connection** — source activity writes into LUT input port `dst_port`. |
| `-1` | **Inhibitory connection** — source activity contributes to the `inhibition` OR gate. Only valid when the source node is type `I`. |

Multiple I sources can simultaneously inhibit the same target (their effects are OR-combined).

## NEAT operators

| Operator | Effect |
|----------|--------|
| `mutate_lut_bits` | Flip individual LUT bits with probability `lut_bit_rate`. |
| `add_connection` | Add a data connection (I source → inhibitory, otherwise → free data port). Self-recurrence is allowed. |
| `add_node` | Split an enabled data connection through a new E or I hidden node. |
| `toggle_connection` | Randomly enable or disable an existing connection. |
| `mutate_node_type` | Flip a hidden node between E and I with probability `p_mutate_node_type`. |

Crossover: NEAT-style innovation-number alignment. Matching connection genes are sampled from either parent. Matching node LUT bits are uniformly mixed. Disjoint/excess genes come from the fitter parent.

## Speciation distance

```
d = c_disjoint × (|disjoint connections| / max_connections)
  + c_lut      × mean(LUT Hamming over matching nodes)
  + c_type     × mean(E≠I mismatch over matching hidden nodes)
```

## Classification readout

```
logit[class c] = spike_count(O_c)   over all T timesteps
prediction     = argmax(logit)
```

## Fitness function

```
fitness = accuracy
        - w_silent     × silent_fraction
        - w_activity   × mean_activity
        - w_complexity × (enabled_data_conns + enabled_inh_conns + enabled_E + enabled_I)
```

## Files

```
eio_neat/
  __init__.py
  genome.py    — NodeGene, ConnectionGene, InnovationRegistry, Genome, crossover, genome_distance
  network.py   — EIONetwork (runtime interpreter), evaluate_genome
  neat.py      — NEATConfig, speciation, reproduction, evolve()

train_nmnist_eio_neat.py
dataset.py
```

## Quick start

```bash
python train_nmnist_eio_neat.py \
  --data-root /path/to/NMNIST \
  --task 7_vs_rest \
  --grid-size 8 \
  --n-time-bins 10 \
  --n-train-per-class 40 \
  --n-val-per-class 20 \
  --pop-size 50 \
  --generations 20 \
  --k 3
```

With inhibitory nodes and hidden E nodes from the start:

```bash
python train_nmnist_eio_neat.py \
  --data-root /path/to/NMNIST \
  --task 7_vs_rest \
  --grid-size 8 \
  --n-time-bins 10 \
  --n-train-per-class 40 \
  --n-val-per-class 20 \
  --pop-size 50 \
  --generations 50 \
  --k 3 \
  --initial-e-nodes 2 \
  --p-new-node-is-inhibitory 0.2 \
  --p-mutate-node-type 0.01
```

---

## NEAT terminology glossary

**Genome** — the encoded description of one network: a set of node genes (type, LUT) and connection genes (src, dst, port, enabled). Evolution operates on genomes, not on network activations.

**Population** — the full set of genomes alive at one time. Size controlled by `pop_size`.

**Generation** — one full cycle: evaluate all genomes → group into species → reproduce the next generation.

**Species** — a group of genomes that are structurally similar, measured by compatibility distance. Speciation protects structural innovations: a newly mutated topology starts with low fitness and would immediately be eliminated by well-optimised incumbents if they competed directly. Grouping by similarity means a new topology only competes against others with the same structure.

**Representative** — one genome stored per species from the previous generation. Incoming genomes are assigned to the first species whose representative is within `compatibility_threshold`. Keeps species membership stable across generations.

**Stale species** — a species that has not improved its best fitness for `max_stale` consecutive generations. Stale species are culled; their offspring quota drops to zero. The *champion* species (the one holding the globally best genome) is always kept regardless.

**Fitness sharing** — each species receives offspring proportional to its mean fitness, not the raw sum. Prevents one large dominant species from consuming the whole population by making every species compete on average quality rather than sheer numbers.

**Tournament selection** — to pick a parent, `tournament_k` genomes are drawn at random from the species and the fittest one wins. Larger `k` = more selection pressure. `tournament_k = 3` is a mild setting.

**Elite / Elitism** — the top `elite_per_species` genomes of a species are copied unchanged into the next generation, bypassing mutation. Preserves the best solution found so far. Only applied to species with at least `elite_min_size` members, to avoid giving free passes to tiny noise clusters.

**Crossover** — sexual reproduction: two parents are combined into a child using innovation-number alignment. Matching connection genes (same innovation ID in both parents) are randomly inherited from either parent; non-matching ones (disjoint/excess) come from the fitter parent. Probability `p_crossover` controls how often crossover is used vs. cloning + mutation.

**Innovation number** — a global counter assigned to every structural event (add-connection, add-node). Two genomes that independently evolved the same connection receive the same number, which is how crossover aligns genes across genomes. This is the core mechanism that makes NEAT work across varied topologies.

**Add-node mutation** — splits an existing enabled connection `A → B` by inserting a new hidden node `N`: `A → N → B`. The original edge is disabled. This is the primary way the network gains depth. The new node is type I with probability `p_new_node_is_inhibitory`, else E.

**Toggle-connection mutation** — enables or disables an existing connection gene. Disabling removes the edge from the active network without deleting it from the genome, so it can be re-enabled later or inherited by offspring.

**LUT bit-rate** — each LUT truth table entry is independently flipped with this probability each generation. This is the analogue of weight perturbation in standard NEAT. Note: the expected number of flips scales as `bit_rate × 2^k`, so mutation is much more disruptive at large `k` (e.g. k=10 → ~20 flips per node per generation).

**Silent fraction** — fraction of input samples for which every output node fires zero spikes across the whole sequence. A silent network makes no prediction and is penalised by `w_silent`. Silent samples also do not count as correct, regardless of class label.

**Mean activity** — average spike rate across all nodes over the sequence. Penalised by `w_activity` to avoid chattering networks. Set to zero for `7_vs_rest` because a good 7-detector should be selective; penalising activity would suppress the correct behaviour.

**Complexity** — count of enabled connections plus enabled hidden nodes. Penalised by `w_complexity` as a lightweight Occam's razor term that favours simpler solutions when accuracy is tied, and indirectly speeds up evaluation.

---

## Experiment hyperparameters

### Dataset splits

| Task | n_train_per_class | n_val_per_class | n_test_per_class | n_time_bins | Input size |
|---|---|---|---|---|---|
| `0_vs_1` | 1000 | 200 | all | 10 | 34×34 = 1156 |
| `7_vs_rest` | 1000 | 200 | all | 10 | 34×34 = 1156 |
| `10class` | 1000 | 100 | all | 10 | 34×34 = 1156 |

`7_vs_rest` mixes all nine non-7 digits into a single "rest" class (~111 examples per digit), producing a 9:1 class imbalance. `first_saccade_only=True` uses only the first of three N-MNIST saccades.

### Population and selection

| Parameter | 0_vs_1 | 7_vs_rest | 10class | Meaning |
|---|---|---|---|---|
| `pop_size` | 100 | 100 | 200 | Number of genomes per generation |
| `generations` | 200 | 200 | 200 | Total generations |
| `elite_per_species` | 1 | 1 | 1 | Top genomes copied unchanged per species |
| `elite_min_size` | 5 | 5 | 5 | Min species size to apply elitism |
| `tournament_k` | 3 | 3 | 3 | Competitors in tournament selection |
| `max_stale` | 15 | 15 | 20 | Generations without improvement before culling |
| `p_crossover` | 0.75 | 0.75 | 0.75 | Probability of sexual reproduction |

### Network architecture

| Parameter | 0_vs_1 | 7_vs_rest | 10class | Meaning |
|---|---|---|---|---|
| `k` | 5 | 10 | 3 | LUT fan-in; truth table size = 2^k |
| `initial_connections_per_output` | 4 | 10 | 5 | Direct input→O connections at init |
| `initial_e_nodes` | 2 | 3 | 6 | Hidden E nodes wired at init |
| `allow_output_feedback` | False | False | True | O nodes can feed back into the network |

`k=10` for `7_vs_rest` gives each node full expressiveness over 10 inputs (1024-entry LUT) to detect complex spatial patterns. `k=3` for `10class` forces simple per-node functions and relies on NEAT composing them. Output feedback is enabled for `10class` to allow the 10 output nodes to interact during the temporal sequence.

### Mutation rates

| Parameter | 0_vs_1 | 7_vs_rest | 10class | Meaning |
|---|---|---|---|---|
| `lut_bit_rate` | 0.02 | 0.02 | 0.02 | Per-bit flip probability in each LUT |
| `p_add_connection` | 0.30 | 0.30 | 0.30 | Probability of adding a connection |
| `p_add_node` | 0.05 | 0.05 | 0.08 | Probability of splitting a connection with a new node |
| `p_toggle_connection` | 0.02 | 0.02 | 0.02 | Probability of enabling/disabling a random connection |
| `p_new_node_is_inhibitory` | 0.2 | 0.2 | 0.2 | Probability a new node is type I |
| `p_mutate_node_type` | 0.01 | 0.01 | 0.01 | Per-node probability of flipping E↔I |

`p_add_node=0.08` for `10class` reflects that the 10-way task needs deeper networks; more aggressive structural growth is needed.

### Speciation

| Parameter | 0_vs_1 | 7_vs_rest | 10class | Meaning |
|---|---|---|---|---|
| `compatibility_threshold` | 1.5 | 4.0 | 2.0 | Max distance to share a species |
| `c_disjoint` | 1.0 | 1.0 | 1.0 | Weight for non-matching connection fraction |
| `c_lut` | 0.4 | 0.4 | 0.4 | Weight for mean LUT Hamming distance |
| `c_type` | 0.2 | 0.2 | 0.2 | Weight for E/I type mismatch |

`7_vs_rest` uses a much looser threshold (4.0) because large k inflates the LUT Hamming term: two random 1024-entry tables differ on ~50% of entries by chance, so distances are structurally larger and the threshold must scale accordingly.

### Fitness penalties

| Parameter | 0_vs_1 | 7_vs_rest | 10class | Meaning |
|---|---|---|---|---|
| `w_silent` | 0.20 | 0.20 | 0.20 | Penalty per unit silent fraction |
| `w_activity` | 0.02 | 0.0 | 0.02 | Penalty per unit mean spike activity |
| `w_complexity` | 0.0002 | 0.0001 | 0.0001 | Penalty per enabled connection/node |
| `fitness_sample` | 0 (all) | 512 | 0 (all) | Training samples used per generation |
| `batch_size` | 128 | 128 | 128 | Batch size during genome evaluation |

`w_activity=0` for `7_vs_rest`: a selective 7-detector should fire sparsely, so penalising activity would suppress correct behaviour. `fitness_sample=512` subsamples training each generation to keep evaluation fast given the large k=10 LUT tables.

---

## Log format

```
gen    0 | train=55.0%  val=50.0% fit=0.472  mean=0.321 | species=3  E=0  I=0  conn=6  silent=0.12 | 2.4s
```

Columns: generation | train acc, val acc, best fitness, mean fitness | species count, hidden E nodes, hidden I nodes, total connections, silent fraction | wall time.
