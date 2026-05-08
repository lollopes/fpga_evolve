"""
evolution.py — Improved (μ+λ) elitist evolution for 3D Boolean lattice networks.

Improvements over the original truncation-selection approach:

  1. (μ+λ) elitism
       The top `elite_size` genomes are carried forward unchanged each generation.
       The best solution found so far is never discarded.

  2. Full-pool ranking
       Fitness is evaluated on the full training pool every generation. This removes
       per-generation batch noise so parent selection is stable from one generation
       to the next.

  3. Activity penalties
       Genomes that produce all-silent (zero spikes) or saturated (mean rate > 0.9)
       top-layer activity are penalised.  fitness = acc - w_silent*silent_frac
       - w_sat*sat_frac.  Stops the search from wasting time on degenerate attractors.

  4. Uniform crossover
       Each child is produced by crossing two randomly chosen elite parents (each
       gene inherited independently with 50% probability) before mutation.  Enables
       building-block combination across lineages.

  5. Separate mutation rates for SEL and LUT
       MUX-selector (SEL) mutations are more disruptive than LUT bit-flips because
       they change *which signal* a node reads.  `sel_mutation_rate` defaults to half
       of `lut_mutation_rate` to reflect this difference.

Generation lifecycle
--------------------
  1. Evaluate all pop_size genomes on the fixed ranking set → fitness = acc - penalties.
  2. Sort by fitness descending; keep top `elite_size` unchanged.
  3. Sample `pop_size - elite_size` child pairs from the elites.
  4. [optional] Uniform crossover between paired parents.
  5. Mutate children (lut_rate for LUT bits, sel_rate for SEL indices).
  6. Next population = elites ++ mutated_children.

Genome layout (unchanged):
  genome[..., :k]   = MUX selectors  in [0, POOL_SIZE)
  genome[..., k:]   = LUT bits        in {0, 1}
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from src.lattice import Lattice3DNetwork


POOL_SIZE = Lattice3DNetwork.POOL_SIZE
N_NODES   = Lattice3DNetwork.N_NODES   # 3 nodes per module: E, I, O


# ---------------------------------------------------------------------------
# Genome helpers
# ---------------------------------------------------------------------------

def random_genome_pack(
    pop_size: int, Z: int, k: int,
    rng: torch.Generator, device: torch.device,
) -> torch.Tensor:
    """Sample a uniformly random initial population.

    Returns
    -------
    [pop_size, Z, 3, k + 2**k] long
        sel part: uniform in [0, POOL_SIZE)
        lut part: uniform in {0, 1}
    """
    lut_size = 2 ** k
    sel = torch.randint(
        0, POOL_SIZE, (pop_size, Z, N_NODES, k),
        generator=rng, device=device, dtype=torch.uint8,
    )
    lut = torch.randint(
        0, 2, (pop_size, Z, N_NODES, lut_size),
        generator=rng, device=device, dtype=torch.uint8,
    )
    return torch.cat([sel, lut], dim=-1)


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def mutate(
    genomes: torch.Tensor,
    k: int,
    lut_rate: float,
    sel_rate: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """Per-element mutation with separate rates for LUT bits and SEL indices.

    Parameters
    ----------
    genomes  : [B, Z, 3, k + 2**k] long
    k        : node arity
    lut_rate : per-bit flip probability for LUT entries
    sel_rate : per-entry resample probability for MUX selectors (SEL)
               Lower than lut_rate by default — SEL changes are more disruptive
               because they rewire which signal a node reads.
    rng      : torch.Generator on the genomes' device

    Returns
    -------
    mutated : [B, Z, 3, k + 2**k] long — new tensor; original not modified
    """
    if not (0.0 <= lut_rate <= 1.0):
        raise ValueError(f"lut_rate must be in [0, 1], got {lut_rate}")
    if not (0.0 <= sel_rate <= 1.0):
        raise ValueError(f"sel_rate must be in [0, 1], got {sel_rate}")

    sel = genomes[..., :k]
    lut = genomes[..., k:]
    dev = genomes.device

    # SEL: resample uniformly in [0, POOL_SIZE)
    sel_flip = torch.rand(sel.shape, generator=rng, device=dev) < sel_rate
    sel_new  = torch.randint(
        0, POOL_SIZE, sel.shape,
        generator=rng, device=dev, dtype=sel.dtype,
    )
    sel_mut = torch.where(sel_flip, sel_new, sel)

    # LUT: bit flip
    lut_flip = torch.rand(lut.shape, generator=rng, device=dev) < lut_rate
    lut_mut  = torch.where(lut_flip, 1 - lut, lut)

    return torch.cat([sel_mut, lut_mut], dim=-1)


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def crossover(
    parents_a: torch.Tensor,   # [N, Z, 3, k + 2**k] long
    parents_b: torch.Tensor,   # [N, Z, 3, k + 2**k] long
    rng: torch.Generator,
) -> torch.Tensor:
    """Uniform crossover between N paired parent genomes.

    Each gene (element) is independently inherited from parent_a or parent_b
    with equal 50% probability.  Both SEL and LUT genes participate.

    Returns
    -------
    children : [N, Z, 3, k + 2**k] long
    """
    mask = torch.rand(parents_a.shape, generator=rng, device=parents_a.device) < 0.5
    return torch.where(mask, parents_a, parents_b)


# ---------------------------------------------------------------------------
# Network build / evaluate
# ---------------------------------------------------------------------------

def build_lattice(
    genome_pack: torch.Tensor,
    Z: int, H: int, W: int, k: int,
    distal_seed: int,
    identity_seed: int,
    use_identity: bool,
    use_positional_cues: bool,
    use_distal: bool,
    device: torch.device,
) -> Lattice3DNetwork:
    """Construct a Lattice3DNetwork from a packed genome population."""
    layer_genomes = [genome_pack[:, z].contiguous() for z in range(Z)]
    net = Lattice3DNetwork(
        Z, H, W, layer_genomes,
        distal_idx=None,
        k=k,
        identity_seed=identity_seed,
        use_identity=use_identity,
        use_positional_cues=use_positional_cues,
        distal_seed=distal_seed,
        use_distal=use_distal,
    )
    return net.to(device)


def evaluate(
    net: Lattice3DNetwork,
    X_batch: torch.Tensor,         # [N, T, H, W] long, binary
    y_batch: torch.Tensor,         # [N] long
    output_sites: torch.Tensor,    # [n_classes, group_size] long
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score every circuit in the population on every sample of the batch.

    Returns
    -------
    accs        : [B] float — argmax accuracy across the batch
    silent_frac : [B] float — fraction of samples where top-layer fired zero spikes
    sat_frac    : [B] float — fraction of samples where top-layer mean rate > 0.9
    """
    B = net.B
    N, T, H, W = X_batch.shape
    dev = X_batch.device

    correct      = torch.zeros(B, device=dev)
    silent_count = torch.zeros(B, device=dev)
    sat_count    = torch.zeros(B, device=dev)

    for n in range(N):
        seq    = X_batch[n:n + 1].expand(B, T, H, W).contiguous()
        traj   = net.run(seq)                                # [B, T, H*W]

        # Classification readout
        feats  = traj.float().sum(dim=1)                     # [B, H*W]
        logits = feats[:, output_sites].sum(dim=-1)          # [B, n_classes]
        pred   = logits.argmax(dim=1)
        correct += (pred == y_batch[n]).float()

        # Activity statistics on the top layer
        total_spikes = traj.float().sum(dim=(1, 2))          # [B]
        mean_rate    = traj.float().mean(dim=(1, 2))         # [B]
        silent_count += (total_spikes == 0).float()
        sat_count    += (mean_rate > 0.9).float()

    return correct / N, silent_count / N, sat_count / N


# ---------------------------------------------------------------------------
# WTA readout
# ---------------------------------------------------------------------------

def _column_features_from_traj(
    traj: torch.Tensor,      # [B, T, H*W] bool/float
    H: int,
    W: int,
    readout_decay: float,
) -> torch.Tensor:
    """Collapse a top-layer trajectory to per-column features.

    If readout_decay == 0, this reduces to a raw time-sum. Otherwise it uses
    the final state of a leaky integrator: s_t = decay * s_{t-1} + spikes_t.
    """
    spikes = traj.float().reshape(traj.shape[0], traj.shape[1], H, W)
    if readout_decay <= 0.0:
        return spikes.sum(dim=1).sum(dim=1)

    state = torch.zeros(
        (spikes.shape[0], H, W), device=spikes.device, dtype=spikes.dtype
    )
    for t in range(spikes.shape[1]):
        state = readout_decay * state + spikes[:, t]
    return state.sum(dim=1)


def classify_wta(
    net: Lattice3DNetwork,
    X_batch: torch.Tensor,      # [N, T, H, W] long, binary
    y_batch: torch.Tensor,      # [N] long
    assignment: torch.Tensor,   # [B, W] long — column-to-class assignment
    n_classes: int,
    readout_decay: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify a batch using a pre-computed per-genome column assignment.

    Each top-layer column is assigned to one class. For each sample the
    logit for class c is the sum of leaky-integrated column activations over
    all columns assigned to c; prediction is argmax.

    Parameters
    ----------
    assignment : [B, W] long — output of evaluate_wta, one class label per column

    Returns
    -------
    accs        : [B] float
    silent_frac : [B] float
    sat_frac    : [B] float
    """
    B = net.B
    N, T, H, W = X_batch.shape
    dev = X_batch.device

    correct      = torch.zeros(B, device=dev)
    silent_count = torch.zeros(B, device=dev)
    sat_count    = torch.zeros(B, device=dev)

    for n in range(N):
        seq      = X_batch[n:n + 1].expand(B, T, H, W).contiguous()
        traj      = net.run(seq)                                 # [B, T, H*W]
        col_feats = _column_features_from_traj(traj, H, W, readout_decay)  # [B, W]

        logits = torch.zeros(B, n_classes, device=dev)
        for c in range(n_classes):
            col_mask = (assignment == c).float()                 # [B, W]
            logits[:, c] = (col_feats * col_mask).sum(dim=1)

        pred = logits.argmax(dim=1)
        correct += (pred == y_batch[n]).float()

        total_spikes = traj.float().sum(dim=(1, 2))
        mean_rate    = traj.float().mean(dim=(1, 2))
        silent_count += (total_spikes == 0).float()
        sat_count    += (mean_rate > 0.9).float()

    return correct / N, silent_count / N, sat_count / N


def evaluate_wta(
    net: Lattice3DNetwork,
    X_batch: torch.Tensor,   # [N, T, H, W] long, binary
    y_batch: torch.Tensor,   # [N] long
    n_classes: int,
    readout_decay: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score every circuit using a data-driven WTA column assignment.

    Two passes over the batch:
      Pass 1 — for each genome, compute mean leaky-integrated column
               activity per class and assign each column to its most-responsive class.
      Pass 2 — classify every sample using that assignment (via classify_wta).

    Unlike the fixed-stripe readout, the network only needs to fire
    *differently* for each class somewhere on the top layer; the assignment
    step discovers where that difference is and exploits it.

    Parameters
    ----------
    n_classes : number of output classes

    Returns
    -------
    accs        : [B] float
    silent_frac : [B] float
    sat_frac    : [B] float
    assignment  : [B, W] long — column-to-class assignment (reusable for val)
    """
    B = net.B
    N, T, H, W = X_batch.shape
    dev = X_batch.device

    # --- Pass 1: accumulate per-class column activity → assignment ----------
    class_col_sums = torch.zeros(B, n_classes, W, device=dev)
    class_counts   = torch.zeros(n_classes, device=dev)

    for n in range(N):
        seq      = X_batch[n:n + 1].expand(B, T, H, W).contiguous()
        traj      = net.run(seq)                                 # [B, T, H*W]
        col_feats = _column_features_from_traj(traj, H, W, readout_decay)  # [B, W]

        lbl = int(y_batch[n])
        class_col_sums[:, lbl, :] += col_feats
        class_counts[lbl] += 1

    # Normalise by per-class sample count, then argmax over classes
    class_col_means = class_col_sums / class_counts.clamp(min=1)[None, :, None]
    assignment = class_col_means.argmax(dim=1)                   # [B, W]

    # --- Pass 2: classify using the computed assignment --------------------
    accs, silent_frac, sat_frac = classify_wta(
        net, X_batch, y_batch, assignment, n_classes, readout_decay=readout_decay
    )

    return accs, silent_frac, sat_frac, assignment


def assignment_diagnostics(
    assignment: torch.Tensor,
    n_classes: int,
) -> tuple[list[int], int, int, float]:
    """Summarise one genome's WTA column assignment."""
    counts = torch.bincount(assignment, minlength=n_classes)
    total = int(counts.sum())
    dominant_class = int(counts.argmax()) if total > 0 else -1
    dominant_frac = float(counts.max() / counts.sum()) if total > 0 else 0.0
    used_classes = int((counts > 0).sum())
    return counts.tolist(), used_classes, dominant_class, dominant_frac


def _evaluate_wta_parallel(
    genomes: torch.Tensor,       # [B, Z, 3, k + 2**k]
    X_batch: torch.Tensor,       # [N, T, H, W] long binary
    y_batch: torch.Tensor,       # [N] long
    n_classes: int,
    Z: int, H: int, W: int, k: int,
    distal_seed: int,
    identity_seed: int,
    use_identity: bool,
    use_positional_cues: bool,
    use_distal: bool,
    readout_decay: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score all B genomes on all N samples in a single forward pass.

    Instead of looping over N samples, tiles the B genomes N times and stacks
    all N inputs into one big batch of size N×B.  One net.run() call replaces
    the N-iteration Python loop, fully saturating the GPU for large N×B.

    Memory scales as N×B×Z×H×W — ensure sufficient VRAM before enabling.
    Use `parallel_samples: false` in config if you hit OOM.

    Returns
    -------
    accs        : [B] float
    silent_frac : [B] float
    sat_frac    : [B] float
    assignment  : [B, W] long
    """
    B = genomes.shape[0]
    N, T, H_, W_ = X_batch.shape
    NB = N * B
    G  = genomes.shape[-1]   # k + 2**k

    # ── Tile genomes and expand input ────────────────────────────────────────
    # Layout: element n*B + b  →  genome b on sample n
    tiled_genomes = (
        genomes.unsqueeze(0)            # [1, B, Z, 3, G]
        .expand(N, -1, -1, -1, -1)     # [N, B, Z, 3, G]
        .reshape(NB, Z, N_NODES, G)    # [NB, Z, 3, G]
        .contiguous()
    )
    expanded_X = (
        X_batch.unsqueeze(1)            # [N, 1, T, H, W]
        .expand(-1, B, -1, -1, -1)     # [N, B, T, H, W]
        .reshape(NB, T, H_, W_)        # [NB, T, H, W]
        .contiguous()
    )

    # ── Single forward pass ──────────────────────────────────────────────────
    big_net = build_lattice(
        tiled_genomes, Z, H_, W_, k,
        distal_seed=distal_seed,
        identity_seed=identity_seed,
        use_identity=use_identity,
        use_positional_cues=use_positional_cues,
        use_distal=use_distal,
        device=device,
    )
    traj = big_net.run(expanded_X)                              # [NB, T, H*W]
    traj_nb = traj.reshape(N, B, T, H_ * W_)                   # [N, B, T, H*W]

    # ── Column features: leaky integration over T, then sum over H rows ───
    col_feats = _column_features_from_traj(
        traj_nb.reshape(N * B, T, H_ * W_), H_, W_, readout_decay
    ).reshape(N, B, W_)

    # ── WTA assignment ───────────────────────────────────────────────────────
    y_oh = F.one_hot(y_batch, n_classes).float()               # [N, n_classes]
    class_col_sums  = torch.einsum("nc,nbw->bcw", y_oh, col_feats)  # [B, n_classes, W]
    class_counts    = y_oh.sum(0).clamp(min=1)                      # [n_classes]
    class_col_means = class_col_sums / class_counts[None, :, None]
    assignment      = class_col_means.argmax(dim=1)                 # [B, W]

    # ── Classification using the WTA assignment ──────────────────────────────
    asgn_oh = F.one_hot(assignment, n_classes).float()          # [B, W, n_classes]
    logits  = torch.einsum("nbw,bwc->nbc", col_feats, asgn_oh) # [N, B, n_classes]
    pred    = logits.argmax(dim=2)                               # [N, B]
    accs    = (pred == y_batch[:, None]).float().mean(dim=0)    # [B]

    # ── Activity statistics ──────────────────────────────────────────────────
    total_spikes = traj_nb.float().sum(dim=(2, 3))              # [N, B]
    mean_rate    = traj_nb.float().mean(dim=(2, 3))             # [N, B]
    silent_frac  = (total_spikes == 0).float().mean(dim=0)      # [B]
    sat_frac     = (mean_rate > 0.9).float().mean(dim=0)        # [B]

    return accs, silent_frac, sat_frac, assignment


def class_readout_sites(n_classes: int, H: int, W: int) -> torch.Tensor:
    """Divide the H×W top layer into n_classes equal vertical stripes.

    Returns
    -------
    sites : [n_classes, H * (W // n_classes)] long
    """
    if W % n_classes != 0:
        raise ValueError(f"W={W} is not divisible by n_classes={n_classes}")
    group_w = W // n_classes
    ys = torch.arange(H)
    sites = []
    for c in range(n_classes):
        xs = torch.arange(c * group_w, (c + 1) * group_w)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        sites.append((yy * W + xx).reshape(-1))
    return torch.stack(sites, dim=0)


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def _compute_fitness(
    accs: torch.Tensor,
    silent_frac: torch.Tensor,
    sat_frac: torch.Tensor,
    w_silent: float,
    w_sat: float,
) -> torch.Tensor:
    """fitness = accuracy - w_silent * silent_frac - w_sat * sat_frac"""
    return accs - w_silent * silent_frac - w_sat * sat_frac


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def evolve(
    pop_size: int,
    n_generations: int,
    Z: int, H: int, W: int, k: int,
    X_pool: torch.Tensor,          # [P, T, H, W] long binary — training pool
    y_pool: torch.Tensor,          # [P] long
    n_classes: int,                # number of output classes (used by WTA readout)
    mutation_rate: float = 0.02,   # backward-compat default; overridden by lut/sel rates
    lut_mutation_rate: float = None,
    sel_mutation_rate: float = None,
    elite_size: int = 20,
    penalty_weights: tuple = (0.3, 0.3),
    use_crossover: bool = True,
    X_val: torch.Tensor = None,    # optional held-out set — logged per generation but NOT used for ranking
    y_val: torch.Tensor = None,
    stagnation_patience: int = 10,
    stagnation_inject_frac: float = 0.3,
    parallel_samples: bool = False,
    readout_decay: float = 0.0,
    seed: int = 42,
    use_identity: bool = True,
    use_positional_cues: bool = False,
    use_distal: bool = True,
    distal_seed: int = 0,
    identity_seed: int = 0,
    device: torch.device = None,
    live_path: Path = None,
    live_meta: dict = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Run (μ+λ) elitist evolution for `n_generations` generations.

    Each generation ranks genomes on the full X_pool. This keeps selection
    pressure consistent across generations instead of introducing mini-batch noise.
    Exploration comes from mutation, crossover, elitism, and stagnation injection.

    If X_val / y_val are provided they are evaluated each generation for
    observability (logged in history as val_acc) but play no role in selection.

    Parameters
    ----------
    pop_size          : even int >= 2
    n_generations     : int >= 1
    Z, H, W, k        : lattice dimensions and node arity
    X_pool            : [P, T, H, W] long binary — training pool
    y_pool            : [P] long
    n_classes         : number of output classes; drives the WTA column assignment
    mutation_rate     : backward-compat; sets both lut and sel rates unless overridden
    lut_mutation_rate : per-bit LUT flip probability (default: mutation_rate)
    sel_mutation_rate : per-entry SEL resample probability (default: mutation_rate * 0.5)
    elite_size             : top genomes kept unchanged each generation (~20% of pop_size)
    penalty_weights        : (w_silent, w_saturated)
    use_crossover          : if True, cross elite pairs before mutation
    X_val / y_val          : optional held-out set; logged per generation, not used for ranking
    stagnation_patience    : generations without improvement before injecting diversity
    stagnation_inject_frac : fraction of population replaced with fresh random genomes
                             on stagnation; bottom-ranked genomes are replaced first
    parallel_samples       : if True, run the full training pool × pop_size genomes in one
                             forward pass (faster on GPU, but uses P×B times more VRAM)
    seed                   : RNG seed
    use_identity, use_positional_cues, use_distal : ablation flags
    distal_seed, identity_seed : seeds for fixed structural bits
    device            : torch device

    Returns
    -------
    final_genomes    : [pop_size, Z, 3, k + 2**k] sorted best→worst on full pool
    final_accs       : [pop_size] float
    history          : list[dict] — per-generation metrics
    best_assignment  : [1, W] long — WTA column assignment of the best genome,
                       computed on the full training pool; pass to classify_wta
                       for held-out evaluation
    """
    if pop_size < 2 or pop_size % 2 != 0:
        raise ValueError(f"pop_size must be an even integer >= 2, got {pop_size}")
    if n_generations < 1:
        raise ValueError(f"n_generations must be >= 1, got {n_generations}")
    if not (1 <= elite_size < pop_size):
        raise ValueError(f"elite_size must be in [1, pop_size), got {elite_size}")
    if X_pool.dim() != 4:
        raise ValueError(f"X_pool must be [P, T, H, W]; got shape {tuple(X_pool.shape)}")
    P, T, Hin, Win = X_pool.shape
    if Hin != H or Win != W:
        raise ValueError(f"X_pool spatial dims ({Hin},{Win}) != lattice ({H},{W})")
    if y_pool.shape[0] != P:
        raise ValueError(f"X_pool and y_pool must agree on axis 0; got {P} vs {y_pool.shape[0]}")
    if (X_val is None) != (y_val is None):
        raise ValueError("X_val and y_val must both be provided or both be None")
    if not (0.0 <= readout_decay <= 1.0):
        raise ValueError(f"readout_decay must be in [0, 1], got {readout_decay}")

    # Resolve mutation rates
    lut_rate = lut_mutation_rate if lut_mutation_rate is not None else mutation_rate
    sel_rate = sel_mutation_rate if sel_mutation_rate is not None else mutation_rate * 0.5
    w_silent, w_sat = penalty_weights

    dev = device if device is not None else X_pool.device
    rng = torch.Generator(device=dev)
    rng.manual_seed(seed)

    genomes    = random_genome_pack(pop_size, Z, k, rng, dev)
    n_children = pop_size - elite_size
    n_inject     = max(1, int(pop_size * stagnation_inject_frac))

    val_available = X_val is not None
    if val_available:
        X_val = X_val.to(dev)
        y_val = y_val.to(dev)

    print(f"Ranking: full train pool n={P} | elite={elite_size} | "
          f"crossover={'on' if use_crossover else 'off'} | "
          f"lut_rate={lut_rate} sel_rate={sel_rate} | "
          f"penalty w_silent={w_silent} w_sat={w_sat} | "
          f"readout_decay={readout_decay:.2f} | "
          f"stagnation patience={stagnation_patience} inject={n_inject} | "
          f"parallel_samples={'on' if parallel_samples else 'off'} | "
          f"val logging={'on' if val_available else 'off'}")

    history: list[dict] = []
    stagnation_count  = 0
    best_fitness_ever = -float("inf")

    for gen in range(n_generations):
        # Rank on the full training pool every generation
        X_batch = X_pool
        y_batch = y_pool

        # Rank on training pool using WTA readout
        if parallel_samples:
            # All N samples × B genomes in one forward pass
            accs, silent_frac, sat_frac, assignments = _evaluate_wta_parallel(
                genomes, X_batch, y_batch, n_classes,
                Z, H, W, k,
                distal_seed=distal_seed,
                identity_seed=identity_seed,
                use_identity=use_identity,
                use_positional_cues=use_positional_cues,
                use_distal=use_distal,
                readout_decay=readout_decay,
                device=dev,
            )
        else:
            net = build_lattice(
                genomes, Z, H, W, k,
                distal_seed=distal_seed,
                identity_seed=identity_seed,
                use_identity=use_identity,
                use_positional_cues=use_positional_cues,
                use_distal=use_distal,
                device=dev,
            )
            accs, silent_frac, sat_frac, assignments = evaluate_wta(
                net, X_batch, y_batch, n_classes, readout_decay=readout_decay
            )
        fitness = _compute_fitness(accs, silent_frac, sat_frac, w_silent, w_sat)

        # Sort best → worst
        order       = torch.argsort(-fitness, stable=True)
        genomes     = genomes[order]
        fitness     = fitness[order]
        accs        = accs[order]
        silent_frac = silent_frac[order]
        sat_frac    = sat_frac[order]
        assignments = assignments[order]

        # ---- Stagnation detection -------------------------------------------
        injected = False
        if float(fitness[0]) > best_fitness_ever + 1e-6:
            best_fitness_ever = float(fitness[0])
            stagnation_count  = 0
        else:
            stagnation_count += 1

        if stagnation_count >= stagnation_patience:
            # Replace bottom n_inject genomes with fresh random individuals
            fresh   = random_genome_pack(n_inject, Z, k, rng, dev)
            genomes = torch.cat([genomes[:pop_size - n_inject], fresh], dim=0)
            stagnation_count = 0
            injected = True

        # Optional: evaluate best genome on held-out val set for observability.
        # Re-use the assignment computed on this generation's training batch.
        val_acc = None
        if val_available:
            val_net = build_lattice(
                genomes[0:1], Z, H, W, k,
                distal_seed=distal_seed,
                identity_seed=identity_seed,
                use_identity=use_identity,
                use_positional_cues=use_positional_cues,
                use_distal=use_distal,
                device=dev,
            )
            val_acc = float(
                classify_wta(
                    val_net, X_val, y_val, assignments[0:1], n_classes, readout_decay=readout_decay
                )[0]
            )

        assign_counts, assign_used, assign_top_class, assign_top_frac = assignment_diagnostics(
            assignments[0], n_classes
        )

        record = {
            "gen":                    gen,
            "best_acc":               float(accs[0]),
            "mean_acc":               float(accs.mean()),
            "best_fitness":           float(fitness[0]),
            "mean_fitness":           float(fitness.mean()),
            "best_silent_frac":       float(silent_frac[0]),
            "best_sat_frac":          float(sat_frac[0]),
            "best_assignment_counts": assign_counts,
            "best_assignment_used":   assign_used,
            "best_assignment_top_class": assign_top_class,
            "best_assignment_top_frac":  assign_top_frac,
            "injected":               injected,
            "best_genome":            genomes[0].cpu().tolist(),  # [Z, 3, k+2**k]
        }
        if val_available:
            record["val_acc"] = val_acc
        history.append(record)

        if live_path is not None:
            try:
                live_data = {"status": "running", "history": history}
                if live_meta:
                    live_data.update(live_meta)
                Path(live_path).write_text(json.dumps(live_data))
            except Exception:
                pass

        val_str = f" | val={val_acc:.1%}" if val_available else ""
        assign_str = (
            f" | assign_used={assign_used}/{n_classes}"
            f" top={assign_top_class}:{assign_top_frac:.1%}"
            f" counts={assign_counts}"
        )
        print(
            f"gen {gen:4d} | "
            f"train best={float(accs[0]):.1%} mean={float(accs.mean()):.1%}"
            f"{val_str} | "
            f"fit={float(fitness[0]):.3f} | "
            f"silent={float(silent_frac[0]):.2f} sat={float(sat_frac[0]):.2f}"
            f"{assign_str}"
            + (" [inject]" if injected else "")
        )

        # ---- (μ+λ) selection ------------------------------------------------
        # Elites: top elite_size genomes pass through unchanged
        elites = genomes[:elite_size]                           # [elite_size, Z, 3, G]

        # Sample parent pairs for children (with replacement from elites)
        idx_a    = torch.randint(0, elite_size, (n_children,), generator=rng, device=dev)
        idx_b    = torch.randint(0, elite_size, (n_children,), generator=rng, device=dev)
        parents_a = elites[idx_a]                              # [n_children, Z, 3, G]
        parents_b = elites[idx_b]

        # Crossover then mutate
        if use_crossover:
            children = crossover(parents_a, parents_b, rng)
        else:
            children = parents_a.clone()

        children = mutate(children, k, lut_rate, sel_rate, rng)

        # New population: elites first (sorted), then children
        genomes = torch.cat([elites, children], dim=0)

    # Final evaluation on the full training pool
    final_net = build_lattice(
        genomes, Z, H, W, k,
        distal_seed=distal_seed,
        identity_seed=identity_seed,
        use_identity=use_identity,
        use_positional_cues=use_positional_cues,
        use_distal=use_distal,
        device=dev,
    )
    final_accs, _, _, final_assignments = evaluate_wta(
        final_net, X_pool, y_pool, n_classes, readout_decay=readout_decay
    )
    order = torch.argsort(-final_accs, stable=True)
    best_assignment = final_assignments[order[0]:order[0] + 1]  # [1, W]

    return genomes[order], final_accs[order], history, best_assignment


# ---------------------------------------------------------------------------
# Demo: evolve on N-MNIST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    POP_SIZE             = 100
    N_GENERATIONS        = 100
    Z                    = 3
    GRID_SIZE            = 16
    H                    = GRID_SIZE
    W                    = GRID_SIZE
    K                    = 2
    N_TIME_BINS          = 10
    TASK                 = "10class"
    N_CLASSES            = 10
    LUT_MUTATION_RATE    = 0.02
    SEL_MUTATION_RATE    = 0.01
    ELITE_SIZE           = 4
    PENALTY_WEIGHTS      = (0.3, 0.3)
    USE_CROSSOVER        = True
    N_POOL_PER_CLASS     = 20
    VAL_FRACTION         = 0.2
    SEED                 = 42
    DISTAL_SEED          = 0
    IDENTITY_SEED        = 0
    USE_IDENTITY         = True
    USE_POSITIONAL_CUES  = False
    USE_DISTAL           = True
    DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_ROOT            = str(Path(__file__).resolve().parent.parent / "datasets")
    CHECKPOINT_PATH      = Path(__file__).resolve().parent / "best_lattice.pt"

    from src.dataset import load_nmnist, n_classes_for_task

    if n_classes_for_task(TASK) != N_CLASSES:
        raise ValueError(
            f"TASK={TASK!r} has {n_classes_for_task(TASK)} classes, "
            f"but N_CLASSES={N_CLASSES} was set"
        )

    print(
        f"Population: {POP_SIZE} | Generations: {N_GENERATIONS} | "
        f"Z={Z} H={H} W={W} k={K} T={N_TIME_BINS} | task={TASK} | "
        f"lut_rate={LUT_MUTATION_RATE} sel_rate={SEL_MUTATION_RATE} | "
        f"elite={ELITE_SIZE} crossover={USE_CROSSOVER} | "
        f"pool/class={N_POOL_PER_CLASS} | device={DEVICE}"
    )

    X_train, y_train, _, _ = load_nmnist(
        data_root=DATA_ROOT,
        task=TASK,
        n_time_bins=N_TIME_BINS,
        grid_size=GRID_SIZE,
        n_train_per_class=N_POOL_PER_CLASS,
        n_val_per_class=1,
        seed=SEED,
        device=torch.device("cpu"),
        first_saccade_only=True,
    )

    X_all = X_train.reshape(-1, N_TIME_BINS, H, W)
    n_total = X_all.shape[0]
    n_pool  = int((1 - VAL_FRACTION) * n_total)
    perm    = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))
    X_pool  = X_all[perm[:n_pool]].to(DEVICE)
    y_pool  = y_train[perm[:n_pool]].to(DEVICE)
    X_val   = X_all[perm[n_pool:]].to(DEVICE)
    y_val   = y_train[perm[n_pool:]].to(DEVICE)

    print(
        f"Loaded N-MNIST pool: X_pool={tuple(X_pool.shape)} X_val={tuple(X_val.shape)}  "
        f"active fraction={float(X_pool.float().mean()):.3f}"
    )

    class_counts = torch.bincount(y_pool, minlength=N_CLASSES)
    majority_acc = float(class_counts.max() / class_counts.sum())
    print(f"Majority-class baseline: {majority_acc:.1%}\n")

    final_genomes, final_accs, history, best_assignment = evolve(
        pop_size=POP_SIZE,
        n_generations=N_GENERATIONS,
        Z=Z, H=H, W=W, k=K,
        X_pool=X_pool,
        y_pool=y_pool,
        n_classes=N_CLASSES,
        lut_mutation_rate=LUT_MUTATION_RATE,
        sel_mutation_rate=SEL_MUTATION_RATE,
        elite_size=ELITE_SIZE,
        penalty_weights=PENALTY_WEIGHTS,
        use_crossover=USE_CROSSOVER,
        X_val=X_val,
        y_val=y_val,
        seed=SEED,
        use_identity=USE_IDENTITY,
        use_positional_cues=USE_POSITIONAL_CUES,
        use_distal=USE_DISTAL,
        distal_seed=DISTAL_SEED,
        identity_seed=IDENTITY_SEED,
        device=DEVICE,
    )

    # Val accuracy using the best genome's WTA assignment
    best_net = build_lattice(
        final_genomes[0:1], Z, H, W, K,
        distal_seed=DISTAL_SEED, identity_seed=IDENTITY_SEED,
        use_identity=USE_IDENTITY, use_positional_cues=USE_POSITIONAL_CUES,
        use_distal=USE_DISTAL, device=DEVICE,
    )
    val_acc = float(classify_wta(best_net, X_val, y_val, best_assignment, N_CLASSES)[0])

    g0 = history[0]
    print(
        f"\n=== Done ===\n"
        f"  initial train : acc best={g0['best_acc']:.1%}  mean={g0['mean_acc']:.1%}\n"
        f"  final on pool : acc best={float(final_accs[0]):.1%}  "
        f"mean={float(final_accs.mean()):.1%}  (pool size={X_pool.shape[0]})\n"
        f"  val (held-out): acc={val_acc:.1%}  (n={X_val.shape[0]})\n"
        f"  baselines     : majority={majority_acc:.1%}  "
        f"random={1.0 / N_CLASSES:.1%}  perfect=100%"
    )

    torch.save({
        "genome":              final_genomes[0].cpu(),
        "best_assignment":     best_assignment.cpu(),
        "Z": Z, "H": H, "W": W, "k": K,
        "n_time_bins":         N_TIME_BINS,
        "task":                TASK,
        "n_classes":           N_CLASSES,
        "distal_seed":         DISTAL_SEED,
        "identity_seed":       IDENTITY_SEED,
        "use_identity":        USE_IDENTITY,
        "use_positional_cues": USE_POSITIONAL_CUES,
        "use_distal":          USE_DISTAL,
        "train_acc":           float(final_accs[0]),
        "val_acc":             val_acc,
    }, CHECKPOINT_PATH)
    print(f"Saved {CHECKPOINT_PATH}")
