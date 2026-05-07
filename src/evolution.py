"""
evolution.py — Truncation-selection evolution for 3D Boolean lattice networks.

One generation:
  1. Draw a fresh random batch of `n_train` examples from the data pool.
  2. Build a Lattice3DNetwork from the current `[B, Z, 3, k + 2**k]` population.
  3. Evaluate L2 classification loss + accuracy per circuit on that batch.
  4. Rank by accuracy desc, ties broken by loss asc.
  5. Survivors  = top half of the population.
  6. Children   = mutated copies of the survivors.
  7. Next pop   = survivors ++ children   (length B).

The whole population sees the same batch within a generation (so per-gen
comparisons are meaningful), but each generation gets a different random
draw. As a side effect, "best-ever" fitness is no longer monotone: a
survivor can rank lower next generation purely because the batch changed.
The final evaluation re-runs on the entire pool so the returned losses /
accs are a stable ranking.

Genome layout (Lattice3DNetwork-specific):
  • genome[..., :k]   = MUX selectors  in [0, POOL_SIZE) — mutation resamples uniformly
  • genome[..., k:]   = LUT bits        in {0, 1}        — mutation flips the bit
  Doing a plain bit-flip on the selector indices (as the original 2D version
  did) would push them out of [0, POOL_SIZE), so we mutate the two halves
  with their type-appropriate operator.

Classification readout:
  • A fixed list of (y, x) sites on the top layer (z = Z-1), one per class.
  • logit_c = sum over time of activity at site c, cast to float.
  • loss   = mean squared error vs one-hot label, averaged over the batch.
  • pred   = argmax(logits).

Network bits that must NOT drift between generations (distal projections,
identity bits) are reseeded deterministically from fixed seeds, so we can
re-instantiate the network every generation cheaply without changing them.
"""

import sys
from pathlib import Path

# Add the repo root to sys.path so `from src.lattice import ...` works whether
# this file is launched as a script or imported. We use the package-qualified
# path (`src.lattice`) because `lattice.py` itself uses a relative import
# (`from .node import node_forward`) that requires it to be imported as part
# of the `src` namespace package — same pattern the smoke test uses.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

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

    Parameters
    ----------
    pop_size : int — number of genomes in the population (axis 0)
    Z, k     : lattice depth and node arity
    rng      : torch.Generator on `device`
    device   : output device

    Returns
    -------
    [pop_size, Z, 3, k + 2**k] long
        sel part: uniform in [0, POOL_SIZE)
        lut part: uniform in {0, 1}
    """
    lut_size = 2 ** k
    sel = torch.randint(
        0, POOL_SIZE, (pop_size, Z, N_NODES, k),
        generator=rng, device=device, dtype=torch.long,
    )
    lut = torch.randint(
        0, 2, (pop_size, Z, N_NODES, lut_size),
        generator=rng, device=device, dtype=torch.long,
    )
    return torch.cat([sel, lut], dim=-1)


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def mutate(
    genomes: torch.Tensor, k: int, mutation_rate: float, rng: torch.Generator,
) -> torch.Tensor:
    """Per-element mutation tailored to the genome layout.

    Parameters
    ----------
    genomes       : [B, Z, 3, k + 2**k] long
    k             : node arity (number of MUX selectors per node)
    mutation_rate : float in [0, 1] — per-element mutation probability
    rng           : torch.Generator on the genomes' device

    Returns
    -------
    mutated : [B, Z, 3, k + 2**k] long — new tensor; original is not modified

    The first k entries along the last axis are MUX selectors and get
    resampled uniformly in [0, POOL_SIZE) when mutated. The remaining
    2**k entries are LUT bits and get flipped when mutated.
    """
    if not (0.0 <= mutation_rate <= 1.0):
        raise ValueError(f"mutation_rate must be in [0, 1], got {mutation_rate}")

    sel = genomes[..., :k]
    lut = genomes[..., k:]
    dev = genomes.device

    sel_flip = torch.rand(sel.shape, generator=rng, device=dev) < mutation_rate
    sel_new  = torch.randint(
        0, POOL_SIZE, sel.shape,
        generator=rng, device=dev, dtype=sel.dtype,
    )
    sel_mut  = torch.where(sel_flip, sel_new, sel)

    lut_flip = torch.rand(lut.shape, generator=rng, device=dev) < mutation_rate
    lut_mut  = torch.where(lut_flip, 1 - lut, lut)

    return torch.cat([sel_mut, lut_mut], dim=-1)


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
    """Construct a Lattice3DNetwork around a packed genome population.

    distal_idx is regenerated from `distal_seed` each call (deterministic),
    and identity bits from `identity_seed`, so re-building the network
    every generation does NOT drift the fixed structural bits.
    """
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
    X_batch: torch.Tensor,            # [N, T, H, W] long, binary
    y_batch: torch.Tensor,            # [N] long
    output_sites_flat: torch.Tensor,  # [n_classes] long, indices into top-layer H*W
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score every circuit in the population on every sample of the batch.

    For each sample we broadcast the input across all B genomes, run the
    lattice for T steps, sum the top-layer activity over time, and read
    out per-class logits at the fixed output sites. Loss is mean squared
    error vs one-hot; prediction is argmax.

    Parameters
    ----------
    net               : Lattice3DNetwork with B = pop_size
    X_batch           : [N, T, H, W] long — binary input frames
    y_batch           : [N] long — class labels
    output_sites_flat : [n_classes] long — flat (y*W + x) indices on layer Z-1

    Returns
    -------
    losses : [B] float — mean L2 loss across the batch
    accs   : [B] float — argmax accuracy across the batch
    """
    B = net.B
    N, T, H, W = X_batch.shape
    n_classes = output_sites_flat.shape[0]
    dev = X_batch.device

    losses  = torch.zeros(B, device=dev)
    correct = torch.zeros(B, device=dev)
    target  = torch.zeros(B, n_classes, device=dev)

    for n in range(N):
        seq = X_batch[n:n + 1].expand(B, T, H, W).contiguous()  # [B, T, H, W]
        traj = net.run(seq)                                      # [B, T, H*W]
        feats = traj.float().sum(dim=1)                          # [B, H*W]
        logits = feats[:, output_sites_flat]                     # [B, n_classes]

        target.zero_()
        label = int(y_batch[n].item())
        target[:, label] = 1.0

        loss = ((logits - target) ** 2).mean(dim=1)              # [B]
        pred = logits.argmax(dim=1)                              # [B]

        losses  += loss
        correct += (pred == y_batch[n]).float()

    return losses / N, correct / N


def class_readout_sites(n_classes: int, H: int, W: int) -> torch.Tensor:
    """Pick `n_classes` evenly-spaced (y, x) sites on the H×W top layer.

    Sites lie along a single horizontal stripe at y = H // 2, with x
    distributed uniformly across [0, W-1]. Returned as flat indices
    `y * W + x` so they can index a flattened readout vector.

    Returns
    -------
    flat_idx : [n_classes] long — distinct flat indices into H*W
    """
    if n_classes > W:
        raise ValueError(
            f"n_classes={n_classes} exceeds top-layer width W={W}; "
            f"need a 2-D layout"
        )
    y = H // 2
    xs = torch.linspace(0, W - 1, n_classes).long()
    return torch.full((n_classes,), y, dtype=torch.long) * W + xs


# ---------------------------------------------------------------------------
# Selection helper
# ---------------------------------------------------------------------------

def _rank_by_acc_then_loss(
    accs: torch.Tensor, losses: torch.Tensor,
) -> torch.Tensor:
    """Return the permutation that sorts by accuracy DESC, ties by loss ASC.

    Equivalent to numpy.lexsort((losses, -accs)). Implemented with two
    stable sorts: first by the secondary key (loss asc), then by the
    primary key (acc desc). torch.argsort(..., stable=True) preserves the
    relative order of ties, so the secondary order survives the second pass.
    """
    order_secondary = torch.argsort(losses, stable=True)
    accs_s = accs[order_secondary]
    order_primary = torch.argsort(-accs_s, stable=True)
    return order_secondary[order_primary]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def evolve(
    pop_size: int,
    n_generations: int,
    Z: int, H: int, W: int, k: int,
    X_pool: torch.Tensor,             # [P, T, H, W] long binary
    y_pool: torch.Tensor,             # [P] long
    n_train: int,
    output_sites_flat: torch.Tensor,  # [n_classes] long
    mutation_rate: float,
    seed: int,
    use_identity: bool,
    use_positional_cues: bool,
    use_distal: bool,
    distal_seed: int,
    identity_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    """Run truncation-selection evolution for `n_generations` generations.

    Each generation draws a fresh random subset of `n_train` examples from
    `(X_pool, y_pool)` (without replacement, within the generation) and
    evaluates the entire population on that subset. The final ranking is
    recomputed on the full pool.

    Parameters
    ----------
    pop_size                       : even int >= 2
    n_generations                  : int >= 1
    Z, H, W, k                     : lattice dimensions and node arity
    X_pool                         : [P, T, H, W] long binary
    y_pool                         : [P] long — class labels in [0, n_classes)
    n_train                        : int — examples sampled per generation
    output_sites_flat              : [n_classes] long — top-layer flat indices
    mutation_rate                  : float — per-element mutation probability
    seed                           : int — RNG seed for genomes / mutation / batch
    use_identity, use_positional_cues, use_distal : ablation flags
    distal_seed, identity_seed     : seeds for fixed structural network bits
    device                         : torch device for genomes / network / data

    Returns
    -------
    final_genomes : [pop_size, Z, 3, k + 2**k] long — sorted best→worst on full pool
    final_losses  : [pop_size] float — pool loss aligned with final_genomes
    final_accs    : [pop_size] float — pool acc  aligned with final_genomes
    history       : list[dict] — per-generation metrics on that gen's batch
    """
    if pop_size < 2 or pop_size % 2 != 0:
        raise ValueError(f"pop_size must be an even integer >= 2, got {pop_size}")
    if n_generations < 1:
        raise ValueError(f"n_generations must be >= 1, got {n_generations}")
    if X_pool.dim() != 4:
        raise ValueError(
            f"X_pool must be [P, T, H, W]; got shape {tuple(X_pool.shape)}"
        )
    P, T, Hin, Win = X_pool.shape
    if Hin != H or Win != W:
        raise ValueError(
            f"X_pool spatial dims ({Hin},{Win}) != lattice ({H},{W})"
        )
    if y_pool.shape[0] != P:
        raise ValueError(
            f"X_pool and y_pool must agree on first axis; got {P} vs {y_pool.shape[0]}"
        )
    if not (1 <= n_train <= P):
        raise ValueError(f"n_train must be in [1, {P}], got {n_train}")

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    half = pop_size // 2

    genomes = random_genome_pack(pop_size, Z, k, rng, device)
    output_sites_flat = output_sites_flat.to(device)
    history: list[dict] = []

    for gen in range(n_generations):
        idx = torch.randperm(P, generator=rng, device=device)[:n_train]
        X_batch = X_pool[idx]
        y_batch = y_pool[idx]

        net = build_lattice(
            genomes, Z, H, W, k,
            distal_seed=distal_seed,
            identity_seed=identity_seed,
            use_identity=use_identity,
            use_positional_cues=use_positional_cues,
            use_distal=use_distal,
            device=device,
        )
        losses, accs = evaluate(net, X_batch, y_batch, output_sites_flat)

        order = _rank_by_acc_then_loss(accs, losses)
        genomes = genomes[order]
        losses  = losses[order]
        accs    = accs[order]

        history.append({
            "gen": gen,
            "best_acc": float(accs[0]),
            "mean_acc": float(accs.mean()),
            "loss_at_best_acc": float(losses[0]),
            "mean_loss": float(losses.mean()),
            "min_loss": float(losses.min()),
        })
        print(
            f"gen {gen:4d} | acc best={float(accs[0]):.1%} mean={float(accs.mean()):.1%} "
            f"| loss={float(losses[0]):.4f} mean={float(losses.mean()):.4f}"
        )

        survivors = genomes[:half]
        children  = mutate(survivors, k, mutation_rate, rng)
        genomes   = torch.cat([survivors, children], dim=0)

    final_net = build_lattice(
        genomes, Z, H, W, k,
        distal_seed=distal_seed,
        identity_seed=identity_seed,
        use_identity=use_identity,
        use_positional_cues=use_positional_cues,
        use_distal=use_distal,
        device=device,
    )
    final_losses, final_accs = evaluate(final_net, X_pool, y_pool, output_sites_flat)
    order = _rank_by_acc_then_loss(final_accs, final_losses)

    return genomes[order], final_losses[order], final_accs[order], history


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
    MUTATION_RATE        = 0.02
    N_POOL_PER_CLASS     = 20      # train pool size = N_POOL_PER_CLASS * N_CLASSES
    N_TRAIN              = 50      # fresh random subset evaluated each generation
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
        f"mutation={MUTATION_RATE} | pool/class={N_POOL_PER_CLASS} batch_per_gen={N_TRAIN} | "
        f"device={DEVICE}"
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

    # X_train is [P, T, F=H*W]; reshape to [P, T, H, W] for the lattice.
    X_pool = X_train.reshape(-1, N_TIME_BINS, H, W).to(DEVICE)
    y_pool = y_train.to(DEVICE)
    print(
        f"Loaded N-MNIST pool: X={tuple(X_pool.shape)}  "
        f"active fraction={float(X_pool.float().mean()):.3f}"
    )

    class_counts = torch.bincount(y_pool, minlength=N_CLASSES)
    majority_acc = float(class_counts.max() / class_counts.sum())
    print(f"Majority-class baseline (always predict most common): {majority_acc:.1%}")

    output_sites_flat = class_readout_sites(N_CLASSES, H, W).to(DEVICE)
    print(f"Output sites (flat indices on top layer): {output_sites_flat.tolist()}\n")

    final_genomes, final_losses, final_accs, history = evolve(
        pop_size=POP_SIZE,
        n_generations=N_GENERATIONS,
        Z=Z, H=H, W=W, k=K,
        X_pool=X_pool,
        y_pool=y_pool,
        n_train=N_TRAIN,
        output_sites_flat=output_sites_flat,
        mutation_rate=MUTATION_RATE,
        seed=SEED,
        use_identity=USE_IDENTITY,
        use_positional_cues=USE_POSITIONAL_CUES,
        use_distal=USE_DISTAL,
        distal_seed=DISTAL_SEED,
        identity_seed=IDENTITY_SEED,
        device=DEVICE,
    )

    g0 = history[0]
    pool_n = X_pool.shape[0]
    print(
        f"\n=== Done ===\n"
        f"  initial batch  : acc best={g0['best_acc']:.1%}  mean={g0['mean_acc']:.1%}  "
        f"| loss={g0['loss_at_best_acc']:.4f}\n"
        f"  final on pool  : acc best={float(final_accs[0]):.1%}  "
        f"mean={float(final_accs.mean()):.1%}  "
        f"| loss={float(final_losses[0]):.4f}  (pool size={pool_n})\n"
        f"  baselines : majority-class={majority_acc:.1%}  "
        f"random={1.0 / N_CLASSES:.1%}  perfect=100.0%"
    )

    # Save the best genome + structural metadata so a viewer/server can
    # rebuild the exact same network without re-running evolution.
    torch.save({
        "genome":              final_genomes[0].cpu(),
        "Z": Z, "H": H, "W": W, "k": K,
        "n_time_bins":         N_TIME_BINS,
        "task":                TASK,
        "output_sites_flat":   output_sites_flat.cpu(),
        "distal_seed":         DISTAL_SEED,
        "identity_seed":       IDENTITY_SEED,
        "use_identity":        USE_IDENTITY,
        "use_positional_cues": USE_POSITIONAL_CUES,
        "use_distal":          USE_DISTAL,
        "train_loss":          float(final_losses[0]),
        "train_acc":           float(final_accs[0]),
    }, CHECKPOINT_PATH)
    print(f"Saved {CHECKPOINT_PATH}")
