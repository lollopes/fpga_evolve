"""
experiments/nmnist/10class_official/run.py — Evolve a Lattice3DNetwork on all
10 N-MNIST digits, using the official train split for selection and the official
test split for final evaluation.

All hyperparameters live in config.json next to this file. Results are written to
    experiments/nmnist/10class_official/results/best_genome.pt   — best genome + metadata
    experiments/nmnist/10class_official/results/history.json     — per-generation metrics

Run from repo root:
    python experiments/nmnist/10class_official/run.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

import torch

from src.dataset import load_nmnist, n_classes_for_task
from src.evolution import build_lattice, classify_wta, evolve

with open(EXP_DIR / "config.json") as f:
    cfg = json.load(f)

TASK = cfg["task"]
N_CLASSES = cfg["n_classes"]
POP_SIZE = cfg["pop_size"]
N_GENERATIONS = cfg["n_generations"]
LUT_MUTATION_RATE = cfg.get("lut_mutation_rate", cfg.get("mutation_rate", 0.02))
SEL_MUTATION_RATE = cfg.get("sel_mutation_rate", LUT_MUTATION_RATE * 0.5)
ELITE_SIZE = cfg.get("elite_size", 20)
PENALTY_WEIGHTS = tuple(cfg.get("penalty_weights", [0.3, 0.3]))
USE_CROSSOVER = cfg.get("use_crossover", True)
STAGNATION_PATIENCE = cfg.get("stagnation_patience", 10)
STAGNATION_INJECT_FRAC = cfg.get("stagnation_inject_frac", 0.3)
PARALLEL_SAMPLES = cfg.get("parallel_samples", False)
N_TRAIN_PER_CLASS = cfg["n_train_per_class"]
N_TEST_PER_CLASS = cfg["n_test_per_class"]
VAL_FRACTION = cfg.get("val_fraction", 0.0)
N_TIME_BINS = cfg["n_time_bins"]
GRID_SIZE = cfg["grid_size"]
FIRST_SACCADE_ONLY = cfg["first_saccade_only"]
SEED = cfg["seed"]
Z = cfg["z"]
K = cfg["k"]
USE_IDENTITY = cfg["use_identity"]
USE_POSITIONAL_CUES = cfg["use_positional_cues"]
USE_DISTAL = cfg["use_distal"]
DISTAL_SEED = cfg["distal_seed"]
IDENTITY_SEED = cfg["identity_seed"]
READOUT_DECAY = cfg.get("readout_decay", 0.0)
VAL_GAP_WEIGHT = cfg.get("val_gap_weight", 0.0)
WARM_START_INPUT = cfg.get("warm_start_input", False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = EXP_DIR / "results"

assert n_classes_for_task(TASK) == N_CLASSES, (
    f"TASK={TASK!r} has {n_classes_for_task(TASK)} classes but n_classes={N_CLASSES} in config"
)
if not (0.0 <= VAL_FRACTION < 1.0):
    raise ValueError(f"val_fraction must be in [0, 1), got {VAL_FRACTION}")
if VAL_GAP_WEIGHT < 0.0:
    raise ValueError(f"val_gap_weight must be >= 0, got {VAL_GAP_WEIGHT}")
if VAL_GAP_WEIGHT > 0.0 and VAL_FRACTION <= 0.0:
    raise ValueError("val_gap_weight > 0 requires val_fraction > 0")


def split_train_val_stratified(X_all, y_all, n_classes, val_fraction, seed):
    """Split the official train set into a train pool and held-out validation set."""
    if val_fraction <= 0.0:
        return X_all, y_all, None, None

    rng = torch.Generator().manual_seed(seed)
    pool_idx_parts = []
    val_idx_parts = []
    for cls in range(n_classes):
        cls_idx = torch.where(y_all == cls)[0]
        if cls_idx.numel() < 2:
            raise ValueError(f"class {cls} has too few samples ({cls_idx.numel()}) for a val split")
        cls_idx = cls_idx[torch.randperm(cls_idx.numel(), generator=rng)]
        n_val_cls = max(1, int(round(val_fraction * cls_idx.numel())))
        n_val_cls = min(cls_idx.numel() - 1, n_val_cls)
        val_idx_parts.append(cls_idx[:n_val_cls])
        pool_idx_parts.append(cls_idx[n_val_cls:])

    pool_idx = torch.cat(pool_idx_parts)
    val_idx = torch.cat(val_idx_parts)
    pool_idx = pool_idx[torch.randperm(pool_idx.numel(), generator=rng)]
    val_idx = val_idx[torch.randperm(val_idx.numel(), generator=rng)]
    return X_all[pool_idx], y_all[pool_idx], X_all[val_idx], y_all[val_idx]


print(
    f"Task: {TASK} | dataset=N-MNIST official split | "
    f"pop={POP_SIZE} gens={N_GENERATIONS} | "
    f"Z={Z} grid_size={GRID_SIZE} k={K} T={N_TIME_BINS} | "
    f"readout_decay={READOUT_DECAY:.2f} val_gap_weight={VAL_GAP_WEIGHT:.2f} | "
    f"first_saccade_only={FIRST_SACCADE_ONLY} train/class={N_TRAIN_PER_CLASS} "
    f"test/class={N_TEST_PER_CLASS} val_fraction={VAL_FRACTION} | device={DEVICE}"
)

X_train, y_train, X_test, y_test = load_nmnist(
    task=TASK,
    n_time_bins=N_TIME_BINS,
    grid_size=GRID_SIZE,
    n_train_per_class=N_TRAIN_PER_CLASS,
    n_val_per_class=N_TEST_PER_CLASS,
    seed=SEED,
    device=torch.device("cpu"),
    first_saccade_only=FIRST_SACCADE_ONLY,
)

F = X_train.shape[2]
H = W = int(F ** 0.5)
assert H * W == F, f"Feature dim {F} is not a perfect square"
assert X_test.shape[2] == F, f"Test feature dim {X_test.shape[2]} does not match train dim {F}"

X_train_4d = X_train.reshape(-1, N_TIME_BINS, H, W)
X_pool_cpu, y_pool_cpu, X_val_cpu, y_val_cpu = split_train_val_stratified(
    X_train_4d, y_train, N_CLASSES, VAL_FRACTION, SEED
)

X_pool = X_pool_cpu.to(DEVICE)
y_pool = y_pool_cpu.to(DEVICE)
X_val = X_val_cpu.to(DEVICE) if X_val_cpu is not None else None
y_val = y_val_cpu.to(DEVICE) if y_val_cpu is not None else None
X_test = X_test.reshape(-1, N_TIME_BINS, H, W).to(DEVICE)
y_test = y_test.to(DEVICE)

print(f"Spatial resolution: H={H} W={W} ({H*W} neurons/layer)")
print(
    f"Train pool   : {tuple(X_pool.shape)}  active fraction={float(X_pool.float().mean()):.3f}"
)
if X_val is not None:
    print(
        f"Val (train) : {tuple(X_val.shape)}  active fraction={float(X_val.float().mean()):.3f}"
    )
print(
    f"Test official: {tuple(X_test.shape)}  active fraction={float(X_test.float().mean()):.3f}"
)

pool_counts = torch.bincount(y_pool, minlength=N_CLASSES)
test_counts = torch.bincount(y_test, minlength=N_CLASSES)
pool_majority = float(pool_counts.max() / pool_counts.sum())
test_majority = float(test_counts.max() / test_counts.sum())
if y_val is not None:
    val_counts = torch.bincount(y_val, minlength=N_CLASSES)
    val_majority = float(val_counts.max() / val_counts.sum())
    print(
        f"Baselines: pool majority={pool_majority:.1%} | "
        f"val majority={val_majority:.1%} | "
        f"test majority={test_majority:.1%} | random={1/N_CLASSES:.1%}\n"
    )
else:
    print(
        f"Baselines: pool majority={pool_majority:.1%} | "
        f"test majority={test_majority:.1%} | random={1/N_CLASSES:.1%}\n"
    )

final_genomes, final_accs, history, best_assignment = evolve(
    pop_size=POP_SIZE,
    n_generations=N_GENERATIONS,
    Z=Z,
    H=H,
    W=W,
    k=K,
    X_pool=X_pool,
    y_pool=y_pool,
    n_classes=N_CLASSES,
    lut_mutation_rate=LUT_MUTATION_RATE,
    sel_mutation_rate=SEL_MUTATION_RATE,
    elite_size=ELITE_SIZE,
    penalty_weights=PENALTY_WEIGHTS,
    use_crossover=USE_CROSSOVER,
    stagnation_patience=STAGNATION_PATIENCE,
    stagnation_inject_frac=STAGNATION_INJECT_FRAC,
    parallel_samples=PARALLEL_SAMPLES,
    readout_decay=READOUT_DECAY,
    val_gap_weight=VAL_GAP_WEIGHT,
    X_val=X_val,
    y_val=y_val,
    seed=SEED,
    use_identity=USE_IDENTITY,
    use_positional_cues=USE_POSITIONAL_CUES,
    use_distal=USE_DISTAL,
    distal_seed=DISTAL_SEED,
    identity_seed=IDENTITY_SEED,
    warm_start_input=WARM_START_INPUT,
    device=DEVICE,
    live_path=REPO_ROOT / "viewer" / "evo_live.json",
    live_meta={"experiment": EXP_DIR.name, "task": TASK},
    live_scene_path=REPO_ROOT / "viewer" / "evo_data.js",
)

best_net = build_lattice(
    final_genomes[0:1],
    Z,
    H,
    W,
    K,
    distal_seed=DISTAL_SEED,
    identity_seed=IDENTITY_SEED,
    use_identity=USE_IDENTITY,
    use_positional_cues=USE_POSITIONAL_CUES,
    use_distal=USE_DISTAL,
    device=DEVICE,
)
val_acc = None
if X_val is not None:
    val_acc = float(
        classify_wta(
            best_net, X_val, y_val, best_assignment, N_CLASSES,
            readout_decay=READOUT_DECAY,
        )[0]
    )
test_acc = float(
    classify_wta(
        best_net, X_test, y_test, best_assignment, N_CLASSES, readout_decay=READOUT_DECAY
    )[0]
)

g0 = history[0]
summary_lines = [
    "\n=== Done ===",
    f"  initial train : acc best={g0['best_acc']:.1%}  mean={g0['mean_acc']:.1%}",
    f"  train pool    : acc best={float(final_accs[0]):.1%}  mean={float(final_accs.mean()):.1%}  (n={X_pool.shape[0]})",
]
if val_acc is not None:
    summary_lines.append(f"  val (train)   : acc={val_acc:.1%}  (n={X_val.shape[0]})")
summary_lines.extend([
    f"  test official : acc={test_acc:.1%}  (n={X_test.shape[0]})",
    f"  baselines     : pool majority={pool_majority:.1%}  test majority={test_majority:.1%}  random={1/N_CLASSES:.1%}",
])
print("\n".join(summary_lines))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

genome_path = RESULTS_DIR / "best_genome.pt"
payload = {
    "genome": final_genomes[0].cpu(),
    "dataset": "nmnist",
    "split": "official_train_test",
    "Z": Z,
    "H": H,
    "W": W,
    "k": K,
    "grid_size": GRID_SIZE,
    "n_time_bins": N_TIME_BINS,
    "task": TASK,
    "first_saccade_only": FIRST_SACCADE_ONLY,
    "readout_decay": READOUT_DECAY,
    "val_fraction": VAL_FRACTION,
    "val_gap_weight": VAL_GAP_WEIGHT,
    "best_assignment": best_assignment.cpu(),
    "distal_seed": DISTAL_SEED,
    "identity_seed": IDENTITY_SEED,
    "use_identity": USE_IDENTITY,
    "use_positional_cues": USE_POSITIONAL_CUES,
    "use_distal": USE_DISTAL,
    "train_acc": float(final_accs[0]),
    "test_acc": test_acc,
}
if val_acc is not None:
    payload["val_acc"] = val_acc
torch.save(payload, genome_path)
print(f"Saved genome  → {genome_path}")

history_path = RESULTS_DIR / "history.json"
with open(history_path, "w") as f:
    json.dump(history, f, indent=2)
print(f"Saved history → {history_path}")
