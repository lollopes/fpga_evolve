"""
experiments/7_vs_rest/run.py — Evolve a Lattice3DNetwork on N-MNIST digit 7 vs all others.

All hyperparameters live in config.json next to this file.  Results are written to
    experiments/7_vs_rest/results/best_genome.pt   — best genome + metadata
    experiments/7_vs_rest/results/history.json     — per-generation metrics

Run from repo root:
    python experiments/7_vs_rest/run.py
"""

import json
import sys
from pathlib import Path

# ── repo root on sys.path so `src.*` imports resolve ────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
EXP_DIR     = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

import torch

from src.evolution import evolve, build_lattice, classify_wta
from src.dataset   import load_nmnist, n_classes_for_task

# ── load config ──────────────────────────────────────────────────────────────
with open(EXP_DIR / "config.json") as f:
    cfg = json.load(f)

TASK                 = cfg["task"]
N_CLASSES            = cfg["n_classes"]
POP_SIZE             = cfg["pop_size"]
N_GENERATIONS        = cfg["n_generations"]
LUT_MUTATION_RATE    = cfg.get("lut_mutation_rate", cfg.get("mutation_rate", 0.02))
SEL_MUTATION_RATE    = cfg.get("sel_mutation_rate", LUT_MUTATION_RATE * 0.5)
ELITE_SIZE              = cfg.get("elite_size", 20)
PENALTY_WEIGHTS         = tuple(cfg.get("penalty_weights", [0.3, 0.3]))
USE_CROSSOVER           = cfg.get("use_crossover", True)
STAGNATION_PATIENCE     = cfg.get("stagnation_patience", 10)
STAGNATION_INJECT_FRAC  = cfg.get("stagnation_inject_frac", 0.3)
PARALLEL_SAMPLES        = cfg.get("parallel_samples", False)
N_POOL_PER_CLASS     = cfg["n_pool_per_class"]
VAL_FRACTION         = cfg["val_fraction"]
N_TIME_BINS          = cfg["n_time_bins"]
FIRST_SACCADE_ONLY   = cfg["first_saccade_only"]
SEED                 = cfg["seed"]
Z                    = cfg["z"]
GRID_SIZE            = cfg["grid_size"]
K                    = cfg["k"]
USE_IDENTITY         = cfg["use_identity"]
USE_POSITIONAL_CUES  = cfg["use_positional_cues"]
USE_DISTAL           = cfg["use_distal"]
DISTAL_SEED          = cfg["distal_seed"]
IDENTITY_SEED        = cfg["identity_seed"]

DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR          = EXP_DIR / "results"

# ── sanity check ─────────────────────────────────────────────────────────────
assert n_classes_for_task(TASK) == N_CLASSES, (
    f"TASK={TASK!r} has {n_classes_for_task(TASK)} classes but n_classes={N_CLASSES} in config"
)

# ── load data ─────────────────────────────────────────────────────────────────
print(
    f"Task: {TASK} | pop={POP_SIZE} gens={N_GENERATIONS} | "
    f"Z={Z} grid_size={GRID_SIZE} k={K} T={N_TIME_BINS} | "
    f"pool/class={N_POOL_PER_CLASS} val_fraction={VAL_FRACTION} | device={DEVICE}"
)

# Load train split only — val is split manually below
X_all, y_all, _, _ = load_nmnist(
    task=TASK,
    n_time_bins=N_TIME_BINS,
    grid_size=GRID_SIZE,
    n_train_per_class=N_POOL_PER_CLASS,
    n_val_per_class=1,
    seed=SEED,
    device=torch.device("cpu"),
    first_saccade_only=FIRST_SACCADE_ONLY,
)

# Derive H, W from actual data — guards against grid_size > sensor size (34×34)
F = X_all.shape[2]
H = W = int(F ** 0.5)
assert H * W == F, f"Feature dim {F} is not a perfect square"

# 80/20 split (seeded for reproducibility)
X_all_4d = X_all.reshape(-1, N_TIME_BINS, H, W)
n_total  = X_all_4d.shape[0]
n_pool   = int((1 - VAL_FRACTION) * n_total)
perm     = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))

X_pool = X_all_4d[perm[:n_pool]].to(DEVICE)
y_pool = y_all[perm[:n_pool]].to(DEVICE)
X_val  = X_all_4d[perm[n_pool:]].to(DEVICE)
y_val  = y_all[perm[n_pool:]].to(DEVICE)

print(f"Spatial resolution: H={H} W={W} ({H*W} neurons/layer)")
print(
    f"Train pool  : {tuple(X_pool.shape)}  "
    f"active fraction={float(X_pool.float().mean()):.3f}"
)
print(f"Val (held-out): {tuple(X_val.shape)}")

class_counts = torch.bincount(y_pool, minlength=N_CLASSES)
majority_acc = float(class_counts.max() / class_counts.sum())
print(f"Majority-class baseline: {majority_acc:.1%}  |  random baseline: {1/N_CLASSES:.1%}\n")

# ── evolve ────────────────────────────────────────────────────────────────────
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
    stagnation_patience=STAGNATION_PATIENCE,
    stagnation_inject_frac=STAGNATION_INJECT_FRAC,
    parallel_samples=PARALLEL_SAMPLES,
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

# ── val evaluation on best genome ─────────────────────────────────────────────
best_net = build_lattice(
    final_genomes[0:1], Z, H, W, K,
    distal_seed=DISTAL_SEED,
    identity_seed=IDENTITY_SEED,
    use_identity=USE_IDENTITY,
    use_positional_cues=USE_POSITIONAL_CUES,
    use_distal=USE_DISTAL,
    device=DEVICE,
)
val_acc = float(classify_wta(best_net, X_val, y_val, best_assignment, N_CLASSES)[0])

# ── report ────────────────────────────────────────────────────────────────────
g0 = history[0]
print(
    f"\n=== Done ===\n"
    f"  initial val   : acc best={g0['best_acc']:.1%}  mean={g0['mean_acc']:.1%}\n"
    f"  train (pool)  : acc best={float(final_accs[0]):.1%}  "
    f"mean={float(final_accs.mean()):.1%}  (n={X_pool.shape[0]})\n"
    f"  val (held-out): acc={val_acc:.1%}  (n={X_val.shape[0]})\n"
    f"  baselines     : majority={majority_acc:.1%}  random={1/N_CLASSES:.1%}  perfect=100%"
)

# ── save results ──────────────────────────────────────────────────────────────
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

genome_path = RESULTS_DIR / "best_genome.pt"
torch.save({
    "genome":              final_genomes[0].cpu(),
    "Z": Z, "H": H, "W": W, "k": K,
    "n_time_bins":         N_TIME_BINS,
    "task":                TASK,
    "best_assignment":     best_assignment.cpu(),
    "distal_seed":         DISTAL_SEED,
    "identity_seed":       IDENTITY_SEED,
    "use_identity":        USE_IDENTITY,
    "use_positional_cues": USE_POSITIONAL_CUES,
    "use_distal":          USE_DISTAL,
    "train_acc":           float(final_accs[0]),
    "val_acc":             val_acc,
}, genome_path)
print(f"Saved genome  → {genome_path}")

history_path = RESULTS_DIR / "history.json"
with open(history_path, "w") as f:
    json.dump(history, f, indent=2)
print(f"Saved history → {history_path}")
