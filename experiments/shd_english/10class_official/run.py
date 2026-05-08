"""
experiments/shd_english/10class_official/run.py — Evolve a Lattice3DNetwork on
all 10 English spoken digits from SHD, using the official train split for
selection and the official test split for final evaluation.

All hyperparameters live in config.json next to this file. Results are written to
    experiments/shd_english/10class_official/results/best_genome.pt   — best genome + metadata
    experiments/shd_english/10class_official/results/history.json     — per-generation metrics

Run from repo root:
    python experiments/shd_english/10class_official/run.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

import torch

from src.dataset import load_shd, n_classes_for_task
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
N_TIME_BINS = cfg["n_time_bins"]
SEED = cfg["seed"]
Z = cfg["z"]
GRID_HEIGHT = cfg["grid_height"]
GRID_WIDTH = cfg["grid_width"]
K = cfg["k"]
USE_IDENTITY = cfg["use_identity"]
USE_POSITIONAL_CUES = cfg["use_positional_cues"]
USE_DISTAL = cfg["use_distal"]
DISTAL_SEED = cfg["distal_seed"]
IDENTITY_SEED = cfg["identity_seed"]
READOUT_DECAY = cfg.get("readout_decay", 0.9)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = EXP_DIR / "results"

assert n_classes_for_task(TASK) == N_CLASSES, (
    f"TASK={TASK!r} has {n_classes_for_task(TASK)} classes but n_classes={N_CLASSES} in config"
)

print(
    f"Task: {TASK} | dataset=SHD English digits official split | "
    f"pop={POP_SIZE} gens={N_GENERATIONS} | "
    f"Z={Z} grid={GRID_HEIGHT}x{GRID_WIDTH} k={K} T={N_TIME_BINS} | "
    f"readout_decay={READOUT_DECAY:.2f} | "
    f"train/class={N_TRAIN_PER_CLASS} test/class={N_TEST_PER_CLASS} | device={DEVICE}"
)

X_train, y_train, X_test, y_test = load_shd(
    task=TASK,
    n_time_bins=N_TIME_BINS,
    n_train_per_class=N_TRAIN_PER_CLASS,
    n_val_per_class=N_TEST_PER_CLASS,
    seed=SEED,
    device=torch.device("cpu"),
    include_test_split=True,
)

F = X_train.shape[2]
H = GRID_HEIGHT
W = GRID_WIDTH
assert H * W == F, f"Configured grid {H}x{W} does not match feature dim {F}"
assert X_test.shape[2] == F, f"Test feature dim {X_test.shape[2]} does not match train dim {F}"

X_pool = X_train.reshape(-1, N_TIME_BINS, H, W).to(DEVICE)
y_pool = y_train.to(DEVICE)
X_test = X_test.reshape(-1, N_TIME_BINS, H, W).to(DEVICE)
y_test = y_test.to(DEVICE)

print(f"Input layout: H={H} W={W} ({H*W} neurons/layer)")
print(
    f"Train official: {tuple(X_pool.shape)}  "
    f"active fraction={float(X_pool.float().mean()):.3f}"
)
print(
    f"Test official : {tuple(X_test.shape)}  "
    f"active fraction={float(X_test.float().mean()):.3f}"
)

train_counts = torch.bincount(y_pool, minlength=N_CLASSES)
test_counts = torch.bincount(y_test, minlength=N_CLASSES)
train_majority = float(train_counts.max() / train_counts.sum())
test_majority = float(test_counts.max() / test_counts.sum())
print(
    f"Baselines: train majority={train_majority:.1%} | "
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
    seed=SEED,
    use_identity=USE_IDENTITY,
    use_positional_cues=USE_POSITIONAL_CUES,
    use_distal=USE_DISTAL,
    distal_seed=DISTAL_SEED,
    identity_seed=IDENTITY_SEED,
    device=DEVICE,
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
test_acc = float(
    classify_wta(
        best_net, X_test, y_test, best_assignment, N_CLASSES, readout_decay=READOUT_DECAY
    )[0]
)

g0 = history[0]
print(
    f"\n=== Done ===\n"
    f"  initial train : acc best={g0['best_acc']:.1%}  mean={g0['mean_acc']:.1%}\n"
    f"  train official: acc best={float(final_accs[0]):.1%}  "
    f"mean={float(final_accs.mean()):.1%}  (n={X_pool.shape[0]})\n"
    f"  test official : acc={test_acc:.1%}  (n={X_test.shape[0]})\n"
    f"  baselines     : train majority={train_majority:.1%}  "
    f"test majority={test_majority:.1%}  random={1/N_CLASSES:.1%}"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

genome_path = RESULTS_DIR / "best_genome.pt"
torch.save({
    "genome": final_genomes[0].cpu(),
    "dataset": "shd_english",
    "split": "official_train_test",
    "Z": Z,
    "H": H,
    "W": W,
    "k": K,
    "n_time_bins": N_TIME_BINS,
    "task": TASK,
    "readout_decay": READOUT_DECAY,
    "best_assignment": best_assignment.cpu(),
    "distal_seed": DISTAL_SEED,
    "identity_seed": IDENTITY_SEED,
    "use_identity": USE_IDENTITY,
    "use_positional_cues": USE_POSITIONAL_CUES,
    "use_distal": USE_DISTAL,
    "train_acc": float(final_accs[0]),
    "test_acc": test_acc,
}, genome_path)
print(f"Saved genome  → {genome_path}")

history_path = RESULTS_DIR / "history.json"
with open(history_path, "w") as f:
    json.dump(history, f, indent=2)
print(f"Saved history → {history_path}")
