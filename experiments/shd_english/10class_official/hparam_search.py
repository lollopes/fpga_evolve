"""
experiments/shd_english/10class_official/hparam_search.py
— Optuna hyperparameter search for SHD English 10-class (official split).

The official test split is NEVER used during search; all selection is done on
a held-out slice of the train split. This search explicitly tests temporal
resolution (`n_time_bins`) and the leaky readout decay (`readout_decay`).
Results are written to:
    results/optuna/study.db        — SQLite study (resumable)
    results/optuna/best_config.json

Run from repo root:
    python experiments/shd_english/10class_official/hparam_search.py
    python experiments/shd_english/10class_official/hparam_search.py --n-trials 80 --trial-gens 30
    python experiments/shd_english/10class_official/hparam_search.py --n-jobs 4
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR   = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

import torch

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    sys.exit("optuna is not installed — run: pip install optuna")

from src.dataset   import load_shd, n_classes_for_task
from src.evolution import evolve, build_lattice, classify_wta

# ── Constants fixed across all trials ────────────────────────────────────────
with open(EXP_DIR / "config.json") as f:
    BASE_CFG = json.load(f)

TASK      = BASE_CFG["task"]           # "10class"
N_CLASSES = BASE_CFG["n_classes"]      # 10
SEED      = BASE_CFG["seed"]
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SHD feature dim is always 700 cochlear channels.
# Valid (H, W) grid shapes that tile exactly 700 neurons per layer.
SHD_F = 700
VALID_GRIDS: list[tuple[int, int]] = [
    (h, w)
    for h in range(1, SHD_F + 1)
    for w in [SHD_F // h]
    if h * w == SHD_F and h <= w          # keep h <= w to avoid duplicates
]

# During HPO we subsample the train split to keep trials fast.
# The official test split is held out entirely — only run.py sees it.
HPO_TRAIN_PER_CLASS = 50   # samples/class used for HPO pool+val
HPO_VAL_FRACTION    = 0.2  # fraction of that subsample held as val

RESULTS_DIR = EXP_DIR / "results" / "optuna"

assert n_classes_for_task(TASK) == N_CLASSES


# ── Data loading (called once per trial, keyed on n_time_bins) ───────────────

_data_cache: dict[int, tuple] = {}

def load_data(n_time_bins: int) -> tuple:
    """Load SHD train subset and split into pool/val.

    The official test split is NOT loaded — we derive our HPO val from the
    train split to avoid leaking test-set information into hyperparameter
    selection.
    """
    if n_time_bins in _data_cache:
        return _data_cache[n_time_bins]

    # include_test_split=False: skip the official test HDF5 file entirely
    X_tr, y_tr, _, _ = load_shd(
        task=TASK,
        n_time_bins=n_time_bins,
        n_train_per_class=HPO_TRAIN_PER_CLASS,
        n_val_per_class=1,          # dummy — we won't use the test split
        seed=SEED,
        device=torch.device("cpu"),
        include_test_split=False,
    )
    # X_tr: [N, T, 700]  — reshape is done per-trial based on chosen (H, W)

    n_total = X_tr.shape[0]
    n_pool  = int((1 - HPO_VAL_FRACTION) * n_total)
    perm    = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))

    result = (X_tr[perm[:n_pool]], y_tr[perm[:n_pool]],
              X_tr[perm[n_pool:]], y_tr[perm[n_pool:]])
    _data_cache[n_time_bins] = result
    return result


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(trial_gens: int):
    def objective(trial: optuna.Trial) -> float:
        # ── Architecture ──────────────────────────────────────────────────────
        grid_idx = trial.suggest_int("grid_idx", 0, len(VALID_GRIDS) - 1)
        H, W     = VALID_GRIDS[grid_idx]
        z        = trial.suggest_int("z", 2, 6)
        k        = trial.suggest_categorical("k", [2, 3])
        use_id   = trial.suggest_categorical("use_identity",        [True, False])
        use_pos  = trial.suggest_categorical("use_positional_cues", [True, False])
        use_dist = trial.suggest_categorical("use_distal",          [True, False])
        # SHD is timing-sensitive; test coarser vs finer framing explicitly.
        n_time_bins = trial.suggest_categorical("n_time_bins", [10, 20, 40])
        # Include 0.0 as the old no-leak baseline for comparison.
        readout_decay = trial.suggest_categorical("readout_decay", [0.0, 0.90, 0.95, 0.98])

        # Log decoded grid so Optuna history is human-readable
        trial.set_user_attr("grid_h", H)
        trial.set_user_attr("grid_w", W)

        # ── Evolution ─────────────────────────────────────────────────────────
        pop_size   = trial.suggest_categorical("pop_size", [50, 100, 150, 200])
        elite_frac = trial.suggest_float("elite_frac", 0.05, 0.25)
        elite_size = max(2, int(pop_size * elite_frac))

        lut_rate  = trial.suggest_float("lut_mutation_rate", 5e-3, 0.1,  log=True)
        sel_rate  = trial.suggest_float("sel_mutation_rate", 2e-3, 0.05, log=True)
        w_silent  = trial.suggest_float("w_silent", 0.0, 0.5)
        w_sat     = trial.suggest_float("w_sat",    0.0, 0.5)
        crossover = trial.suggest_categorical("use_crossover", [False])
        stag_pat  = trial.suggest_int("stagnation_patience",    5, 20)
        stag_frac = trial.suggest_float("stagnation_inject_frac", 0.1, 0.5)

        # ── Data ──────────────────────────────────────────────────────────────
        X_pool_flat, y_pool, X_val_flat, y_val = load_data(n_time_bins)

        # Reshape flat 700-dim features into chosen (H, W) grid
        X_pool = X_pool_flat.reshape(-1, n_time_bins, H, W).to(DEVICE)
        X_val  = X_val_flat.reshape(-1,  n_time_bins, H, W).to(DEVICE)
        y_pool = y_pool.to(DEVICE)
        y_val  = y_val.to(DEVICE)

        # ── Evolve ────────────────────────────────────────────────────────────
        try:
            final_genomes, final_accs, history, best_assignment = evolve(
                pop_size=pop_size,
                n_generations=trial_gens,
                Z=z, H=H, W=W, k=k,
                X_pool=X_pool,
                y_pool=y_pool,
                n_classes=N_CLASSES,
                lut_mutation_rate=lut_rate,
                sel_mutation_rate=sel_rate,
                elite_size=elite_size,
                penalty_weights=(w_silent, w_sat),
                use_crossover=crossover,
                stagnation_patience=stag_pat,
                stagnation_inject_frac=stag_frac,
                parallel_samples=False,
                readout_decay=readout_decay,
                X_val=X_val,
                y_val=y_val,
                seed=SEED,
                use_identity=use_id,
                use_positional_cues=use_pos,
                use_distal=use_dist,
                distal_seed=BASE_CFG["distal_seed"],
                identity_seed=BASE_CFG["identity_seed"],
                device=DEVICE,
            )
        except Exception as e:
            raise optuna.exceptions.TrialPruned(str(e))

        # ── Final val accuracy ────────────────────────────────────────────────
        best_net = build_lattice(
            final_genomes[0:1], z, H, W, k,
            distal_seed=BASE_CFG["distal_seed"],
            identity_seed=BASE_CFG["identity_seed"],
            use_identity=use_id,
            use_positional_cues=use_pos,
            use_distal=use_dist,
            device=DEVICE,
        )
        val_acc = float(
            classify_wta(
                best_net, X_val, y_val, best_assignment, N_CLASSES,
                readout_decay=readout_decay,
            )[0]
        )

        # Report intermediate val accs for median pruning
        for record in history:
            if "val_acc" in record:
                trial.report(record["val_acc"], step=record["gen"])
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return val_acc

    return objective


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optuna HPO for shd_english/10class_official"
    )
    parser.add_argument("--n-trials",   type=int, default=50,
                        help="Total number of Optuna trials (default: 50)")
    parser.add_argument("--trial-gens", type=int, default=30,
                        help="Evolution generations per trial (default: 30)")
    parser.add_argument("--n-jobs",     type=int, default=1,
                        help="Parallel trial workers (default: 1)")
    parser.add_argument("--study-name", type=str,
                        default="shd_english_10class_hpo_timebins_decay_v2",
                        help="Optuna study name")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    db_path = RESULTS_DIR / f"{args.study_name}.db"

    print(
        f"HPO config: n_trials={args.n_trials}  trial_gens={args.trial_gens}  "
        f"n_jobs={args.n_jobs}  device={DEVICE}\n"
        f"Valid grids (H×W, H*W=700): {VALID_GRIDS}\n"
        f"HPO pool: {HPO_TRAIN_PER_CLASS} samples/class "
        f"({int((1-HPO_VAL_FRACTION)*HPO_TRAIN_PER_CLASS*N_CLASSES)} pool / "
        f"{int(HPO_VAL_FRACTION*HPO_TRAIN_PER_CLASS*N_CLASSES)} val)\n"
        f"Search: n_time_bins in [10, 20, 40], "
        f"readout_decay in [0.0, 0.90, 0.95, 0.98]\n"
        f"Study name: {args.study_name}\n"
        f"Study DB: {db_path}\n"
    )

    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    study.optimize(
        make_objective(args.trial_gens),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    # ── Report results ────────────────────────────────────────────────────────
    best = study.best_trial
    H_best = best.user_attrs["grid_h"]
    W_best = best.user_attrs["grid_w"]

    print(f"\n=== Best trial ===")
    print(f"  val_acc : {best.value:.1%}  (HPO train-subset val, NOT official test)")
    print(f"  grid    : H={H_best} W={W_best}  (grid_idx={best.params['grid_idx']})")
    print(f"  params  :")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    pop_size   = best.params["pop_size"]
    elite_frac = best.params["elite_frac"]
    elite_size = max(2, int(pop_size * elite_frac))

    best_cfg = {
        **BASE_CFG,
        "z":                      best.params["z"],
        "grid_height":            H_best,
        "grid_width":             W_best,
        "k":                      best.params["k"],
        "use_identity":           best.params["use_identity"],
        "use_positional_cues":    best.params["use_positional_cues"],
        "use_distal":             best.params["use_distal"],
        "n_time_bins":            best.params["n_time_bins"],
        "readout_decay":         best.params["readout_decay"],
        "pop_size":               pop_size,
        "elite_size":             elite_size,
        "lut_mutation_rate":      best.params["lut_mutation_rate"],
        "sel_mutation_rate":      best.params["sel_mutation_rate"],
        "penalty_weights":        [best.params["w_silent"], best.params["w_sat"]],
        "use_crossover":          best.params["use_crossover"],
        "stagnation_patience":    best.params["stagnation_patience"],
        "stagnation_inject_frac": best.params["stagnation_inject_frac"],
        "_hpo_val_acc":           best.value,
        "_hpo_note":              "val measured on HPO train-subset, not official test",
        "_hpo_trial_gens":        args.trial_gens,
        "_hpo_n_trials_run":      len(study.trials),
    }

    cfg_path = RESULTS_DIR / "best_config.json"
    with open(cfg_path, "w") as f:
        json.dump(best_cfg, f, indent=2)
    print(f"\nSaved best config → {cfg_path}")
    print("Copy to config.json and run run.py (full n_generations) for final evaluation.")


if __name__ == "__main__":
    main()
