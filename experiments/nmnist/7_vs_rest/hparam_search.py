"""
experiments/7_vs_rest/hparam_search.py — Optuna hyperparameter search for 7_vs_rest.

Each trial runs a shortened evolution and reports held-out val accuracy.
Results are written to:
    experiments/7_vs_rest/results/optuna/study.db      — SQLite study (resumable)
    experiments/7_vs_rest/results/optuna/best_config.json

Run from repo root:
    python experiments/7_vs_rest/hparam_search.py
    python experiments/7_vs_rest/hparam_search.py --n-trials 100 --trial-gens 30
    python experiments/7_vs_rest/hparam_search.py --n-jobs 4   # parallel trials
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

from src.evolution import evolve, build_lattice, classify_wta
from src.dataset   import load_nmnist, n_classes_for_task

# ── Constants fixed across all trials ────────────────────────────────────────
with open(EXP_DIR / "config.json") as f:
    BASE_CFG = json.load(f)

TASK           = BASE_CFG["task"]
N_CLASSES      = BASE_CFG["n_classes"]
GRID_SIZE      = BASE_CFG["grid_size"]          # spatial resolution stays fixed
N_POOL_PER_CLASS = BASE_CFG["n_pool_per_class"] # data size stays fixed
VAL_FRACTION   = BASE_CFG["val_fraction"]
FIRST_SACCADE  = BASE_CFG["first_saccade_only"]
SEED           = BASE_CFG["seed"]
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR    = EXP_DIR / "results" / "optuna"

assert n_classes_for_task(TASK) == N_CLASSES


# ── Data loading (done once, shared across trials) ────────────────────────────

def load_data(n_time_bins: int) -> tuple:
    """Load and split N-MNIST for a given temporal resolution."""
    X_all, y_all, _, _ = load_nmnist(
        task=TASK,
        n_time_bins=n_time_bins,
        grid_size=GRID_SIZE,
        n_train_per_class=N_POOL_PER_CLASS,
        n_val_per_class=1,
        seed=SEED,
        device=torch.device("cpu"),
        first_saccade_only=FIRST_SACCADE,
    )
    F = X_all.shape[2]
    H = W = int(F ** 0.5)
    assert H * W == F

    X_all_4d = X_all.reshape(-1, n_time_bins, H, W)
    n_total   = X_all_4d.shape[0]
    n_pool    = int((1 - VAL_FRACTION) * n_total)
    perm      = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))

    X_pool = X_all_4d[perm[:n_pool]].to(DEVICE)
    y_pool = y_all[perm[:n_pool]].to(DEVICE)
    X_val  = X_all_4d[perm[n_pool:]].to(DEVICE)
    y_val  = y_all[perm[n_pool:]].to(DEVICE)
    return X_pool, y_pool, X_val, y_val, H, W


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(trial_gens: int):
    def objective(trial: optuna.Trial) -> float:
        # ── Architecture ──────────────────────────────────────────────────────
        z          = trial.suggest_int("z",  2, 6)
        k          = trial.suggest_categorical("k", [2, 3])
        use_id     = trial.suggest_categorical("use_identity",        [True, False])
        use_pos    = trial.suggest_categorical("use_positional_cues", [True, False])
        use_dist   = trial.suggest_categorical("use_distal",          [True, False])
        n_time_bins = trial.suggest_categorical("n_time_bins",        [5, 10, 15, 20])

        # ── Evolution ─────────────────────────────────────────────────────────
        pop_size   = trial.suggest_categorical("pop_size", [50, 100, 150, 200])
        elite_frac = trial.suggest_float("elite_frac", 0.05, 0.25)
        elite_size = max(2, int(pop_size * elite_frac))
        # ensure elite_size < pop_size (always true given frac <= 0.25)

        lut_rate   = trial.suggest_float("lut_mutation_rate", 5e-3, 0.1,  log=True)
        sel_rate   = trial.suggest_float("sel_mutation_rate", 2e-3, 0.05, log=True)
        w_silent   = trial.suggest_float("w_silent", 0.0, 0.5)
        w_sat      = trial.suggest_float("w_sat",    0.0, 0.5)
        crossover  = trial.suggest_categorical("use_crossover", [True, False])
        stag_pat   = trial.suggest_int("stagnation_patience",    5, 20)
        stag_frac  = trial.suggest_float("stagnation_inject_frac", 0.1, 0.5)

        # ── Data ──────────────────────────────────────────────────────────────
        X_pool, y_pool, X_val, y_val, H, W = load_data(n_time_bins)

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
            # e.g. OOM or invalid config — treat as a pruned trial
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
        val_acc = float(classify_wta(best_net, X_val, y_val, best_assignment, N_CLASSES)[0])

        # Log intermediate val accs per generation for pruning support
        for record in history:
            if "val_acc" in record:
                trial.report(record["val_acc"], step=record["gen"])
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return val_acc

    return objective


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for 7_vs_rest")
    parser.add_argument("--n-trials",   type=int, default=50,
                        help="Total number of Optuna trials (default: 50)")
    parser.add_argument("--trial-gens", type=int, default=40,
                        help="Evolution generations per trial (default: 40)")
    parser.add_argument("--n-jobs",     type=int, default=1,
                        help="Parallel trial workers (default: 1)")
    parser.add_argument("--study-name", type=str, default="7_vs_rest_hpo",
                        help="Optuna study name (default: 7_vs_rest_hpo)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    db_path = RESULTS_DIR / "study.db"

    print(
        f"HPO config: n_trials={args.n_trials}  trial_gens={args.trial_gens}  "
        f"n_jobs={args.n_jobs}  device={DEVICE}\n"
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
    print(f"\n=== Best trial ===")
    print(f"  val_acc : {best.value:.1%}")
    print(f"  params  :")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Reconstruct elite_size from best params for the saved config
    pop_size   = best.params["pop_size"]
    elite_frac = best.params["elite_frac"]
    elite_size = max(2, int(pop_size * elite_frac))

    best_cfg = {
        **BASE_CFG,
        "z":                      best.params["z"],
        "k":                      best.params["k"],
        "use_identity":           best.params["use_identity"],
        "use_positional_cues":    best.params["use_positional_cues"],
        "use_distal":             best.params["use_distal"],
        "n_time_bins":            best.params["n_time_bins"],
        "pop_size":               pop_size,
        "elite_size":             elite_size,
        "lut_mutation_rate":      best.params["lut_mutation_rate"],
        "sel_mutation_rate":      best.params["sel_mutation_rate"],
        "penalty_weights":        [best.params["w_silent"], best.params["w_sat"]],
        "use_crossover":          best.params["use_crossover"],
        "stagnation_patience":    best.params["stagnation_patience"],
        "stagnation_inject_frac": best.params["stagnation_inject_frac"],
        "_hpo_val_acc":           best.value,
        "_hpo_trial_gens":        args.trial_gens,
        "_hpo_n_trials_run":      len(study.trials),
    }

    cfg_path = RESULTS_DIR / "best_config.json"
    with open(cfg_path, "w") as f:
        json.dump(best_cfg, f, indent=2)
    print(f"\nSaved best config → {cfg_path}")
    print(f"(Copy to config.json and run run.py with full n_generations to train.)")


if __name__ == "__main__":
    main()
