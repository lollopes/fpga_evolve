"""
experiments/nmnist/10class_official/hparam_search.py — Optuna hyperparameter
search for full 10-class N-MNIST using only a train-derived validation split.

The official N-MNIST test split is NEVER used during search; all selection is
done on a held-out slice of the train split. Results are written to:
    results/optuna/<study-name>.db  — SQLite study (resumable)
    results/optuna/best_config.json

Run from repo root:
    python experiments/nmnist/10class_official/hparam_search.py
    python experiments/nmnist/10class_official/hparam_search.py --n-trials 80 --trial-gens 30
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

import torch

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    sys.exit("optuna is not installed — run: pip install optuna")

from src.dataset import load_nmnist, n_classes_for_task
from src.evolution import evolve, build_lattice, classify_wta

with open(EXP_DIR / "config.json") as f:
    BASE_CFG = json.load(f)

TASK          = BASE_CFG["task"]
N_CLASSES     = BASE_CFG["n_classes"]
GRID_SIZE     = BASE_CFG["grid_size"]           # fixed: 32
N_TIME_BINS   = BASE_CFG["n_time_bins"]         # fixed: 10
FIRST_SACCADE = BASE_CFG["first_saccade_only"]  # fixed: True
WARM_START    = BASE_CFG.get("warm_start_input", True)  # fixed: True
SEED          = BASE_CFG["seed"]
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_GAP_WEIGHT = BASE_CFG.get("val_gap_weight", 0.0)

HPO_TRAIN_PER_CLASS = 200
HPO_VAL_FRACTION    = 0.15
RESULTS_DIR = EXP_DIR / "results" / "optuna"

assert n_classes_for_task(TASK) == N_CLASSES


def load_fixed_data():
    """Load the HPO dataset once. grid_size and n_time_bins are fixed."""
    X_tr, y_tr, _, _ = load_nmnist(
        task=TASK,
        n_time_bins=N_TIME_BINS,
        grid_size=GRID_SIZE,
        n_train_per_class=HPO_TRAIN_PER_CLASS,
        n_val_per_class=1,
        seed=SEED,
        device=torch.device("cpu"),
        first_saccade_only=FIRST_SACCADE,
    )
    F = X_tr.shape[2]
    H = W = int(F ** 0.5)
    assert H * W == F

    X_all_4d = X_tr.reshape(-1, N_TIME_BINS, H, W)
    n_total   = X_all_4d.shape[0]
    n_pool    = int((1 - HPO_VAL_FRACTION) * n_total)
    perm      = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))

    return (
        X_all_4d[perm[:n_pool]].to(DEVICE), y_tr[perm[:n_pool]].to(DEVICE),
        X_all_4d[perm[n_pool:]].to(DEVICE), y_tr[perm[n_pool:]].to(DEVICE),
        H, W,
    )


def make_objective(trial_gens, X_pool, y_pool, X_val, y_val, H, W):
    def objective(trial):
        # ── Architecture (sensible ranges for 32×32 N-MNIST) ─────────────────
        z    = trial.suggest_int("z", 2, 5)              # 2–5 layers
        k    = trial.suggest_categorical("k", [2, 3, 4]) # LUT arity

        # ── Evolution ─────────────────────────────────────────────────────────
        pop_size   = trial.suggest_categorical("pop_size", [50, 100, 150, 200])
        elite_frac = trial.suggest_float("elite_frac", 0.05, 0.25)
        elite_size = max(2, int(pop_size * elite_frac))
        lut_rate   = trial.suggest_float("lut_mutation_rate", 5e-3, 0.1,  log=True)
        sel_rate   = trial.suggest_float("sel_mutation_rate", 2e-3, 0.05, log=True)
        use_id        = BASE_CFG["use_identity"]
        use_pos       = BASE_CFG["use_positional_cues"]
        use_dist      = BASE_CFG["use_distal"]
        readout_decay = BASE_CFG["readout_decay"]
        w_silent, w_sat = BASE_CFG["penalty_weights"]
        crossover     = BASE_CFG["use_crossover"]
        stag_pat      = BASE_CFG["stagnation_patience"]
        stag_frac     = BASE_CFG["stagnation_inject_frac"]

        try:
            final_genomes, final_accs, history, best_assignment = evolve(
                pop_size=pop_size,
                n_generations=trial_gens,
                Z=z,
                H=H,
                W=W,
                k=k,
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
                val_gap_weight=VAL_GAP_WEIGHT,
                warm_start_input=WARM_START,
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

        for record in history:
            if "val_acc" in record:
                trial.report(record["val_acc"], step=record["gen"])
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return val_acc

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for nmnist/10class_official")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--trial-gens", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--study-name",
        type=str,
        default="nmnist_10class_hpo_v1",
        help="Optuna study name",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    db_path = RESULTS_DIR / f"{args.study_name}.db"

    # Load dataset once — fixed for all trials
    print("Loading HPO dataset (fixed for all trials)...")
    X_pool, y_pool, X_val, y_val, H, W = load_fixed_data()
    print(
        f"  pool: {tuple(X_pool.shape)}  val: {tuple(X_val.shape)}\n"
        f"  grid: {H}×{W}  n_time_bins={N_TIME_BINS}  "
        f"warm_start_input={WARM_START}\n"
    )

    print(
        f"HPO config: n_trials={args.n_trials}  trial_gens={args.trial_gens}  "
        f"n_jobs={args.n_jobs}  device={DEVICE}\n"
        f"Fixed: grid_size={GRID_SIZE}  n_time_bins={N_TIME_BINS}  "
        f"first_saccade_only={FIRST_SACCADE}  warm_start_input={WARM_START}  "
        f"use_identity={BASE_CFG['use_identity']}  "
        f"use_positional_cues={BASE_CFG['use_positional_cues']}  "
        f"use_distal={BASE_CFG['use_distal']}  readout_decay={BASE_CFG['readout_decay']}\n"
        f"Fixed evo: penalty_weights={tuple(BASE_CFG['penalty_weights'])}  "
        f"use_crossover={BASE_CFG['use_crossover']}  "
        f"stagnation_patience={BASE_CFG['stagnation_patience']}  "
        f"stagnation_inject_frac={BASE_CFG['stagnation_inject_frac']}\n"
        f"Search: z in [2,5]  k in [2,3,4]  pop_size in [50,100,150,200]  "
        f"elite_frac in [0.05,0.25]  elite_size=max(2,int(pop_size*elite_frac))  "
        f"lut_mutation_rate in [5e-3,1e-1] log  "
        f"sel_mutation_rate in [2e-3,5e-2] log\n"
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
        make_objective(args.trial_gens, X_pool, y_pool, X_val, y_val, H, W),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    best = study.best_trial
    print("\n=== Best trial ===")
    print(f"  val_acc : {best.value:.1%}  (HPO train-subset val, NOT official test)")
    print("  params  :")
    for key, value in best.params.items():
        print(f"    {key}: {value}")

    pop_size = best.params["pop_size"]
    elite_frac = best.params["elite_frac"]
    elite_size = max(2, int(pop_size * elite_frac))

    best_cfg = {
        **BASE_CFG,
        "z":                      best.params["z"],
        "k":                      best.params["k"],
        "use_identity":           BASE_CFG["use_identity"],
        "use_positional_cues":    BASE_CFG["use_positional_cues"],
        "use_distal":             BASE_CFG["use_distal"],
        "readout_decay":          BASE_CFG["readout_decay"],
        "warm_start_input":       WARM_START,
        "pop_size":               pop_size,
        "elite_frac":             elite_frac,
        "elite_size":             elite_size,
        "lut_mutation_rate":      best.params["lut_mutation_rate"],
        "sel_mutation_rate":      best.params["sel_mutation_rate"],
        "penalty_weights":        BASE_CFG["penalty_weights"],
        "use_crossover":          BASE_CFG["use_crossover"],
        "stagnation_patience":    BASE_CFG["stagnation_patience"],
        "stagnation_inject_frac": BASE_CFG["stagnation_inject_frac"],
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
