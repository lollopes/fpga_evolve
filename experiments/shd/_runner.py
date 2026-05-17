"""
experiments/shd/_runner.py
Shared runner logic for all SHD experiments.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset import load_shd_train_val_test, n_classes_for_task  # noqa: E402
from eio_neat import NEATConfig, evolve, EIONetwork              # noqa: E402

DATA_ROOT = str(ROOT / "datasets" / "SHD" / "data")


@torch.no_grad()
def accuracy(net: EIONetwork, X: torch.Tensor, y: torch.Tensor,
             batch_size: int, device: torch.device) -> float:
    correct, n = 0, X.shape[0]
    for s in range(0, n, batch_size):
        xb = X[s:s + batch_size].to(device)
        yb = y[s:s + batch_size].to(device)
        pred, _ = net.forward_counts(xb)
        preds = pred.argmax(1)
        silent = pred.sum(1) == 0
        correct += int(((preds == yb) & ~silent).sum())
    return correct / max(1, n)


def _run_single(config: dict, seed: int, device: torch.device, out_dir: Path,
                on_generation=None) -> dict:
    """Run one evolution with a fixed seed, save results to out_dir."""
    task    = config["task"]
    _RUNNER_ONLY_KEYS = {"n_seeds"}
    neat_kw = {k: v for k, v in config["neat"].items() if k not in _RUNNER_ONLY_KEYS}
    neat_kw["seed"] = seed
    ds_kw   = config["dataset"]

    X_train, y_train, X_val, y_val, X_test, y_test = load_shd_train_val_test(
        data_root=DATA_ROOT,
        task=task,
        time_window=ds_kw["time_window"],
        n_train_per_class=ds_kw.get("n_train_per_class"),
        n_val_per_class=ds_kw.get("n_val_per_class"),
        n_test_per_class=ds_kw.get("n_test_per_class"),
        n_channels=ds_kw.get("n_channels"),
        seed=seed,
        device=torch.device("cpu"),
    )

    neat_config = NEATConfig(**neat_kw)
    n_outputs   = n_classes_for_task(task)

    t0 = time.time()
    best, history, _reg = evolve(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        n_outputs=n_outputs,
        config=neat_config,
        on_generation=on_generation,
        device=device,
        X_test=X_test,   y_test=y_test,
    )
    elapsed = time.time() - t0

    net       = EIONetwork(best, device)
    bs        = neat_kw["batch_size"]
    train_acc = accuracy(net, X_train, y_train, bs, device)
    val_acc   = accuracy(net, X_val,   y_val,   bs, device)
    test_acc  = accuracy(net, X_test,  y_test,  bs, device)

    bm = best.metrics
    print(f"\n  train={train_acc:.1%}  val={val_acc:.1%}  test={test_acc:.1%}  "
          f"H={int(bm.get('enabled_h', 0))}  nodes={int(bm.get('n_nodes', 0))}  "
          f"conn={int(bm.get('enabled_data_conns', 0))}  "
          f"time={elapsed:.0f}s")

    result = dict(
        task=task,
        seed=seed,
        train_acc=train_acc,
        val_acc=val_acc,
        test_acc=test_acc,
        elapsed_s=elapsed,
        history=history,
        config=config,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / "genome.json", "w") as f:
        json.dump(best.to_dict(), f, indent=2)
    torch.save(result, out_dir / "checkpoint.pt")
    print(f"  saved → {out_dir}")

    return result


def run_experiment(config: dict, device: torch.device, exp_dir: Path,
                   on_generation=None) -> dict:
    task    = config["task"]
    neat_kw = config["neat"]
    n_seeds = neat_kw.get("n_seeds", 1)
    master_seed = neat_kw["seed"]

    print(f"\n{'='*60}")
    print(f"  TASK : {task}")
    print(f"  DIR  : {exp_dir}")
    print(f"  SEEDS: {n_seeds}")
    print(f"{'='*60}")

    if n_seeds == 1:
        return _run_single(config, master_seed, device,
                           exp_dir / "results", on_generation)

    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(0, 2**31, size=n_seeds).tolist()

    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n--- seed {seed}  ({i+1}/{n_seeds}) ---")
        result = _run_single(config, seed, device,
                             exp_dir / "results" / f"seed_{seed}", on_generation)
        all_results.append(result)

    summary = _summarise(all_results, config)
    summary_path = exp_dir / "results" / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  summary saved → {summary_path}")
    print(f"  test  {summary['test_acc_mean']:.1%} ± {summary['test_acc_std']:.1%}")
    print(f"  val   {summary['val_acc_mean']:.1%} ± {summary['val_acc_std']:.1%}")
    return summary


def _summarise(results: list, config: dict) -> dict:
    train_accs = [r["train_acc"] for r in results]
    val_accs   = [r["val_acc"]   for r in results]
    test_accs  = [r["test_acc"]  for r in results]
    return dict(
        task=results[0]["task"],
        seeds=[r["seed"] for r in results],
        train_acc_mean=statistics.mean(train_accs),
        train_acc_std=statistics.stdev(train_accs) if len(train_accs) > 1 else 0.0,
        val_acc_mean=statistics.mean(val_accs),
        val_acc_std=statistics.stdev(val_accs) if len(val_accs) > 1 else 0.0,
        test_acc_mean=statistics.mean(test_accs),
        test_acc_std=statistics.stdev(test_accs) if len(test_accs) > 1 else 0.0,
        config=config,
    )
