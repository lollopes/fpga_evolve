#!/usr/bin/env python3
"""
experiments/nmnist/run.py
Run E/I/O NEAT on N-MNIST: 0_vs_1, 7_vs_rest, 10class.

Usage:
    python experiments/nmnist/run.py                  # all three
    python experiments/nmnist/run.py --task 0_vs_1
    python experiments/nmnist/run.py --task 7_vs_rest
    python experiments/nmnist/run.py --task 10class
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset import load_nmnist_train_val_test, n_classes_for_task  # noqa: E402
from eio_neat import NEATConfig, evolve, EIONetwork               # noqa: E402

DATA_ROOT = str(Path(__file__).resolve().parents[2] / "datasets")
RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Per-task configs — matched exactly to README_EIO_NEAT.md tables
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "0_vs_1": dict(
        pop_size=100,
        generations=200,
        k=5,
        initial_connections_per_output=4,
        initial_e_nodes=2,
        lut_bit_rate=0.02,
        p_add_connection=0.30,
        p_add_node=0.05,
        p_toggle_connection=0.02,
        p_crossover=0.75,
        p_new_node_is_inhibitory=0.2,
        p_mutate_node_type=0.01,
        allow_output_feedback=False,
        compatibility_threshold=1.5,
        c_disjoint=1.0,
        c_lut=0.4,
        c_type=0.2,
        max_stale=15,
        elite_min_size=5,
        batch_size=128,
        fitness_sample=0,
        seed=42,
    ),
    "7_vs_rest": dict(
        pop_size=100,
        generations=200,
        k=10,
        initial_connections_per_output=10,
        initial_e_nodes=3,
        lut_bit_rate=0.02,
        p_add_connection=0.30,
        p_add_node=0.05,
        p_toggle_connection=0.02,
        p_crossover=0.75,
        p_new_node_is_inhibitory=0.2,
        p_mutate_node_type=0.01,
        allow_output_feedback=False,
        compatibility_threshold=4.0,
        c_disjoint=1.0,
        c_lut=0.4,
        c_type=0.2,
        max_stale=15,
        elite_min_size=5,
        batch_size=128,
        fitness_sample=512,
        seed=42,
    ),
    "10class": dict(
        pop_size=200,
        generations=200,
        k=3,
        initial_connections_per_output=5,
        initial_e_nodes=6,
        lut_bit_rate=0.02,
        p_add_connection=0.30,
        p_add_node=0.08,
        p_toggle_connection=0.02,
        p_crossover=0.75,
        p_new_node_is_inhibitory=0.2,
        p_mutate_node_type=0.01,
        allow_output_feedback=True,
        compatibility_threshold=2.0,
        c_disjoint=1.0,
        c_lut=0.4,
        c_type=0.2,
        max_stale=20,
        elite_min_size=5,
        batch_size=128,
        fitness_sample=0,
        seed=42,
    ),
}

DATASET_KWARGS = {
    "0_vs_1":    dict(n_train_per_class=1000, n_val_per_class=200, n_test_per_class=None),
    "7_vs_rest": dict(n_train_per_class=1000, n_val_per_class=200, n_test_per_class=None),
    "10class":   dict(n_train_per_class=1000, n_val_per_class=100, n_test_per_class=None),
}


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


def run_experiment(task: str, device: torch.device) -> dict:
    print(f"\n{'='*60}")
    print(f"  TASK: {task}")
    print(f"{'='*60}")

    cfg_kw = EXPERIMENTS[task]
    ds_kw  = DATASET_KWARGS[task]

    X_train, y_train, X_val, y_val, X_test, y_test = load_nmnist_train_val_test(
        task=task,
        n_time_bins=10,
        grid_size=8,
        data_root=DATA_ROOT,
        seed=cfg_kw["seed"],
        device=torch.device("cpu"),
        first_saccade_only=True,
        **ds_kw,
    )
    print(f"  train={tuple(X_train.shape)}  val={tuple(X_val.shape)}  test={tuple(X_test.shape)}")

    config = NEATConfig(**cfg_kw)
    n_outputs = n_classes_for_task(task)

    t0 = time.time()
    best, history, _reg = evolve(
        X_train=X_train, y_train=y_train,
        X_val=X_val,   y_val=y_val,
        n_outputs=n_outputs,
        config=config,
        device=device,
    )
    elapsed = time.time() - t0

    net = EIONetwork(best, device)
    bs  = cfg_kw["batch_size"]
    train_acc = accuracy(net, X_train, y_train, bs, device)
    val_acc   = accuracy(net, X_val,   y_val,   bs, device)
    test_acc  = accuracy(net, X_test,  y_test,  bs, device)

    bm = best.metrics
    print(f"\n  train={train_acc:.1%}  val={val_acc:.1%}  test={test_acc:.1%}  "
          f"E={int(bm.get('enabled_e',0))} I={int(bm.get('enabled_i',0))} "
          f"conn={int(bm.get('enabled_data_conns',0)+bm.get('enabled_inh_conns',0))}  "
          f"time={elapsed:.0f}s")

    result = dict(
        task=task,
        train_acc=train_acc,
        val_acc=val_acc,
        test_acc=test_acc,
        elapsed_s=elapsed,
        history=history,
        config=cfg_kw,
        genome=best.to_dict(),
    )

    out = RESULTS_DIR / task
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "result.json", "w") as f:
        json.dump({k: v for k, v in result.items() if k != "genome"}, f, indent=2)
    with open(out / "genome.json", "w") as f:
        json.dump(best.to_dict(), f, indent=2)
    torch.save(result, out / "checkpoint.pt")
    print(f"  saved → {out}")

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(EXPERIMENTS) + ["all"], default="all")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    tasks  = list(EXPERIMENTS) if args.task == "all" else [args.task]

    summary = {}
    for task in tasks:
        r = run_experiment(task, device)
        summary[task] = dict(train=r["train_acc"], val=r["val_acc"], test=r["test_acc"])

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for task, accs in summary.items():
        print(f"  {task:<12}  train={accs['train']:.1%}  val={accs['val']:.1%}  test={accs['test']:.1%}")


if __name__ == "__main__":
    main()
