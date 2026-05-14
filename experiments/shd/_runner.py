"""
experiments/shd/_runner.py
Shared runner logic for all SHD experiments.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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


def run_experiment(config: dict, device: torch.device, exp_dir: Path,
                   on_generation=None) -> dict:
    task    = config["task"]
    neat_kw = config["neat"]
    ds_kw   = config["dataset"]

    print(f"\n{'='*60}")
    print(f"  TASK : {task}")
    print(f"  DIR  : {exp_dir}")
    print(f"{'='*60}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_shd_train_val_test(
        data_root=DATA_ROOT,
        task=task,
        time_window=ds_kw["time_window"],
        n_train_per_class=ds_kw.get("n_train_per_class"),
        n_val_per_class=ds_kw.get("n_val_per_class"),
        n_test_per_class=ds_kw.get("n_test_per_class"),
        seed=neat_kw["seed"],
        device=torch.device("cpu"),
    )
    print(f"  train={tuple(X_train.shape)}  val={tuple(X_val.shape)}  test={tuple(X_test.shape)}")

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
    )
    elapsed = time.time() - t0

    net       = EIONetwork(best, device)
    bs        = neat_kw["batch_size"]
    train_acc = accuracy(net, X_train, y_train, bs, device)
    val_acc   = accuracy(net, X_val,   y_val,   bs, device)
    test_acc  = accuracy(net, X_test,  y_test,  bs, device)

    bm = best.metrics
    print(f"\n  train={train_acc:.1%}  val={val_acc:.1%}  test={test_acc:.1%}  "
          f"E={int(bm.get('enabled_e', 0))}  I={int(bm.get('enabled_i', 0))}  "
          f"conn={int(bm.get('enabled_data_conns', 0) + bm.get('enabled_inh_conns', 0))}  "
          f"time={elapsed:.0f}s")

    result = dict(
        task=task,
        train_acc=train_acc,
        val_acc=val_acc,
        test_acc=test_acc,
        elapsed_s=elapsed,
        history=history,
        config=config,
    )

    out_dir = exp_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / "genome.json", "w") as f:
        json.dump(best.to_dict(), f, indent=2)
    torch.save(result, out_dir / "checkpoint.pt")
    print(f"  saved → {out_dir}")

    return result
