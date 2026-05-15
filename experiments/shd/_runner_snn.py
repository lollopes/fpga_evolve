"""
experiments/shd/_runner_snn.py
Shared training loop for SHD SNN (LIF backprop) experiments.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset import load_shd_train_val_test, n_classes_for_task  # noqa: E402
from lif.bp_mlp import BP_MLP                                     # noqa: E402

DATA_ROOT = str(ROOT / "datasets" / "SHD" / "data")


def _forward_sequence(model: BP_MLP, X: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Run a batch of spike sequences through the network one timestep at a time.

    X       : [B, T, F]  float
    returns : [B, n_outputs]  accumulated LI membrane potential over T steps
    """
    B, T, _ = X.shape
    model.reset_potential()
    output_sum = torch.zeros(B, model.network[-1].fc.out_features, device=device)
    for t in range(T):
        out = model.inference(X[:, t, :])   # [B, n_outputs]
        output_sum = output_sum + out
    return output_sum


@torch.no_grad()
def _evaluate(model: BP_MLP, X: torch.Tensor, y: torch.Tensor,
              batch_size: int, device: torch.device) -> tuple[float, float]:
    """Return (accuracy, mean_ce) over the full split."""
    model.eval()
    n = X.shape[0]
    correct = 0
    ce_sum = 0.0
    for s in range(0, n, batch_size):
        xb = X[s:s + batch_size].float().to(device)
        yb = y[s:s + batch_size].to(device)
        logits = _forward_sequence(model, xb, device)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum())
        ce_sum += F.cross_entropy(logits, yb).item() * xb.shape[0]
    return correct / max(1, n), ce_sum / max(1, n)


def run_experiment(config: dict, device: torch.device, exp_dir: Path,
                   on_epoch=None) -> dict:
    task      = config["task"]
    ds_kw     = config["dataset"]
    tr_kw     = config["training"]
    snn_kw    = config["snn_params"]
    model_kw  = config["model"]

    torch.manual_seed(tr_kw["seed"])

    print(f"\n{'='*60}")
    print(f"  TASK : {task}  (SNN baseline)")
    print(f"  DIR  : {exp_dir}")
    print(f"{'='*60}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_shd_train_val_test(
        data_root=DATA_ROOT,
        task=task,
        time_window=ds_kw["time_window"],
        n_train_per_class=ds_kw.get("n_train_per_class"),
        n_val_per_class=ds_kw.get("n_val_per_class"),
        n_test_per_class=ds_kw.get("n_test_per_class"),
        seed=tr_kw["seed"],
        device=torch.device("cpu"),
    )
    print(f"  train={tuple(X_train.shape)}  val={tuple(X_val.shape)}  test={tuple(X_test.shape)}")

    n_outputs   = n_classes_for_task(task)
    input_size  = int(X_train.shape[-1])
    batch_size  = tr_kw["batch_size"]

    model = BP_MLP(
        hidden_layers=model_kw["hidden_layers"],
        input_size=input_size,
        output_size=n_outputs,
        batch_size=batch_size,
        snn_params=snn_kw,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=tr_kw["lr"])

    N_train = X_train.shape[0]
    history = []
    best_val_acc = -1.0
    best_state   = None
    t0 = time.time()

    for epoch in range(tr_kw["epochs"]):
        model.train()
        perm = torch.randperm(N_train)
        epoch_loss = 0.0
        correct    = 0
        n_seen     = 0

        for i in range(0, N_train - batch_size + 1, batch_size):   # drop last incomplete
            idx = perm[i:i + batch_size]
            xb  = X_train[idx].float().to(device)   # [B, T, F]
            yb  = y_train[idx].to(device)

            optimizer.zero_grad()
            logits = _forward_sequence(model, xb, device)   # [B, n_outputs]
            loss   = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

            B = xb.shape[0]
            epoch_loss += loss.item() * B
            correct    += int((logits.argmax(1) == yb).sum())
            n_seen     += B

        train_acc  = correct / max(1, n_seen)
        train_loss = epoch_loss / max(1, n_seen)
        val_acc, val_ce = _evaluate(model, X_val, y_val, batch_size, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        rec = {
            "epoch":      epoch,
            "train_acc":  train_acc,
            "train_loss": train_loss,
            "val_acc":    val_acc,
            "val_ce":     val_ce,
            "elapsed_s":  time.time() - t0,
        }
        history.append(rec)
        if on_epoch is not None:
            on_epoch(rec)

        print(f"epoch {epoch:4d} | train={train_acc:.1%}  val={val_acc:.1%}  "
              f"loss={train_loss:.4f}  {rec['elapsed_s']:.0f}s")

    # Restore best checkpoint and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)

    train_acc_final, _ = _evaluate(model, X_train, y_train, batch_size, device)
    val_acc_final,   _ = _evaluate(model, X_val,   y_val,   batch_size, device)
    test_acc,        _ = _evaluate(model, X_test,  y_test,  batch_size, device)

    print(f"\n  train={train_acc_final:.1%}  val={val_acc_final:.1%}  test={test_acc:.1%}  "
          f"time={time.time()-t0:.0f}s")

    result = dict(
        task=task,
        train_acc=train_acc_final,
        val_acc=val_acc_final,
        test_acc=test_acc,
        elapsed_s=time.time() - t0,
        history=history,
        config=config,
    )

    out_dir = exp_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    torch.save(best_state, out_dir / "model.pt")
    print(f"  saved → {out_dir}")

    return result
