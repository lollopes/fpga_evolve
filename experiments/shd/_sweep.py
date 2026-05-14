"""
experiments/shd/_sweep.py
Shared wandb sweep logic for all SHD experiments.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _runner import run_experiment


def _apply_overrides(base: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(base)
    for key, value in overrides.items():
        parts = key.split(".")
        d = cfg
        for part in parts[:-1]:
            d = d[part]
        d[parts[-1]] = value
    return cfg


def run_sweep(base_config: dict, sweep_json: dict, exp_dir: Path,
              sweep_id: str | None = None, create_only: bool = False) -> None:

    project   = sweep_json.get("project", f"eio-neat-shd-{base_config['task']}")
    sw_gen    = sweep_json.get("generations", 60)
    run_count = sweep_json.get("count", 50)
    device    = torch.device(sweep_json.get("device", "cpu"))

    if sweep_id is None:
        sweep_cfg = {
            "method":     sweep_json["method"],
            "metric":     sweep_json["metric"],
            "parameters": sweep_json["parameters"],
        }
        if "early_terminate" in sweep_json:
            sweep_cfg["early_terminate"] = sweep_json["early_terminate"]

        sweep_id = wandb.sweep(sweep_cfg, project=project)
        print(f"SWEEP_ID={sweep_id}")
        print(f"  project : {project}")
        print(f"  attach  : wandb agent {project}/{sweep_id}")

    if create_only:
        return

    def train():
        with wandb.init() as run:
            overrides = {f"neat.{k}": v for k, v in wandb.config.items()}
            overrides["neat.generations"] = sw_gen
            overrides["neat.n_workers"]   = 1  # parallelism is at SLURM job level

            cfg     = _apply_overrides(base_config, overrides)
            out_dir = exp_dir / "sweep_runs" / run.id

            def on_generation(rec: dict) -> None:
                wandb.log({
                    "train_acc":   rec["best_train_acc"],
                    "val_acc":     rec["best_val_acc"],
                    "ce":          rec["best_ce"],
                    "n_species":   rec["n_species"],
                    "E_nodes":     rec["best_hidden_E"],
                    "I_nodes":     rec["best_hidden_I"],
                    "connections": rec["best_connections"],
                    "silent_frac": rec["best_silent_frac"],
                }, step=rec["generation"])

            result = run_experiment(cfg, device, out_dir, on_generation=on_generation)

            wandb.summary["best_val_acc"]   = result["val_acc"]
            wandb.summary["best_train_acc"] = result["train_acc"]
            wandb.summary["test_acc"]       = result["test_acc"]

    wandb.agent(sweep_id, function=train, project=project, count=run_count)
