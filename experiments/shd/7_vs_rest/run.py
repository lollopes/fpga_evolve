#!/usr/bin/env python3
import argparse
import json
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _runner import run_experiment

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    exp_dir = Path(__file__).parent
    config  = json.loads((exp_dir / "config.json").read_text())
    run_experiment(config, torch.device(args.device), exp_dir)
