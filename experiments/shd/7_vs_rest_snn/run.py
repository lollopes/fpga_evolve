#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.shd._runner_snn import run_experiment  # noqa: E402

if __name__ == "__main__":
    exp_dir = Path(__file__).parent
    config  = json.loads((exp_dir / "config.json").read_text())
    device  = torch.device("cpu")

    run_experiment(config, device, exp_dir)
