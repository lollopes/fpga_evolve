#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _sweep import run_sweep

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-id", default=None, help="Attach to existing sweep instead of creating one")
    p.add_argument("--create-only", action="store_true", help="Register sweep on wandb and print ID, then exit")
    args = p.parse_args()

    exp_dir     = Path(__file__).parent
    base_config = json.loads((exp_dir / "config.json").read_text())
    sweep_json  = json.loads((exp_dir / "sweep.json").read_text())
    run_sweep(base_config, sweep_json, exp_dir, sweep_id=args.sweep_id, create_only=args.create_only)
