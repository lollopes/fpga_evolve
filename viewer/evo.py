"""
evo.py — export a Lattice3DNetwork evolution run into a static JS data file
for evo.html.

Run from repo root:
    python viewer/evo.py --experiment experiments/0_vs_1

Writes viewer/evo_data.js which is loaded by viewer/evo.html (no server needed).
"""

import json
import sys
from pathlib import Path

import torch

HERE      = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.append(str(REPO_ROOT))
sys.path.append(str(HERE))   # so we can import visuals directly

from src.evolution import build_lattice
from visuals import build_scene   # reuse existing scene builder


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export an evolution run to evo_data.js for evo.html."
    )
    parser.add_argument(
        "--experiment", metavar="DIR", required=True,
        help="Path to an experiment directory (e.g. experiments/0_vs_1).",
    )
    parser.add_argument(
        "--out", metavar="FILE", default=None,
        help="Output path (default: viewer/evo_data.js).",
    )
    args = parser.parse_args()

    exp_dir  = Path(args.experiment).resolve()
    out_path = Path(args.out).resolve() if args.out else HERE / "evo_data.js"

    # Load champion checkpoint
    ckpt = torch.load(exp_dir / "results" / "best_genome.pt", map_location="cpu")

    genome = ckpt["genome"]           # [Z, 3, k+2**k]  (batch dim stripped when saved)
    if genome.dim() == 3:
        genome = genome.unsqueeze(0)  # → [1, Z, 3, k+2**k]

    Z, H, W, K = ckpt["Z"], ckpt["H"], ckpt["W"], ckpt["k"]

    net = build_lattice(
        genome, Z, H, W, K,
        distal_seed         = ckpt["distal_seed"],
        identity_seed       = ckpt["identity_seed"],
        use_identity        = ckpt["use_identity"],
        use_positional_cues = ckpt["use_positional_cues"],
        use_distal          = ckpt["use_distal"],
        device              = torch.device("cpu"),
    )

    # Load per-generation history
    history_path = exp_dir / "results" / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []

    # Build the 3D scene for the champion network
    scene = build_scene(net, batch_index=0)

    # Bundle everything into a single JS payload
    data = {
        "experiment": exp_dir.name,
        "task":       ckpt.get("task", ""),
        "train_acc":  ckpt.get("train_acc"),
        "val_acc":    ckpt.get("val_acc"),
        "history":    history,
        "scene":      scene,
    }

    out_path.write_text("window.EVO_DATA = " + json.dumps(data) + ";\n", encoding="utf-8")

    print(f"wrote {out_path}")
    print(f"  experiment={data['experiment']}  task={data['task']}")
    if data["train_acc"] is not None:
        print(f"  train={data['train_acc']:.1%}  val={data['val_acc']:.1%}")
    print(f"  {len(history)} generations")
    print(f"  open {HERE / 'evo.html'} in a browser.")
