"""
visuals.py — bake a random Lattice3DNetwork into a static JS scene file.

Run this from the repo root or from anywhere — it locates its own folder.

    python viewer/visuals.py

It writes ``lattice_scene.js`` next to ``viewer.html`` (both in this folder).
The HTML loads the JS via ``<script src>`` (works from file:// — no local
server required) and renders the lattice on a plain 2D canvas with
hand-rolled 3D rotation. No PyVista, no trame, no three.js, no build step.

The exported scene is a flat list of nodes and a flat list of edges. Each
edge is one MUX selector resolved to its (3D source pos, target node pos,
connection kind). Connection kinds correspond to the entries of the 16-entry
pool used by ``src/lattice.py``.

To regenerate with different settings, edit the constants in the
``if __name__ == "__main__":`` block and rerun.
"""

import json
import sys
from pathlib import Path

import torch

HERE      = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.append(str(REPO_ROOT))

from src.lattice import Lattice3DNetwork


# ---------------------------------------------------------------------------
# Visual layout constants (must match viewer.html's expectations)
# ---------------------------------------------------------------------------

SPACING_X = 4.0
SPACING_Y = 4.0
SPACING_Z = 5.0

# E / I / O offsets relative to the module center (matches plot_lattice.py).
NODE_OFFSETS = {
    0: (-0.55, -0.35, 0.0),  # E
    1: ( 0.55, -0.35, 0.0),  # I
    2: ( 0.00,  0.55, 0.0),  # O
}
NODE_NAMES  = ["E", "I", "O"]
NODE_COLORS = ["#4169E1", "#DC143C", "#FF8C00"]  # royalblue, crimson, darkorange

KIND_COLORS = {
    "lateral":       "#88c0d0",
    "vertical_up":   "#a3be8c",
    "vertical_down": "#bf616a",
    "self":          "#e5e7eb",
    "feedforward":   "#ebcb8b",
    "distal0":       "#b48ead",
    "distal1":       "#d08770",
    "identity0":     "#5e81ac",
    "identity1":     "#5e81ac",
    "positional_x":  "#4c566a",
    "positional_y":  "#4c566a",
    "zero":          "#3b4252",
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def module_center(x: int, y: int, z: int) -> list:
    return [x * SPACING_X, y * SPACING_Y, z * SPACING_Z]


def node_pos(x: int, y: int, z: int, n: int) -> list:
    cx, cy, cz = module_center(x, y, z)
    ox, oy, oz = NODE_OFFSETS[n]
    return [cx + ox, cy + oy, cz + oz]


def input_pos(x: int, y: int) -> list:
    """Position for the input neuron sitting below z=0 at column (x, y)."""
    return [x * SPACING_X, y * SPACING_Y, -SPACING_Z]


# ---------------------------------------------------------------------------
# Pool index -> (source 3D position, kind)
# ---------------------------------------------------------------------------

def resolve_source(net: Lattice3DNetwork, x: int, y: int, z: int, p: int):
    """
    Map a pool index ``p ∈ [0, 16)`` for site ``(x, y, z)`` to a 3D source
    position and a connection-kind string. Returns ``(None, "zero")`` for
    out-of-bounds neighbours and disabled pool entries (so the caller can
    drop them or render them as zero/dead).
    """
    if p < 6:
        # Pool layout (from lattice.py):
        #   P[0] left  (x-1)        P[1] right (x+1)
        #   P[2] up    (y-1)        P[3] down  (y+1)
        #   P[4] above (z-1) → vertical_up
        #   P[5] below (z+1) → vertical_down
        # 4 lateral + 2 vertical kinds so the viewer can toggle them
        # independently. (Convention matches plot_lattice.py.)
        offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        kinds   = ["lateral", "lateral", "lateral", "lateral",
                   "vertical_up", "vertical_down"]
        dx, dy, dz = offsets[p]
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < net.W and 0 <= ny < net.H and 0 <= nz < net.Z:
            return module_center(nx, ny, nz), kinds[p]
        return None, "zero"

    if p == 6:
        return module_center(x, y, z), "self"

    if p == 7:
        if z == 0:
            return input_pos(x, y), "feedforward"
        return None, "zero"

    if p in (8, 9):
        if not net.use_distal:
            return None, "zero"
        sz, sy, sx = net.distal_idx[z, y, x, p - 8].tolist()
        return module_center(int(sx), int(sy), int(sz)), f"distal{p - 8}"

    if p in (10, 11):
        if not net.use_identity:
            return None, "zero"
        return module_center(x, y, z), f"identity{p - 10}"

    if p in (12, 13):
        if not net.use_positional_cues:
            return None, "zero"
        return module_center(x, y, z), "positional_x" if p == 12 else "positional_y"

    return None, "zero"


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------

def build_scene(net: Lattice3DNetwork, batch_index: int) -> dict:
    nodes = []
    edges = []

    for z in range(net.Z):
        for y in range(net.H):
            for x in range(net.W):
                cx, cy, cz = module_center(x, y, z)
                nodes.append({
                    "pos": [cx, cy, cz],
                    "kind": "module",
                    "color": "#4b5563",
                    "label": f"({x},{y},{z})",
                    "z": z,
                    "site": [x, y, z],
                    "radius": 1.4,
                })
                for n in range(3):
                    nodes.append({
                        "pos": node_pos(x, y, z, n),
                        "kind": "node",
                        "color": NODE_COLORS[n],
                        "label": f"{NODE_NAMES[n]}@({x},{y},{z})",
                        "z": z,
                        "node_idx": n,
                        "site": [x, y, z],
                        "radius": 2.6,
                    })

    for y in range(net.H):
        for x in range(net.W):
            nodes.append({
                "pos": input_pos(x, y),
                "kind": "input",
                "color": "#fcd34d",
                "label": f"in({x},{y})",
                "z": -1,
                "site": [x, y],
                "radius": 2.0,
            })

    sel = net.layer_sel  # [Z, B, 3, k]
    for z in range(net.Z):
        for y in range(net.H):
            for x in range(net.W):
                for n in range(3):
                    target = node_pos(x, y, z, n)
                    for s in range(net.k):
                        p = int(sel[z, batch_index, n, s].item())
                        src, kind = resolve_source(net, x, y, z, p)
                        if src is None:
                            continue
                        edges.append({
                            "from": src,
                            "to": target,
                            "kind": kind,
                            "color": KIND_COLORS.get(kind, "#888888"),
                            "z": z,
                            "node_idx": n,
                            "selector_idx": s,
                            "pool_idx": p,
                        })

    return {
        "Z": net.Z, "H": net.H, "W": net.W, "k": net.k,
        "pool_size": int(net.POOL_SIZE),
        "spacing": [SPACING_X, SPACING_Y, SPACING_Z],
        "node_offsets": [list(NODE_OFFSETS[i]) for i in range(3)],
        "node_names": NODE_NAMES,
        "node_colors": NODE_COLORS,
        "kind_colors": KIND_COLORS,
        "use_identity": bool(net.use_identity),
        "use_positional_cues": bool(net.use_positional_cues),
        "use_distal": bool(net.use_distal),
        "nodes": nodes,
        "edges": edges,
        "layer_sel":     net.layer_sel[:, batch_index].tolist(),   # [Z, 3, k]
        "layer_lut":     net.layer_lut[:, batch_index].tolist(),   # [Z, 3, 2**k]
        "distal_idx":    net.distal_idx.tolist(),                  # [Z, H, W, 2, 3]  (z, y, x)
        "identity_bits": net.identity_bits.tolist(),               # [Z, H, W, 2]
    }


def write_scene_js(scene: dict, out_path: Path) -> None:
    payload = "window.LATTICE_DATA = " + json.dumps(scene) + ";\n"
    out_path.write_text(payload, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    Z = 3
    H = 4
    W = 4
    B = 1
    K_ARITY = 2
    SEED = 0

    USE_IDENTITY = True
    USE_POSITIONAL_CUES = False
    USE_DISTAL = True

    BATCH_INDEX = 0
    OUT_PATH = HERE / "lattice_scene.js"

    torch.manual_seed(SEED)
    net = Lattice3DNetwork.random(Z=Z, H=H, W=W, B=B, k=K_ARITY, seed=SEED)
    net.use_identity = USE_IDENTITY
    net.use_positional_cues = USE_POSITIONAL_CUES
    net.use_distal = USE_DISTAL

    scene = build_scene(net, batch_index=BATCH_INDEX)
    write_scene_js(scene, OUT_PATH)

    n_nodes = len(scene["nodes"])
    n_edges = len(scene["edges"])
    n_zero  = sum(1 for e in scene["edges"] if e["kind"] == "zero")
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}  "
          f"({n_nodes} nodes, {n_edges} edges, {n_zero} zero)")
    print(f"  Z={Z}  H={H}  W={W}  k={K_ARITY}  seed={SEED}")
    print(f"  open {(HERE / 'viewer.html').relative_to(REPO_ROOT)} in a browser.")
