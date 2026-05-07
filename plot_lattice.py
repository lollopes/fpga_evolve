# visualize_lattice_pyvista.py
#
# Usage:
#   python visualize_lattice_pyvista.py
#
# Requirements:
#   pip install pyvista numpy torch
#
# Assumptions:
#   - your project has src/lattice.py
#   - Lattice3DNetwork.random(...) exists
#   - net.layer_sel has shape [Z, B, N_nodes, k]
#   - pool size is 16 with the semantics described in your architecture
#
# Notes:
#   - This visualizer is intentionally robust to both:
#       * old 12-node-per-module lattice
#       * newer 3-node E/I/O microcircuit lattice
#   - It visualizes genome-selected edges, not runtime activity dynamics.

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.render_env import configure_pyvista_env

# Decide the rendering path before importing PyVista/VTK.
FORCE_OFFSCREEN = os.environ.get("FORCE_OFFSCREEN")
DEFAULT_OFFSCREEN = FORCE_OFFSCREEN != "0"
configure_pyvista_env(offscreen=DEFAULT_OFFSCREEN)

import numpy as np
import pyvista as pv
import torch

from src.lattice import Lattice3DNetwork


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Small lattice first, otherwise the plot gets too dense.
Z = 3
H = 4
W = 4
B = 1

# If your random() supports K or other args, adapt below if needed.
RANDOM_SEED = 0

# Show only a subregion if needed
X_RANGE = None   # e.g. (0, 3)
Y_RANGE = None   # e.g. (0, 3)
Z_RANGE = None   # e.g. (0, 2)

# Visual spacing
MODULE_SPACING_X = 4.0
MODULE_SPACING_Y = 4.0
MODULE_SPACING_Z = 5.0

# Internal node offsets relative to module center
INTERNAL_NODE_RADIUS = 0.22
MODULE_RADIUS = 0.13
EDGE_TUBE_RADIUS = 0.03
EDGE_ARROW_SCALE = 0.35
EDGE_ARROW_RADIUS = 0.12
EDGE_ARROW_TARGET_GAP = 0.10
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1000

# Whether to draw labels
SHOW_MODULE_LABELS = False
SHOW_NODE_LABELS = False

# Whether to draw only external selected edges, or also internal helper edges
DRAW_INTERNAL_HELPER_EDGES = True

# Whether to draw a spike-input layer below z=0
DRAW_INPUT_NEURONS = True
INPUT_NEURON_COLOR = "gold"

CONNECTION_KINDS = [
    "lateral",
    "vertical_up",
    "vertical_down",
    "self",
    "feedforward",
    "distal0",
    "distal1",
    "identity0",
    "identity1",
    "positional_x",
    "positional_y",
    "zero",
    "internal",
]

CONNECTION_LABELS = {
    "lateral": "lateral",
    "vertical_up": "vertical up",
    "vertical_down": "vertical down",
    "self": "self",
    "feedforward": "feedforward",
    "distal0": "distal 0",
    "distal1": "distal 1",
    "identity0": "identity 0",
    "identity1": "identity 1",
    "positional_x": "positional x",
    "positional_y": "positional y",
    "zero": "zero / out-of-bounds",
    "internal": "internal",
}

# Rendering behavior:
# - Default to off-screen because DISPLAY can be present but still unusable for VTK/GL.
# - You can override with FORCE_OFFSCREEN=1 or FORCE_OFFSCREEN=0.
SCREENSHOT_PATH = os.environ.get("PLOT_SCREENSHOT_PATH", "lattice_plot.png")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

@dataclass
class SourceSpec:
    kind: str
    coord: Optional[Tuple[int, int, int]] = None  # (x, y, z)
    label: Optional[str] = None


@dataclass
class LatticeScene:
    plotter: pv.Plotter
    connection_actors: Dict[str, List[Any]]
    node_items: List[Tuple[str, str]]
    connection_items: List[Tuple[str, str]]
    title: str
    n_nodes: int
    k: int
    node_actors: Dict[Tuple[int, int], List[Any]]  # (z, node_idx) -> sphere actors
    node_names: List[str]
    module_pos: Dict[Tuple[int, int, int], Any]
    internal_pos: Dict[Tuple[int, int, int, int], Any]
    layer_connection_actors: Dict[int, Dict[str, List[Any]]]  # z -> kind -> actors


def get_node_names(n_nodes: int) -> List[str]:
    """
    Robust naming:
    - if 3 nodes: E, I, O
    - if 12 nodes: E0..E8, I0..I2
    - otherwise generic N0..N{n-1}
    """
    if n_nodes == 3:
        return ["E", "I", "O"]
    if n_nodes == 12:
        return [f"E{i}" for i in range(9)] + [f"I{i}" for i in range(3)]
    return [f"N{i}" for i in range(n_nodes)]


def get_node_legend_items(n_nodes: int) -> List[Tuple[str, str]]:
    if n_nodes == 3:
        items = [("E node", node_color("E")), ("I node", node_color("I")), ("O node", node_color("O"))]
    elif n_nodes == 12:
        items = [("E nodes", node_color("E0")), ("I nodes", node_color("I0"))]
    else:
        items = [("Generic node", node_color("N0"))]
    if DRAW_INPUT_NEURONS:
        items.append(("Input neuron", INPUT_NEURON_COLOR))
    return items


def get_connection_legend_items() -> List[Tuple[str, str]]:
    return [(CONNECTION_LABELS[kind], edge_color(kind)) for kind in CONNECTION_KINDS]


def in_range(v: int, rng: Optional[Tuple[int, int]]) -> bool:
    if rng is None:
        return True
    a, b = rng
    return a <= v < b


def keep_module(x: int, y: int, z: int) -> bool:
    return in_range(x, X_RANGE) and in_range(y, Y_RANGE) and in_range(z, Z_RANGE)


def module_center(x: int, y: int, z: int) -> np.ndarray:
    return np.array([
        x * MODULE_SPACING_X,
        y * MODULE_SPACING_Y,
        z * MODULE_SPACING_Z,
    ], dtype=float)


def internal_offsets(n_nodes: int) -> Dict[int, np.ndarray]:
    """
    Place internal nodes around the module center.
    For 3-node case: triangle.
    For 12-node case: 3x4-ish arrangement.
    Generic fallback: small circle.
    """
    offsets = {}

    if n_nodes == 3:
        offsets[0] = np.array([-0.55, -0.35,  0.00])  # E
        offsets[1] = np.array([ 0.55, -0.35,  0.00])  # I
        offsets[2] = np.array([ 0.00,  0.55,  0.00])  # O
        return offsets

    if n_nodes == 12:
        # 9 E nodes in a 3x3-ish pattern + 3 I nodes on top
        coords = [
            (-0.8, -0.8, 0.0), (0.0, -0.8, 0.0), (0.8, -0.8, 0.0),
            (-0.8,  0.0, 0.0), (0.0,  0.0, 0.0), (0.8,  0.0, 0.0),
            (-0.8,  0.8, 0.0), (0.0,  0.8, 0.0), (0.8,  0.8, 0.0),
            (-0.8,  1.7, 0.0), (0.0,  1.7, 0.0), (0.8,  1.7, 0.0),
        ]
        for i, c in enumerate(coords):
            offsets[i] = np.array(c)
        return offsets

    # Generic circular placement
    r = 0.9
    for i in range(n_nodes):
        theta = 2.0 * math.pi * i / n_nodes
        offsets[i] = np.array([r * math.cos(theta), r * math.sin(theta), 0.0])
    return offsets


def node_color(node_name: str) -> str:
    if node_name.startswith("E"):
        return "royalblue"
    if node_name.startswith("I"):
        return "crimson"
    if node_name.startswith("O"):
        return "darkorange"
    return "white"


def edge_color(kind: str) -> str:
    colors = {
        "lateral": "deepskyblue",
        "vertical_up": "limegreen",
        "vertical_down": "forestgreen",
        "self": "gray",
        "feedforward": "gold",
        "distal0": "magenta",
        "distal1": "violet",
        "identity0": "orange",
        "identity1": "sandybrown",
        "positional_x": "cyan",
        "positional_y": "turquoise",
        "zero": "dimgray",
        "internal": "white",
        "unknown": "lightgray",
    }
    return colors.get(kind, "lightgray")


def add_tube(plotter: pv.Plotter, p0: np.ndarray, p1: np.ndarray, color: str, radius: float = EDGE_TUBE_RADIUS):
    line = pv.Line(p0, p1)
    tube = line.tube(radius=radius)
    return plotter.add_mesh(tube, color=color)


def add_arrow(plotter: pv.Plotter, p0: np.ndarray, p1: np.ndarray, color: str, scale: float = 0.35):
    direction = p1 - p0
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return None
    unit = direction / norm
    height = min(scale, max(norm - EDGE_ARROW_TARGET_GAP, scale * 0.5))
    center = p1 - unit * (0.5 * height + EDGE_ARROW_TARGET_GAP)
    arrow = pv.Cone(
        center=center,
        direction=unit,
        height=height,
        radius=EDGE_ARROW_RADIUS,
        resolution=24,
    )
    return plotter.add_mesh(arrow, color=color)


def get_output_node_index(node_names: List[str]) -> Optional[int]:
    for idx, name in enumerate(node_names):
        if name == "O":
            return idx
    return None


def get_external_source_point(
    module_pos: Dict[Tuple[int, int, int], np.ndarray],
    internal_pos: Dict[Tuple[int, int, int, int], np.ndarray],
    coord: Tuple[int, int, int],
    node_names: List[str],
) -> np.ndarray:
    output_idx = get_output_node_index(node_names)
    if output_idx is not None:
        return internal_pos[(coord[0], coord[1], coord[2], output_idx)]
    return module_pos[coord]


def register_actor(actor_groups: Dict[str, List[object]], kind: str, actor: object) -> None:
    if actor is None:
        return
    actor_groups.setdefault(kind, []).append(actor)


def infer_offscreen() -> bool:
    if FORCE_OFFSCREEN == "1":
        return True
    if FORCE_OFFSCREEN == "0":
        return False
    return True


def get_activity_source_anchor(center: np.ndarray, kind: str) -> np.ndarray:
    """
    Anchor positions for non-module sources like self/feedforward/identity.
    """
    offsets = {
        "self":         np.array([0.0,  0.0, -1.2]),
        "feedforward":  np.array([0.0, -1.4,  0.0]),
        "distal0":      np.array([-1.2, 0.0,  0.0]),
        "distal1":      np.array([ 1.2, 0.0,  0.0]),
        "identity0":    np.array([-1.0, 1.1,  0.0]),
        "identity1":    np.array([ 1.0, 1.1,  0.0]),
        "positional_x": np.array([-1.2, -1.0, 0.0]),
        "positional_y": np.array([ 1.2, -1.0, 0.0]),
        "zero":         np.array([0.0, 0.0,  1.2]),
        "unknown":      np.array([0.0, 0.0,  1.2]),
    }
    return center + offsets.get(kind, np.array([0.0, 0.0, 1.2]))


def pool_source_for_module(
    net: Lattice3DNetwork,
    x: int,
    y: int,
    z: int,
    node_idx: int,
    pool_idx: int,
    n_nodes: int,
) -> SourceSpec:
    """
    Decode what a pool index means for module(x,y,z).

    This follows the pool convention from your new 3D architecture.
    It also includes an optional special case:
    if n_nodes == 3 and node_idx == 2 (O node) and pool_idx == 0,
    we interpret it as the internal E->O coupling from microcircuit.py.

    If your current lattice implementation does NOT actually replace P[0] with E
    for O, remove that special case.
    """
    # Special case for the minimal 3-node microcircuit:
    # O node sees E injected at P[0].
    if n_nodes == 3 and node_idx == 2 and pool_idx == 0:
        return SourceSpec(kind="internal", coord=(x, y, z), label="E")

    if pool_idx == 0:
        src = (x - 1, y, z)
        return SourceSpec("lateral", src, "left")
    if pool_idx == 1:
        src = (x + 1, y, z)
        return SourceSpec("lateral", src, "right")
    if pool_idx == 2:
        src = (x, y - 1, z)
        return SourceSpec("lateral", src, "up")
    if pool_idx == 3:
        src = (x, y + 1, z)
        return SourceSpec("lateral", src, "down")
    if pool_idx == 4:
        src = (x, y, z - 1)
        return SourceSpec("vertical_up", src, "z-1")
    if pool_idx == 5:
        src = (x, y, z + 1)
        return SourceSpec("vertical_down", src, "z+1")
    if pool_idx == 6:
        return SourceSpec("self", (x, y, z), "self_prev")
    if pool_idx == 7:
        if z == 0 and DRAW_INPUT_NEURONS:
            return SourceSpec("feedforward", (x, y, -1), "input")
        return SourceSpec("feedforward", None, "feedforward")
    if pool_idx == 8:
        # Distal 0
        if hasattr(net, "distal_idx"):
            dz, dy, dx = net.distal_idx[z, y, x, 0].tolist()
            return SourceSpec("distal0", (int(dx), int(dy), int(dz)), "distal0")
        return SourceSpec("distal0", None, "distal0")
    if pool_idx == 9:
        # Distal 1
        if hasattr(net, "distal_idx"):
            dz, dy, dx = net.distal_idx[z, y, x, 1].tolist()
            return SourceSpec("distal1", (int(dx), int(dy), int(dz)), "distal1")
        return SourceSpec("distal1", None, "distal1")
    if pool_idx == 10:
        return SourceSpec("identity0", None, "id0")
    if pool_idx == 11:
        return SourceSpec("identity1", None, "id1")
    if pool_idx == 12:
        return SourceSpec("positional_x", None, "xcue")
    if pool_idx == 13:
        return SourceSpec("positional_y", None, "ycue")
    if pool_idx == 14 or pool_idx == 15:
        return SourceSpec("zero", None, "zero")

    return SourceSpec("unknown", None, f"P[{pool_idx}]")


# ---------------------------------------------------------------------
# Connection helpers (can be called independently for live updates)
# ---------------------------------------------------------------------

def clear_connections(
    plotter: pv.Plotter,
    connection_actors: Dict[str, List[Any]],
    layer_connection_actors: Optional[Dict[int, Dict[str, List[Any]]]] = None,
    z_only: Optional[int] = None,
) -> None:
    if z_only is not None and layer_connection_actors is not None:
        layer_actors = layer_connection_actors.get(z_only, {})
        remove_ids = {id(a) for actors in layer_actors.values() for a in actors}
        for actors in layer_actors.values():
            for actor in actors:
                plotter.remove_actor(actor)
        for kind in connection_actors:
            connection_actors[kind] = [a for a in connection_actors[kind] if id(a) not in remove_ids]
        for kind in layer_actors:
            layer_actors[kind].clear()
    else:
        for kind in connection_actors:
            for actor in connection_actors[kind]:
                plotter.remove_actor(actor)
            connection_actors[kind].clear()
        if layer_connection_actors is not None:
            for z_actors in layer_connection_actors.values():
                for actors in z_actors.values():
                    actors.clear()


def draw_connections(
    plotter: pv.Plotter,
    net: "Lattice3DNetwork",
    batch_index: int,
    connection_actors: Dict[str, List[Any]],
    module_pos: Dict[Tuple[int, int, int], Any],
    internal_pos: Dict[Tuple[int, int, int, int], Any],
    node_names: List[str],
    n_nodes: int,
    layer_connection_actors: Optional[Dict[int, Dict[str, List[Any]]]] = None,
    z_only: Optional[int] = None,
) -> None:
    Z_ = net.layer_sel.shape[0]
    layer_sel = net.layer_sel[:, batch_index].detach().cpu().numpy()

    for z in range(Z_):
        if z_only is not None and z != z_only:
            continue

        if layer_connection_actors is not None:
            layer_connection_actors.setdefault(z, {kind: [] for kind in CONNECTION_KINDS})

        def _reg(kind: str, actor: Any, _z: int = z) -> None:
            register_actor(connection_actors, kind, actor)
            if layer_connection_actors is not None:
                layer_connection_actors[_z][kind].append(actor)

        for y in range(net.H):
            for x in range(net.W):
                if not keep_module(x, y, z):
                    continue
                for node_idx in range(n_nodes):
                    target = internal_pos[(x, y, z, node_idx)]
                    selectors = layer_sel[z, node_idx]
                    for pool_idx in selectors:
                        src = pool_source_for_module(net, x, y, z, node_idx, int(pool_idx), n_nodes)

                        if src.kind == "internal":
                            if src.label == "E":
                                source_point = internal_pos[(x, y, z, 0)]
                                _reg("internal", add_tube(plotter, source_point, target, edge_color("internal")))
                                _reg("internal", add_arrow(plotter, source_point, target, edge_color("internal")))
                            continue

                        if src.coord is not None:
                            sx, sy, sz = src.coord
                            # Input-neuron layer lives at virtual z = -1
                            if sz == -1:
                                if (sx, sy, sz) in module_pos:
                                    source_point = internal_pos[(sx, sy, sz, 0)]
                                    _reg(src.kind, add_tube(plotter, source_point, target, edge_color(src.kind)))
                                    _reg(src.kind, add_arrow(plotter, source_point, target, edge_color(src.kind), scale=EDGE_ARROW_SCALE))
                                continue
                            if not (0 <= sx < net.W and 0 <= sy < net.H and 0 <= sz < Z_):
                                anchor = get_activity_source_anchor(module_pos[(x, y, z)], "zero")
                                sph = pv.Sphere(radius=0.08, center=anchor)
                                _reg("zero", plotter.add_mesh(sph, color=edge_color("zero")))
                                _reg("zero", add_tube(plotter, anchor, target, edge_color("zero"), radius=0.02))
                                _reg("zero", add_arrow(plotter, anchor, target, edge_color("zero"), scale=0.2))
                                continue
                            if not keep_module(sx, sy, sz):
                                continue
                            source_point = get_external_source_point(module_pos, internal_pos, (sx, sy, sz), node_names)
                            _reg(src.kind, add_tube(plotter, source_point, target, edge_color(src.kind)))
                            _reg(src.kind, add_arrow(plotter, source_point, target, edge_color(src.kind), scale=EDGE_ARROW_SCALE))
                        else:
                            anchor = get_activity_source_anchor(module_pos[(x, y, z)], src.kind)
                            sph = pv.Sphere(radius=0.08, center=anchor)
                            _reg(src.kind, plotter.add_mesh(sph, color=edge_color(src.kind)))
                            _reg(src.kind, add_tube(plotter, anchor, target, edge_color(src.kind), radius=0.02))
                            _reg(src.kind, add_arrow(plotter, anchor, target, edge_color(src.kind), scale=0.2))


# ---------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------

def build_scene(
    net: Lattice3DNetwork,
    batch_index: int = 0,
    offscreen: Optional[bool] = None,
    window_size: Tuple[int, int] = (WINDOW_WIDTH, WINDOW_HEIGHT),
) -> LatticeScene:
    # Infer lattice/genome structure
    # Expect layer_sel shape [Z, B, N_nodes, k]
    if not hasattr(net, "layer_sel"):
        raise AttributeError("net.layer_sel not found. The visualizer expects layer_sel in the lattice.")

    Z_, B_, n_nodes, k = net.layer_sel.shape
    node_names = get_node_names(n_nodes)
    offsets = internal_offsets(n_nodes)

    print("Detected:")
    print(f"  Z={Z_}, B={B_}, n_nodes={n_nodes}, k={k}")
    print(f"  node names: {node_names}")

    if offscreen is None:
        offscreen = infer_offscreen()

    print(
        "Rendering mode:",
        "off-screen" if offscreen else "interactive",
        f"(DISPLAY={os.environ.get('DISPLAY')!r}, FORCE_OFFSCREEN={FORCE_OFFSCREEN!r})",
    )

    pv.OFF_SCREEN = offscreen
    plotter = pv.Plotter(off_screen=offscreen, window_size=window_size)
    plotter.set_background("black")
    connection_actors: Dict[str, List[Any]] = {kind: [] for kind in CONNECTION_KINDS}
    layer_connection_actors: Dict[int, Dict[str, List[Any]]] = {}
    node_actors: Dict[Tuple[int, int], List[Any]] = {}

    # Cache positions
    module_pos: Dict[Tuple[int, int, int], np.ndarray] = {}
    internal_pos: Dict[Tuple[int, int, int, int], np.ndarray] = {}

    # -----------------------------------------------------------------
    # Draw module centers and internal nodes
    # -----------------------------------------------------------------
    module_points = []
    module_labels = []

    for z in range(Z_):
        for y in range(net.H):
            for x in range(net.W):
                if not keep_module(x, y, z):
                    continue

                c = module_center(x, y, z)
                module_pos[(x, y, z)] = c
                module_points.append(c)
                module_labels.append(f"M({x},{y},{z})")

                # Internal nodes
                for i in range(n_nodes):
                    p = c + offsets[i]
                    internal_pos[(x, y, z, i)] = p
                    sph = pv.Sphere(radius=INTERNAL_NODE_RADIUS, center=p)
                    actor = plotter.add_mesh(sph, color=node_color(node_names[i]), smooth_shading=True)
                    node_actors.setdefault((z, i), []).append(actor)

                    if SHOW_NODE_LABELS:
                        plotter.add_point_labels(
                            np.array([p]),
                            [node_names[i]],
                            point_size=1,
                            font_size=10,
                            text_color="white",
                            shape=None,
                            always_visible=True,
                        )

                # Optional helper edges inside the module
                if DRAW_INTERNAL_HELPER_EDGES and n_nodes == 3:
                    # Just a simple visual hint for the 3-node motif
                    # E -> O helper
                    register_actor(
                        connection_actors,
                        "internal",
                        add_tube(
                            plotter,
                            internal_pos[(x, y, z, 0)],
                            internal_pos[(x, y, z, 2)],
                            "white",
                            radius=0.02,
                        ),
                    )
                    register_actor(
                        connection_actors,
                        "internal",
                        add_arrow(
                            plotter,
                            internal_pos[(x, y, z, 0)],
                            internal_pos[(x, y, z, 2)],
                            "white",
                            scale=EDGE_ARROW_SCALE,
                        ),
                    )

    if module_points and SHOW_MODULE_LABELS:
        module_points = np.array(module_points)
        plotter.add_point_labels(
            module_points,
            module_labels,
            point_size=1,
            font_size=10,
            text_color="white",
            shape=None,
            always_visible=False,
        )

    # -----------------------------------------------------------------
    # Draw input neuron layer (spike inputs, virtual z = -1)
    # -----------------------------------------------------------------
    if DRAW_INPUT_NEURONS:
        for y in range(net.H):
            for x in range(net.W):
                if not keep_module(x, y, 0):
                    continue
                c = module_center(x, y, -1)
                module_pos[(x, y, -1)] = c
                internal_pos[(x, y, -1, 0)] = c
                sph = pv.Sphere(radius=INTERNAL_NODE_RADIUS, center=c)
                actor = plotter.add_mesh(sph, color=INPUT_NEURON_COLOR, smooth_shading=True)
                node_actors.setdefault((-1, 0), []).append(actor)
                if SHOW_NODE_LABELS:
                    plotter.add_point_labels(
                        np.array([c]),
                        [f"In({x},{y})"],
                        point_size=1,
                        font_size=10,
                        text_color="white",
                        shape=None,
                        always_visible=True,
                    )

    # -----------------------------------------------------------------
    # Draw selected edges from genome
    # -----------------------------------------------------------------
    draw_connections(plotter, net, batch_index, connection_actors, module_pos, internal_pos, node_names, n_nodes, layer_connection_actors)

    # -----------------------------------------------------------------
    # Cosmetics
    # -----------------------------------------------------------------
    plotter.add_axes()
    plotter.show_grid(color="gray")
    plotter.camera_position = "iso"

    title = (
        f"3D Lattice Visualization | "
        f"Z={Z_}, H={net.H}, W={net.W}, nodes/module={n_nodes}, k={k}"
    )
    plotter.add_text(title, position="upper_left", font_size=10, color="white")

    return LatticeScene(
        plotter=plotter,
        connection_actors=connection_actors,
        node_items=get_node_legend_items(n_nodes),
        connection_items=get_connection_legend_items(),
        title=title,
        n_nodes=n_nodes,
        k=k,
        node_actors=node_actors,
        node_names=node_names,
        module_pos=module_pos,
        internal_pos=internal_pos,
        layer_connection_actors=layer_connection_actors,
    )


def wire_input_neurons_for_plot(net: Lattice3DNetwork) -> None:
    """
    Plot-only startup: force every z=0 node's first selector to P[7]
    and set the LUT so the node fires whenever P[7]=1.

    This makes all input→z0 gold wires visible regardless of the evolved
    genome.  It mutates layer_sel / layer_lut in-place on the CPU copy
    and is called only by build_and_plot, never during training/eval.
    """
    k        = net.layer_sel.shape[-1]
    lut_size = net.layer_lut.shape[-1]   # 2**k

    # Entries in the LUT where the P[7]-carrying selector bit is 1.
    # Selector 0 is the MSB, so P[7]=1  →  address >= 2**(k-1).
    half = lut_size // 2
    I_NODE = 1  # inhibitory node index — must NOT fire on input or it suppresses O
    with torch.no_grad():
        for node in range(net.N_NODES):
            if node == I_NODE:
                continue
            # First selector → P[7]
            net.layer_sel[0, :, node, 0] = 7
            # Upper half of the LUT → output 1 (P[7]=1 addresses)
            net.layer_lut[0, :, node, half:] = 1


def apply_activity_display(
    scene: LatticeScene,
    net: "Lattice3DNetwork",
    batch_index: int,
    input_spikes=None,  # np.ndarray [H, W] long, or None
) -> None:
    """
    Colour lattice spheres to show current spike state.
    Firing modules glow warm-white (high ambient); silent ones revert to base colour.
    Actors are stored in node_actors[(z, node_idx)] in row-major (y outer, x inner) order,
    so flat_idx = y * W + x picks the right actor per site.
    """
    state_np = net.state[batch_index].detach().cpu().numpy()  # [Z, H, W, 1]

    GLOW_COLOR = (1.0, 1.0, 0.75)
    GLOW_AMB   = 0.9
    BASE_AMB   = 0.1

    def _set(actor, rgb, amb):
        prop = actor.GetProperty()
        prop.SetColor(*rgb)
        prop.SetAmbient(amb)

    for z in range(net.Z):
        for y in range(net.H):
            for x in range(net.W):
                active   = bool(state_np[z, y, x, 0])
                flat_idx = y * net.W + x
                for ni in range(scene.n_nodes):
                    actors = scene.node_actors.get((z, ni), [])
                    if flat_idx < len(actors):
                        if active:
                            _set(actors[flat_idx], GLOW_COLOR, GLOW_AMB)
                        else:
                            base = tuple(pv.Color(node_color(scene.node_names[ni])).float_rgb)
                            _set(actors[flat_idx], base, BASE_AMB)

    # Input-neuron layer
    if input_spikes is not None:
        inp_glow = (1.0, 1.0, 0.3)
        inp_base = tuple(pv.Color(INPUT_NEURON_COLOR).float_rgb)
        for y in range(net.H):
            for x in range(net.W):
                active   = bool(input_spikes[y, x])
                flat_idx = y * net.W + x
                actors   = scene.node_actors.get((-1, 0), [])
                if flat_idx < len(actors):
                    _set(actors[flat_idx],
                         inp_glow if active else inp_base,
                         GLOW_AMB if active else BASE_AMB)


def build_and_plot(net: Lattice3DNetwork, batch_index: int = 0):
    if DRAW_INPUT_NEURONS:
        wire_input_neurons_for_plot(net)
    scene = build_scene(net, batch_index=batch_index)
    plotter = scene.plotter
    offscreen = infer_offscreen()

    if offscreen:
        plotter.screenshot(SCREENSHOT_PATH)
        print(f"Off-screen render saved to: {SCREENSHOT_PATH}")
    else:
        plotter.show()


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------

def make_random_net() -> Lattice3DNetwork:
    """
    Try to instantiate a random lattice in a robust way.
    Your current src/lattice.py may still have the older signature, so this
    function tries a couple of common possibilities.
    """
    try:
        # Newer/minimal style
        return Lattice3DNetwork.random(Z=Z, H=H, W=W, B=B, seed=RANDOM_SEED)
    except TypeError:
        try:
            # Older signature with K
            return Lattice3DNetwork.random(Z=Z, H=H, W=W, B=B, seed=RANDOM_SEED, K=1)
        except TypeError:
            # Last fallback: maybe device is required
            return Lattice3DNetwork.random(Z=Z, H=H, W=W, B=B, device=torch.device("cpu"), seed=RANDOM_SEED)


if __name__ == "__main__":
    pv.global_theme.allow_empty_mesh = True
    net = make_random_net()

    print("Created lattice.")
    if hasattr(net, "layer_sel"):
        print("layer_sel shape:", tuple(net.layer_sel.shape))
    if hasattr(net, "layer_lut"):
        print("layer_lut shape:", tuple(net.layer_lut.shape))
    if hasattr(net, "state"):
        print("state shape:", tuple(net.state.shape))

    build_and_plot(net, batch_index=0)
