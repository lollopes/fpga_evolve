from __future__ import annotations

import argparse
from importlib import import_module
from typing import Any, Callable

from src.render_env import configure_pyvista_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the lattice viewer in a browser with a separate control sidebar."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind.")
    parser.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Batch index to visualize from the generated random network.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the default browser after the server starts.",
    )
    return parser.parse_args()


def require_module(name: str, install_hint: str):
    try:
        return import_module(name)
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency '{name}'. Install it with: {install_hint}") from exc


def load_plotter_ui() -> Callable[..., Any]:
    candidates = (
        "pyvista.trame.ui.vuetify3",
        "pyvista.trame.ui.vuetify",
        "pyvista.trame.ui",
    )
    for module_name in candidates:
        try:
            module = import_module(module_name)
            return module.plotter_ui
        except (ModuleNotFoundError, AttributeError):
            continue
    raise SystemExit(
        "PyVista trame UI helpers are unavailable. "
        "Install a PyVista build with trame support, for example: "
        "pip install pyvista trame trame-vuetify trame-vtk"
    )


def main() -> None:
    args = parse_args()

    require_module("numpy", "pip install numpy")
    configure_pyvista_env(offscreen=True)
    pv = require_module("pyvista", "pip install pyvista")
    require_module("torch", "pip install torch")
    trame_app = require_module("trame.app", "pip install trame trame-vuetify trame-vtk")
    trame_layout = require_module("trame.ui.vuetify3", "pip install trame-vuetify")
    trame_html = require_module("trame.widgets.html", "pip install trame")
    trame_v3 = require_module("trame.widgets.vuetify3", "pip install trame-vuetify")
    plotter_ui = load_plotter_ui()

    import asyncio
    import base64
    import io
    import threading
    import time
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plot_lattice import (
        CONNECTION_KINDS, build_scene, make_random_net,
        clear_connections, draw_connections,
        wire_input_neurons_for_plot, apply_activity_display,
        DRAW_INPUT_NEURONS, INPUT_NEURON_COLOR,
    )

    get_server = trame_app.get_server
    SinglePageLayout = trame_layout.SinglePageLayout
    html = trame_html
    v3 = trame_v3

    pv.global_theme.allow_empty_mesh = True

    net = make_random_net()
    if DRAW_INPUT_NEURONS:
        wire_input_neurons_for_plot(net)
    net.reset_state()
    scene = build_scene(net, batch_index=args.batch_index, offscreen=True)

    POOL_SIZE = 16
    SEL_BITS = 4  # bits needed to represent 0-15

    Z_ = net.layer_sel.shape[0]
    n_nodes = scene.n_nodes
    k = scene.k
    lut_size = 2 ** k
    node_names = scene.node_names

    lut_np = net.layer_lut[:, args.batch_index].detach().cpu().numpy()
    sel_np = net.layer_sel[:, args.batch_index].detach().cpu().numpy()

    server = get_server(client_type="vue3")
    state = server.state
    ctrl = server.controller

    for kind in CONNECTION_KINDS:
        state[f"show_{kind}"] = True

    for z in range(Z_):
        for n in range(n_nodes):
            for b in range(lut_size):
                state[f"bit_{z}_{n}_{b}"] = int(lut_np[z, n, b])
            for s in range(k):
                state[f"sel_{z}_{n}_{s}"] = int(sel_np[z, n, s])

    state["sel_node"] = ""

    # ------------------------------------------------------------------
    # Simulation state
    # ------------------------------------------------------------------
    import torch as _torch
    _input_spikes = np.zeros((net.H, net.W), dtype=np.int64)
    _render_lock  = threading.Lock()
    _is_playing   = False
    _play_thread  = [None]

    state["is_playing"] = False
    for _y in range(net.H):
        for _x in range(net.W):
            state[f"input_{_y}_{_x}"] = 0

    # ------------------------------------------------------------------
    # Raster history
    # ------------------------------------------------------------------
    N_HIST   = 80
    _H, _W   = net.H, net.W
    # shape: [Z, H, W, 3, N_HIST]  — node axis: 0=E, 1=I, 2=O
    _hist    = np.zeros((Z_, _H, _W, 3, N_HIST), dtype=np.int8)
    _hist_t  = [0]  # total steps written (use mod N_HIST for index)

    state["raster_z"]   = 0
    state["raster_y"]   = 0
    state["raster_x"]   = 0
    state["raster_img"] = ""

    _NODE_COLORS = ["#4169E1", "#DC143C", "#FF8C00"]  # E royalblue, I crimson, O darkorange
    _NODE_LABELS = ["E", "I", "O"]

    def _update_history() -> None:
        t = _hist_t[0] % N_HIST
        if hasattr(net, "last_E"):
            bi = args.batch_index
            _hist[:, :, :, 0, t] = net.last_E[bi].cpu().numpy()
            _hist[:, :, :, 1, t] = net.last_I[bi].cpu().numpy()
            _hist[:, :, :, 2, t] = net.last_O_raw[bi].cpu().numpy()
        _hist_t[0] += 1

    def _render_raster() -> None:
        rz = int(state["raster_z"])
        ry = int(state["raster_y"])
        rx = int(state["raster_x"])
        n_filled = min(_hist_t[0], N_HIST)
        if n_filled == 0:
            state["raster_img"] = ""
            return
        t_start = _hist_t[0] % N_HIST if _hist_t[0] >= N_HIST else 0
        idx = [(t_start + i) % N_HIST for i in range(n_filled)]
        raster = _hist[rz, ry, rx, :, :][:, idx]  # [3, n_filled]

        fig, ax = plt.subplots(figsize=(6, 1.8))
        fig.patch.set_facecolor("#000000")
        ax.set_facecolor("#0d0d0d")
        for ni in range(3):
            spikes = np.where(raster[ni] == 1)[0]
            if len(spikes):
                ax.vlines(spikes, ni + 0.05, ni + 0.9,
                          color=_NODE_COLORS[ni], linewidth=2.5, alpha=0.9)
        ax.set_yticks([0.47, 1.47, 2.47])
        ax.set_yticklabels(_NODE_LABELS, color="white", fontsize=9, fontweight="bold")
        ax.set_ylim(-0.1, 3.1)
        ax.set_xlim(-0.5, n_filled - 0.5)
        ax.set_xlabel("step", color="#64748b", fontsize=7)
        ax.tick_params(axis="x", colors="#64748b", labelsize=6)
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e293b")
        fig.tight_layout(pad=0.4)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=110)
        plt.close(fig)
        buf.seek(0)
        state["raster_img"] = "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    view = None

    def update_view() -> None:
        if view is None:
            return
        if hasattr(view, "update"):
            view.update()
        elif hasattr(scene.plotter, "render"):
            scene.plotter.render()

    def set_connection_visibility(kind: str, visible: bool) -> None:
        for actor in scene.connection_actors.get(kind, []):
            actor.SetVisibility(bool(visible))
        update_view()

    def _highlight_node(sel_z: int, sel_n: int) -> None:
        for (az, an), actors in scene.node_actors.items():
            opacity = 1.0 if (az == sel_z and an == sel_n) else 0.12
            for actor in actors:
                actor.GetProperty().SetOpacity(opacity)
        update_view()

    def _clear_highlight() -> None:
        for actors in scene.node_actors.values():
            for actor in actors:
                actor.GetProperty().SetOpacity(1.0)
        update_view()

    def _redraw_connections(z_only: int = None) -> None:
        clear_connections(
            scene.plotter, scene.connection_actors,
            scene.layer_connection_actors, z_only,
        )
        draw_connections(
            scene.plotter, net, args.batch_index,
            scene.connection_actors,
            scene.module_pos, scene.internal_pos,
            scene.node_names, scene.n_nodes,
            scene.layer_connection_actors, z_only,
        )
        for kind in CONNECTION_KINDS:
            visible = bool(state[f"show_{kind}"])
            for actor in scene.connection_actors.get(kind, []):
                actor.SetVisibility(visible)
        update_view()

    @ctrl.trigger("toggle_bit")
    def _toggle_bit(z: int, n: int, b: int) -> None:
        z, n, b = int(z), int(n), int(b)
        key = f"bit_{z}_{n}_{b}"
        new_val = 1 - int(state[key])
        state[key] = new_val
        net.layer_lut[z, args.batch_index, n, b] = new_val
        _highlight_node(z, n)

    @ctrl.trigger("flip_sel_bit")
    def _flip_sel_bit(z: int, n: int, s: int, b: int) -> None:
        z, n, s, b = int(z), int(n), int(s), int(b)
        key = f"sel_{z}_{n}_{s}"
        new_val = int(state[key]) ^ (1 << b)
        state[key] = new_val
        net.layer_sel[z, args.batch_index, n, s] = new_val
        _redraw_connections(z_only=z)
        _highlight_node(z, n)

    # ------------------------------------------------------------------
    # Simulation triggers
    # ------------------------------------------------------------------

    def _do_step() -> None:
        with _render_lock:
            inp = _torch.from_numpy(_input_spikes).unsqueeze(0).long()
            with _torch.no_grad():
                net.step(inp)
            _input_spikes[:] = 0
            _update_history()
            apply_activity_display(scene, net, args.batch_index, _input_spikes)
        for _y in range(net.H):
            for _x in range(net.W):
                state[f"input_{_y}_{_x}"] = 0
        _render_raster()
        update_view()

    async def _play_loop() -> None:
        # Runs on the asyncio event loop — same thread as the GL context.
        # This avoids the GLX BadAccess crash caused by rendering from a
        # background OS thread.
        while _is_playing:
            _do_step()
            await asyncio.sleep(0.12)  # ~8 fps

    @ctrl.trigger("sim_step")
    def _sim_step() -> None:
        _do_step()

    @ctrl.trigger("sim_play")
    async def _sim_play() -> None:
        nonlocal _is_playing
        _is_playing = not _is_playing
        state["is_playing"] = _is_playing
        if _is_playing:
            task = asyncio.ensure_future(_play_loop())
            _play_thread[0] = task

    @ctrl.trigger("sim_reset")
    def _sim_reset() -> None:
        nonlocal _is_playing
        _is_playing = False
        state["is_playing"] = False
        with _render_lock:
            net.reset_state()
            _hist[:] = 0
            _hist_t[0] = 0
            apply_activity_display(scene, net, args.batch_index, _input_spikes)
        state["raster_img"] = ""
        update_view()

    @ctrl.trigger("toggle_input")
    def _toggle_input(y: int, x: int) -> None:
        y, x = int(y), int(x)
        _input_spikes[y, x] = 1 - _input_spikes[y, x]
        state[f"input_{y}_{x}"] = int(_input_spikes[y, x])
        flat = y * net.W + x
        with _render_lock:
            import pyvista as _pv
            actors = scene.node_actors.get((-1, 0), [])
            if flat < len(actors):
                prop = actors[flat].GetProperty()
                if _input_spikes[y, x]:
                    prop.SetColor(1.0, 1.0, 0.3)
                    prop.SetAmbient(0.9)
                else:
                    base = tuple(_pv.Color(INPUT_NEURON_COLOR).float_rgb)
                    prop.SetColor(*base)
                    prop.SetAmbient(0.1)
        update_view()

    # ------------------------------------------------------------------
    # Connection-visibility toggles
    # ------------------------------------------------------------------
    for kind in CONNECTION_KINDS:
        key = f"show_{kind}"

        def _register_toggle(connection_kind: str, state_key: str) -> None:
            @state.change(state_key)
            def _toggle(**kwargs: Any) -> None:
                visible = kwargs.get(state_key, state[state_key])
                set_connection_visibility(connection_kind, visible)

        _register_toggle(kind, key)

    # ------------------------------------------------------------------
    # Raster selection change handlers
    # ------------------------------------------------------------------
    @state.change("raster_z", "raster_y", "raster_x")
    def _on_raster_sel_change(**_kwargs: Any) -> None:
        _render_raster()

    with SinglePageLayout(server) as layout:
        layout.title.set_text("Lattice Viewer")

        with layout.content:
            html.Style(
                """
                html, body, #app, .v-application, .v-application__wrap {
                    margin: 0;
                    width: 100%;
                    height: 100%;
                    background: #000 !important;
                    color: #e5e7eb;
                    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
                }
                .panel-title {
                    font-size: 1rem;
                    font-weight: 600;
                    letter-spacing: 0.02em;
                    margin-bottom: 0.18rem;
                }
                .panel-copy {
                    color: #94a3b8;
                    line-height: 1.3;
                    font-size: 0.82rem;
                    margin-bottom: 0.55rem;
                }
                .section-label {
                    margin-top: 0.4rem;
                    margin-bottom: 0.15rem;
                    font-size: 0.6rem;
                    font-weight: 700;
                    letter-spacing: 0.08em;
                    text-transform: uppercase;
                    color: #fff;
                }
                .node-grid {
                    display: grid;
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    gap: 0.05rem 0.4rem;
                }
                .legend-row {
                    display: flex;
                    align-items: center;
                    gap: 0.3rem;
                    padding: 0.03rem 0;
                    min-width: 0;
                }
                .legend-swatch {
                    width: 0.55rem;
                    height: 0.55rem;
                    border-radius: 999px;
                    flex: 0 0 auto;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }
                .legend-text {
                    font-size: 0.7rem;
                    white-space: nowrap;
                }
                .connection-toggle {
                    margin: 0;
                    padding: 0;
                }
                .connection-toggle .v-label {
                    font-size: 0.7rem;
                    line-height: 1.1;
                    opacity: 1;
                    white-space: nowrap;
                }
                .connection-toggle .v-selection-control {
                    min-height: 20px;
                }
                .direction-note {
                    margin-top: auto;
                    padding: 0.3rem 0.5rem;
                    border-radius: 0.5rem;
                    background: rgba(15, 118, 110, 0.12);
                    border: 1px solid rgba(45, 212, 191, 0.22);
                    color: #ccfbf1;
                    font-size: 0.65rem;
                }
                @media (max-width: 767px) {
                    .node-grid {
                        grid-template-columns: 1fr;
                    }
                }
                """
            )

            with v3.VContainer(
                fluid=True,
                classes="pa-0",
                style="height:calc(100vh - 64px); display:flex; align-items:flex-start; justify-content:flex-start; padding:20px; background:#000;",
            ):
                with html.Div(style="display:flex; flex-direction:row; align-items:flex-start; gap:10px;"):
                    with html.Div(
                        style="width:min(80vw,700px); height:calc(100vh - 104px); border-radius:18px; overflow:hidden; border:1px solid rgba(148,163,184,0.18); box-shadow:0 24px 80px rgba(0,0,0,0.5); background:#000;",
                    ):
                        view = plotter_ui(
                            scene.plotter,
                            mode="server",
                            default_server_rendering=True,
                            collapse_menu=True,
                        )
                    # --- Simulation panel ---
                    with html.Div(
                        style="width:200px; height:calc(100vh - 104px); overflow-y:auto; overflow-x:hidden; border-radius:18px; border:1px solid rgba(148,163,184,0.18); box-shadow:0 24px 80px rgba(0,0,0,0.5); background:#000; padding:12px; flex-shrink:0;",
                    ):
                        html.Div("Simulation", style="color:#fff; font-size:0.6rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:8px;")
                        with html.Div(style="display:flex; flex-direction:column; gap:4px; margin-bottom:10px;"):
                            v3.VBtn(
                                ("{{ is_playing ? '⏸ Pause' : '▶ Play' }}",),
                                color=("is_playing ? 'warning' : 'success'", "success"),
                                variant="flat",
                                density="compact",
                                block=True,
                                style="font-size:0.62rem; font-weight:700;",
                                click="trigger('sim_play')",
                            )
                            with html.Div(style="display:flex; gap:4px;"):
                                v3.VBtn(
                                    "Step",
                                    color="primary",
                                    variant="flat",
                                    density="compact",
                                    style="font-size:0.62rem; font-weight:700; flex:1;",
                                    click="trigger('sim_step')",
                                )
                                v3.VBtn(
                                    "Reset",
                                    color="grey-darken-3",
                                    variant="flat",
                                    density="compact",
                                    style="font-size:0.62rem; font-weight:700; flex:1;",
                                    click="trigger('sim_reset')",
                                )
                        html.Div("Input spikes", style="color:#94a3b8; font-size:0.6rem; font-weight:600; margin-bottom:6px;")
                        html.Div("Click a neuron to toggle its spike on/off.", style="color:#64748b; font-size:0.55rem; line-height:1.3; margin-bottom:6px;")
                        with html.Div(style=f"display:grid; grid-template-columns:repeat({net.W}, 1fr); gap:3px;"):
                            for _gy in range(net.H):
                                for _gx in range(net.W):
                                    _key = f"input_{_gy}_{_gx}"
                                    v3.VBtn(
                                        "",
                                        color=(f"{_key} ? 'warning' : 'grey-darken-4'", "grey-darken-4"),
                                        variant="flat",
                                        density="compact",
                                        min_width="24",
                                        height="24",
                                        style="padding:0; min-width:0;",
                                        click=f"trigger('toggle_input', [{_gy}, {_gx}])",
                                    )

                    # --- Legend panel ---
                    with html.Div(
                        style="width:190px; height:calc(100vh - 104px); overflow:hidden; border-radius:18px; border:1px solid rgba(148,163,184,0.18); box-shadow:0 24px 80px rgba(0,0,0,0.5); background:#000; padding:12px;",
                    ):
                        with html.Div(style="display:flex; flex-direction:column; height:100%;"):
                            html.Div("Node Types", classes="section-label", style="color:#fff !important;")
                            with html.Div(classes="node-grid"):
                                for label, color in scene.node_items:
                                    with html.Div(classes="legend-row"):
                                        html.Span(classes="legend-swatch", style=f"background: {color};")
                                        html.Span(label, classes="legend-text", style=f"color: {color};")

                            html.Div("Connections", classes="section-label", style="color:#fff !important;")
                            for kind, (label, color) in zip(CONNECTION_KINDS, scene.connection_items):
                                v3.VCheckbox(
                                    v_model=(f"show_{kind}", True),
                                    color=color,
                                    label=label,
                                    hide_details=True,
                                    density="compact",
                                    classes="connection-toggle",
                                    style=f"color: {color};",
                                )

                            html.Div("Arrows: source -> target.", classes="direction-note")

                    # --- Selectors panel ---
                    with html.Div(
                        style="width:175px; height:calc(100vh - 104px); overflow-y:auto; overflow-x:hidden; border-radius:18px; border:1px solid rgba(148,163,184,0.18); box-shadow:0 24px 80px rgba(0,0,0,0.5); background:#000; padding:12px; flex-shrink:0;",
                    ):
                        html.Div("Selectors", style="color:#fff; font-size:0.6rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:4px;")
                        html.Div("Flip a bit to rewire connections in 3D.", style="color:#64748b; font-size:0.58rem; line-height:1.3; margin-bottom:8px;")
                        for z in range(Z_):
                            html.Div(f"Layer {z}", style="color:#94a3b8; font-size:0.6rem; font-weight:600; margin-top:8px; margin-bottom:4px;")
                            for n in range(n_nodes):
                                n_color = scene.node_items[n][1] if n < len(scene.node_items) else "#fff"
                                for s in range(k):
                                    key = f"sel_{z}_{n}_{s}"
                                    with html.Div(style="display:flex; align-items:center; gap:3px; margin-bottom:3px;"):
                                        html.Span(f"{node_names[n]}{s}", style=f"color:{n_color}; font-size:0.58rem; width:20px; flex-shrink:0; font-weight:600;")
                                        for b in range(SEL_BITS - 1, -1, -1):  # MSB first
                                            v3.VBtn(
                                                f"{{{{ ({key} >> {b}) & 1 }}}}",
                                                color=(f"({key} >> {b}) & 1 ? 'primary' : 'grey-darken-4'", "grey-darken-4"),
                                                variant="flat",
                                                density="compact",
                                                min_width="22",
                                                height="22",
                                                style="font-size:0.6rem; font-weight:700; padding:0;",
                                                click=f"trigger('flip_sel_bit', [{z}, {n}, {s}, {b}])",
                                            )

                    # --- LUT bits panel ---
                    with html.Div(
                        style="width:155px; height:calc(100vh - 104px); overflow-y:auto; overflow-x:hidden; border-radius:18px; border:1px solid rgba(148,163,184,0.18); box-shadow:0 24px 80px rgba(0,0,0,0.5); background:#000; padding:12px; flex-shrink:0;",
                    ):
                        html.Div("LUT bits", style="color:#fff; font-size:0.6rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:4px;")
                        html.Div("Click to toggle 0↔1. Updates the truth table.", style="color:#64748b; font-size:0.58rem; line-height:1.3; margin-bottom:8px;")
                        for z in range(Z_):
                            html.Div(f"Layer {z}", style="color:#94a3b8; font-size:0.6rem; font-weight:600; margin-top:8px; margin-bottom:4px;")
                            for n in range(n_nodes):
                                n_color = scene.node_items[n][1] if n < len(scene.node_items) else "#fff"
                                with html.Div(style="display:flex; align-items:center; gap:3px; margin-bottom:3px;"):
                                    html.Span(node_names[n], style=f"color:{n_color}; font-size:0.62rem; width:16px; flex-shrink:0; font-weight:600;")
                                    for b in range(lut_size):
                                        key = f"bit_{z}_{n}_{b}"
                                        v3.VBtn(
                                            f"{{{{ {key} }}}}",
                                            color=(f"{key} ? 'success' : 'grey-darken-4'", "grey-darken-4"),
                                            variant="flat",
                                            density="compact",
                                            min_width="22",
                                            height="22",
                                            style="font-size:0.6rem; font-weight:700; padding:0;",
                                            click=f"trigger('toggle_bit', [{z}, {n}, {b}])",
                                        )

                    # --- Raster panel ---
                    with html.Div(
                        style="width:260px; height:calc(100vh - 104px); overflow-y:auto; overflow-x:hidden; border-radius:18px; border:1px solid rgba(148,163,184,0.18); box-shadow:0 24px 80px rgba(0,0,0,0.5); background:#000; padding:12px; flex-shrink:0;",
                    ):
                        html.Div("Raster", style="color:#fff; font-size:0.6rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:8px;")
                        html.Div("Select a module to trace E / I / O spikes over time.", style="color:#64748b; font-size:0.55rem; line-height:1.3; margin-bottom:10px;")
                        with html.Div(style="display:flex; gap:6px; margin-bottom:10px; align-items:center;"):
                            html.Span("z", style="color:#94a3b8; font-size:0.6rem; font-weight:700; width:10px;")
                            v3.VSelect(
                                v_model=("raster_z", 0),
                                items=([{"title": str(z), "value": z} for z in range(Z_)],),
                                density="compact",
                                variant="outlined",
                                hide_details=True,
                                style="font-size:0.6rem; flex:1;",
                            )
                            html.Span("y", style="color:#94a3b8; font-size:0.6rem; font-weight:700; width:10px;")
                            v3.VSelect(
                                v_model=("raster_y", 0),
                                items=([{"title": str(y), "value": y} for y in range(_H)],),
                                density="compact",
                                variant="outlined",
                                hide_details=True,
                                style="font-size:0.6rem; flex:1;",
                            )
                            html.Span("x", style="color:#94a3b8; font-size:0.6rem; font-weight:700; width:10px;")
                            v3.VSelect(
                                v_model=("raster_x", 0),
                                items=([{"title": str(x), "value": x} for x in range(_W)],),
                                density="compact",
                                variant="outlined",
                                hide_details=True,
                                style="font-size:0.6rem; flex:1;",
                            )
                        html.Img(
                            src=("raster_img", ""),
                            style="width:100%; border-radius:6px; display:block;",
                            v_if="raster_img",
                        )
                        html.Div(
                            "No data yet — press Step or Play.",
                            style="color:#475569; font-size:0.6rem; text-align:center; margin-top:20px;",
                            v_else=True,
                        )

    server.start(host=args.host, port=args.port, open_browser=args.open_browser)


if __name__ == "__main__":
    main()
