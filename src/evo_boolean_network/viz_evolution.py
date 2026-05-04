"""
viz_evolution.py — Live Dash/Plotly/Cytoscape viewer for Boolean network evolution.

Opens a browser dashboard at http://localhost:8050 that updates every second.

Panels
------
  Top-left   Fitness curve   — best + mean accuracy over generations (Plotly).
  Top-right  LUT histogram   — function-type distribution of the best genome (Plotly).
  Middle     Network graph   — output nodes + ancestors up to depth 4 (Dash Cytoscape).
  Bottom     Connectivity    — full n_nodes × (n_inputs + n_nodes) heatmap (Plotly).

Usage
-----
    from evo_boolean_network import EvoConfig, evolve_binary_classifier
    from evo_boolean_network.viz_evolution import EvolutionViewer

    config = EvoConfig(...)
    viewer = EvolutionViewer(config)          # starts server, open browser

    evolve_binary_classifier(
        config, X, y, ...,
        snapshot_callback=viewer.update,
        snapshot_every=10,
    )

    viewer.wait()                             # block until Ctrl+C

Requires: dash  dash-cytoscape  plotly  networkx
    pip install dash dash-cytoscape plotly networkx
"""

from __future__ import annotations

import threading
import time
import webbrowser
import socket

import numpy as np

from .config import EvoConfig
from .genome import describe_genome


# ---------------------------------------------------------------------------
# Optional visualization stack import
# ---------------------------------------------------------------------------

def _patch_comm_for_dash() -> None:
    """
    Patch comm.create_comm so Dash import does not crash in non-kernel contexts.

    Some environments ship a comm package where create_comm raises
    NotImplementedError by default (for example comm==0.1.2). Dash imports this
    during module import for Jupyter integration. For normal browser-hosted Dash
    apps we can safely treat that case as "no comm available".
    """
    try:
        import comm  # type: ignore[import-not-found]
    except Exception:
        return

    create_comm = getattr(comm, "create_comm", None)
    if create_comm is None:
        return

    def _safe_create_comm(*args, **kwargs):
        try:
            return create_comm(*args, **kwargs)
        except NotImplementedError:
            return None

    comm.create_comm = _safe_create_comm


_DASH_IMPORT_ERROR: Exception | None = None

try:
    _patch_comm_for_dash()
    import networkx as nx
    import dash
    from dash import Input, Output, dcc, html
    import dash_cytoscape as cyto
    import plotly.graph_objects as go
except Exception as exc:  # pragma: no cover - environment-dependent
    _DASH_IMPORT_ERROR = exc
    nx = None
    dash = None
    Input = Output = dcc = html = cyto = go = None


# ---------------------------------------------------------------------------
# LUT palette
# ---------------------------------------------------------------------------

_LUT_ORDER = [
    "CONST_0", "CONST_1",
    "COPY_A",  "COPY_B",
    "NOT_A",   "NOT_B",
    "AND",     "NAND",
    "OR",      "NOR",
    "XOR",     "XNOR",
    "CUSTOM",
]

_LUT_COLOR = {
    "CONST_0": "#aaaaaa", "CONST_1": "#666666",
    "COPY_A":  "#4caf50", "COPY_B":  "#8bc34a",
    "NOT_A":   "#f44336", "NOT_B":   "#ff7043",
    "AND":     "#2196f3", "NAND":    "#03a9f4",
    "OR":      "#ff9800", "NOR":     "#fb8c00",
    "XOR":     "#9c27b0", "XNOR":   "#673ab7",
    "CUSTOM":  "#607d8b",
}


# ---------------------------------------------------------------------------
# Thread-safe shared state
# ---------------------------------------------------------------------------

class _State:
    def __init__(self, zero_genome: np.ndarray) -> None:
        self._lock    = threading.Lock()
        self.gens:   list[int]   = []
        self.best:   list[float] = []
        self.mean:   list[float] = []
        self.genome: np.ndarray  = zero_genome.copy()
        self.latest_gen: int     = -1
        self.done:   bool        = False

    def push(self, gen: int, genome: np.ndarray, fitnesses: np.ndarray) -> None:
        with self._lock:
            self.gens.append(gen)
            self.best.append(float(fitnesses.max()))
            self.mean.append(float(fitnesses.mean()))
            self.genome      = genome.copy()
            self.latest_gen  = gen

    def snapshot(self):
        with self._lock:
            return (
                list(self.gens),
                list(self.best),
                list(self.mean),
                self.genome.copy(),
                self.latest_gen,
                self.done,
            )


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

class EvolutionViewer:
    """
    Live web dashboard for Boolean network evolution.

    Pass ``viewer.update`` as ``snapshot_callback`` to evolve_binary_classifier.
    Call ``viewer.wait()`` after evolution finishes to keep the page alive.
    """

    def __init__(
        self,
        config: EvoConfig,
        port: int = 8050,
        open_browser: bool = True,
        auto_port: bool = False,
    ) -> None:
        if _DASH_IMPORT_ERROR is not None or dash is None:
            raise ImportError(
                "EvolutionViewer requires optional visualization dependencies. "
                "Install with: pip install dash dash-cytoscape plotly networkx. "
                "If Dash import fails with comm-related errors, upgrade comm: "
                "pip install --upgrade comm"
            ) from _DASH_IMPORT_ERROR

        self.config = config
        self.port   = self._resolve_port(port, auto_port)

        zero = np.zeros(config.total_genome_bits, dtype=np.uint8)
        self._state         = _State(zero)
        self._last_rendered = -2   # sentinel so first render always fires

        self._app = self._build_app()
        self._thread = threading.Thread(
            target=self._app.run,
            kwargs={"port": self.port, "debug": False, "use_reloader": False},
            daemon=True,
        )
        self._thread.start()
        url = f"http://localhost:{self.port}"
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        print(f"[EvolutionViewer] Dashboard → {url}")

    @staticmethod
    def _is_port_available(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False

    @classmethod
    def _resolve_port(cls, requested_port: int, auto_port: bool) -> int:
        if cls._is_port_available(requested_port):
            return requested_port

        if not auto_port:
            raise OSError(
                f"Port {requested_port} is already in use. "
                "Pick another port (e.g. EvolutionViewer(config, port=8051)) "
                "or pass auto_port=True."
            )

        # Find the next available port in a small local range.
        for port in range(requested_port + 1, requested_port + 200):
            if cls._is_port_available(port):
                return port

        raise OSError(
            f"Could not find a free port near {requested_port}. "
            "Please provide an explicit free port."
        )

    # ------------------------------------------------------------------
    # Dash app
    # ------------------------------------------------------------------

    def _build_app(self) -> dash.Dash:
        app = dash.Dash(__name__, title="BoolNet Evolution")

        app.layout = html.Div([
            # ── header ──────────────────────────────────────────────
            html.Div([
                html.H3("Boolean Network Evolution",
                        style={"margin": "0", "fontSize": "16px"}),
                html.Span(id="status-bar",
                          style={"fontSize": "13px", "color": "#555", "marginLeft": "20px"}),
            ], style={"display": "flex", "alignItems": "center",
                      "padding": "8px 16px", "background": "#f4f4f4",
                      "borderBottom": "1px solid #ccc"}),

            # ── top row: fitness | LUT histogram ────────────────────
            html.Div([
                html.Div(dcc.Graph(id="fitness-plot", style={"height": "280px"}),
                         style={"flex": "2", "minWidth": 0}),
                html.Div(dcc.Graph(id="lut-plot",     style={"height": "280px"}),
                         style={"flex": "1", "minWidth": 0}),
            ], style={"display": "flex", "gap": "8px", "padding": "8px 16px"}),

            # ── middle: Cytoscape network graph ─────────────────────
            html.Div([
                html.Div(
                    "Full network — inputs left, outputs right"
                    "  — blue edges = A input, red edges = B input",
                    style={"fontSize": "11px", "color": "#666", "marginBottom": "4px"},
                ),
                cyto.Cytoscape(
                    id="network-graph",
                    layout={"name": "preset"},
                    style={"width": "100%", "height": "480px",
                            "border": "1px solid #ddd", "borderRadius": "4px"},
                    elements=[],
                    stylesheet=self._cyto_stylesheet(),
                    userZoomingEnabled=True,
                    userPanningEnabled=True,
                ),
            ], style={"padding": "0 16px 8px"}),

            # ── node info panel (populated on click) ─────────────────
            html.Div(
                id="node-info",
                children="Click a node to inspect its LUT.",
                style={"padding": "4px 16px 8px", "fontSize": "12px",
                       "color": "#444", "minHeight": "60px"},
            ),

            # ── bottom: connectivity heatmap ─────────────────────────
            html.Div(
                dcc.Graph(id="conn-heatmap", style={"height": "340px"}),
                style={"padding": "0 16px 8px"},
            ),

            dcc.Interval(id="tick", interval=1000, n_intervals=0),
        ], style={"fontFamily": "sans-serif"})

        # ── single callback that refreshes everything ────────────────
        @app.callback(
            Output("status-bar",    "children"),
            Output("fitness-plot",  "figure"),
            Output("lut-plot",      "figure"),
            Output("network-graph", "elements"),
            Output("conn-heatmap",  "figure"),
            Input("tick", "n_intervals"),
        )
        def _refresh(_n):
            gens, best, mean, genome, latest_gen, done = self._state.snapshot()

            if not gens:
                empty = go.Figure()
                return "Waiting for first snapshot…", empty, empty, [], empty

            status = (f"Gen {latest_gen}  │  best {best[-1]:.4f}"
                      f"  │  mean {mean[-1]:.4f}"
                      + ("  ✓ done" if done else ""))

            # skip expensive rebuilds if nothing new arrived
            if latest_gen == self._last_rendered:
                return status, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            self._last_rendered = latest_gen

            G   = self._build_nx_graph(genome)
            pos = self._compute_positions(G)

            return (
                status,
                self._fitness_fig(gens, best, mean),
                self._lut_fig(genome),
                self._to_cyto_elements(G, pos),
                self._conn_fig(genome),
            )

        @app.callback(
            Output("node-info", "children"),
            Input("network-graph", "tapNodeData"),
        )
        def _show_node_info(data):
            if not data:
                return "Click a node to inspect its LUT."

            label     = data.get("label", data["id"])
            node_type = data.get("node_type", "internal")

            if node_type == "input":
                return html.Span([html.B(label), "  —  external input"])

            lut_name = data.get("lut", "?")
            src_a    = data.get("src_a", "?")
            src_b    = data.get("src_b", "?")
            bits     = data.get("lut_bits", [])

            def _src_label(s):
                if s == "" or s is None:
                    return "?"
                return f"IN{-s - 1}" if s < 0 else f"N{s}"

            rows = [
                html.Tr([html.Td(a, style={"padding": "1px 8px"}),
                         html.Td(b, style={"padding": "1px 8px"}),
                         html.Td(bits[i] if i < len(bits) else "?",
                                 style={"padding": "1px 8px", "fontWeight": "bold",
                                        "color": "#2196f3" if (i < len(bits) and bits[i]) else "#f44336"})],
                        style={"background": "#f9f9f9" if i % 2 else "#fff"})
                for i, (a, b) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)])
            ]

            return html.Div([
                html.Span(html.B(label)),
                html.Span(f"  LUT: {lut_name}  │  A ← {_src_label(src_a)}  │  B ← {_src_label(src_b)}",
                          style={"marginLeft": "10px", "color": "#555"}),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("A", style={"padding": "2px 8px", "textAlign": "left"}),
                        html.Th("B", style={"padding": "2px 8px", "textAlign": "left"}),
                        html.Th("Out", style={"padding": "2px 8px", "textAlign": "left"}),
                    ]), style={"borderBottom": "1px solid #ccc"}),
                    html.Tbody(rows),
                ], style={"borderCollapse": "collapse", "marginTop": "6px",
                          "border": "1px solid #ddd", "borderRadius": "4px"}),
            ])

        return app

    # ------------------------------------------------------------------
    # NetworkX graph helpers
    # ------------------------------------------------------------------

    def _build_nx_graph(self, genome: np.ndarray) -> nx.DiGraph:
        cfg = self.config
        G   = nx.DiGraph()

        for i in range(cfg.n_inputs):
            G.add_node(f"I{i}", node_type="input", label=f"IN{i}", lut="")

        for d in describe_genome(genome, cfg):
            nid      = d["node_id"]
            ntype    = "output" if nid in set(cfg.output_nodes) else "internal"
            label    = f"OUT{nid}" if ntype == "output" else f"N{nid}"
            G.add_node(f"N{nid}", node_type=ntype, label=label, lut=d["lut_name"],
                       src_a=d["src_a"], src_b=d["src_b"],
                       lut_bits=[int(b) for b in d["lut_bits"]])

            for src, port in [(d["src_a"], "A"), (d["src_b"], "B")]:
                src_id = f"I{-src - 1}" if src < 0 else f"N{src}"
                G.add_edge(src_id, f"N{nid}", port=port)

        return G

    @staticmethod
    def _compute_positions(
        G: nx.DiGraph,
        x_spacing: int = 160,
        y_spacing: int = 70,
    ) -> dict[str, dict]:
        if not G.nodes:
            return {}

        try:
            layer: dict[str, int] = {}
            for node in nx.topological_sort(G):
                preds = list(G.predecessors(node))
                layer[node] = 0 if not preds else max(layer[p] for p in preds) + 1
        except nx.NetworkXUnfeasible:
            # Cyclic graph fallback — scale spring layout to same coordinate range
            pos = nx.spring_layout(G, seed=42)
            return {n: {"x": float(xy[0]) * 600, "y": float(xy[1]) * 400}
                    for n, xy in pos.items()}

        # Force output nodes to the rightmost layer so they land on the right
        output_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "output"]
        if output_nodes:
            max_layer = max(layer.values())
            for n in output_nodes:
                layer[n] = max_layer

        by_layer: dict[int, list] = {}
        for node, l in layer.items():
            by_layer.setdefault(l, []).append(node)

        positions: dict[str, dict] = {}
        for l, nodes in sorted(by_layer.items()):
            nodes = sorted(nodes)
            for i, node in enumerate(nodes):
                positions[node] = {
                    "x": l * x_spacing,
                    "y": (i - (len(nodes) - 1) / 2) * y_spacing,
                }
        return positions

    @staticmethod
    def _to_cyto_elements(G: nx.DiGraph, positions: dict | None = None) -> list[dict]:
        elements = []
        for node, data in G.nodes(data=True):
            elem: dict = {"data": {
                "id":        node,
                "label":     data.get("label", node),
                "node_type": data.get("node_type", "internal"),
                "lut":       data.get("lut", "CUSTOM"),
                "src_a":     data.get("src_a", ""),
                "src_b":     data.get("src_b", ""),
                "lut_bits":  data.get("lut_bits", []),
            }}
            if positions and node in positions:
                elem["position"] = positions[node]
            elements.append(elem)
        for src, tgt, data in G.edges(data=True):
            elements.append({"data": {
                "source": src,
                "target": tgt,
                "port":   data.get("port", "A"),
            }})
        return elements

    # ------------------------------------------------------------------
    # Plotly figure builders
    # ------------------------------------------------------------------

    @staticmethod
    def _fitness_fig(gens, best, mean) -> go.Figure:
        fig = go.Figure([
            go.Scatter(x=gens, y=best, name="best",
                       line={"color": "#2196f3", "width": 2}),
            go.Scatter(x=gens, y=mean, name="mean",
                       line={"color": "#f44336", "width": 1.5, "dash": "dash"},
                       opacity=0.7),
        ])
        fig.update_layout(
            title="Population fitness",
            xaxis_title="Generation", yaxis_title="Accuracy",
            yaxis={"range": [0, 1.05]},
            legend={"orientation": "h", "y": -0.25},
            margin={"t": 36, "b": 50, "l": 48, "r": 12},
        )
        return fig

    def _lut_fig(self, genome: np.ndarray) -> go.Figure:
        counts = {name: 0 for name in _LUT_ORDER}
        for d in describe_genome(genome, self.config):
            key = d["lut_name"] if d["lut_name"] in counts else "CUSTOM"
            counts[key] += 1

        fig = go.Figure(go.Bar(
            x=_LUT_ORDER,
            y=[counts[n] for n in _LUT_ORDER],
            marker_color=[_LUT_COLOR[n] for n in _LUT_ORDER],
        ))
        fig.update_layout(
            title="LUT distribution (best genome)",
            yaxis_title="# nodes",
            xaxis={"tickangle": -40},
            margin={"t": 36, "b": 80, "l": 48, "r": 12},
        )
        return fig

    def _conn_fig(self, genome: np.ndarray) -> go.Figure:
        cfg              = self.config
        n_nodes, n_in    = cfg.n_nodes, cfg.n_inputs
        mat              = np.zeros((n_nodes, n_in + n_nodes), dtype=np.float32)

        for d in describe_genome(genome, cfg):
            nid  = d["node_id"]
            sa, sb = d["src_a"], d["src_b"]
            col_a = (-sa - 1)   if sa < 0 else (n_in + sa)
            col_b = (-sb - 1)   if sb < 0 else (n_in + sb)
            mat[nid, col_a] =  1.0   # A → blue
            mat[nid, col_b] = -1.0   # B → red

        x_labels = [f"I{i}" for i in range(n_in)] + [f"N{i}" for i in range(n_nodes)]
        y_labels = [f"N{i}" for i in range(n_nodes)]

        fig = go.Figure(go.Heatmap(
            z=mat, x=x_labels, y=y_labels,
            colorscale="RdBu_r", zmin=-1, zmax=1,
        ))
        if n_in > 0:
            fig.add_vline(x=n_in - 0.5, line_color="black",
                          line_width=1, line_dash="dash", opacity=0.35)
        fig.update_layout(
            title="Connectivity  (blue = A input, red = B input)",
            xaxis_title=f"Source  [ I0–I{n_in-1} = ext inputs │ N0–N{n_nodes-1} = nodes ]",
            yaxis_title="Destination node",
            yaxis={"autorange": "reversed"},
            margin={"t": 36, "b": 60, "l": 56, "r": 12},
        )
        return fig

    # ------------------------------------------------------------------
    # Cytoscape stylesheet
    # ------------------------------------------------------------------

    @staticmethod
    def _cyto_stylesheet() -> list[dict]:
        styles = [
            {"selector": "node", "style": {
                "label":        "data(label)",
                "font-size":    "9px",
                "width":        "44px",
                "height":       "44px",
                "text-valign":  "center",
                "text-halign":  "center",
                "text-wrap":    "wrap",
                "color":        "#fff",
            }},
            {"selector": "node[node_type='input']", "style": {
                "background-color": "#bdbdbd",
                "color":            "#333",
                "shape":            "rectangle",
                "width":            "56px",
                "height":           "36px",
            }},
            {"selector": "node[node_type='output']", "style": {
                "background-color": "#ffb300",
                "border-width":     "3px",
                "border-color":     "#e65100",
                "color":            "#333",
                "shape":            "star",
                "width":            "60px",
                "height":           "60px",
            }},
            {"selector": "edge[port='A']", "style": {
                "line-color":          "#2196f3",
                "target-arrow-color":  "#2196f3",
                "target-arrow-shape":  "triangle",
                "curve-style":         "bezier",
                "width":               2,
            }},
            {"selector": "edge[port='B']", "style": {
                "line-color":          "#f44336",
                "target-arrow-color":  "#f44336",
                "target-arrow-shape":  "triangle",
                "curve-style":         "bezier",
                "width":               2,
            }},
        ]
        # per-LUT node colours
        for lut_name, color in _LUT_COLOR.items():
            styles.append({
                "selector": f"node[lut='{lut_name}'][node_type='internal']",
                "style":    {"background-color": color},
            })
        return styles

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, gen: int, best_genome: np.ndarray, fitnesses: np.ndarray) -> None:
        """Snapshot callback — pass directly to evolve_binary_classifier."""
        self._state.push(gen, best_genome, fitnesses)

    def wait(self) -> None:
        """Mark evolution as done and block until Ctrl+C."""
        self._state.done = True
        print(f"[EvolutionViewer] Evolution done.  "
              f"Dashboard still live → http://localhost:{self.port}  (Ctrl+C to exit)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
