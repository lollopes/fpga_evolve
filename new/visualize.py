"""
visualize.py — Render a Circuit as an Obsidian-style interactive HTML graph.

Generates a single self-contained .html file driven by vis-network (loaded
from a CDN) with:
  - dark background, muted node colors by LUT function,
  - force-directed physics (drag, zoom, hover),
  - arrow heads on every edge so direction is explicit,
  - tooltip showing LUT name and the (A, B) signal sources for each node.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np

from circuit import Circuit


_LUT_NAMES = {
    (0, 0, 0, 0): "CONST_0",
    (1, 1, 1, 1): "CONST_1",
    (0, 0, 1, 1): "COPY_A",
    (0, 1, 0, 1): "COPY_B",
    (1, 1, 0, 0): "NOT_A",
    (1, 0, 1, 0): "NOT_B",
    (0, 0, 0, 1): "AND",
    (1, 1, 1, 0): "NAND",
    (0, 1, 1, 1): "OR",
    (1, 0, 0, 0): "NOR",
    (0, 1, 1, 0): "XOR",
    (1, 0, 0, 1): "XNOR",
}

# One coherent palette inspired by Obsidian's dark graph view: muted defaults
# for the static layout, plus contrasting "ON" variants used during activity
# playback (bit = 1 lights up; bit = 0 stays in the muted base color).
_BG_COLOR = "#191919"
_LABEL_COLOR = "#d8dadc"

_NODE_COLOR = "#3d4a5c"          # OFF (state == 0) — computation node
_NODE_BORDER = "#2a3343"
_NODE_ON_COLOR = "#9fd2ff"       # ON (state == 1)
_NODE_ON_BORDER = "#7fb8e8"

_INPUT_COLOR = "#4a4d50"         # OFF — external input
_INPUT_BORDER = "#33343a"
_INPUT_ON_COLOR = "#f0d870"      # ON
_INPUT_ON_BORDER = "#c9b04a"

_EDGE_COLOR = "#3a3a3a"          # OFF wire
_EDGE_HIGHLIGHT = "#e8e8e8"
_EDGE_ON_COLOR = "#e8e8e8"       # live wire (source carries a 1)


def lut_name(lut_row: np.ndarray) -> str:
    """Identify the Boolean function from a 4-bit truth table row."""
    return _LUT_NAMES.get(tuple(int(b) for b in lut_row), "CUSTOM")


def candidate_to_source(circuit: Circuit, node_id: int, candidate_pos: int) -> tuple[str, int]:
    """Resolve a candidate-pool position into ('ext'|'node', idx)."""
    if candidate_pos < circuit.n_inputs:
        return ("ext", candidate_pos)
    internal_pos = candidate_pos - circuit.n_inputs
    if internal_pos >= node_id:
        internal_pos += 1
    return ("node", internal_pos)


def _source_label(kind: str, idx: int) -> str:
    return f"I{idx}" if kind == "ext" else f"N{idx}"


def _build_nodes(circuit: Circuit, degree: dict[str, int], batch_id: int) -> list[dict]:
    nodes: list[dict] = []
    for k in range(circuit.n_inputs):
        node_id = f"I{k}"
        nodes.append({
            "id": node_id,
            "label": node_id,
            "title": f"External input {k}",
            "color": {"background": _INPUT_COLOR, "border": _INPUT_BORDER},
            "shape": "dot",
            "value": degree.get(node_id, 0),
        })
    for i in range(circuit.n_nodes):
        node_id = f"N{i}"
        name = lut_name(circuit.luts[batch_id, i])
        a_kind, a_idx = candidate_to_source(circuit, i, int(circuit.sel_a[batch_id, i]))
        b_kind, b_idx = candidate_to_source(circuit, i, int(circuit.sel_b[batch_id, i]))
        tooltip = (
            f"N{i} — {name}\n"
            f"A ← {_source_label(a_kind, a_idx)}\n"
            f"B ← {_source_label(b_kind, b_idx)}"
        )
        nodes.append({
            "id": node_id,
            "label": node_id,
            "title": tooltip,
            "color": {"background": _NODE_COLOR, "border": _NODE_BORDER},
            "shape": "dot",
            "value": degree.get(node_id, 0),
        })
    return nodes


def _build_edges(circuit: Circuit, batch_id: int) -> list[dict]:
    edges: list[dict] = []
    for i in range(circuit.n_nodes):
        sel_pair = (
            ("A", int(circuit.sel_a[batch_id, i])),
            ("B", int(circuit.sel_b[batch_id, i])),
        )
        for port, sel in sel_pair:
            kind, idx = candidate_to_source(circuit, i, sel)
            src = _source_label(kind, idx)
            edges.append({
                "id": f"e{len(edges)}",
                "from": src,
                "to": f"N{i}",
                "title": f"{src} → N{i}  (port {port})",
                "port": port,
            })
    return edges


def _node_degrees(edges: list[dict]) -> dict[str, int]:
    degree: dict[str, int] = {}
    for e in edges:
        degree[e["from"]] = degree.get(e["from"], 0) + 1
        degree[e["to"]] = degree.get(e["to"], 0) + 1
    return degree


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    html, body {{
      margin: 0; padding: 0;
      width: 100vw; height: 100vh;
      background: {bg};
      color: {label};
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      overflow: hidden;
    }}
    #network {{ width: 100vw; height: 100vh; }}
    #header {{
      position: fixed; top: 12px; left: 12px;
      background: rgba(40, 40, 40, 0.85);
      padding: 8px 14px;
      border-radius: 6px;
      font-size: 12px;
      pointer-events: none;
      backdrop-filter: blur(6px);
      z-index: 100;
    }}
    #header b {{ color: #fff; font-size: 13px; }}
    .vis-tooltip {{
      background: #2a2a2a !important;
      color: #ddd !important;
      border: 1px solid #444 !important;
      border-radius: 4px !important;
      font-family: inherit !important;
      font-size: 12px !important;
      white-space: pre !important;
      padding: 6px 10px !important;
    }}
    #controls {{
      position: fixed; top: 12px; right: 12px;
      width: 240px;
      background: rgba(40, 40, 40, 0.92);
      border-radius: 6px;
      padding: 10px 12px 12px;
      font-size: 12px;
      backdrop-filter: blur(6px);
      user-select: none;
      box-shadow: 0 4px 16px rgba(0,0,0,0.4);
      z-index: 100;
      pointer-events: auto;
    }}
    #controls .ctrl-header {{
      display: flex; justify-content: space-between; align-items: center;
      font-weight: 600; color: #fff; font-size: 13px;
      margin-bottom: 6px;
    }}
    #controls button.ctrl-icon {{
      background: transparent; color: #ddd; border: none;
      font-size: 16px; line-height: 1; cursor: pointer; padding: 0 4px;
    }}
    #controls.collapsed #ctrl-body {{ display: none; }}
    #controls .ctrl-row {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 2px 8px;
      align-items: center;
      margin: 8px 0 2px;
    }}
    #controls .ctrl-row .ctrl-label {{ color: #c0c0c0; }}
    #controls .ctrl-row .ctrl-value {{
      color: #8a8a8a; font-size: 11px; font-variant-numeric: tabular-nums;
    }}
    #controls input[type=range] {{
      grid-column: 1 / span 2;
      width: 100%;
      accent-color: #7c93b7;
      background: transparent;
      margin: 2px 0 0;
    }}
    #controls .ctrl-buttons {{
      display: flex; gap: 6px; margin-top: 12px;
    }}
    #controls button.ctrl-btn {{
      flex: 1;
      background: #2f2f2f; color: #ddd;
      border: 1px solid #444; border-radius: 4px;
      padding: 4px 8px; cursor: pointer; font-size: 11px;
      font-family: inherit;
    }}
    #controls button.ctrl-btn:hover {{ background: #3a3a3a; }}
    #playback {{
      position: fixed; bottom: 14px; left: 50%; transform: translateX(-50%);
      background: rgba(40, 40, 40, 0.92);
      border-radius: 6px;
      padding: 8px 14px;
      font-size: 12px;
      display: flex; align-items: center; gap: 12px;
      backdrop-filter: blur(6px);
      user-select: none;
      box-shadow: 0 4px 16px rgba(0,0,0,0.4);
      z-index: 100;
      width: min(640px, 92vw);
    }}
    #playback button.pb-btn {{
      background: #2f2f2f; color: #ddd;
      border: 1px solid #444; border-radius: 4px;
      padding: 4px 12px; cursor: pointer; font-size: 11px;
      font-family: inherit;
      min-width: 56px;
    }}
    #playback button.pb-btn:hover {{ background: #3a3a3a; }}
    #playback input[type=range] {{
      flex: 1;
      accent-color: #9fd2ff;
      background: transparent;
      margin: 0;
    }}
    #playback #playback-speed {{ flex: 0 0 110px; accent-color: #f0d870; }}
    #playback .pb-time {{
      color: #c0c0c0; font-variant-numeric: tabular-nums;
      min-width: 84px; text-align: center;
    }}
    #playback .pb-sep {{ color: #555; }}
    #playback .pb-label {{ color: #888; }}
  </style>
</head>
<body>
  <div id="header">
    <b>{title}</b><br />
    {summary}
  </div>
  <div id="controls">
    <div class="ctrl-header">
      <span>Forces</span>
      <button class="ctrl-icon" id="ctrl-toggle" title="Collapse">&minus;</button>
    </div>
    <div id="ctrl-body">
      <div class="ctrl-row">
        <span class="ctrl-label">Center force</span>
        <span class="ctrl-value" id="val-centralGravity">0.01</span>
        <input type="range" min="0" max="1" step="0.01" value="0.01"
               data-param="centralGravity" data-mul="1" data-precision="2" />
      </div>
      <div class="ctrl-row">
        <span class="ctrl-label">Repel force</span>
        <span class="ctrl-value" id="val-gravitationalConstant">55</span>
        <input type="range" min="0" max="300" step="1" value="55"
               data-param="gravitationalConstant" data-mul="-1" data-precision="0" />
      </div>
      <div class="ctrl-row">
        <span class="ctrl-label">Link force</span>
        <span class="ctrl-value" id="val-springConstant">0.080</span>
        <input type="range" min="0" max="0.5" step="0.005" value="0.08"
               data-param="springConstant" data-mul="1" data-precision="3" />
      </div>
      <div class="ctrl-row">
        <span class="ctrl-label">Link distance</span>
        <span class="ctrl-value" id="val-springLength">110</span>
        <input type="range" min="10" max="500" step="5" value="110"
               data-param="springLength" data-mul="1" data-precision="0" />
      </div>
      <div class="ctrl-row">
        <span class="ctrl-label">Damping</span>
        <span class="ctrl-value" id="val-damping">0.50</span>
        <input type="range" min="0" max="1" step="0.05" value="0.5"
               data-param="damping" data-mul="1" data-precision="2" />
      </div>
      <div class="ctrl-buttons">
        <button class="ctrl-btn" id="ctrl-pause">Pause</button>
        <button class="ctrl-btn" id="ctrl-reset">Reset</button>
      </div>
    </div>
  </div>
  <div id="network"></div>
  <div id="playback">
    <button class="pb-btn" id="playback-play">Play</button>
    <input type="range" id="playback-scrub" min="0" max="{t_max}" value="0" step="1" />
    <span class="pb-time" id="playback-time">t = 0 / {t_max}</span>
    <span class="pb-sep">|</span>
    <span class="pb-label">Speed</span>
    <input type="range" id="playback-speed" min="50" max="2000" step="50" value="200" />
    <span class="pb-time" id="playback-speed-val">200ms</span>
  </div>
  <script>
    const NODES = {nodes_json};
    const EDGES = {edges_json};

    const data = {{
      nodes: new vis.DataSet(NODES),
      edges: new vis.DataSet(EDGES),
    }};

    const options = {{
      nodes: {{
        borderWidth: 1.5,
        scaling: {{
          min: 10,
          max: 32,
          label: {{ enabled: false }},
        }},
        font: {{
          color: "{label}",
          size: 12,
          face: "inherit",
          strokeWidth: 0,
          vadjust: -22,
        }},
        shadow: {{ enabled: true, color: "rgba(0,0,0,0.55)", size: 6, x: 0, y: 0 }},
      }},
      edges: {{
        color: {{ color: "{edge}", highlight: "{edge_hi}", hover: "{edge_hi}", opacity: 0.6 }},
        width: 1.0,
        selectionWidth: 1.6,
        hoverWidth: 1.6,
        smooth: false,
        arrows: {{
          to: {{ enabled: true, scaleFactor: 0.55, type: "arrow" }}
        }},
      }},
      physics: {{
        solver: "forceAtlas2Based",
        forceAtlas2Based: {{
          gravitationalConstant: -55,
          centralGravity: 0.01,
          springLength: 110,
          springConstant: 0.08,
          damping: 0.5,
          avoidOverlap: 0.6,
        }},
        stabilization: {{ iterations: 600, fit: true }},
      }},
      interaction: {{
        hover: true,
        dragNodes: true,
        zoomView: true,
        navigationButtons: false,
        tooltipDelay: 120,
      }},
    }};

    const network = new vis.Network(
      document.getElementById("network"),
      data,
      options,
    );

    network.once("stabilizationIterationsDone", () => {{
      network.setOptions({{ physics: {{ enabled: true }} }});
    }});

    const physicsDefaults = {{
      centralGravity: 0.01,
      gravitationalConstant: -55,
      springLength: 110,
      springConstant: 0.08,
      damping: 0.5,
    }};

    const controlsEl = document.getElementById("controls");
    ["pointerdown", "mousedown", "touchstart", "wheel"].forEach(ev => {{
      controlsEl.addEventListener(ev, e => e.stopPropagation(), {{ passive: true }});
    }});

    const sliders = document.querySelectorAll("#controls input[type=range]");
    sliders.forEach(slider => {{
      slider.addEventListener("input", () => {{
        const param = slider.dataset.param;
        const mul = parseFloat(slider.dataset.mul);
        const precision = parseInt(slider.dataset.precision, 10);
        const display = parseFloat(slider.value);
        const real = display * mul;
        const valEl = document.getElementById("val-" + param);
        if (valEl) valEl.textContent = display.toFixed(precision);
        network.setOptions({{
          physics: {{ forceAtlas2Based: {{ [param]: real }} }}
        }});
      }});
    }});

    let physicsOn = true;
    const pauseBtn = document.getElementById("ctrl-pause");
    pauseBtn.addEventListener("click", () => {{
      physicsOn = !physicsOn;
      network.setOptions({{ physics: {{ enabled: physicsOn }} }});
      pauseBtn.textContent = physicsOn ? "Pause" : "Resume";
    }});

    document.getElementById("ctrl-reset").addEventListener("click", () => {{
      sliders.forEach(slider => {{
        const param = slider.dataset.param;
        const mul = parseFloat(slider.dataset.mul);
        const def = physicsDefaults[param];
        slider.value = def / mul;
        slider.dispatchEvent(new Event("input"));
      }});
    }});

    const toggleBtn = document.getElementById("ctrl-toggle");
    toggleBtn.addEventListener("click", () => {{
      const ctrl = document.getElementById("controls");
      ctrl.classList.toggle("collapsed");
      toggleBtn.innerHTML = ctrl.classList.contains("collapsed") ? "+" : "&minus;";
    }});

    // ── activity playback ────────────────────────────────────────────
    const STATE_TRAJ = {state_json};   // (T+1) x n_nodes
    const EXT_TRAJ   = {ext_json};     // (T+1) x n_inputs
    const T_FRAMES   = STATE_TRAJ.length;

    const NODE_OFF  = {{ background: "{node_off}", border: "{node_off_border}" }};
    const NODE_ON   = {{ background: "{node_on}",  border: "{node_on_border}"  }};
    const INPUT_OFF = {{ background: "{input_off}", border: "{input_off_border}" }};
    const INPUT_ON  = {{ background: "{input_on}",  border: "{input_on_border}"  }};
    const EDGE_OFF  = "{edge_off}";
    const EDGE_ON   = "{edge_on}";

    let curT = 0;
    let playing = false;
    let playerHandle = null;
    let stepMs = 200;

    function applyFrame(t) {{
      const stateRow = STATE_TRAJ[t];
      const extRow   = EXT_TRAJ[t];

      const nUpdates = [];
      for (let i = 0; i < stateRow.length; i++) {{
        nUpdates.push({{ id: "N" + i, color: stateRow[i] ? NODE_ON : NODE_OFF }});
      }}
      for (let k = 0; k < extRow.length; k++) {{
        nUpdates.push({{ id: "I" + k, color: extRow[k] ? INPUT_ON : INPUT_OFF }});
      }}
      data.nodes.update(nUpdates);

      const eUpdates = [];
      const allEdges = data.edges.get();
      for (const e of allEdges) {{
        const from = e.from;
        const live = (from[0] === "I")
          ? (extRow[parseInt(from.slice(1), 10)] | 0)
          : (stateRow[parseInt(from.slice(1), 10)] | 0);
        eUpdates.push({{
          id: e.id,
          color: {{
            color: live ? EDGE_ON : EDGE_OFF,
            opacity: live ? 0.95 : 0.35,
          }},
          width: live ? 1.8 : 1.0,
        }});
      }}
      data.edges.update(eUpdates);

      const tEl = document.getElementById("playback-time");
      if (tEl) tEl.textContent = `t = ${{t}} / ${{T_FRAMES - 1}}`;
      const scrub = document.getElementById("playback-scrub");
      if (scrub && parseInt(scrub.value, 10) !== t) scrub.value = t;
    }}

    function startPlay() {{
      if (T_FRAMES <= 1) return;
      if (playerHandle) clearInterval(playerHandle);
      playing = true;
      playerHandle = setInterval(() => {{
        curT = (curT + 1) % T_FRAMES;
        applyFrame(curT);
      }}, stepMs);
      document.getElementById("playback-play").textContent = "Pause";
    }}

    function stopPlay() {{
      playing = false;
      if (playerHandle) {{ clearInterval(playerHandle); playerHandle = null; }}
      document.getElementById("playback-play").textContent = "Play";
    }}

    document.getElementById("playback-play").addEventListener("click", () => {{
      if (playing) stopPlay(); else startPlay();
    }});

    document.getElementById("playback-scrub").addEventListener("input", (e) => {{
      curT = parseInt(e.target.value, 10);
      applyFrame(curT);
    }});

    document.getElementById("playback-speed").addEventListener("input", (e) => {{
      stepMs = parseInt(e.target.value, 10);
      document.getElementById("playback-speed-val").textContent = stepMs + "ms";
      if (playing) {{ stopPlay(); startPlay(); }}
    }});

    const playbackEl = document.getElementById("playback");
    ["pointerdown", "mousedown", "touchstart", "wheel"].forEach(ev => {{
      playbackEl.addEventListener(ev, e => e.stopPropagation(), {{ passive: true }});
    }});

    applyFrame(0);
  </script>
</body>
</html>
"""


def render_html(
    circuit: Circuit,
    batch_id: int,
    state_traj: np.ndarray,
    ext_traj: np.ndarray,
    title: str,
) -> str:
    """Return a self-contained HTML document rendering one circuit from the batch.

    Parameters
    ----------
    circuit : Circuit
    batch_id : int
        Which batch element to draw.
    state_traj : np.ndarray, shape (T+1, n_nodes), dtype uint8
        Per-step internal node states for the chosen batch element. Frame 0 is
        the initial state, frames 1..T are after each clock cycle.
    ext_traj : np.ndarray, shape (T+1, n_inputs), dtype uint8
        Per-step external input values aligned with state_traj.
    title : str
    """
    if not (0 <= batch_id < circuit.batch_size):
        raise IndexError(
            f"batch_id {batch_id} out of range for batch_size {circuit.batch_size}"
        )
    s = np.asarray(state_traj, dtype=np.uint8)
    e = np.asarray(ext_traj, dtype=np.uint8)
    if s.ndim != 2 or s.shape[1] != circuit.n_nodes:
        raise ValueError(
            f"state_traj must have shape (T+1, {circuit.n_nodes}), got {s.shape}"
        )
    if e.ndim != 2 or e.shape[1] != circuit.n_inputs:
        raise ValueError(
            f"ext_traj must have shape (T+1, {circuit.n_inputs}), got {e.shape}"
        )
    if s.shape[0] != e.shape[0]:
        raise ValueError(
            f"state_traj and ext_traj must have matching first dim, "
            f"got {s.shape[0]} vs {e.shape[0]}"
        )
    t_max = s.shape[0] - 1

    edges = _build_edges(circuit, batch_id)
    degree = _node_degrees(edges)
    nodes = _build_nodes(circuit, degree, batch_id)
    summary = (
        f"batch {batch_id}/{circuit.batch_size} &middot; "
        f"{circuit.n_inputs} inputs &middot; {circuit.n_nodes} nodes &middot; "
        f"{circuit.total_genome_bits} genome bits &middot; "
        f"sel_bits={circuit.sel_bits} &middot; T={t_max}"
    )
    return _HTML_TEMPLATE.format(
        title=title,
        summary=summary,
        bg=_BG_COLOR,
        label=_LABEL_COLOR,
        edge=_EDGE_COLOR,
        edge_hi=_EDGE_HIGHLIGHT,
        nodes_json=json.dumps(nodes, indent=2),
        edges_json=json.dumps(edges, indent=2),
        state_json=json.dumps(s.tolist()),
        ext_json=json.dumps(e.tolist()),
        t_max=t_max,
        node_off=_NODE_COLOR,
        node_off_border=_NODE_BORDER,
        node_on=_NODE_ON_COLOR,
        node_on_border=_NODE_ON_BORDER,
        input_off=_INPUT_COLOR,
        input_off_border=_INPUT_BORDER,
        input_on=_INPUT_ON_COLOR,
        input_on_border=_INPUT_ON_BORDER,
        edge_off=_EDGE_COLOR,
        edge_on=_EDGE_ON_COLOR,
    )


def write_html(
    circuit: Circuit,
    output_path: Path,
    batch_id: int,
    state_traj: np.ndarray,
    ext_traj: np.ndarray,
    title: str,
) -> Path:
    """Write the HTML rendering (with activity playback) and return its path."""
    output_path.write_text(
        render_html(circuit, batch_id, state_traj, ext_traj, title),
        encoding="utf-8",
    )
    return output_path


# ---------------------------------------------------------------------------
# Live (server-driven) variant: same network drawing + animation, plus a
# Sample panel that fetches a fresh MNIST trajectory from a backing server.
# ---------------------------------------------------------------------------

_LIVE_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    html, body {{
      margin: 0; padding: 0;
      width: 100vw; height: 100vh;
      background: {bg};
      color: {label};
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      overflow: hidden;
    }}
    #network {{ width: 100vw; height: 100vh; }}
    #header {{
      position: fixed; top: 12px; left: 12px;
      background: rgba(40, 40, 40, 0.85);
      padding: 8px 14px;
      border-radius: 6px;
      font-size: 12px;
      pointer-events: none;
      backdrop-filter: blur(6px);
      z-index: 100;
    }}
    #header b {{ color: #fff; font-size: 13px; }}
    .vis-tooltip {{
      background: #2a2a2a !important;
      color: #ddd !important;
      border: 1px solid #444 !important;
      border-radius: 4px !important;
      font-family: inherit !important;
      font-size: 12px !important;
      white-space: pre !important;
      padding: 6px 10px !important;
    }}
    .right-rail {{
      position: fixed; top: 12px; right: 12px; bottom: 80px;
      display: flex; flex-direction: column; gap: 12px;
      width: 260px;
      z-index: 100;
      pointer-events: none;          /* let clicks pass through gaps */
      overflow: hidden;              /* keep panels clipped if window shrinks */
    }}
    .right-rail > * {{
      pointer-events: auto;          /* re-enable on the panels themselves */
    }}
    .floating-panel {{
      background: rgba(40, 40, 40, 0.92);
      border-radius: 6px;
      padding: 10px 12px 12px;
      font-size: 12px;
      backdrop-filter: blur(6px);
      user-select: none;
      box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    }}
    #sample-panel {{ flex: 0 0 auto; max-height: 100%; overflow-y: auto; }}
    #controls    {{ flex: 0 0 auto; }}
    .panel-header {{
      display: flex; justify-content: space-between; align-items: center;
      font-weight: 600; color: #fff; font-size: 13px;
      margin-bottom: 6px;
    }}
    #digit-canvas {{
      width: 236px; height: 236px;
      background: #0f0f0f; border-radius: 4px;
      image-rendering: pixelated; image-rendering: crisp-edges;
      display: block;
    }}
    #sample-status {{
      display: flex; justify-content: space-between; align-items: center;
      margin-top: 8px; font-size: 12px; color: #d8dadc;
    }}
    #sample-status .pred-ok    {{ color: #9fd2ff; font-weight: 600; }}
    #sample-status .pred-bad   {{ color: #f0a070; font-weight: 600; }}
    #class-bars {{ margin-top: 8px; }}
    .cb-row {{
      display: grid; grid-template-columns: 18px 1fr 28px;
      gap: 4px; align-items: center;
      font-size: 11px; line-height: 1.5;
    }}
    .cb-label {{ color: #b8b8b8; font-variant-numeric: tabular-nums; }}
    .cb-track {{
      height: 6px; background: #2c2c2c; border-radius: 3px; overflow: hidden;
    }}
    .cb-fill {{
      height: 100%; background: #7c93b7; border-radius: 3px;
      transition: width 0.18s ease;
    }}
    .cb-row.is-pred .cb-fill   {{ background: #9fd2ff; }}
    .cb-row.is-true .cb-label  {{ color: #f0d870; font-weight: 600; }}
    .cb-count {{ color: #888; font-size: 10px; text-align: right; font-variant-numeric: tabular-nums; }}
    #sample-buttons {{
      display: flex; gap: 6px; margin-top: 10px;
    }}
    button.pb-btn {{
      flex: 1;
      background: #2f2f2f; color: #ddd;
      border: 1px solid #444; border-radius: 4px;
      padding: 5px 8px; cursor: pointer; font-size: 11px;
      font-family: inherit;
    }}
    button.pb-btn:hover {{ background: #3a3a3a; }}
    button.pb-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
    #controls .ctrl-header {{
      display: flex; justify-content: space-between; align-items: center;
      font-weight: 600; color: #fff; font-size: 13px;
      margin-bottom: 6px;
    }}
    #controls button.ctrl-icon {{
      background: transparent; color: #ddd; border: none;
      font-size: 16px; line-height: 1; cursor: pointer; padding: 0 4px;
    }}
    #controls.collapsed #ctrl-body {{ display: none; }}
    #controls .ctrl-row {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 2px 8px;
      align-items: center;
      margin: 8px 0 2px;
    }}
    #controls .ctrl-row .ctrl-label {{ color: #c0c0c0; }}
    #controls .ctrl-row .ctrl-value {{
      color: #8a8a8a; font-size: 11px; font-variant-numeric: tabular-nums;
    }}
    #controls input[type=range] {{
      grid-column: 1 / span 2;
      width: 100%;
      accent-color: #7c93b7;
      background: transparent;
      margin: 2px 0 0;
    }}
    #controls .ctrl-buttons {{
      display: flex; gap: 6px; margin-top: 12px;
    }}
    #controls button.ctrl-btn {{
      flex: 1;
      background: #2f2f2f; color: #ddd;
      border: 1px solid #444; border-radius: 4px;
      padding: 4px 8px; cursor: pointer; font-size: 11px;
      font-family: inherit;
    }}
    #controls button.ctrl-btn:hover {{ background: #3a3a3a; }}
    #playback {{
      position: fixed; bottom: 14px; left: 50%; transform: translateX(-50%);
      background: rgba(40, 40, 40, 0.92);
      border-radius: 6px;
      padding: 8px 14px;
      font-size: 12px;
      display: flex; align-items: center; gap: 12px;
      backdrop-filter: blur(6px);
      user-select: none;
      box-shadow: 0 4px 16px rgba(0,0,0,0.4);
      z-index: 100;
      width: min(640px, 92vw);
    }}
    #playback button.pb-btn {{ flex: 0 0 auto; min-width: 56px; }}
    #playback input[type=range] {{
      flex: 1; accent-color: #9fd2ff; background: transparent; margin: 0;
    }}
    #playback #playback-speed {{ flex: 0 0 110px; accent-color: #f0d870; }}
    #playback .pb-time {{
      color: #c0c0c0; font-variant-numeric: tabular-nums;
      min-width: 84px; text-align: center;
    }}
    #playback .pb-sep   {{ color: #555; }}
    #playback .pb-label {{ color: #888; }}
  </style>
</head>
<body>
  <div id="header">
    <b>{title}</b><br />
    {summary}
  </div>

  <div class="right-rail">
    <div id="sample-panel" class="floating-panel">
      <div class="panel-header"><span>Sample</span><span id="sample-idx" style="color:#888;font-weight:400">--</span></div>
      <canvas id="digit-canvas" width="280" height="280"></canvas>
      <div id="sample-status">
        <span>true: <b id="true-label">-</b></span>
        <span>pred: <b id="pred-label" class="pred-ok">-</b></span>
      </div>
      <div id="class-bars"></div>
      <div id="sample-buttons">
        <button class="pb-btn" id="btn-random">Random</button>
        <button class="pb-btn" id="btn-prev">◀ Prev</button>
        <button class="pb-btn" id="btn-next">Next ▶</button>
      </div>
    </div>

    <div id="controls" class="floating-panel">
      <div class="ctrl-header">
        <span>Forces</span>
        <button class="ctrl-icon" id="ctrl-toggle" title="Collapse">&minus;</button>
      </div>
      <div id="ctrl-body">
        <div class="ctrl-row">
          <span class="ctrl-label">Center force</span>
          <span class="ctrl-value" id="val-centralGravity">0.01</span>
          <input type="range" min="0" max="1" step="0.01" value="0.01"
                 data-param="centralGravity" data-mul="1" data-precision="2" />
        </div>
        <div class="ctrl-row">
          <span class="ctrl-label">Repel force</span>
          <span class="ctrl-value" id="val-gravitationalConstant">55</span>
          <input type="range" min="0" max="300" step="1" value="55"
                 data-param="gravitationalConstant" data-mul="-1" data-precision="0" />
        </div>
        <div class="ctrl-row">
          <span class="ctrl-label">Link distance</span>
          <span class="ctrl-value" id="val-springLength">110</span>
          <input type="range" min="10" max="500" step="5" value="110"
                 data-param="springLength" data-mul="1" data-precision="0" />
        </div>
        <div class="ctrl-buttons">
          <button class="ctrl-btn" id="ctrl-pause">Pause</button>
          <button class="ctrl-btn" id="ctrl-reset">Reset</button>
        </div>
      </div>
    </div>
  </div>

  <div id="network"></div>

  <div id="playback">
    <button class="pb-btn" id="playback-play">Play</button>
    <input type="range" id="playback-scrub" min="0" max="1" value="0" step="1" />
    <span class="pb-time" id="playback-time">t = 0 / 0</span>
    <span class="pb-sep">|</span>
    <span class="pb-label">Speed</span>
    <input type="range" id="playback-speed" min="50" max="2000" step="50" value="200" />
    <span class="pb-time" id="playback-speed-val">200ms</span>
  </div>

  <script>
    const NODES = {nodes_json};
    const EDGES = {edges_json};
    const N_CLASSES = {n_classes};
    const N_INPUTS  = {n_inputs};
    const N_NODES   = {n_nodes};
    const API_BASE  = "{api_base}";

    const data = {{
      nodes: new vis.DataSet(NODES),
      edges: new vis.DataSet(EDGES),
    }};

    const options = {{
      nodes: {{
        borderWidth: 1.5,
        scaling: {{ min: 10, max: 32, label: {{ enabled: false }} }},
        font: {{ color: "{label}", size: 12, face: "inherit", strokeWidth: 0, vadjust: -22 }},
        shadow: {{ enabled: true, color: "rgba(0,0,0,0.55)", size: 6, x: 0, y: 0 }},
      }},
      edges: {{
        color: {{ color: "{edge}", highlight: "{edge_hi}", hover: "{edge_hi}", opacity: 0.6 }},
        width: 1.0, selectionWidth: 1.6, hoverWidth: 1.6, smooth: false,
        arrows: {{ to: {{ enabled: true, scaleFactor: 0.55, type: "arrow" }} }},
      }},
      physics: {{
        solver: "forceAtlas2Based",
        forceAtlas2Based: {{
          gravitationalConstant: -55, centralGravity: 0.01,
          springLength: 110, springConstant: 0.08, damping: 0.5,
          avoidOverlap: 0.6,
        }},
        stabilization: {{ iterations: 600, fit: true }},
      }},
      interaction: {{ hover: true, dragNodes: true, zoomView: true,
                     navigationButtons: false, tooltipDelay: 120 }},
    }};

    const network = new vis.Network(
      document.getElementById("network"), data, options,
    );
    network.once("stabilizationIterationsDone", () => {{
      network.setOptions({{ physics: {{ enabled: true }} }});
    }});

    // Stop pointer events on floating panels from reaching the network.
    document.querySelectorAll(".floating-panel, #playback").forEach(el => {{
      ["pointerdown", "mousedown", "touchstart", "wheel"].forEach(ev => {{
        el.addEventListener(ev, e => e.stopPropagation(), {{ passive: true }});
      }});
    }});

    // Forces panel sliders.
    const sliders = document.querySelectorAll("#controls input[type=range]");
    sliders.forEach(slider => {{
      slider.addEventListener("input", () => {{
        const param = slider.dataset.param;
        const mul = parseFloat(slider.dataset.mul);
        const precision = parseInt(slider.dataset.precision, 10);
        const display = parseFloat(slider.value);
        const valEl = document.getElementById("val-" + param);
        if (valEl) valEl.textContent = display.toFixed(precision);
        network.setOptions({{
          physics: {{ forceAtlas2Based: {{ [param]: display * mul }} }}
        }});
      }});
    }});
    let physicsOn = true;
    document.getElementById("ctrl-pause").addEventListener("click", e => {{
      physicsOn = !physicsOn;
      network.setOptions({{ physics: {{ enabled: physicsOn }} }});
      e.target.textContent = physicsOn ? "Pause" : "Resume";
    }});
    document.getElementById("ctrl-reset").addEventListener("click", () => {{
      const defaults = {{ centralGravity: 0.01, gravitationalConstant: -55,
                          springLength: 110 }};
      sliders.forEach(slider => {{
        slider.value = defaults[slider.dataset.param] / parseFloat(slider.dataset.mul);
        slider.dispatchEvent(new Event("input"));
      }});
    }});
    document.getElementById("ctrl-toggle").addEventListener("click", e => {{
      const ctrl = document.getElementById("controls");
      ctrl.classList.toggle("collapsed");
      e.target.innerHTML = ctrl.classList.contains("collapsed") ? "+" : "&minus;";
    }});

    // ── activity playback ────────────────────────────────────────────
    let STATE_TRAJ = [new Array(N_NODES).fill(0)];
    let EXT_TRAJ   = [new Array(N_INPUTS).fill(0)];
    let T_FRAMES   = STATE_TRAJ.length;

    const NODE_OFF  = {{ background: "{node_off}",  border: "{node_off_border}" }};
    const NODE_ON   = {{ background: "{node_on}",   border: "{node_on_border}"  }};
    const INPUT_OFF = {{ background: "{input_off}", border: "{input_off_border}" }};
    const INPUT_ON  = {{ background: "{input_on}",  border: "{input_on_border}"  }};
    const EDGE_OFF  = "{edge_off}";
    const EDGE_ON   = "{edge_on}";

    let curT = 0, playing = false, playerHandle = null, stepMs = 200;

    function applyFrame(t) {{
      if (T_FRAMES <= 0) return;
      t = Math.max(0, Math.min(T_FRAMES - 1, t));
      const stateRow = STATE_TRAJ[t];
      const extRow   = EXT_TRAJ[t];
      const nUpdates = [];
      for (let i = 0; i < stateRow.length; i++) {{
        nUpdates.push({{ id: "N" + i, color: stateRow[i] ? NODE_ON : NODE_OFF }});
      }}
      for (let k = 0; k < extRow.length; k++) {{
        nUpdates.push({{ id: "I" + k, color: extRow[k] ? INPUT_ON : INPUT_OFF }});
      }}
      data.nodes.update(nUpdates);
      const eUpdates = [];
      for (const e of data.edges.get()) {{
        const from = e.from;
        const live = (from[0] === "I")
          ? (extRow[parseInt(from.slice(1), 10)] | 0)
          : (stateRow[parseInt(from.slice(1), 10)] | 0);
        eUpdates.push({{
          id: e.id,
          color: {{ color: live ? EDGE_ON : EDGE_OFF, opacity: live ? 0.95 : 0.35 }},
          width: live ? 1.8 : 1.0,
        }});
      }}
      data.edges.update(eUpdates);

      const tEl = document.getElementById("playback-time");
      if (tEl) tEl.textContent = `t = ${{t}} / ${{T_FRAMES - 1}}`;
      const scrub = document.getElementById("playback-scrub");
      if (scrub && parseInt(scrub.value, 10) !== t) scrub.value = t;
    }}

    function startPlay() {{
      if (T_FRAMES <= 1) return;
      if (playerHandle) clearInterval(playerHandle);
      playing = true;
      playerHandle = setInterval(() => {{
        curT = (curT + 1) % T_FRAMES;
        applyFrame(curT);
      }}, stepMs);
      document.getElementById("playback-play").textContent = "Pause";
    }}
    function stopPlay() {{
      playing = false;
      if (playerHandle) {{ clearInterval(playerHandle); playerHandle = null; }}
      document.getElementById("playback-play").textContent = "Play";
    }}
    document.getElementById("playback-play").addEventListener("click", () => {{
      if (playing) stopPlay(); else startPlay();
    }});
    document.getElementById("playback-scrub").addEventListener("input", e => {{
      curT = parseInt(e.target.value, 10);
      applyFrame(curT);
    }});
    document.getElementById("playback-speed").addEventListener("input", e => {{
      stepMs = parseInt(e.target.value, 10);
      document.getElementById("playback-speed-val").textContent = stepMs + "ms";
      if (playing) {{ stopPlay(); startPlay(); }}
    }});

    // ── sample panel ─────────────────────────────────────────────────
    const digitCanvas = document.getElementById("digit-canvas");
    const digitCtx = digitCanvas.getContext("2d");
    digitCtx.imageSmoothingEnabled = false;
    const offCanvas = document.createElement("canvas");
    offCanvas.width = 28; offCanvas.height = 28;
    const offCtx = offCanvas.getContext("2d");

    function drawDigit(image28x28) {{
      const im = offCtx.createImageData(28, 28);
      for (let r = 0; r < 28; r++) {{
        for (let c = 0; c < 28; c++) {{
          const v = image28x28[r][c] ? 230 : 16;
          const i = (r * 28 + c) * 4;
          im.data[i] = im.data[i + 1] = im.data[i + 2] = v;
          im.data[i + 3] = 255;
        }}
      }}
      offCtx.putImageData(im, 0, 0);
      digitCtx.clearRect(0, 0, digitCanvas.width, digitCanvas.height);
      digitCtx.drawImage(offCanvas, 0, 0, digitCanvas.width, digitCanvas.height);
    }}

    function drawClassBars(counts, trueLabel, predLabel) {{
      const container = document.getElementById("class-bars");
      const max = Math.max(1, ...counts);
      let html = "";
      for (let c = 0; c < counts.length; c++) {{
        const w = (counts[c] / max) * 100;
        const cls = (c === predLabel ? " is-pred" : "")
                  + (c === trueLabel ? " is-true" : "");
        html +=
          `<div class="cb-row${{cls}}">`
          + `<span class="cb-label">${{c}}</span>`
          + `<span class="cb-track"><span class="cb-fill" style="width:${{w}}%"></span></span>`
          + `<span class="cb-count">${{counts[c]}}</span>`
          + `</div>`;
      }}
      container.innerHTML = html;
    }}

    let currentIdx = -1;
    let lastRequestedIdx = null;

    async function loadSample(query) {{
      const url = query === "random"
        ? `${{API_BASE}}/api/sample/random`
        : `${{API_BASE}}/api/sample/${{query}}`;
      try {{
        ["btn-random", "btn-prev", "btn-next"].forEach(id => {{
          document.getElementById(id).disabled = true;
        }});
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${{resp.status}}`);
        const sample = await resp.json();

        currentIdx = sample.idx;
        STATE_TRAJ = sample.state_traj;
        EXT_TRAJ   = sample.ext_traj;
        T_FRAMES   = STATE_TRAJ.length;

        const scrub = document.getElementById("playback-scrub");
        scrub.max = Math.max(0, T_FRAMES - 1);
        scrub.value = 0;
        curT = 0;

        document.getElementById("sample-idx").textContent = `idx ${{currentIdx}}`;
        document.getElementById("true-label").textContent = sample.true_label;
        const predEl = document.getElementById("pred-label");
        predEl.textContent = sample.predicted;
        predEl.className = (sample.true_label === sample.predicted)
          ? "pred-ok" : "pred-bad";

        drawDigit(sample.image);
        drawClassBars(sample.counts, sample.true_label, sample.predicted);
        applyFrame(0);
        if (!playing) startPlay();
      }} catch (err) {{
        console.error("loadSample failed:", err);
        document.getElementById("sample-idx").textContent = "(error)";
      }} finally {{
        ["btn-random", "btn-prev", "btn-next"].forEach(id => {{
          document.getElementById(id).disabled = false;
        }});
      }}
    }}

    document.getElementById("btn-random").addEventListener("click", () => loadSample("random"));
    document.getElementById("btn-next").addEventListener("click", () => {{
      if (currentIdx >= 0) loadSample(currentIdx + 1);
    }});
    document.getElementById("btn-prev").addEventListener("click", () => {{
      if (currentIdx > 0) loadSample(currentIdx - 1);
    }});

    // Auto-load a first sample once the page is up.
    loadSample("random");
  </script>
</body>
</html>
"""


def render_html_live(circuit: Circuit, batch_id: int, title: str, api_base: str) -> str:
    """Return an HTML page that fetches MNIST samples from a backing server.

    The page renders the circuit graph and animation panel exactly like
    `render_html`, but instead of a baked-in trajectory it calls the server's
    `/api/sample/...` endpoint to get fresh state/ext trajectories on demand.

    Parameters
    ----------
    circuit : Circuit (a single circuit, batch_id must be in range)
    batch_id : int
    title : str
    api_base : str — base URL for the API (e.g. "" if same origin)
    """
    if not (0 <= batch_id < circuit.batch_size):
        raise IndexError(
            f"batch_id {batch_id} out of range for batch_size {circuit.batch_size}"
        )

    edges = _build_edges(circuit, batch_id)
    degree = _node_degrees(edges)
    nodes = _build_nodes(circuit, degree, batch_id)
    summary = (
        f"{circuit.n_inputs} inputs &middot; {circuit.n_nodes} nodes &middot; "
        f"{circuit.total_genome_bits} genome bits &middot; "
        f"sel_bits={circuit.sel_bits}"
    )
    return _LIVE_HTML_TEMPLATE.format(
        title=title,
        summary=summary,
        bg=_BG_COLOR,
        label=_LABEL_COLOR,
        edge=_EDGE_COLOR,
        edge_hi=_EDGE_HIGHLIGHT,
        nodes_json=json.dumps(nodes, indent=2),
        edges_json=json.dumps(edges, indent=2),
        node_off=_NODE_COLOR,
        node_off_border=_NODE_BORDER,
        node_on=_NODE_ON_COLOR,
        node_on_border=_NODE_ON_BORDER,
        input_off=_INPUT_COLOR,
        input_off_border=_INPUT_BORDER,
        input_on=_INPUT_ON_COLOR,
        input_on_border=_INPUT_ON_BORDER,
        edge_off=_EDGE_COLOR,
        edge_on=_EDGE_ON_COLOR,
        n_classes=10,
        n_inputs=circuit.n_inputs,
        n_nodes=circuit.n_nodes,
        api_base=api_base,
    )

