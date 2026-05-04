"""
server.py — Live demo of the best-evolved Boolean circuit on MNIST.

Loads a checkpoint produced by `evolution.py` (`best_circuit.npz`) plus the
MNIST test split, and serves an interactive HTML page where:

  - The graph view shows the circuit (vis-network) with the same animation
    playback as the static viewer.
  - The Sample panel shows the digit being classified, the true label,
    the network's predicted class (argmax of per-class spike counts), and
    a small bar chart of those counts.
  - Buttons fetch a new sample (random / next / prev). On each fetch the
    server runs the circuit on that sample and returns the trajectory; the
    front-end then animates that trajectory.

Pure stdlib server (`http.server` + `socketserver`) — no Flask, no extra deps.
"""

import http.server
import json
import socketserver
import sys
import urllib.parse
import webbrowser
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np

from circuit import Circuit
from evaluate import load_mnist_binary, run_static
from visualize import render_html_live


# ---------------------------------------------------------------------------
# Server state — populated once at startup, then read-only
# ---------------------------------------------------------------------------

class ServerState:
    """Container for everything the request handler needs to read."""

    def __init__(
        self,
        circuit: Circuit,
        n_steps: int,
        output_node_ids: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        html: str,
        train_loss: float,
        train_acc: float,
    ) -> None:
        self.circuit = circuit
        self.n_steps = n_steps
        self.output_node_ids = output_node_ids
        self.X_test = X_test
        self.y_test = y_test
        self.html = html
        self.train_loss = train_loss
        self.train_acc = train_acc
        self.rng = np.random.default_rng()


def build_state(
    checkpoint_path: Path,
    mnist_cache: Path,
    n_test_samples: int,
    threshold: float,
    api_base: str,
) -> ServerState:
    """Load checkpoint + test data and pre-render the static HTML shell."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            f"Run `python new/evolution.py` first to produce it."
        )

    data = np.load(checkpoint_path)
    genome = data["genome"]
    n_nodes = int(data["n_nodes"])
    n_inputs = int(data["n_inputs"])
    n_steps = int(data["n_steps"])
    output_node_ids = data["output_node_ids"].astype(np.int64)
    train_loss = float(data["train_loss"])
    train_acc = float(data["train_acc"])

    circuit = Circuit(genome[None, :], n_nodes, n_inputs)

    _, _, X_test, y_test = load_mnist_binary(1, n_test_samples, threshold, mnist_cache)
    print(
        f"[server] checkpoint  : {checkpoint_path.name}  "
        f"(N={n_nodes}, K={n_inputs}, T={n_steps}, train_acc={train_acc:.1%})"
    )
    print(f"[server] test split  : {X_test.shape[0]} samples")

    title = (
        f"Best evolved circuit | N={n_nodes} K={n_inputs} T={n_steps} | "
        f"train acc={train_acc:.1%}"
    )
    html = render_html_live(circuit, batch_id=0, title=title, api_base=api_base)

    return ServerState(
        circuit=circuit,
        n_steps=n_steps,
        output_node_ids=output_node_ids,
        X_test=X_test,
        y_test=y_test,
        html=html,
        train_loss=train_loss,
        train_acc=train_acc,
    )


# ---------------------------------------------------------------------------
# Sample evaluation: run the circuit on one test image
# ---------------------------------------------------------------------------

def evaluate_sample(state: ServerState, idx: int) -> dict:
    """Run circuit on test sample `idx`, return JSON-ready dict."""
    n_test = state.X_test.shape[0]
    if not (0 <= idx < n_test):
        raise IndexError(f"sample idx {idx} out of range [0, {n_test})")

    K = state.circuit.n_inputs
    N = state.circuit.n_nodes
    T = state.n_steps

    x = state.X_test[idx]                                       # (K,) uint8
    true_label = int(state.y_test[idx])

    # Run static: hold the binarized digit at the inputs for T cycles.
    ext = np.broadcast_to(x, (1, K))                            # (1, K)
    trajectory = run_static(state.circuit, ext, T)              # (T+1, 1, N)

    state_seq = trajectory[:, 0, :]                             # (T+1, N)
    ext_seq = np.broadcast_to(x, (T + 1, K))                    # (T+1, K)

    # Predicted class: argmax of spike counts over t=1..T at the output nodes
    # (ignore t=0 which is the all-zero initial state).
    out_bits = state_seq[1:, state.output_node_ids]             # (T, n_classes)
    counts = out_bits.sum(axis=0)                               # (n_classes,)
    predicted = int(counts.argmax())

    return {
        "idx": idx,
        "true_label": true_label,
        "predicted": predicted,
        "image": x.reshape(28, 28).astype(int).tolist(),
        "counts": counts.astype(int).tolist(),
        "state_traj": state_seq.astype(int).tolist(),
        "ext_traj": ext_seq.astype(int).tolist(),
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

def make_handler(state: ServerState) -> type:
    class Handler(http.server.BaseHTTPRequestHandler):
        # Suppress default access-log noise on stderr.
        def log_message(self, format: str, *args) -> None:
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")

        def _send_json(self, obj: dict, status: int) -> None:
            body = json.dumps(obj).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path == "/" or path == "/index.html":
                self._send_html(state.html)
                return

            if path == "/api/sample/random":
                idx = int(state.rng.integers(0, state.X_test.shape[0]))
                self._send_json(evaluate_sample(state, idx), 200)
                return

            if path.startswith("/api/sample/"):
                tail = path[len("/api/sample/"):]
                try:
                    idx = int(tail)
                except ValueError:
                    self._send_json({"error": f"bad idx '{tail}'"}, 400)
                    return
                try:
                    self._send_json(evaluate_sample(state, idx), 200)
                except IndexError as e:
                    self._send_json({"error": str(e)}, 400)
                return

            self.send_error(404, f"not found: {path}")

    return Handler


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PORT = 8000
    HOST = "127.0.0.1"
    OPEN_BROWSER = True
    N_TEST_SAMPLES = 1000
    THRESHOLD = 0.5

    REPO_ROOT = Path(__file__).resolve().parent.parent
    CHECKPOINT_PATH = Path(__file__).resolve().parent / "best_circuit.npz"
    MNIST_CACHE = REPO_ROOT / "datasets" / "mnist"
    API_BASE = ""   # same-origin

    state = build_state(
        checkpoint_path=CHECKPOINT_PATH,
        mnist_cache=MNIST_CACHE,
        n_test_samples=N_TEST_SAMPLES,
        threshold=THRESHOLD,
        api_base=API_BASE,
    )

    handler_cls = make_handler(state)
    with socketserver.ThreadingTCPServer((HOST, PORT), handler_cls) as httpd:
        httpd.allow_reuse_address = True
        url = f"http://{HOST}:{PORT}"
        print(f"[server] listening on {url}")
        if OPEN_BROWSER:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[server] stopping")
