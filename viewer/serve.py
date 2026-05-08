"""
serve.py — Serve the viewer directory over HTTP for live evolution monitoring.

Run from repo root:
    python viewer/serve.py

Then open http://localhost:8765/evo.html in a browser.
While an experiment is running the page polls evo_live.json every 2 seconds.
"""

import http.server
import os
from pathlib import Path

PORT = 8765
os.chdir(Path(__file__).resolve().parent)

print(f"Serving viewer at http://localhost:{PORT}/evo.html")
print("Press Ctrl+C to stop.")
http.server.test(HandlerClass=http.server.SimpleHTTPRequestHandler, port=PORT, bind="127.0.0.1")
