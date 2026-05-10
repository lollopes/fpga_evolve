"""
serve.py — Serve the viewer directory over HTTP for live evolution monitoring.

Run from repo root:
    python viewer/serve.py

Then open http://localhost:8765/evo.html in a browser.
While an experiment is running the page polls evo_live.json every 2 seconds.
Use the experiment selector dropdown to switch between saved runs.
"""

import http.server
import json
import os
from pathlib import Path

PORT = 8765
os.chdir(Path(__file__).resolve().parent)


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/api/experiments'):
            evo_dir = Path('evo_data')
            experiments = sorted(
                d.name for d in evo_dir.iterdir()
                if d.is_dir() and (d / 'evo_live.json').exists()
            ) if evo_dir.exists() else []
            body = json.dumps(experiments).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def log_message(self, fmt, *args):
        pass  # suppress per-request noise


print(f"Serving viewer at http://localhost:{PORT}/evo.html")
print("Press Ctrl+C to stop.")
http.server.test(HandlerClass=ViewerHandler, port=PORT, bind="127.0.0.1")
