"""Tiny stdlib web server for the validation dashboard.

Run: ``python dashboard/server.py`` then open http://localhost:8000 .
No third-party dependencies — just the standard library plus the engine. Every
number the UI shows is computed by the real engine via :mod:`bridge`.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(__file__))

import montecarlo as mc  # noqa: E402
from accounts import list_registry  # noqa: E402
from bridge import evaluate  # noqa: E402

_HERE = os.path.dirname(__file__)
PORT = int(os.environ.get("PORT", "8000"))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._file("index.html")
        elif self.path in ("/montecarlo", "/montecarlo.html"):
            self._file("montecarlo.html")
        elif self.path == "/api/registry":
            self._send(200, list_registry())
        elif self.path == "/api/mc/registry":
            self._send(200, mc.registry())
        else:
            self._send(404, {"error": "not found"})

    def _file(self, name):
        with open(os.path.join(_HERE, name), "rb") as f:
            self._send(200, f.read(), "text/html")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) or b"{}"
        try:
            req = json.loads(raw)
            if self.path == "/api/evaluate":
                result = evaluate(req["firm"], req["atype"], req["size"],
                                  req["role"], req.get("days", []),
                                  intraday_mode=req.get("intraday_mode", "strict"))
            elif self.path == "/api/montecarlo":
                result = mc.run(req)
            else:
                self._send(404, {"error": "not found"})
                return
            self._send(200, result)
        except Exception as exc:  # surface engine/validation errors to the UI
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, *args):  # keep the console quiet
        pass


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Prop-firm engine dashboard running at http://localhost:{PORT}")
    print("Every number is computed by the real reference simulator. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
