"""Loopback server for replay, account trace and joint target research."""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

sys.path.insert(0, os.path.dirname(__file__))
import replay

_HERE = os.path.dirname(__file__)
PORT = int(os.environ.get("PORT", "8000"))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/index.html", "/trace", "/trace.html"):
            self._file("replay.html", "text/html")
        elif path in ("/research", "/research.html"):
            self._file("research.html", "text/html")
        elif path in ("/replay.js", "/history.js", "/trace.js", "/rolling.js", "/risk.js", "/research.js", "/replay.css"):
            self._file(path[1:], "text/css" if path.endswith(".css") else "text/javascript")
        else:
            self._send(404, {"error": "not found"})

    def _file(self, name, ctype):
        with open(os.path.join(_HERE, name), "rb") as stream:
            content = stream.read()
        if name in ("replay.html", "research.html"):
            content = content.replace(b'data-runtime="browser"', b'data-runtime="server"')
        self._send(200, content, ctype)

    def do_POST(self):
        handlers = {"/api/replay": replay.run, "/api/generate": replay.generate, "/api/manual": replay.manual}
        handler = handlers.get(urlsplit(self.path).path)
        if handler is None:
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 6_000_000:
                self._send(413, {"error": "Request exceeds the 6 MB limit"})
                return
            request = json.loads(self.rfile.read(length) or b"{}")
            self._send(200, handler(request))
        except Exception as exc:
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, *args):
        pass


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Propfirm Engine: http://localhost:{PORT} (replay) /trace.html (trace) /research.html (target search)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
