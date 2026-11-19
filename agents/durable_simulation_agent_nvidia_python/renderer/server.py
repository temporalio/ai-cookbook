"""The renderer's HTTP API over the job registry. Standard library only: it runs in Isaac Sim's own Python.

POST /jobs {key, op, params} -> 202 (created) | 409 (attached: existing status+result) | 400 (invalid)
GET  /jobs/<key>            -> 200 job | 404
DELETE /jobs/<key>          -> 202 cancelling | 409 already finished | 404
GET  /healthz               -> {ready, queue_depth, backend}
The runner (Kit's main loop on the GPU host, a thread in tests) drains the registry; handlers never block on it.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from renderer.jobs import Backend, JobRegistry


def make_handler(registry: JobRegistry, info: dict[str, Any]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:  # quiet
            pass

        def _send(self, code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body, sort_keys=True).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send(200, {"ready": True, "queue_depth": registry.queue_depth(), **info})
                return
            if self.path.startswith("/jobs/"):
                job = registry.get(self.path.split("/", 2)[2])
                self._send(200, job.public()) if job else self._send(404, {"error": {"type": "NotFound"}})
                return
            self._send(404, {"error": {"type": "NotFound"}})

        def do_POST(self) -> None:
            if self.path != "/jobs":
                self._send(404, {"error": {"type": "NotFound"}})
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n < 0:
                    raise ValueError("invalid Content-Length")
                body = json.loads(self.rfile.read(n) or b"{}")
                job, created = registry.submit(body["key"], body["op"], body.get("params", {}))
            except (KeyError, TypeError, ValueError) as e:
                self._send(400, {"error": {"type": "InvalidParams", "msg": str(e)}})
                return
            self._send(202 if created else 409, {**job.public(), "attached": not created})

        def do_DELETE(self) -> None:
            if not self.path.startswith("/jobs/"):
                self._send(404, {"error": {"type": "NotFound"}})
                return
            key = self.path.split("/", 2)[2]
            outcome = registry.cancel(key)
            code = {"cancelling": 202, "done": 409, "missing": 404}[outcome]
            self._send(code, {"key": key, "status": outcome})

    return Handler


def serve(registry: JobRegistry, host: str = "127.0.0.1", port: int = 8799, info: dict[str, Any] | None = None) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(registry, info or {}))
    threading.Thread(target=server.serve_forever, daemon=True, name="renderer-http").start()
    return server


def run_forever_in_thread(registry: JobRegistry, backend: Backend, stop: threading.Event, idle_sleep: float = 0.05) -> threading.Thread:
    """Test and laptop runner: a thread draining the registry. Inside Isaac Sim, the main loop does this between Kit updates."""
    import time

    def loop() -> None:
        while not stop.is_set():
            if not registry.run_one(backend):
                time.sleep(idle_sleep)

    t = threading.Thread(target=loop, daemon=True, name="renderer-runner")
    t.start()
    return t
