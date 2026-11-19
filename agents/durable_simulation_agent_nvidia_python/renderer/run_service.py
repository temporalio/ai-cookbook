"""Run the renderer's job service with a backend: `kit` inside Isaac Sim's Python on the GPU host (the Dockerfile's
entrypoint), or `fake` on a laptop to try the job API.

    /isaac-sim/python.sh -m renderer.run_service --backend kit --host 0.0.0.0 --port 8799 --out /srv/renderer/out
    uv run python -m renderer.run_service --backend fake --port 8799 --out /tmp/renderer-out
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

from renderer.jobs import JobRegistry
from renderer.server import run_forever_in_thread, serve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["fake", "kit"], default="fake")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--out", default="/srv/renderer/out", help="job outputs; the gpu Worker copies them from here, so it must see the same path")
    ap.add_argument("--cache", default=None, help="finished-job cache (default: <out>/jobcache)")
    args = ap.parse_args()
    out = Path(args.out)
    registry = JobRegistry(cache_dir=Path(args.cache) if args.cache else out / "jobcache")
    if args.backend == "fake":
        from renderer.fake_backend import FakeKit

        stop = threading.Event()
        run_forever_in_thread(registry, FakeKit(out), stop)
        srv = serve(registry, host=args.host, port=args.port, info={"backend": "fake"})
        print(f"[renderer] fake backend on http://{args.host}:{srv.server_address[1]}", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            stop.set()
            srv.shutdown()
    else:
        from renderer.kit_service import run_kit_service  # Isaac Sim's Python only

        run_kit_service(registry, host=args.host, port=args.port, out=out)


if __name__ == "__main__":
    main()
