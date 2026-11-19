"""The entrypoint inside Isaac Sim's Python: a SimulationApp, the HTTP job API, and the main-thread job loop."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def run_kit_service(registry, *, host: str, port: int, out: Path) -> None:
    from isaacsim import SimulationApp  # must come before any other omni import

    app = SimulationApp({"headless": True, "renderer": "RealTimePathTracing", "width": 1920, "height": 1080})
    sys.path.insert(0, os.environ.get("RENDERER_HOME", "/opt/renderer"))
    from renderer.jobs import InvalidJob
    from renderer.kit_backend import KitBackend
    from renderer.server import serve
    from validate.lint import lint_layer

    asset_root = os.environ.get("ISAACSIM_ASSET_ROOT", "/opt/assets")
    catalog = Path(os.environ.get("RENDERER_SITE_LAYER", "/opt/renderer/site/000_site.usda")).with_name("catalog.json")  # baked into the image

    def lint(path: str) -> None:  # the gpu Worker linted every layer already; Kit still never opens one that fails
        report = lint_layer(path, catalog=catalog, asset_root=asset_root)
        if not report.ok:
            raise InvalidJob(f"lint rejected {path}: {report.errors[:3]}")

    backend = KitBackend(app, out, lint=lint)
    srv = serve(registry, host=host, port=port, info={"backend": "kit"})
    print(f"[renderer] kit backend on http://{host}:{srv.server_address[1]}", flush=True)
    try:
        while app.is_running():
            if not registry.run_one(backend):  # synchronous; Replicator pumps the app during a render
                app.update()
    finally:
        srv.shutdown()
        app.close()
