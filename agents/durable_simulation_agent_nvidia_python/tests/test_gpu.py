"""The gpu Task Queue's Activities: the CPU plan view and the geometry lane check on real layers, and the same Activities
as renderer jobs when RENDERER_URL points at a renderer service (here: a stub of the demo's job API)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from temporalio.testing import ActivityEnvironment

from activities import gpu
from activities.openshell import FakeSandboxes
from models import LayerRef, RenderInput, SimulateInput
from sceneops.hashing import sha256_file

CODE = "h.place_asset('packing_table', -3.0, 2.0, 90, name='station_1')\nh.place_asset('pallet', 1.7, -3.0, 0, name='staging_1')\n"


@pytest.fixture
async def layer(fake_env) -> LayerRef:
    fake = FakeSandboxes()
    await fake.ensure("g-exec", "exec", [])
    env = (await fake.apply("g-exec", 1, CODE, {}, 0.6)).envelope
    path = fake_env / "wf-gpu" / "layers" / "001.usda"
    path.parent.mkdir(parents=True)
    await fake.download_layer("g-exec", 1, path)
    return LayerRef(step=1, path=str(path), sha256=env["data"]["sha256"], bytes=path.stat().st_size, summary="step 1")


async def test_cpu_plan_view_and_geometry_lane_check(fake_env, layer):
    env = ActivityEnvironment()
    ref = await env.run(gpu.render_preview, RenderInput(run_id="wf-gpu", step=1, layers=[layer], camera="Station", width=640, height=360))
    assert ref.mode == "plan" and ref.path.endswith("001-Station-plan.png") and ref.sha256 == sha256_file(ref.path)
    again = await env.run(gpu.render_preview, RenderInput(run_id="wf-gpu", step=1, layers=[layer], camera="Station", width=640, height=360))
    assert again.sha256 == ref.sha256, "idempotent by output path"
    chk = await env.run(gpu.simulate_lane, SimulateInput(run_id="wf-gpu", step=1, layers=[layer], lane="L2", robot="carter", min_clearance_m=0.6))
    assert chk.passed is False and chk.contacts == ["/World/Agent/staging_1"] and chk.method == "geometry" and chk.clip_path is None
    assert json.loads((fake_env / "wf-gpu" / "lane" / "001-L2.json").read_text())["contacts"] == ["/World/Agent/staging_1"]


class _StubRenderer(BaseHTTPRequestHandler):
    """The demo renderer's job API: POST /jobs (202, or 409 when the key is known), GET /jobs/<key>."""

    jobs: dict[str, dict] = {}
    root = None

    def log_message(self, *args):  # quiet
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        known = body["key"] in self.jobs
        if not known:
            self.jobs[body["key"]] = {"status": "running", "op": body["op"], "params": body["params"], "polls": 0}
        self.send_response(409 if known else 202)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"key": body["key"], "status": "running", "adopters": int(known)}).encode())

    def do_GET(self):
        key = self.path.rsplit("/", 1)[-1]
        job = self.jobs[key]
        job["polls"] += 1
        if job["polls"] >= 2:
            d = self.root / key
            d.mkdir(parents=True, exist_ok=True)
            if job["op"] == "render_still":
                (d / "rgb_0000.png").write_bytes(b"\x89PNG\r\n\x1a\n")
                job.update(status="done", result={"output_path": str(d / "rgb_0000.png")})
            else:
                (d / "lane.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
                job.update(status="done", result={"passed": True, "min_clearance_m": 0.9, "contacts": [], "clip_path": str(d / "lane.mp4"), "method": "physx",
                                                  "drive": {"ok": True, "reached": True}})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"key": key, "status": job["status"], "progress": {"phase": "render"}, "result": job.get("result")}).encode())


async def test_renderer_jobs_when_a_renderer_is_configured(fake_env, layer, monkeypatch):
    _StubRenderer.jobs, _StubRenderer.root = {}, fake_env / "renderer-out"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubRenderer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("RENDERER_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("RENDERER_SHARED_DIR", str(fake_env / "shared"))
    try:
        env = ActivityEnvironment()
        ref = await env.run(gpu.render_preview, RenderInput(run_id="wf-gpu", step=1, layers=[layer], camera="Station", width=1280, height=720))
        assert ref.mode == "rt2" and ref.path.endswith("001-Station-rt2.png") and (fake_env / "wf-gpu" / "renders" / "001-Station-rt2.png").read_bytes().startswith(b"\x89PNG")
        chk = await env.run(gpu.simulate_lane, SimulateInput(run_id="wf-gpu", step=1, layers=[layer], lane="L2", robot="carter", min_clearance_m=0.6))
        assert chk.passed and chk.method == "physx" and chk.reached is True and chk.clip_path.endswith("001-L2.mp4")
        jobs = list(_StubRenderer.jobs.values())
        assert [j["op"] for j in jobs] == ["render_still", "simulate_lane"]
        assert jobs[0]["params"]["camera"] == "/World/Cameras/Station" and jobs[0]["params"]["mode"] == "rt2" and jobs[0]["params"]["base_ref"] == gpu.BASE_REF
        assert jobs[0]["params"]["layers"] == [str(fake_env / "shared" / "layers" / f"{layer.sha256}.usda")], "the renderer reads linted copies by content hash"
        assert jobs[1]["params"]["lane"] == "L2" and jobs[1]["params"]["render_clip"] is True
    finally:
        server.shutdown()
