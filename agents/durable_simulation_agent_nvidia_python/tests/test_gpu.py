"""The gpu Task Queue's Activities: the CPU plan view and the geometry lane check on real layers, and the same Activities
as renderer jobs when RENDERER_URL points at the recipe's renderer service (here with its fake backend)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from temporalio.testing import ActivityEnvironment

from activities import gpu
from activities.openshell import FakeSandboxes
from models import LayerRef, RenderInput, SimulateInput
from renderer.fake_backend import FakeKit
from renderer.jobs import JobRegistry
from renderer.server import run_forever_in_thread, serve
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


async def test_renderer_jobs_when_a_renderer_is_configured(fake_env, layer, monkeypatch):
    """The same Activities against the recipe's renderer service, with its fake backend standing in for Isaac Sim."""
    registry = JobRegistry(cache_dir=fake_env / "renderer-cache")
    backend = FakeKit(fake_env / "renderer-out", lane_passes_after_step=1)
    server = serve(registry, port=0)
    stop = threading.Event()
    run_forever_in_thread(registry, backend, stop)
    monkeypatch.setenv("RENDERER_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("RENDERER_SHARED_DIR", str(fake_env / "shared"))
    try:
        env = ActivityEnvironment()
        ref = await env.run(gpu.render_preview, RenderInput(run_id="wf-gpu", step=1, layers=[layer], camera="Station", width=1280, height=720))
        assert ref.mode == "rt2" and ref.path.endswith("001-Station-rt2.png") and Path(ref.path).read_bytes().startswith(b"\x89PNG")
        chk = await env.run(gpu.simulate_lane, SimulateInput(run_id="wf-gpu", step=1, layers=[layer], lane="L2", robot="carter", min_clearance_m=0.6))
        assert chk.passed and chk.method == "fake" and chk.reached is True and chk.clip_path.endswith("001-L2.mp4")
        jobs = backend.jobs
        assert [j.op for j in jobs] == ["render_still", "simulate_lane"]
        assert jobs[0].params["camera"] == "/World/Cameras/Station" and jobs[0].params["mode"] == "rt2" and jobs[0].params["base_ref"] == gpu.BASE_REF
        assert jobs[0].params["layers"] == [str(fake_env / "shared" / "layers" / f"{layer.sha256}.usda")], "the renderer reads linted copies by content hash"
        assert jobs[1].params["lane"] == "L2" and jobs[1].params["render_clip"] is True and jobs[1].key != jobs[0].key
    finally:
        stop.set()
        server.shutdown()
