"""The renderer's job API with a fake backend: submit, attach by key, progress, the disk cache of finished jobs,
cancellation, one job at a time, and the client the gpu Activities use. The Isaac Sim backend itself needs a GPU."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from renderer.fake_backend import FakeKit
from renderer.jobs import CancelledJob, JobRegistry
from renderer.server import run_forever_in_thread, serve
from renderer_client import RendererClient, RendererError


class Frames:
    """Renders `frames` frames at `per_frame` seconds each."""

    def __init__(self, per_frame: float = 0.05) -> None:
        self.per_frame = per_frame
        self.calls = 0

    def run(self, job, progress, cancelled):
        self.calls += 1
        frames = int(job.params.get("frames", 4))
        progress({"phase": "render", "frames_total": frames})
        for i in range(frames):
            if cancelled():
                raise CancelledJob()
            time.sleep(self.per_frame)
            progress({"frames_done": i + 1, "pct": int(100 * (i + 1) / frames)})
        return {"output_path": f"out/{job.key}.png", "frames": frames}


def req(method: str, url: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture
def service(tmp_path):
    started = []

    def start(backend):
        reg = JobRegistry(cache_dir=tmp_path / "cache")
        srv = serve(reg, port=0)
        stop = threading.Event()
        run_forever_in_thread(reg, backend, stop)
        started.append((srv, stop))
        return reg, f"http://127.0.0.1:{srv.server_address[1]}"

    yield start
    for srv, stop in started:
        stop.set()
        srv.shutdown()


def test_submit_attach_progress_and_cached_result(service, tmp_path):
    backend = Frames()
    reg, base = service(backend)
    c1, j1 = req("POST", f"{base}/jobs", {"key": "k1", "op": "render_still", "params": {"frames": 6}})
    assert c1 == 202 and j1["status"] in ("queued", "running") and j1["attached"] is False
    c2, j2 = req("POST", f"{base}/jobs", {"key": "k1", "op": "render_still", "params": {"frames": 6}})
    assert c2 == 409 and j2["attached"] is True and j2["adopters"] == 1, "a retried Activity attaches to the running job"
    for _ in range(100):
        _, j = req("GET", f"{base}/jobs/k1")
        if j["status"] == "done":
            break
        time.sleep(0.05)
    assert j["status"] == "done" and j["result"]["frames"] == 6 and j["progress"]["pct"] == 100
    assert backend.calls == 1, "attaching never renders twice"
    assert req("GET", f"{base}/healthz")[1]["ready"] is True
    assert JobRegistry(cache_dir=tmp_path / "cache").get("k1").result["frames"] == 6, "a restarted service still knows the finished job"


def test_cancel_running_and_queued_jobs(service):
    reg, base = service(Frames(per_frame=0.1))
    req("POST", f"{base}/jobs", {"key": "slow", "op": "render_still", "params": {"frames": 50}})
    req("POST", f"{base}/jobs", {"key": "queued", "op": "render_still", "params": {"frames": 1}})
    time.sleep(0.25)
    assert req("DELETE", f"{base}/jobs/queued")[0] == 202
    assert req("DELETE", f"{base}/jobs/slow")[0] == 202
    for _ in range(50):
        if req("GET", f"{base}/jobs/slow")[1]["status"] == "cancelled":
            break
        time.sleep(0.05)
    assert req("GET", f"{base}/jobs/slow")[1]["status"] == "cancelled" and req("GET", f"{base}/jobs/queued")[1]["status"] == "cancelled"
    assert req("DELETE", f"{base}/jobs/slow")[0] == 409 and req("DELETE", f"{base}/jobs/nope")[0] == 404
    assert req("POST", f"{base}/jobs", {"key": "slow", "op": "render_still", "params": {"frames": 2}})[0] == 202, "a retried Activity runs the job again"


def test_one_job_at_a_time(service):
    _, base = service(Frames(per_frame=0.05))
    for k in ("a", "b", "c"):
        req("POST", f"{base}/jobs", {"key": k, "op": "render_still", "params": {"frames": 2}})
    running = set()
    for _ in range(60):
        states = [req("GET", f"{base}/jobs/{k}")[1]["status"] for k in ("a", "b", "c")]
        running.add(states.count("running"))
        if states == ["done"] * 3:
            break
        time.sleep(0.03)
    assert states == ["done"] * 3 and max(running) <= 1


async def test_the_client_runs_both_operations_and_reports_failures(service, tmp_path: Path):
    _, base = service(FakeKit(tmp_path / "out", lane_passes_after_step=2))
    client = RendererClient(base)
    try:
        seen: list[dict] = []
        still = await client.run("r1", "render_still", {"camera": "/World/Cameras/Station", "layers": ["a.usda"]}, on_progress=seen.append, poll=0.05)
        assert Path(still["output_path"]).read_bytes().startswith(b"\x89PNG") and any(p.get("status") == "done" for p in seen)
        lane = await client.run("l1", "simulate_lane", {"lane": "L2", "layers": ["a.usda"]}, poll=0.05)
        assert lane["passed"] is False and lane["contacts"] == ["/World/Agent/staging_1"] and lane["drive"]["reached"] is False
        with pytest.raises(RendererError, match="unknown op"):
            await client.run("x1", "export", {}, poll=0.05)
    finally:
        await client.aclose()
