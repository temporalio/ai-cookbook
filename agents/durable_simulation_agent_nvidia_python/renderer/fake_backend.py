"""A backend that pretends to be Isaac Sim: tiny PNG and MP4 outputs, progress, cancellation. It exercises the job API on
a laptop and in the tests (`uv run python -m renderer.run_service --backend fake`)."""

from __future__ import annotations

import time
from pathlib import Path

from renderer.jobs import CancelledJob, Job

PNG_1x1 = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360000002000154a24f5f0000000049454e44ae426082")


class FakeKit:
    def __init__(self, out: Path, per_frame: float = 0.02, lane_passes_after_step: int = 1) -> None:
        self.out = out
        self.per_frame = per_frame
        self.lane_passes_after_step = lane_passes_after_step  # the lane check passes once the stack has at least this many layers
        self.jobs: list[Job] = []

    def run(self, job: Job, progress, cancelled):
        self.jobs.append(job)
        d = self.out / "jobs" / job.key
        d.mkdir(parents=True, exist_ok=True)
        frames = 2 if job.op == "render_still" else 1
        progress({"phase": "render", "frames_total": frames})
        for i in range(frames):
            if cancelled():
                raise CancelledJob()
            time.sleep(self.per_frame)
            progress({"frames_done": i + 1, "pct": int(100 * (i + 1) / frames)})
        if job.op == "render_still":
            p = d / "rgb_0000.png"
            p.write_bytes(PNG_1x1)
            return {"output_path": str(p), "mode": job.params.get("mode", "rt2"), "layers": job.params["layers"]}
        if job.op == "simulate_lane":
            passed = len(job.params.get("layers", [])) >= self.lane_passes_after_step
            p = d / "lane.mp4"
            p.write_bytes(b"\x00\x00\x00\x18ftypmp42")
            return {"passed": passed, "min_clearance_m": 0.9 if passed else 0.41, "contacts": [] if passed else ["/World/Agent/staging_1"],
                    "clip_path": str(p), "method": "fake", "drive": {"ok": True, "reached": passed}, "trajectory": [[0.0, 3.4, -11.0], [1.0, 3.4, -10.7]]}
        raise ValueError(f"unknown op {job.op}")
