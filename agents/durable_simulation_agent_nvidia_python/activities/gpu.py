"""The `gpu` Task Queue: previews and lane checks. With RENDERER_URL set, both are jobs for the Isaac Sim renderer service
of the full demo (github.com/temporalio/openshell-omniverse-demo, `renderer/`): an RTX still from a named camera and a
PhysX drive-through of the Nova Carter along the lane. Without it, the same Activities draw a plan view and check the lane
geometry on the CPU. Either way the Workflow sees a RenderRef and a LaneCheck, and the records land in runs/<workflow-id>/."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from temporalio import activity
from temporalio.exceptions import ApplicationError

import relay
from activities.sandbox import _layer_file, _phase, heartbeating
from models import (
    CloseInput,
    LaneCheck,
    LayerRef,
    RenderInput,
    RenderRef,
    SimulateInput,
)
from paths import ASSET_ROOT, CATALOG, SITE_LAYER, run_dir, runs_dir
from renderer_client import RendererClient, RendererError
from sceneops.hashing import sha256_file
from validate.judge import lane_check
from validate.lint import lint_layer
from validate.plan import render_plan

BASE_REF = os.environ.get("RENDERER_BASE_REF", "Isaac/Environments/Simple_Warehouse/full_warehouse.usd")  # the warehouse stage the renderer opens
PREVIEW = {"mode": "rt2", "total_spp": 64, "subframes": 16}


def renderer_url() -> str | None:
    return os.environ.get("RENDERER_URL") or None


def _job_key(op: str, layers: list[LayerRef], params: dict[str, Any]) -> str:
    """Idempotency key for a renderer job, computed here so a retried Activity attaches to the running job."""
    doc = {"op": op, "base_ref": BASE_REF, "layers": [layer.sha256 for layer in layers], "params": params}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _layer_paths(layers: list[LayerRef]) -> list[Path]:
    """Every layer verified against its recorded hash and linted again before any renderer opens it (defense in depth)."""
    paths = []
    for layer in layers:
        p = _layer_file(layer, "render")
        report = lint_layer(p, asset_root=ASSET_ROOT)
        if not report.ok:
            raise ApplicationError(f"layer {layer.step} rejected by lint: {'; '.join(report.errors)[:500]}", type="LintRejected", non_retryable=True)
        paths.append(p)
    return paths


def _atomic_copy(src: str | Path, dst: Path) -> None:
    """Copy, then rename into place: a Worker killed mid-copy never leaves a truncated file that a retry would trust."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.part")
    shutil.copyfile(src, tmp)
    tmp.replace(dst)


def _shared_paths(paths: list[Path]) -> list[str]:
    """The renderer reads layers from a directory it shares with this Worker (RENDERER_SHARED_DIR), by content hash."""
    shared = Path(os.environ.get("RENDERER_SHARED_DIR", runs_dir() / "_renderer")) / "layers"
    out = []
    for p in paths:
        sha = sha256_file(p)
        dst = shared / f"{sha}.usda"
        if not dst.exists() or sha256_file(dst) != sha:
            _atomic_copy(p, dst)
        out.append(str(dst))
    return out


async def _job(op: str, params: dict[str, Any], key: str, hb: dict[str, Any], timeout: float) -> dict[str, Any]:
    client = RendererClient(renderer_url())
    try:
        return await client.run(key, op, params, on_progress=lambda u: _phase(hb, f"{op}:{u.get('status') or u.get('phase')}"), timeout=timeout,
                                cancelled=activity.is_cancelled)
    except RendererError as e:
        raise ApplicationError(f"{op}: {e}", type="RenderFailed", non_retryable=not e.retryable) from e
    finally:
        await client.aclose()


@activity.defn
@relay.traced
async def render_preview(inp: RenderInput) -> RenderRef:
    """A still of the current layer stack from a named camera of the site layer: Isaac Sim RTX through the renderer
    service, or the CPU plan view. Idempotent by output path: every output is renamed into place only once complete."""
    cam = inp.camera.rsplit("/", 1)[-1]
    mode = PREVIEW["mode"] if renderer_url() else "plan"
    out = run_dir(inp.run_id) / "renders" / f"{inp.step:03d}-{cam}-{mode}.png"
    if out.exists():
        return RenderRef(step=inp.step, camera=cam, mode=mode, path=str(out), sha256=sha256_file(out))
    paths = _layer_paths(inp.layers)
    async with heartbeating() as hb:
        if renderer_url():
            params = {**PREVIEW, "camera": f"/World/Cameras/{cam}", "width": inp.width, "height": inp.height}
            key = _job_key("render_still", inp.layers, params)
            _phase(hb, "submit")
            result = await _job("render_still", {**params, "base_ref": BASE_REF, "layers": _shared_paths(paths), "job_key": key}, key, hb, timeout=1800)
            _atomic_copy(result["output_path"], out)
        else:
            _phase(hb, "plan")
            render_plan(paths, site_layer=SITE_LAYER, catalog=CATALOG, out=out, asset_root=ASSET_ROOT, width=inp.width, height=inp.height,
                        title=f"step {inp.step} · {cam} · plan view (CPU stand-in for the Isaac Sim render)")
    return RenderRef(step=inp.step, camera=cam, mode=mode, path=str(out), sha256=sha256_file(out))


@activity.defn
@relay.traced
async def simulate_lane(inp: SimulateInput) -> LaneCheck:
    """Can the AMR still use the lane? Isaac Sim drives the Nova Carter through the layout (PhysX contacts, clearance,
    goal reached, a clip), or the CPU checks footprints against the lane volume. The result is recorded next to the layers."""
    paths = _layer_paths(inp.layers)
    clip = None
    async with heartbeating() as hb:
        if renderer_url():
            params = {"lane": inp.lane, "robot": inp.robot, "min_clearance_m": inp.min_clearance_m, "render_clip": True}
            key = _job_key("simulate_lane", inp.layers, params)
            _phase(hb, "submit")
            result = await _job("simulate_lane", {**params, "base_ref": BASE_REF, "layers": _shared_paths(paths), "job_key": key}, key, hb, timeout=1800)
            if result.get("clip_path"):
                clip = run_dir(inp.run_id) / "clips" / f"{inp.step:03d}-{inp.lane}.mp4"
                _atomic_copy(result["clip_path"], clip)
            drive = result.get("drive") or {}
            check = {"passed": bool(result["passed"]), "min_clearance_m": float(result.get("min_clearance_m") if result.get("min_clearance_m") is not None else 99.0),
                     "contacts": [str(c) for c in result.get("contacts", [])], "reached": drive.get("reached") if drive.get("ok") else None,
                     "method": str(result.get("method") or "physx")}
        else:
            _phase(hb, "geometry")
            check = lane_check(paths, site_layer=SITE_LAYER, catalog=CATALOG, asset_root=ASSET_ROOT, lane=inp.lane, robot=inp.robot, min_clearance_m=inp.min_clearance_m)
    record = run_dir(inp.run_id) / "lane" / f"{inp.step:03d}-{inp.lane}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    tmp = record.with_name(f".{record.name}.part")
    tmp.write_text(json.dumps({"step": inp.step, "lane": inp.lane, "robot": inp.robot, "min_clearance_m": inp.min_clearance_m, **check}, indent=2, sort_keys=True))
    tmp.replace(record)
    return LaneCheck(step=inp.step, lane=inp.lane, passed=check["passed"], min_clearance_m=check["min_clearance_m"], contacts=check["contacts"],
                     reached=check.get("reached"), clip_path=str(clip) if clip else None, method=check["method"])


@activity.defn
async def close_trace(inp: CloseInput) -> list[str]:
    """The gpu Worker's last Activity for the run: closes its NeMo Relay scope and writes its final trajectory."""
    return await relay.close_here(reason=inp.reason)
