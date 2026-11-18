"""The Isaac Sim backend: runs inside Isaac Sim 6.1's Python (`/isaac-sim/python.sh -m renderer.run_service --backend kit`).

No thread touches Kit. The main thread alternates `registry.run_one(backend)` (synchronous Replicator and USD calls that
pump the app themselves) with `app.update()`; the HTTP thread only enqueues jobs and reads their status. Every job lints
its layers again before Kit opens them. Two operations: `render_still`, an RTX frame from a named camera, and
`simulate_lane`, the lane against the real geometry plus a Nova Carter drive-through with PhysX.
"""

from __future__ import annotations

import glob
import os
import time
from pathlib import Path
from typing import Any, Callable

from renderer import DEFAULT_BASE_REF
from renderer.jobs import CancelledJob, InvalidJob, Job

SITE_LAYER = os.environ.get("RENDERER_SITE_LAYER", "/opt/renderer/site/000_site.usda")  # the trusted copy, baked into the image
ASSET_ROOT = os.environ.get("ISAACSIM_ASSET_ROOT", "/opt/assets")
BASE_REF = os.environ.get("RENDERER_BASE_REF", DEFAULT_BASE_REF)


class KitBackend:
    def __init__(self, app: Any, out: Path, lint: Callable[[str], None] | None = None) -> None:
        self.app = app
        self.out = out
        self.layers_dir = (Path(os.environ.get("RENDERER_SHARED_DIR", out.parent)) / "layers").resolve()
        self.lint = lint
        self._stack_hash: str | None = None
        import carb.settings  # noqa: F401 — inside Kit only
        import omni.replicator.core as rep
        import omni.usd

        self.rep, self.omni_usd = rep, omni.usd
        self.settings = carb.settings.get_settings()
        rep.orchestrator.set_capture_on_play(False)  # Isaac Sim 6.1's base app turns capture-on-play on

    # ---------------------------------------------------------------- stage
    def ensure_stage(self, base_ref: str, layer_paths: list[str], progress: Callable[[dict[str, Any]], None], cancelled: Callable[[], bool]) -> str:
        import hashlib

        from pxr import Sdf, Usd  # noqa: F401

        if cancelled():
            raise CancelledJob()
        if base_ref != BASE_REF:
            raise InvalidJob("base_ref does not match the renderer's configured base scene")
        asset_root = Path(ASSET_ROOT).resolve()
        base = (asset_root / BASE_REF).resolve()
        if not base.is_relative_to(asset_root) or not base.is_file():
            raise InvalidJob("the configured base scene must be a file inside the asset root")
        if not isinstance(layer_paths, list) or any(not isinstance(p, str) for p in layer_paths):
            raise InvalidJob("layers must be a list of paths")
        layers = [Path(p).resolve() for p in layer_paths]
        if any(p.parent != self.layers_dir or p.suffix != ".usda" or not p.is_file() for p in layers):
            raise InvalidJob("layers must be .usda files in the shared layers directory")
        h = hashlib.sha256("\n".join(str(p) for p in [*layers, base]).encode()).hexdigest()
        if h == self._stack_hash and self.omni_usd.get_context().get_stage() is not None:
            return h
        progress({"phase": "stage_load"})
        for p in layers:
            if cancelled():
                raise CancelledJob()
            if self.lint:
                self.lint(str(p))
        root_path = str(self.out / "stage" / f"{h[:16]}.usda")
        Path(root_path).parent.mkdir(parents=True, exist_ok=True)
        root = Sdf.Layer.FindOrOpen(root_path) or Sdf.Layer.CreateNew(root_path)
        root.Clear()
        # newest step first, then the site layer, then the base warehouse as the weakest sublayer
        root.subLayerPaths = [*(str(p) for p in reversed(layers)), SITE_LAYER, str(base)]
        # stage metadata resolves across the root layer stack; pin it so the agent's layers cannot flip units/up-axis
        root.pseudoRoot.SetInfo("metersPerUnit", 1.0)
        root.pseudoRoot.SetInfo("upAxis", "Z")
        root.defaultPrim = "World"
        root.Save()
        ctx = self.omni_usd.get_context()
        self._stack_hash = None
        ctx.open_stage(root_path)
        self._pump_until_loaded(cancelled)
        for _ in range(2):  # warm-up frames after a stage change (materials/RTX)
            self.app.update()
        self._stack_hash = h
        return h

    def _pump_until_loaded(self, cancelled: Callable[[], bool], timeout_s: float = 600.0) -> None:
        from isaacsim.core.experimental.utils.stage import is_stage_loading

        t0 = time.time()
        while is_stage_loading():
            if cancelled():
                raise CancelledJob()
            self.app.update()
            if time.time() - t0 > timeout_s:
                raise RuntimeError("stage load timed out")

    # ---------------------------------------------------------------- render settings
    def _set_mode(self, mode: str, total_spp: int, subframes: int) -> int:
        s = self.settings
        if mode == "pt":
            s.set("/rtx/rendermode", "PathTracing")
            per_frame = max(1, total_spp // max(1, subframes))
            s.set("/rtx/pathtracing/spp", per_frame)
            s.set("/rtx/pathtracing/clampSpp", per_frame)
            s.set("/rtx/pathtracing/totalSpp", total_spp)
            s.set("/rtx/pathtracing/optixDenoiser/enabled", True)
        else:
            s.set("/rtx/rendermode", "RealTimePathTracing")
            s.set("/rtx/post/dlss/execMode", 2)  # DLSS Quality
        return subframes

    # ---------------------------------------------------------------- ops
    def run(self, job: Job, progress: Callable[[dict[str, Any]], None], cancelled: Callable[[], bool]) -> dict[str, Any]:
        p = job.params
        if cancelled():
            raise CancelledJob()
        if job.op == "render_still":
            return self.render_still(job, progress, cancelled)
        if job.op == "simulate_lane":
            return self.simulate_lane(p, progress, cancelled)
        raise InvalidJob(f"unknown op {job.op}")

    def render_still(self, job: Job, progress, cancelled) -> dict[str, Any]:
        p = job.params
        rep = self.rep
        self.ensure_stage(p["base_ref"], p["layers"], progress, cancelled)
        if cancelled():
            raise CancelledJob()
        subframes = self._set_mode(p.get("mode", "rt2"), int(p.get("total_spp", 64)), int(p.get("subframes", 16)))
        out_dir = self.out / "jobs" / job.key
        out_dir.mkdir(parents=True, exist_ok=True)
        progress({"phase": "render", "frames_total": 1})
        rp = rep.create.render_product(p["camera"], (int(p.get("width", 1920)), int(p.get("height", 1080))))
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=str(out_dir))
        writer = rep.writers.get("BasicWriter")
        writer.initialize(backend=backend, rgb=True)
        writer.attach([rp])
        try:
            if cancelled():
                raise CancelledJob()
            rep.orchestrator.step(rt_subframes=subframes, delta_time=0.0)
            rep.orchestrator.wait_until_complete()
            if cancelled():
                raise CancelledJob()
        finally:
            writer.detach()
            rp.destroy()
        pngs = sorted(glob.glob(str(out_dir / "**" / "rgb_*.png"), recursive=True))
        if not pngs:
            raise InvalidJob(f"no rgb output in {out_dir}")
        progress({"frames_done": 1, "pct": 100})
        return {"output_path": pngs[-1], "mode": p.get("mode", "rt2"), "subframes": subframes, "layers": p["layers"]}

    def simulate_lane(self, p: dict[str, Any], progress, cancelled) -> dict[str, Any]:
        """Lane check = static real-geometry clearance (bbox vs lane volume) + a Nova Carter drive-through (PhysX) when
        robot != "none": the sample's own planner/controller graph drives the robot along the lane; contacts with agent prims
        or a path clearance below the threshold fail the check. Returns the sampled trajectory for the replay clip."""
        from pxr import Usd, UsdGeom

        self.ensure_stage(p["base_ref"], p["layers"], progress, cancelled)
        if cancelled():
            raise CancelledJob()
        stage = self.omni_usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy])
        lane = stage.GetPrimAtPath(f"/World/Site/Lanes/{p['lane']}")
        if not lane:
            raise InvalidJob(f"lane {p['lane']} not in site layer")
        lr = cache.ComputeWorldBound(lane).ComputeAlignedRange()
        la = [lr.GetMin()[0], lr.GetMin()[1], lr.GetMax()[0], lr.GetMax()[1]]
        min_clearance = float(p.get("min_clearance_m", 0.6))
        contacts, min_gap, boxes = [], 1e9, []
        agent_root = stage.GetPrimAtPath("/World/Agent")
        for prim in (agent_root.GetChildren() if agent_root else []):
            if not prim.IsActive():
                continue
            r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if r.IsEmpty() or r.GetMin()[2] > 2.0:  # overhead (task lights) never obstructs a lane
                continue
            a = [r.GetMin()[0], r.GetMin()[1], r.GetMax()[0], r.GetMax()[1]]
            gx = max(0.0, max(a[0], la[0]) - min(a[2], la[2]))
            gy = max(0.0, max(a[1], la[1]) - min(a[3], la[3]))
            gap = (gx * gx + gy * gy) ** 0.5
            flat = r.GetMax()[2] < 0.02  # painted floor markings: overlap-only (they are not obstacles), same rule as sceneops
            if not flat:
                min_gap = min(min_gap, gap)
                boxes.append((prim.GetPath().pathString, a))
            if (gap == 0.0) if flat else (gap < min_clearance):
                contacts.append(prim.GetPath().pathString)
        result: dict[str, Any] = {"passed": not contacts, "min_clearance_m": round(min_gap, 3) if min_gap < 1e9 else None, "contacts": contacts,
                                  "clip_path": None, "method": "bbox-vs-lane (real geometry)", "drive": None, "trajectory": []}
        if p.get("robot", "carter") in ("none", "", None):
            return result
        try:
            drive = self._drive_lane(p["lane"], la, boxes, min_clearance, progress, cancelled)
        except CancelledJob:
            raise
        except Exception as e:  # noqa: BLE001 — the static check still stands; the drive is reported as unavailable
            drive = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        result["drive"] = {k: v for k, v in drive.items() if k != "trajectory"}
        result["trajectory"] = drive.get("trajectory", [])
        result["method"] = "bbox-vs-lane + Nova Carter drive-through (PhysX)" if drive.get("ok") else result["method"]
        if drive.get("ok"):
            drive_fail = (not drive["reached"]) or bool(drive["contacts"]) or (drive["min_path_clearance_m"] is not None and drive["min_path_clearance_m"] < min_clearance)
            result["passed"] = result["passed"] and not drive_fail
            result["contacts"] = sorted(set(contacts) | set(drive["contacts"]) | set(drive["too_close"]))
            if drive["min_path_clearance_m"] is not None:
                result["min_clearance_m"] = round(min(result["min_clearance_m"] if result["min_clearance_m"] is not None else 1e9, drive["min_path_clearance_m"]), 3)
        return result

    def _drive_lane(self, lane: str, la: list[float], boxes: list[tuple[str, list[float]]], min_clearance: float, progress, cancelled) -> dict[str, Any]:
        """Reference the Nova Carter nav sample at one end of the lane, point its planner at a world-space goal near the other
        end, play PhysX (rendering minimal), sample the chassis at 10 Hz, collect PhysX contacts with agent prims, clean up."""
        import math

        import omni.timeline
        from pxr import Gf, Sdf, Usd, UsdGeom

        stage = self.omni_usd.get_context().get_stage()
        long_x = (la[2] - la[0]) >= (la[3] - la[1])
        cx, cy = (la[0] + la[2]) / 2, (la[1] + la[3]) / 2
        if long_x:
            goal = (la[2] - 1.5, cy)
            start = (max(la[0] + 1.0, goal[0] - 18.0), cy)
            yaw = 0.0
        else:
            goal = (cx, la[3] - 1.5)
            start = (cx, max(la[1] + 1.0, goal[1] - 18.0))
            yaw = 90.0
        length = math.hypot(goal[0] - start[0], goal[1] - start[1])
        sim_seconds = min(90.0, length / 0.3 + 10.0)
        robot_r = 0.45  # half of the Carter's longer side, plus a little
        sample = f"{ASSET_ROOT}/Isaac/Samples/Replicator/OmniGraph/nova_carter_nav_only.usd"
        session = stage.GetSessionLayer()
        contacts: list[str] = []
        sub = None
        try:
            with Usd.EditContext(stage, session):
                UsdGeom.Xform.Define(stage, "/World/Sim")
                carter = stage.DefinePrim("/World/Sim/Carter", "Xform")
                carter.GetReferences().AddReference(sample, "/nova_carter")
                xf = UsdGeom.Xformable(carter)
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(start[0], start[1], 0.0))
                xf.AddRotateZOp().Set(yaw)
                for prim in Usd.PrimRange(carter):  # the sample payloads the sensor variant; sensors render every frame (85 s/frame)
                    n = prim.GetName().lower()
                    if prim != carter and any(k in n for k in ("camera", "lidar", "hawk", "radar", "imu", "sensor", "owl", "fisheye", "stereo")):
                        prim.SetActive(False)
                goal_prim = UsdGeom.Xform.Define(stage, "/World/Sim/Goal")
                goal_prim.AddTranslateOp().Set(Gf.Vec3d(goal[0], goal[1], 0.0))
                planner = None
                for prim in Usd.PrimRange(carter):
                    if prim.GetTypeName() == "OmniGraphNode" and str(prim.GetAttribute("node:type").Get()) == "isaacsim.robot.wheeled_robots.QuinticPathPlanner":
                        planner = prim
                if planner is None:
                    raise RuntimeError("QuinticPathPlanner node not found in the Carter sample")
                planner.GetRelationship("inputs:targetPrim").SetTargets([Sdf.Path("/World/Sim/Goal")])
            for _ in range(3):
                self.app.update()
            chassis = stage.GetPrimAtPath("/World/Sim/Carter/chassis_link")
            try:
                from isaacsim.core.experimental.prims import XformPrim

                xp = XformPrim("/World/Sim/Carter/chassis_link")

                def pos():
                    q, _ = xp.get_world_poses()
                    q = q.numpy() if hasattr(q, "numpy") else q
                    return float(q[0][0]), float(q[0][1])
            except Exception:  # noqa: BLE001
                def pos():
                    t = UsdGeom.Xformable(chassis).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
                    return float(t[0]), float(t[1])
            try:
                import omni.physx
                from pxr import PhysxSchema

                with Usd.EditContext(stage, session):
                    PhysxSchema.PhysxContactReportAPI.Apply(chassis).CreateThresholdAttr(0.0)

                def _on_contact(headers, data):
                    for h in headers:
                        a, b = str(h.actor0), str(h.actor1)
                        other = b if "/World/Sim/Carter" in a else a
                        if other.startswith("/World/Agent"):
                            contacts.append(other.split("/World/Agent/")[1].split("/")[0] and "/World/Agent/" + other.split("/World/Agent/")[1].split("/")[0])

                sub = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(_on_contact)
            except Exception:  # noqa: BLE001
                sub = None
            prev_mode = self.settings.get("/rtx/rendermode")
            self.settings.set("/rtx/rendermode", "MinimalRendering")  # the physics pass renders nothing
            tl = omni.timeline.get_timeline_interface()
            tl.play()
            traj, reached, min_path, too_close = [], False, 1e9, set()
            total = int(sim_seconds * 60)
            t0 = time.time()
            for frame in range(total):
                if frame % 120 == 0 and cancelled():
                    raise CancelledJob()
                self.app.update()
                if frame % 6:
                    continue
                x, y = pos()
                d = math.hypot(x - goal[0], y - goal[1])
                for path, b in boxes:
                    gx = max(0.0, max(b[0], x - robot_r) - min(b[2], x + robot_r))
                    gy = max(0.0, max(b[1], y - robot_r) - min(b[3], y + robot_r))
                    gap = (gx * gx + gy * gy) ** 0.5
                    min_path = min(min_path, gap)
                    if gap < min_clearance:
                        too_close.add(path)
                traj.append([round(frame / 60, 2), round(x, 3), round(y, 3)])
                if frame % 120 == 0:
                    progress({"phase": "drive", "pct": int(100 * frame / total), "goal_m": round(d, 2)})
                if d < 0.6:
                    reached = True
                    break
            wall = time.time() - t0
            tl.stop()
            return {"ok": True, "robot": "nova_carter", "lane": lane, "start": [round(start[0], 2), round(start[1], 2)], "goal": [round(goal[0], 2), round(goal[1], 2)],
                    "reached": reached, "sim_seconds": round(len(traj) * 0.1, 1), "wall_seconds": round(wall, 1), "contacts": sorted(set(contacts)),
                    "too_close": sorted(too_close), "min_path_clearance_m": round(min_path, 3) if boxes else None, "trajectory": traj}
        finally:
            try:
                omni.timeline.get_timeline_interface().stop()
            except Exception:  # noqa: BLE001
                pass
            if sub is not None:
                try:
                    sub.unsubscribe() if hasattr(sub, "unsubscribe") else None
                except Exception:  # noqa: BLE001
                    pass
            try:
                self.settings.set("/rtx/rendermode", prev_mode if "prev_mode" in locals() and prev_mode else "RealTimePathTracing")
            except Exception:  # noqa: BLE001
                pass
            if session.GetPrimAtPath("/World/Sim"):
                with Usd.EditContext(stage, session):  # RemovePrim deletes the spec from the current edit target (the session layer)
                    stage.RemovePrim("/World/Sim")
            for _ in range(2):
                self.app.update()
