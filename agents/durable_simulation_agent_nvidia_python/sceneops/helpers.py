"""The helper API the model's code uses inside the exec sandbox (and the trusted judge uses outside it).

The generated code runs with ``h`` (a SceneHelpers bound to the stage whose
edit target is the current step layer), plus ``stage``, ``Usd``, ``Sdf``,
``UsdGeom``, ``Gf`` in scope. Everything here is CPU-only pxr; the warehouse
geometry is NOT present in the sandbox, so footprints come from catalog bboxes.
"""

from __future__ import annotations

import math
from typing import Any

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

from sceneops.catalog import (
    load_catalog,  # noqa: F401 — re-exported for apply.py and the judge
)

AGENT_ROOT = "/World/Agent"
SITE_ROOT = "/World/Site"
LANES = f"{SITE_ROOT}/Lanes"
KEEP_CLEAR = f"{SITE_ROOT}/KeepClear"
DOCKS_PREFIX = f"{SITE_ROOT}/Dock_"


class SceneHelpers:
    def __init__(self, stage: Usd.Stage, catalog: dict[str, dict[str, Any]], asset_root: str = "/opt/assets") -> None:
        self.stage = stage
        self.catalog = catalog
        self.asset_root = asset_root.rstrip("/")
        self._bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])

    # ---------- catalog ----------
    def search(self, query: str, k: int = 8) -> list[dict[str, Any]]:
        q = {w for w in query.lower().replace(",", " ").split() if w}
        scored = []
        for a in self.catalog.values():
            text = " ".join([a["id"], a["name"], *a["tags"]]).lower()
            score = sum(1 for w in q if w in text)
            if score:
                scored.append((score, a))
        scored.sort(key=lambda s: (-s[0], s[1]["id"]))
        return [{"id": a["id"], "name": a["name"], "size_m": a["size_m"], "tags": a["tags"]} for _, a in scored[:k]]

    def resolve_id(self, ref: str) -> str | None:
        """Catalog id for an id, a USD path/URL (matched by file name, then by folder), or None."""
        if ref in self.catalog:
            return ref
        key = ref.strip().rstrip("/").split("/")[-1]
        if key.endswith(".usd") or key.endswith(".usda") or key.endswith(".usdc"):
            stem = key.rsplit(".", 1)[0].lower()
            for a in self.catalog.values():
                if a["usd"].rsplit("/", 1)[-1].rsplit(".", 1)[0].lower() == stem:
                    return a["id"]
            folder = ref.rsplit("/", 1)[0].rstrip("/").split("/")[-1].lower()
            for a in self.catalog.values():
                if a["usd"].rsplit("/", 2)[-2].lower() == folder:
                    return a["id"]
        low = ref.lower()
        if low in self.catalog:
            return low
        return None

    def _entry(self, catalog_id: str) -> dict[str, Any]:
        rid = self.resolve_id(catalog_id)
        if rid is None:
            near = [a["id"] for a in self.search(catalog_id.replace("_", " ").replace("/", " "), 3)]
            raise ValueError(f"unknown catalog id {catalog_id!r} (only catalog ids are placeable, never file paths); nearest: {near}")
        return self.catalog[rid]

    # ---------- site ----------
    def dock(self, n: int) -> tuple[float, float]:
        prim = self.stage.GetPrimAtPath(f"{DOCKS_PREFIX}{n}")
        if not prim:
            raise ValueError(f"no dock {n}")
        t = UsdGeom.XformCommonAPI(prim).GetXformVectors(Usd.TimeCode.Default())[0]
        return (float(t[0]), float(t[1]))

    def site_volumes(self) -> list[dict[str, Any]]:
        out = []
        for scope, kind in ((LANES, "lane"), (KEEP_CLEAR, "keep_clear")):
            root = self.stage.GetPrimAtPath(scope)
            if not root:
                continue
            for prim in root.GetChildren():
                r = self._bbox.ComputeWorldBound(prim).ComputeAlignedRange()
                lo, hi = r.GetMin(), r.GetMax()
                out.append({"name": prim.GetName(), "kind": kind, "aabb": [lo[0], lo[1], hi[0], hi[1]]})
        return out

    # ---------- authoring ----------
    def _unique(self, base: str) -> str:
        root = self.stage.GetPrimAtPath(AGENT_ROOT)
        existing = {c.GetName() for c in root.GetChildren()} if root else set()
        name, i = base, 1
        while name in existing:
            i += 1
            name = f"{base}_{i}"
        return name

    def _named(self, name: str) -> str:
        """An explicit name replaces any existing agent prim of that name (re-placing is an edit, not a duplicate)."""
        path = f"{AGENT_ROOT}/{name}"
        if self.stage.GetPrimAtPath(path):
            self.remove(path)
        return name

    def clear(self) -> list[str]:
        """Remove every agent-authored prim (older layers' prims are deactivated)."""
        root = self.stage.GetPrimAtPath(AGENT_ROOT)
        removed = []
        for prim in list(root.GetChildren()) if root else []:
            if prim.IsActive():
                removed.append(self.remove(prim.GetPath().pathString))
        return removed

    def agent_prims(self) -> list[dict[str, Any]]:
        """Everything active under /World/Agent: assets (with footprint), floor markings and lights (with aabb)."""
        root = self.stage.GetPrimAtPath(AGENT_ROOT)
        out = []
        for prim in (root.GetChildren() if root else []):
            if not prim.IsActive():
                continue
            cid = prim.GetCustomDataByKey("agent:catalog_id")
            kind = "asset" if cid else (prim.GetCustomDataByKey("agent:kind") or "other")
            if cid:
                box = self.footprint(prim.GetPath().pathString)
            else:
                r = self._bbox.ComputeWorldBound(prim).ComputeAlignedRange()
                box = None if r.IsEmpty() else [round(r.GetMin()[0], 2), round(r.GetMin()[1], 2), round(r.GetMax()[0], 2), round(r.GetMax()[1], 2)]
            out.append({"path": prim.GetPath().pathString, "kind": kind, "catalog_id": cid, "aabb": box})
        return out

    def place_asset(self, catalog_id: str, x: float, y: float, yaw_deg: float = 0.0, name: str | None = None) -> str:
        """Reference a catalog asset at (x, y) on the floor, rotated yaw_deg about Z. Returns the prim path."""
        entry = self._entry(catalog_id)
        name = self._named(name) if name else self._unique(entry["id"])
        path = f"{AGENT_ROOT}/{name}"
        xf = UsdGeom.Xform.Define(self.stage, path)
        prim = xf.GetPrim()
        prim.GetReferences().AddReference(f"{self.asset_root}/{entry['usd']}")
        prim.SetActive(True)  # explicit: a re-placed name must override an older layer's `active = false`
        prim.SetCustomDataByKey("agent:catalog_id", entry["id"])
        api = UsdGeom.XformCommonAPI(xf)
        api.SetTranslate(Gf.Vec3d(float(x), float(y), float(entry.get("z", 0.0))))
        api.SetRotate(Gf.Vec3f(0.0, 0.0, float(yaw_deg)))
        return path

    def move(self, prim_path: str, dx: float = 0.0, dy: float = 0.0) -> str:
        prim = self._agent_prim(prim_path)
        api = UsdGeom.XformCommonAPI(prim)
        t = api.GetXformVectors(Usd.TimeCode.Default())[0]
        api.SetTranslate(Gf.Vec3d(t[0] + dx, t[1] + dy, t[2]))
        return prim_path

    def set_position(self, prim_path: str, x: float, y: float) -> str:
        prim = self._agent_prim(prim_path)
        api = UsdGeom.XformCommonAPI(prim)
        t = api.GetXformVectors(Usd.TimeCode.Default())[0]
        api.SetTranslate(Gf.Vec3d(float(x), float(y), t[2]))
        return prim_path

    def set_rotation(self, prim_path: str, yaw_deg: float) -> str:
        prim = self._agent_prim(prim_path)
        UsdGeom.XformCommonAPI(prim).SetRotate(Gf.Vec3f(0.0, 0.0, float(yaw_deg)))
        return prim_path

    def remove(self, prim_path: str) -> str:
        """Remove an agent prim: drop its spec if this step's layer authored it, else deactivate it (older layer)."""
        prim = self._agent_prim(prim_path)
        layer = self.stage.GetEditTarget().GetLayer()
        if layer.GetPrimAtPath(prim.GetPath()) is not None and layer.GetPrimAtPath(prim.GetPath()).specifier == Sdf.SpecifierDef:
            self.stage.RemovePrim(prim.GetPath())
        else:
            prim.SetActive(False)
        return prim_path

    def add_floor_marking(self, name: str, points: list[tuple[float, float]], width: float = 0.08,
                          color: tuple[float, float, float] = (1.0, 0.85, 0.0)) -> str:
        """A painted line on the floor through points [(x,y),...] as a thin mesh strip."""
        if len(points) < 2:
            raise ValueError("need at least two points")
        path = f"{AGENT_ROOT}/{self._named(name)}"
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        pts, counts, idx = [], [], []
        z = 0.005
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            dx, dy = x1 - x0, y1 - y0
            L = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / L * width / 2, dx / L * width / 2
            base = len(pts)
            pts += [Gf.Vec3f(x0 + nx, y0 + ny, z), Gf.Vec3f(x1 + nx, y1 + ny, z),
                    Gf.Vec3f(x1 - nx, y1 - ny, z), Gf.Vec3f(x0 - nx, y0 - ny, z)]
            counts.append(4)
            idx += [base, base + 1, base + 2, base + 3]
        mesh.CreatePointsAttr(pts)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(idx)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        mesh.GetPrim().SetActive(True)  # a same-name replacement must override this layer's own deactivation
        mesh.GetPrim().SetCustomDataByKey("agent:kind", "floor_marking")
        return path

    def add_light(self, name: str, x: float, y: float, z: float = 3.5, intensity: float = 20000.0,
                  color: tuple[float, float, float] = (1.0, 0.98, 0.9), radius: float = 0.15) -> str:
        path = f"{AGENT_ROOT}/{self._named(name)}"
        light = UsdLux.SphereLight.Define(self.stage, path)
        light.CreateIntensityAttr(float(intensity))
        light.CreateRadiusAttr(float(radius))
        light.CreateColorAttr(Gf.Vec3f(*color))
        UsdGeom.XformCommonAPI(light).SetTranslate(Gf.Vec3d(float(x), float(y), float(z)))
        light.GetPrim().SetActive(True)
        light.GetPrim().SetCustomDataByKey("agent:kind", "light")
        return path

    # ---------- geometry / validation ----------
    def _agent_prim(self, prim_path: str) -> Usd.Prim:
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim_path.startswith(AGENT_ROOT + "/"):
            raise ValueError(f"{prim_path} is not an agent-authored prim under {AGENT_ROOT}")
        return prim

    def placed_assets(self) -> list[dict[str, Any]]:
        root = self.stage.GetPrimAtPath(AGENT_ROOT)
        out = []
        for prim in (root.GetChildren() if root else []):
            if not prim.IsActive():
                continue
            cid = prim.GetCustomDataByKey("agent:catalog_id")
            if not cid:
                continue
            t, r, *_ = UsdGeom.XformCommonAPI(prim).GetXformVectors(Usd.TimeCode.Default())
            out.append({"path": prim.GetPath().pathString, "catalog_id": cid, "x": float(t[0]), "y": float(t[1]),
                        "yaw_deg": float(r[2]), "footprint": self.footprint(prim.GetPath().pathString)})
        return out

    def footprint(self, prim_path: str) -> list[float]:
        """Conservative world-space AABB [xmin, ymin, xmax, ymax] from the catalog size and yaw."""
        prim = self._agent_prim(prim_path)
        entry = self._entry(prim.GetCustomDataByKey("agent:catalog_id"))
        t, r, *_ = UsdGeom.XformCommonAPI(prim).GetXformVectors(Usd.TimeCode.Default())
        return self.footprint_at(entry["id"], float(t[0]), float(t[1]), float(r[2]))

    def footprint_at(self, catalog_id: str, x: float, y: float, yaw_deg: float = 0.0) -> list[float]:
        """World AABB [xmin, ymin, xmax, ymax] a catalog asset would occupy at (x, y, yaw): its local bbox (pivots are not
        centred — a conveyor spans 0..4 m from its origin) rotated about Z and translated. Falls back to size/2 if unmeasured."""
        entry = self._entry(catalog_id)
        lb = entry.get("local_bbox")
        if lb:
            corners = ((lb[0], lb[1]), (lb[2], lb[1]), (lb[2], lb[3]), (lb[0], lb[3]))
        else:
            w, d, _ = entry["size_m"]
            corners = ((-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2))
        yaw = math.radians(float(yaw_deg))
        c, s = math.cos(yaw), math.sin(yaw)
        xs = [x + px * c - py * s for px, py in corners]
        ys = [y + px * s + py * c for px, py in corners]
        return [min(xs), min(ys), max(xs), max(ys)]

    @staticmethod
    def _aabb_gap(a: list[float], b: list[float]) -> float:
        gx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
        gy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
        return math.hypot(gx, gy)

    def check_clearance(self, min_clearance_m: float = 0.6) -> list[dict[str, Any]]:
        """Violations of lane / keep-clear clearance for every placed asset (cheap pre-check; the
        trusted side re-validates with real geometry)."""
        vols = self.site_volumes()
        out = []
        for asset in self.placed_assets():
            for v in vols:
                gap = self._aabb_gap(asset["footprint"], v["aabb"])
                if gap < min_clearance_m:
                    out.append({"asset": asset["path"], "volume": v["name"], "kind": v["kind"],
                                "clearance_m": round(gap, 3), "required_m": min_clearance_m,
                                "status": "encroaches" if gap == 0.0 else "too_close"})
        for prim in self.agent_prims():  # painted markings may touch a lane but never lie inside it (the trusted judge agrees)
            if prim["kind"] == "floor_marking" and prim["aabb"]:
                for v in vols:
                    if self._aabb_gap(prim["aabb"], v["aabb"]) == 0.0:
                        out.append({"asset": prim["path"], "volume": v["name"], "kind": v["kind"], "clearance_m": 0.0, "required_m": 0.0,
                                    "status": "encroaches"})
        return out

    def build_area(self) -> list[float] | None:
        """[xmin, ymin, xmax, ymax] of /World/Site/Candidates/BuildArea (the open floor meant for the brief), or None."""
        prim = self.stage.GetPrimAtPath(f"{SITE_ROOT}/Candidates/BuildArea")
        if not prim:
            return None
        r = self._bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        return [r.GetMin()[0], r.GetMin()[1], r.GetMax()[0], r.GetMax()[1]]

    @staticmethod
    def _rect(x: float, y: float, w: float, d: float, yaw_deg: float) -> list[float]:
        yaw = math.radians(float(yaw_deg))
        c, sn = math.cos(yaw), math.sin(yaw)
        xs, ys = [], []
        for px, py in ((-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)):
            xs.append(x + px * c - py * sn)
            ys.append(y + px * sn + py * c)
        return [min(xs), min(ys), max(xs), max(ys)]

    def conflicts(self, x: float, y: float, w: float, d: float, yaw_deg: float = 0.0, exclude: str | None = None,
                  min_clearance_m: float = 0.6) -> list[dict[str, Any]]:
        """What a w x d footprint centred at (x, y) would hit: lanes/keep_clear volumes (clearance) and other placed assets
        (overlap) and the build area (if any). [] means the spot is free. Authors nothing."""
        fp = self._rect(x, y, w, d, yaw_deg)
        return self._fp_conflicts(fp, exclude, min_clearance_m)

    def _fp_conflicts(self, fp: list[float], exclude: str | None, min_clearance_m: float) -> list[dict[str, Any]]:
        out = []
        for v in self.site_volumes():
            gap = self._aabb_gap(fp, v["aabb"])
            if gap < min_clearance_m:
                out.append({"with": v["name"], "kind": v["kind"], "clearance_m": round(gap, 3), "required_m": min_clearance_m})
        for a in self.placed_assets():
            if a["path"] == exclude:
                continue
            fb = a["footprint"]
            if fp[0] < fb[2] and fb[0] < fp[2] and fp[1] < fb[3] and fb[1] < fp[3]:
                out.append({"with": a["path"], "kind": "asset_overlap"})
        ba = self.build_area()
        if ba and not (fp[0] >= ba[0] and fp[1] >= ba[1] and fp[2] <= ba[2] and fp[3] <= ba[3]):
            out.append({"with": "BuildArea", "kind": "outside_build_area", "aabb": [round(v, 2) for v in ba]})
        return out

    def conflicts_for(self, catalog_id: str, x: float, y: float, yaw_deg: float = 0.0, exclude: str | None = None,
                      min_clearance_m: float = 0.6) -> list[dict[str, Any]]:
        """conflicts() for a catalog asset's real footprint at (x, y, yaw). [] means free. Authors nothing."""
        return self._fp_conflicts(self.footprint_at(catalog_id, x, y, yaw_deg), exclude, min_clearance_m)

    def find_spot(self, catalog_id: str, region: list[float] | None = None, yaw_deg: float = 0.0, step: float = 0.25,
                  min_clearance_m: float = 0.6, prefer: tuple[float, float] | None = None) -> tuple[float, float] | None:
        """First free (x, y) for a catalog asset inside region [xmin,ymin,xmax,ymax] (default: the build area), trying the
        grid position nearest `prefer` first (default: the region's max-x edge, i.e. toward the docks). None if nothing fits."""
        self._entry(catalog_id)
        region = region or self.build_area()
        if not region:
            raise ValueError("no region given and the site has no BuildArea")
        fp0 = self.footprint_at(catalog_id, 0.0, 0.0, yaw_deg)   # offsets of the footprint around the pivot
        xs = [region[0] - fp0[0] + i * step for i in range(int(max(0.0, (region[2] - region[0]) - (fp0[2] - fp0[0])) / step) + 1)]
        ys = [region[1] - fp0[1] + i * step for i in range(int(max(0.0, (region[3] - region[1]) - (fp0[3] - fp0[1])) / step) + 1)]
        px, py = prefer if prefer else (region[2], (region[1] + region[3]) / 2)
        cands = sorted(((x, y) for x in xs for y in ys), key=lambda c: (c[0] - px) ** 2 + (c[1] - py) ** 2)
        for x, y in cands:
            if not self.conflicts_for(catalog_id, x, y, yaw_deg, min_clearance_m=min_clearance_m):
                return (round(x, 3), round(y, 3))
        return None

    def nudge_clear(self, prim_path: str, max_m: float = 3.0, step: float = 0.25, min_clearance_m: float = 0.6,
                    region: list[float] | None = None) -> dict[str, Any]:
        """Move an asset the least distance (grid search up to max_m) to a spot with no lane/keep-clear/overlap/build-area
        conflicts, keeping its yaw. Returns {path, moved, from, to, conflicts_left}. Markings/lights are left alone."""
        prim = self._agent_prim(prim_path)
        cid = prim.GetCustomDataByKey("agent:catalog_id")
        if not cid:
            return {"path": prim_path, "moved": False, "reason": "not a catalog asset"}
        t, r, *_ = UsdGeom.XformCommonAPI(prim).GetXformVectors(Usd.TimeCode.Default())
        x0, y0, yaw = float(t[0]), float(t[1]), float(r[2])
        region = region or self.build_area()
        if not self.conflicts_for(cid, x0, y0, yaw, exclude=prim_path, min_clearance_m=min_clearance_m):
            return {"path": prim_path, "moved": False, "from": [x0, y0], "to": [x0, y0], "conflicts_left": []}
        n = int(max_m / step)
        cands = sorted(((x0 + i * step, y0 + j * step) for i in range(-n, n + 1) for j in range(-n, n + 1) if (i, j) != (0, 0)),
                       key=lambda c: (c[0] - x0) ** 2 + (c[1] - y0) ** 2)
        for x, y in cands:
            if region and not (region[0] <= x <= region[2] and region[1] <= y <= region[3]):
                continue
            if not self.conflicts_for(cid, x, y, yaw, exclude=prim_path, min_clearance_m=min_clearance_m):
                self.set_position(prim_path, x, y)
                return {"path": prim_path, "moved": True, "from": [x0, y0], "to": [round(x, 3), round(y, 3)], "conflicts_left": []}
        return {"path": prim_path, "moved": False, "from": [x0, y0], "to": [x0, y0],
                "conflicts_left": self.conflicts_for(cid, x0, y0, yaw, exclude=prim_path, min_clearance_m=min_clearance_m)}

    def fix_all(self, rounds: int = 4, min_clearance_m: float = 0.6) -> list[dict[str, Any]]:
        """nudge_clear every asset that has a conflict (up to `rounds` passes) and remove floor markings painted into a lane or
        keep-clear volume (redraw them outside afterwards). Returns the moves/removals made."""
        moves: list[dict[str, Any]] = []
        vols = self.site_volumes()
        for prim in self.agent_prims():
            if prim["kind"] == "floor_marking" and prim["aabb"] and any(self._aabb_gap(prim["aabb"], v["aabb"]) == 0.0 for v in vols):
                self.remove(prim["path"])
                moves.append({"path": prim["path"], "removed": True, "reason": "floor marking inside a lane/keep-clear volume; redraw it outside"})
        for _ in range(rounds):
            dirty = False
            for a in self.placed_assets():
                if self.conflicts_for(a["catalog_id"], a["x"], a["y"], a["yaw_deg"], exclude=a["path"], min_clearance_m=min_clearance_m):
                    res = self.nudge_clear(a["path"], min_clearance_m=min_clearance_m)
                    moves.append(res)
                    dirty = dirty or res.get("moved", False)
            if not dirty:
                break
        return moves

    def distance_to_dock(self, prim_path: str, dock_n: int) -> float:
        prim = self._agent_prim(prim_path)
        t = UsdGeom.XformCommonAPI(prim).GetXformVectors(Usd.TimeCode.Default())[0]
        dx, dy = self.dock(dock_n)
        return math.hypot(t[0] - dx, t[1] - dy)

    def summary(self, min_clearance_m: float = 0.6) -> dict[str, Any]:
        assets = self.placed_assets()
        return {
            "placed_assets": assets,
            "agent_prims": self.agent_prims(),
            "violations": self.check_clearance(min_clearance_m),
            "site_volumes": self.site_volumes(),
            "docks": {n: self.dock(n) for n in (1, 2, 3, 4) if self.stage.GetPrimAtPath(f"{DOCKS_PREFIX}{n}")},
        }
