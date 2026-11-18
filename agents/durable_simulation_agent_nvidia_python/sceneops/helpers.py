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
OVERHEAD_M = 2.0  # anything whose bottom is higher (ceiling lamps, signs) never blocks a lane, as in the renderer's check
FLAT_M = 0.02  # anything lower than this is paint on the floor


class SceneHelpers:
    def __init__(self, stage: Usd.Stage, catalog: dict[str, dict[str, Any]], asset_root: str = "/opt/assets") -> None:
        self.stage = stage
        self.catalog = catalog
        self.asset_root = asset_root.rstrip("/")
        self._bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])  # the site's volumes are guides
        self._render_bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])  # what gets rendered

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

    @staticmethod
    def _not_an_asset(prim: Usd.Prim, kind: str) -> None:
        """A marking or light that replaces an older asset of the same name keeps none of it: no reference, no catalog id."""
        prim.GetReferences().SetReferences([])
        prim.SetCustomDataByKey("agent:catalog_id", "")
        prim.SetCustomDataByKey("agent:kind", kind)

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
        """Everything active under /World/Agent: assets (with footprint), floor markings, lights and other geometry (with aabb)."""
        root = self.stage.GetPrimAtPath(AGENT_ROOT)
        out = []
        for prim in (root.GetChildren() if root else []):
            if not prim.IsActive():
                continue
            ids = self._asset_ids(prim)
            if ids:
                kind, box = "asset", [round(v, 2) for v in self._asset_box(prim, ids)[0]]
            else:
                b = self._geometry_box(prim)
                box = None if b is None else [round(v, 2) for v in b[0]]
                kind = (prim.GetCustomDataByKey("agent:kind") or "other") if b is None else ("floor_marking" if b[2] < FLAT_M else "geometry")
                if prim.IsA(UsdLux.BoundableLightBase) or prim.IsA(UsdLux.NonboundableLightBase):
                    kind = "light"
            out.append({"path": prim.GetPath().pathString, "kind": kind, "catalog_id": ids[0] if ids else None, "aabb": box})
        return out

    # ---------- what stands on the floor: assets found by their references, measured with full transforms ----------
    def _catalog_files(self) -> dict[str, str]:
        return {f"{self.asset_root}/{a['usd']}": a["id"] for a in self.catalog.values()}

    def _references(self, prim: Usd.Prim) -> list[str]:
        """The prim's composed references from this stage's own layers. The referenced files may not exist on this host, so
        the prim index cannot be asked; applying the local opinions' list edits, weakest first, gives the same list."""
        local = {layer.identifier for layer in self.stage.GetLayerStack()}
        refs: list = []
        for spec in reversed(prim.GetPrimStack()):
            if spec.layer.identifier in local:
                refs = spec.referenceList.ApplyEditsToList(refs)
        return [r.assetPath for r in refs if r.assetPath]

    def _asset_ids(self, prim: Usd.Prim) -> list[str]:
        """Catalog ids of the assets the prim references, whatever its custom data says."""
        files = self._catalog_files()
        return [files[r] for r in self._references(prim) if r in files]

    def _asset_box(self, prim: Usd.Prim, ids: list[str]) -> tuple[list[float], float, float]:
        """([xmin, ymin, xmax, ymax], zmin, zmax) of the referenced assets' measured local boxes under the prim's full world
        transform: translate, rotate, scale and matrix ops alike."""
        m = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
        pts = []
        for cid in ids:
            entry = self._entry(cid)
            lb = entry.get("local_bbox")
            if not lb:
                w, d, _ = entry["size_m"]
                lb = [-w / 2, -d / 2, w / 2, d / 2]
            z0, z1 = entry.get("local_z") or [0.0, float(entry["size_m"][2])]
            pts += [m.Transform(Gf.Vec3d(x, y, z)) for x in (lb[0], lb[2]) for y in (lb[1], lb[3]) for z in (z0, z1)]
        xs, ys, zs = [q[0] for q in pts], [q[1] for q in pts], [q[2] for q in pts]
        return [min(xs), min(ys), max(xs), max(ys)], min(zs), max(zs)

    def _geometry_box(self, prim: Usd.Prim) -> tuple[list[float], float, float] | None:
        """The world box of geometry authored in the layers (render and default purposes), or None if the prim has none."""
        r = self._render_bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        if r.IsEmpty():
            return None
        lo, hi = r.GetMin(), r.GetMax()
        return [lo[0], lo[1], hi[0], hi[1]], lo[2], hi[2]

    def obstacles(self) -> list[dict[str, Any]]:
        """What the clearance rules apply to: every asset and every piece of geometry on the floor, with its footprint.
        Overhead items (bottom above OVERHEAD_M) never block a lane; flat geometry is a floor marking, which may touch a
        lane but not lie inside it."""
        root = self.stage.GetPrimAtPath(AGENT_ROOT)
        out = []
        for prim in (root.GetChildren() if root else []):
            if not prim.IsActive():
                continue
            ids = self._asset_ids(prim)
            box = self._asset_box(prim, ids) if ids else self._geometry_box(prim)
            if box is None or box[1] > OVERHEAD_M:
                continue
            kind = "asset" if ids else ("floor_marking" if box[2] < FLAT_M else "geometry")
            out.append({"path": prim.GetPath().pathString, "kind": kind, "catalog_id": ids[0] if ids else None, "aabb": box[0]})
        return out

    def place_asset(self, catalog_id: str, x: float, y: float, yaw_deg: float = 0.0, name: str | None = None) -> str:
        """Reference a catalog asset at (x, y) on the floor, rotated yaw_deg about Z. Returns the prim path."""
        entry = self._entry(catalog_id)
        name = self._named(name) if name else self._unique(entry["id"])
        path = f"{AGENT_ROOT}/{name}"
        xf = UsdGeom.Xform.Define(self.stage, path)
        prim = xf.GetPrim()
        # Explicit, not prepended: a name re-placed over an older layer's prim must not keep that layer's reference too.
        prim.GetReferences().SetReferences([Sdf.Reference(f"{self.asset_root}/{entry['usd']}")])
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
        self._not_an_asset(mesh.GetPrim(), "floor_marking")
        mesh.ClearXformOpOrder()  # the points are in world space: no transform inherited from an older prim of this name
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
        self._not_an_asset(light.GetPrim(), "light")
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
            ids = self._asset_ids(prim) if prim.IsActive() else []
            if not ids:
                continue
            m = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
            t, ax = m.ExtractTranslation(), m.TransformDir(Gf.Vec3d(1, 0, 0))
            out.append({"path": prim.GetPath().pathString, "catalog_id": ids[0], "x": float(t[0]), "y": float(t[1]),
                        "yaw_deg": math.degrees(math.atan2(ax[1], ax[0])), "footprint": self._asset_box(prim, ids)[0]})
        return out

    def footprint(self, prim_path: str) -> list[float]:
        """World-space AABB [xmin, ymin, xmax, ymax] of the asset at prim_path, under its full transform."""
        prim = self._agent_prim(prim_path)
        ids = self._asset_ids(prim)
        if not ids:
            raise ValueError(f"{prim_path} does not reference a catalog asset")
        return self._asset_box(prim, ids)[0]

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
        """Lane and keep-clear violations of everything on the floor: assets and geometry need the clearance, floor
        markings may touch a volume but not lie inside it. [] means OK."""
        vols = self.site_volumes()
        out = []
        for ob in self.obstacles():
            for v in vols:
                gap = self._aabb_gap(ob["aabb"], v["aabb"])
                required = 0.0 if ob["kind"] == "floor_marking" else min_clearance_m
                if (gap == 0.0) if ob["kind"] == "floor_marking" else (gap < min_clearance_m):
                    out.append({"asset": ob["path"], "volume": v["name"], "kind": v["kind"], "clearance_m": round(gap, 3), "required_m": required,
                                "status": "encroaches" if gap == 0.0 else "too_close"})
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
        for ob in self.obstacles():
            if ob["path"] == exclude or ob["kind"] == "floor_marking":
                continue
            fb = ob["aabb"]
            if fp[0] < fb[2] and fb[0] < fp[2] and fp[1] < fb[3] and fb[1] < fp[3]:
                out.append({"with": ob["path"], "kind": "asset_overlap"})
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
                if self._fp_conflicts(a["footprint"], a["path"], min_clearance_m):
                    if not self._plain_placement(a["path"]):
                        moves.append({"path": a["path"], "moved": False, "reason": "scaled or matrix-transformed: re-place it with h.place_asset"})
                        continue
                    res = self.nudge_clear(a["path"], min_clearance_m=min_clearance_m)
                    moves.append(res)
                    dirty = dirty or res.get("moved", False)
            if not dirty:
                break
        return moves

    def _plain_placement(self, prim_path: str) -> bool:
        """True when the prim's transform is what place_asset writes (translate and rotate, unit scale), which is what
        nudge_clear and set_position can move."""
        prim = self._agent_prim(prim_path)
        api = UsdGeom.XformCommonAPI(prim)
        if not api:
            return False
        scale = api.GetXformVectors(Usd.TimeCode.Default())[2]
        return all(abs(c - 1.0) < 1e-9 for c in scale)

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
