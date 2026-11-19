"""The trusted judge: the sandbox's clearance pre-check, re-run outside the sandbox on the composed layers with the
trusted copies of the site layer and the catalog. `lane_check` is the CPU stand-in for the Isaac Sim drive-through:
footprints against the lane volume, no physics.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Iterator

from pxr import Usd

from sceneops.helpers import SceneHelpers, load_catalog

ROBOTS = {"carter": {"width_m": 0.5}}  # NVIDIA Nova Carter, the AMR Isaac Sim drives


@contextlib.contextmanager
def _quiet_missing_assets(asset_root: str) -> Iterator[None]:
    """On a host without the Isaac Sim assets, usd-core warns on stderr for every reference it cannot open while it
    composes. Footprints come from the catalog, not from the referenced geometry, so the warnings carry no information
    here; they are dropped at the file-descriptor level (they are written by C++, past Python's sys.stderr)."""
    if os.path.isdir(asset_root):
        yield
        return
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def compose(layer_paths: list[str | Path], *, site_layer: str | Path, catalog: str | Path, asset_root: str = "/opt/assets") -> SceneHelpers:
    """[newest .. oldest layer, site] composed in memory, with the trusted site layer and catalog."""
    stage = Usd.Stage.CreateInMemory()
    with _quiet_missing_assets(asset_root):
        stage.GetRootLayer().subLayerPaths = [str(p) for p in reversed(list(layer_paths))] + [str(site_layer)]
    return SceneHelpers(stage, load_catalog(catalog), asset_root=asset_root)


def judge_layers(layer_paths: list[str | Path], *, site_layer: str | Path, catalog: str | Path, asset_root: str = "/opt/assets",
                 min_clearance_m: float = 0.6) -> dict[str, Any]:
    """Placed assets, lane / keep-clear violations and asset overlaps, with trusted inputs."""
    h = compose(layer_paths, site_layer=site_layer, catalog=catalog, asset_root=asset_root)
    s = h.summary(min_clearance_m)
    overlaps = _asset_overlaps(s["placed_assets"])
    return {
        "placed_assets": [{k: a[k] for k in ("path", "catalog_id", "x", "y", "yaw_deg")} for a in s["placed_assets"]],
        "agent_prims": s["agent_prims"],
        "violations": s["violations"],
        "asset_overlaps": overlaps,
        "min_clearance_m": min((v["clearance_m"] for v in s["violations"]), default=None),
        "passed": not s["violations"] and not overlaps,
    }


def _asset_overlaps(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, a in enumerate(assets):
        for b in assets[i + 1:]:
            fa, fb = a["footprint"], b["footprint"]
            if fa[0] < fb[2] and fb[0] < fa[2] and fa[1] < fb[3] and fb[1] < fa[3]:
                out.append({"a": a["path"], "b": b["path"]})
    return out


def lane_check(layer_paths: list[str | Path], *, site_layer: str | Path, catalog: str | Path, lane: str, robot: str = "carter",
               min_clearance_m: float = 0.6, asset_root: str = "/opt/assets") -> dict[str, Any]:
    """Geometry only: every asset and piece of geometry on the floor against the lane volume, measured with full
    transforms. `passed` needs the required clearance from everything; floor markings may touch the lane but not lie in
    it; `reached` is false when something stands in the lane itself. Isaac Sim replaces this with a PhysX drive-through
    when a renderer is configured (activities/gpu.py)."""
    h = compose(layer_paths, site_layer=site_layer, catalog=catalog, asset_root=asset_root)
    vol = next((v for v in h.site_volumes() if v["name"] == lane), None)
    if vol is None:
        raise ValueError(f"no lane {lane!r} in the site layer")
    gaps, contacts, blocked = [], [], False
    for ob in h.obstacles():
        gap = SceneHelpers._aabb_gap(ob["aabb"], vol["aabb"])
        if ob["kind"] == "floor_marking":
            if gap == 0.0:
                contacts.append(ob["path"])
            continue
        gaps.append(gap)
        if gap < min_clearance_m:
            contacts.append(ob["path"])
            blocked = blocked or gap == 0.0
    return {"lane": lane, "robot": robot, "passed": not contacts, "min_clearance_m": round(min(gaps), 3) if gaps else 99.0,
            "contacts": contacts, "reached": not blocked, "method": "geometry"}
