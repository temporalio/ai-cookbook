"""Sdf-level lint of agent-authored layers, run on the TRUSTED side before Kit ever opens them.

Kit evaluates code embedded in USD (OmniGraph script nodes, omni:scripting attributes) and resolves
arbitrary asset paths. An agent layer may only: define prims under /World/Agent, use an allowlisted set
of prim types / API schemas / shader ids, and reference assets inside the baked asset root. Pure
path-prefix ``over`` prims (/World, /World/Agent) with no properties are allowed so the layer composes.
No composition is performed here (Sdf only), so nothing in the layer can execute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pxr import Sdf

AGENT_ROOT = Sdf.Path("/World/Agent")
PREFIX_OVERS = {Sdf.Path("/World"), AGENT_ROOT}

ALLOWED_PRIM_TYPES = {
    "", "Xform", "Scope", "Mesh", "BasisCurves", "Cube", "Sphere", "Cylinder", "Cone", "Capsule",
    "SphereLight", "RectLight", "DiskLight", "CylinderLight", "Material", "Shader",
}
ALLOWED_API_SCHEMAS = {"MaterialBindingAPI", "PhysicsCollisionAPI", "CollectionAPI"}
ALLOWED_SHADER_IDS = {"UsdPreviewSurface", "UsdPrimvarReader_float2", "UsdPrimvarReader_float3"}
FORBIDDEN_ATTR_PREFIXES = ("omni:scripting", "omni:graph", "node:", "inputs:script", "outputs:")
FORBIDDEN_TYPE_PREFIXES = ("OmniGraph", "Omni", "Physics", "Camera", "Py", "Script")
DEFAULT_LIMITS = {"max_bytes": 2_000_000, "max_prims": 2000}


@dataclass
class LintReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "stats": self.stats}


def _asset_path_ok(asset: str, asset_root: str) -> bool:
    if not asset:
        return True
    if "${" in asset or ".." in asset.split("/") or "://" in asset or asset.startswith(("http", "omniverse:", "file:")):
        return False
    return asset.startswith(asset_root.rstrip("/") + "/")


def lint_layer(path: str | Path, *, asset_root: str = "/opt/assets", limits: dict | None = None) -> LintReport:
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    p = Path(path)
    rep = LintReport(ok=True)
    size = p.stat().st_size
    if size > limits["max_bytes"]:
        rep.errors.append(f"layer is {size} bytes > {limits['max_bytes']}")
    try:
        layer = Sdf.Layer.FindOrOpen(str(p))
    except Exception as e:  # noqa: BLE001 — pxr raises Tf.ErrorException on parse errors
        layer = None
        rep.errors.append(f"layer failed to parse: {str(e).splitlines()[-1][:200]}")
    if layer is None:
        rep.errors.append("layer failed to parse")
        rep.ok = False
        return rep
    if layer.subLayerPaths:
        rep.errors.append(f"agent layers may not have sublayers: {list(layer.subLayerPaths)}")
    if layer.relocates:
        rep.errors.append("relocates are not allowed")

    prims = 0

    def visit(sp: Sdf.Path) -> None:
        nonlocal prims
        if sp.IsPropertyPath():
            prop = layer.GetPropertyAtPath(sp)
            name = sp.name
            if name.startswith(FORBIDDEN_ATTR_PREFIXES):
                rep.errors.append(f"{sp}: forbidden property")
            if isinstance(prop, Sdf.AttributeSpec) and name == "info:id":
                v = prop.default
                if v not in ALLOWED_SHADER_IDS:
                    rep.errors.append(f"{sp}: shader id {v!r} not allowed")
            if isinstance(prop, Sdf.AttributeSpec) and prop.typeName == Sdf.ValueTypeNames.Asset:
                v = prop.default
                if v is not None and not _asset_path_ok(str(v.path), asset_root):
                    rep.errors.append(f"{sp}: asset path {v.path!r} outside {asset_root}")
            return
        if not sp.IsPrimPath():
            return
        prims += 1
        spec = layer.GetPrimAtPath(sp)
        under_agent = sp.HasPrefix(AGENT_ROOT) and sp != AGENT_ROOT
        if not under_agent:
            if sp in PREFIX_OVERS and spec.specifier == Sdf.SpecifierOver and not spec.properties and not spec.referenceList.GetAddedOrExplicitItems():
                return  # pure path prefix
            rep.errors.append(f"{sp}: prims are only allowed under {AGENT_ROOT} (found {spec.specifier})")
            return
        t = spec.typeName
        if t.startswith(FORBIDDEN_TYPE_PREFIXES) or t not in ALLOWED_PRIM_TYPES:
            rep.errors.append(f"{sp}: prim type {t!r} not allowed")
        for api in spec.GetInfo("apiSchemas").GetAddedOrExplicitItems() if spec.HasInfo("apiSchemas") else []:
            base = api.split(":")[0]
            if base not in ALLOWED_API_SCHEMAS:
                rep.errors.append(f"{sp}: API schema {api!r} not allowed")
        if spec.payloadList.GetAddedOrExplicitItems():
            rep.errors.append(f"{sp}: payloads are not allowed")
        for ref in spec.referenceList.GetAddedOrExplicitItems():
            if not _asset_path_ok(ref.assetPath, asset_root):
                rep.errors.append(f"{sp}: reference {ref.assetPath!r} outside {asset_root}")
        if spec.variantSelections:
            rep.warnings.append(f"{sp}: variant selections present")

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    if prims > limits["max_prims"]:
        rep.errors.append(f"{prims} prims > {limits['max_prims']}")
    rep.stats = {"bytes": size, "prims": prims}
    rep.ok = not rep.errors
    return rep
