"""Sdf-level lint of agent-authored layers, run on the TRUSTED side before Kit ever opens them.

Kit evaluates code embedded in USD (OmniGraph script nodes, omni:scripting attributes) and resolves
arbitrary asset paths. An agent layer may only: define prims under /World/Agent with allowlisted metadata,
prim types, API schemas and shader ids, and reference assets inside the baked asset root. Bare path-prefix
``over`` prims (/World, /World/Agent) are allowed so the layer composes. No variants, payloads, inherits,
specializes, value clips or sublayers anywhere. No composition is performed here (Sdf only), so nothing in
the layer can execute.
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
# Prim metadata the helper API writes (the rest is refused: payload, inheritPaths, specializes, variantSetNames,
# variantSelection, clips, assetInfo, ...). Prefix overs carry nothing but their specifier.
AGENT_INFO_KEYS = {"specifier", "typeName", "references", "active", "customData", "apiSchemas", "kind", "documentation", "hidden"}
PREFIX_INFO_KEYS = {"specifier"}


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


def _asset_values(layer: Sdf.Layer, sp: Sdf.Path, prop: Sdf.AttributeSpec) -> list[str]:
    """Every asset path an asset or asset[] attribute holds: its default and every time sample."""
    values = [prop.default] + [layer.QueryTimeSample(sp, t) for t in layer.ListTimeSamplesForPath(sp)]
    out: list[str] = []
    for v in values:
        if v is None:
            continue
        out += [item.path for item in ([v] if isinstance(v, Sdf.AssetPath) else list(v))]
    return out


def lint_layer(path: str | Path, *, asset_root: str = "/opt/assets", limits: dict | None = None) -> LintReport:
    """An allowlist, not a denylist: agent layers may only carry what the helper API writes. Anything else, a payload,
    a variant, an inherit, a value clip, a sublayer, is an error, wherever in the layer it appears."""
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
        if sp.ContainsPrimVariantSelection():
            if sp.IsPrimVariantSelectionPath():
                rep.errors.append(f"{sp}: variants are not allowed")
            return  # everything inside a variant is reported once, at the variant
        if sp.IsPropertyPath():
            prop = layer.GetPropertyAtPath(sp)
            name = sp.name
            if name.startswith(FORBIDDEN_ATTR_PREFIXES):
                rep.errors.append(f"{sp}: forbidden property")
            if isinstance(prop, Sdf.AttributeSpec) and name == "info:id":
                v = prop.default
                if v not in ALLOWED_SHADER_IDS:
                    rep.errors.append(f"{sp}: shader id {v!r} not allowed")
            if isinstance(prop, Sdf.AttributeSpec) and prop.typeName in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray):
                for v in _asset_values(layer, sp, prop):
                    if not _asset_path_ok(v, asset_root):
                        rep.errors.append(f"{sp}: asset path {v!r} outside {asset_root}")
            return
        if not sp.IsPrimPath():
            return
        prims += 1
        spec = layer.GetPrimAtPath(sp)
        keys = set(spec.ListInfoKeys())
        under_agent = sp.HasPrefix(AGENT_ROOT) and sp != AGENT_ROOT
        if not under_agent:
            # /World and /World/Agent may appear only as bare `over` path prefixes: no metadata, no properties.
            if sp in PREFIX_OVERS and spec.specifier == Sdf.SpecifierOver and keys <= PREFIX_INFO_KEYS and not spec.properties:
                return
            rep.errors.append(f"{sp}: prims are only allowed under {AGENT_ROOT}, and its prefixes only as bare overs")
            return
        for k in sorted(keys - AGENT_INFO_KEYS):
            rep.errors.append(f"{sp}: metadata {k!r} not allowed")
        t = spec.typeName
        if t.startswith(FORBIDDEN_TYPE_PREFIXES) or t not in ALLOWED_PRIM_TYPES:
            rep.errors.append(f"{sp}: prim type {t!r} not allowed")
        for api in spec.GetInfo("apiSchemas").GetAddedOrExplicitItems() if spec.HasInfo("apiSchemas") else []:
            base = api.split(":")[0]
            if base not in ALLOWED_API_SCHEMAS:
                rep.errors.append(f"{sp}: API schema {api!r} not allowed")
        for ref in spec.referenceList.GetAddedOrExplicitItems():
            if not _asset_path_ok(ref.assetPath, asset_root):
                rep.errors.append(f"{sp}: reference {ref.assetPath!r} outside {asset_root}")

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    if prims > limits["max_prims"]:
        rep.errors.append(f"{prims} prims > {limits['max_prims']}")
    rep.stats = {"bytes": size, "prims": prims}
    rep.ok = not rep.errors
    return rep
