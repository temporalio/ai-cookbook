"""Sdf-level lint of agent-authored layers, run on the TRUSTED side before Kit ever opens them.

Kit evaluates code embedded in USD (OmniGraph script nodes, omni:scripting attributes) and resolves
arbitrary asset paths. An agent layer may only: define prims under /World/Agent with allowlisted metadata,
prim types, API schemas and shader ids, only as direct children, and reference the catalog's own asset files. The asset root also holds
assets that carry OmniGraph nodes (the Nova Carter sample, for one), so a path under the root is not enough;
the catalog's assets and their dependencies carry none. Bare path-prefix ``over`` prims (/World, /World/Agent)
are allowed so the layer composes. No variants, payloads, inherits, specializes, value clips or sublayers
anywhere. No composition is performed here (Sdf only), so nothing in the layer can execute.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
FORBIDDEN_PROPERTIES = {"extent", "extentsHint", "purpose"}
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


def _asset_values(prop: Sdf.AttributeSpec) -> list[str]:
    """Asset paths in a static attribute. A value block authors no path."""
    value = prop.default
    if value is None or isinstance(value, Sdf.ValueBlock):
        return []
    if isinstance(value, Sdf.AssetPath):
        return [value.path]
    return [item.path for item in value]


def catalog_references(catalog: dict[str, dict[str, Any]] | str | Path, asset_root: str) -> set[str]:
    """The asset paths an agent layer may reference: the catalog's files under the asset root, nothing else."""
    entries = catalog.values() if isinstance(catalog, dict) else json.loads(Path(catalog).read_text())["assets"]
    return {f"{asset_root.rstrip('/')}/{a['usd']}" for a in entries}


def lint_layer(path: str | Path, *, catalog: dict[str, dict[str, Any]] | str | Path, asset_root: str = "/opt/assets",
               limits: dict | None = None) -> LintReport:
    """An allowlist, not a denylist: agent layers may only carry what the helper API writes. Anything else, a payload,
    a variant, an inherit, a value clip, a sublayer, is an error, wherever in the layer it appears."""
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    allowed_refs = catalog_references(catalog, asset_root)
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
            if name in FORBIDDEN_PROPERTIES or name.startswith(FORBIDDEN_ATTR_PREFIXES):
                rep.errors.append(f"{sp}: forbidden property")
            if not isinstance(prop, Sdf.AttributeSpec):
                return
            if layer.GetNumTimeSamplesForPath(sp) or "spline" in prop.ListInfoKeys():
                rep.errors.append(f"{sp}: time samples and splines are not allowed")
            if name == "info:id":
                shader_id = prop.default
                if not isinstance(shader_id, str) or shader_id not in ALLOWED_SHADER_IDS:
                    rep.errors.append(f"{sp}: shader id {shader_id!r} not allowed")
            if prop.typeName in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray):
                for v in _asset_values(prop):
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
        if sp.GetParentPath() != AGENT_ROOT:
            # Only direct children, which is all the helper API writes: nothing nested under a scaled group, nothing
            # authored inside a referenced asset, nothing the judge's measurements cannot see.
            rep.errors.append(f"{sp}: only direct children of {AGENT_ROOT} are allowed")
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
        references = spec.referenceList.GetAddedOrExplicitItems()
        if references and (spec.specifier != Sdf.SpecifierDef or t != "Xform"):
            rep.errors.append(f"{sp}: an asset reference requires a defined Xform")
        for ref in references:
            if not ref.assetPath:
                rep.errors.append(f"{sp}: internal references are not allowed")
            elif ref.assetPath not in allowed_refs:
                rep.errors.append(f"{sp}: reference {ref.assetPath!r} is not a catalog asset")

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    if prims > limits["max_prims"]:
        rep.errors.append(f"{prims} prims > {limits['max_prims']}")
    rep.stats = {"bytes": size, "prims": prims}
    rep.ok = not rep.errors
    return rep
