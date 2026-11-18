"""sceneops.apply — the EXEC-sandbox entrypoint. No network, ever.

    python3 -I -m sceneops.apply apply --step N [--meta k=v ...] < code.py
    python3 -I -m sceneops.apply rehydrate < '{"layers":[{"step":1,"sha256":"..."}]}'
    python3 -I -m sceneops.apply summary

Prints exactly one JSON envelope as the LAST stdout line; everything else goes to stderr.
Exit codes: 0 ok · 2 validation failed (the agent can fix its code) · 4 internal.
Workdir layout: <work>/scene.usda, <work>/layers/NNN.usda, <work>/layers/NNN.result.json, <work>/layers/.lock
"""

from __future__ import annotations

import argparse
import fcntl
import io
import json
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom

from .hashing import layer_stack_hash, sha256_bytes, sha256_file
from .helpers import SceneHelpers, load_catalog

EXIT_OK, EXIT_VALIDATION, EXIT_INTERNAL = 0, 2, 4
SITE_LAYER = "000_site.usda"


def emit(op: str, data: dict | None, *, ok: bool, error: dict | None = None, warnings: list[str] | None = None,
         started: float, code: int = EXIT_OK, **extra) -> None:
    env = {"ok": ok, "op": op, "elapsed_ms": int((time.time() - started) * 1000), "data": data, "warnings": warnings or [],
           "error": error, **extra}
    sys.stdout.write(json.dumps(env, sort_keys=True) + "\n")
    sys.stdout.flush()
    sys.exit(code)


@contextmanager
def locked(work: Path):
    work.joinpath("layers").mkdir(parents=True, exist_ok=True)
    with open(work / "layers" / ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def layer_path(work: Path, step: int) -> Path:
    return work / "layers" / f"{step:03d}.usda"


def existing_steps(work: Path) -> list[int]:
    return sorted(int(p.stem) for p in (work / "layers").glob("[0-9][0-9][0-9].usda"))


def rebuild_root(work: Path, site: Path, steps: list[int]) -> Sdf.Layer:
    """scene.usda sublayers = [newest step ... oldest step, site layer]. Relative paths keep it portable."""
    root_path = work / "scene.usda"
    root = Sdf.Layer.FindOrOpen(str(root_path)) or Sdf.Layer.CreateNew(str(root_path))
    root.Clear()
    root.subLayerPaths = [f"layers/{s:03d}.usda" for s in sorted(steps, reverse=True)] + [str(site / SITE_LAYER)]
    root.Save()
    return root


def compile_lenient(code: str, name: str):
    """Compile model-written code; if it fails only because quotes were JSON-escaped (a common LLM slip:
    print(f\\"..\\")), unescape and retry. Returns (code_object | None, repaired_flag | error_message)."""
    try:
        return compile(code, name, "exec"), False
    except SyntaxError as e:
        first = str(e)
    if "line continuation character" in first or "unterminated" in first:
        fixed = code.replace('\\"', '"').replace("\\'", "'")
        try:
            return compile(fixed, name, "exec"), True
        except SyntaxError:
            pass
    return None, first


def prim_specs(layer: Sdf.Layer) -> tuple[list[str], list[str]]:
    added, modified = [], []

    def visit(path: Sdf.Path) -> None:
        if path.IsPrimPath():
            spec = layer.GetPrimAtPath(path)
            (added if spec.specifier == Sdf.SpecifierDef else modified).append(path.pathString)

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    return added, modified


def flatten_sha256(stage: Usd.Stage) -> str:
    return sha256_bytes(stage.Flatten().ExportToString().encode())


def stack_from_root(work: Path, root: Sdf.Layer) -> list[dict]:
    out = []
    for sub in reversed(root.subLayerPaths):  # oldest first
        if sub.startswith("layers/"):
            step = int(Path(sub).stem)
            p = layer_path(work, step)
            out.append({"step": step, "sha256": sha256_file(p), "bytes": p.stat().st_size})
    return out


def cmd_apply(args: argparse.Namespace, started: float) -> None:
    work, site = Path(args.work), Path(args.site)
    code = sys.stdin.read()
    code_sha = sha256_bytes(code.encode())
    meta = dict(kv.split("=", 1) for kv in args.meta)
    with locked(work):
        res_path = work / "layers" / f"{args.step:03d}.result.json"
        lp = layer_path(work, args.step)
        if res_path.exists() and lp.exists():
            prev = json.loads(res_path.read_text())
            if prev.get("code_sha256") == code_sha and sha256_file(lp) == prev["data"]["sha256"]:
                prev["cached"] = True
                prev["elapsed_ms"] = int((time.time() - started) * 1000)
                sys.stdout.write(json.dumps(prev, sort_keys=True) + "\n")
                sys.exit(EXIT_OK)
        prior = [s for s in existing_steps(work) if s < args.step]
        root = rebuild_root(work, site, prior)
        stage = Usd.Stage.Open(str(work / "scene.usda"))
        if lp.exists():
            layer = Sdf.Layer.FindOrOpen(str(lp))
            layer.Clear()
        else:
            layer = Sdf.Layer.CreateNew(str(lp))
        root.subLayerPaths.insert(0, f"layers/{args.step:03d}.usda")
        root_subs_before = list(root.subLayerPaths)
        stage.SetEditTarget(Usd.EditTarget(layer))
        catalog = load_catalog(site / "catalog.json")
        h = SceneHelpers(stage, catalog, asset_root=args.asset_root)
        ns = {"h": h, "stage": stage, "Usd": Usd, "Sdf": Sdf, "UsdGeom": UsdGeom, "Gf": Gf}
        compiled, repaired = compile_lenient(code, f"step_{args.step:03d}")
        if compiled is None:
            emit("apply", None, ok=False, error={"type": "SyntaxError", "msg": repaired}, started=started, code=EXIT_VALIDATION,
                 step=args.step, code_sha256=code_sha)
        captured = io.StringIO()
        try:
            with redirect_stdout(captured):
                exec(compiled, ns)  # noqa: S102 — model-written code; this is what the exec sandbox is for
        except Exception as e:  # noqa: BLE001
            emit("apply", None, ok=False, error={"type": type(e).__name__, "msg": str(e), "trace": traceback.format_exc(limit=4)},
                 started=started, code=EXIT_VALIDATION, step=args.step, code_sha256=code_sha, stdout_tail=captured.getvalue()[-1500:])
        escaped = [lay.identifier for lay in stage.GetLayerStack(includeSessionLayers=False)
                   if lay.dirty and lay.identifier not in (layer.identifier, root.identifier)]
        if list(root.subLayerPaths) != root_subs_before:
            escaped.append(root.identifier)
        if escaped:
            emit("apply", None, ok=False, error={"type": "EditEscaped", "msg": f"code modified layers other than step {args.step}", "detail": escaped},
                 started=started, code=EXIT_VALIDATION, step=args.step, code_sha256=code_sha)
        layer.customLayerData = {"agent": {**meta, "step": args.step, "code_sha256": code_sha}}
        layer.Save()
        root.Save()
        sha = sha256_file(lp)
        added, modified = prim_specs(layer)
        summary = h.summary(args.min_clearance)
        stack = stack_from_root(work, root)
        data = {
            "layer_path": str(lp), "sha256": sha, "bytes": lp.stat().st_size,
            "prims_added": added, "prims_modified": modified,
            "placed_assets": [{k: a[k] for k in ("path", "catalog_id", "x", "y", "yaw_deg")} for a in summary["placed_assets"]],
            "agent_prims": summary["agent_prims"],
            "precheck_violations": summary["violations"],
            "layer_stack": stack, "layer_stack_hash": layer_stack_hash([(s["step"], s["sha256"]) for s in stack]),
            "stdout_tail": captured.getvalue()[-1500:], "code_repaired": repaired,
        }
        warnings = [f"precheck: {v['asset']} {v['status']} {v['volume']} ({v['clearance_m']} m < {v['required_m']} m)" for v in summary["violations"]]
        env = {"ok": True, "op": "apply", "step": args.step, "code_sha256": code_sha, "data": data, "warnings": warnings,
               "error": None, "cached": False}
        res_path.write_text(json.dumps(env, sort_keys=True))
        emit("apply", data, ok=True, warnings=warnings, started=started, step=args.step, code_sha256=code_sha, cached=False)


def cmd_rehydrate(args: argparse.Namespace, started: float) -> None:
    work, site = Path(args.work), Path(args.site)
    spec = json.loads(sys.stdin.read() or "{}")
    wanted = sorted(spec.get("layers", []), key=lambda entry: entry["step"])
    with locked(work):
        mismatches = []
        for entry in wanted:
            p = layer_path(work, entry["step"])
            if not p.exists():
                mismatches.append({"step": entry["step"], "reason": "missing"})
            elif sha256_file(p) != entry["sha256"]:
                mismatches.append({"step": entry["step"], "reason": "sha256 mismatch", "have": sha256_file(p)})
        if mismatches:
            emit("rehydrate", None, ok=False, error={"type": "RehydrateMismatch", "msg": "layer files do not match the manifest", "detail": mismatches},
                 started=started, code=EXIT_VALIDATION)
        root = rebuild_root(work, site, [entry["step"] for entry in wanted])
        stage = Usd.Stage.Open(str(work / "scene.usda"))
        stack = stack_from_root(work, root)
        emit("rehydrate", {"layers_applied": stack, "layer_stack_hash": layer_stack_hash([(s["step"], s["sha256"]) for s in stack]),
                           "flatten_sha256": flatten_sha256(stage)}, ok=True, started=started)


def cmd_summary(args: argparse.Namespace, started: float) -> None:
    work, site = Path(args.work), Path(args.site)
    root_path = work / "scene.usda"
    if not root_path.exists():
        rebuild_root(work, site, existing_steps(work))
    stage = Usd.Stage.Open(str(root_path))
    root = stage.GetRootLayer()
    h = SceneHelpers(stage, load_catalog(site / "catalog.json"), asset_root=args.asset_root)
    s = h.summary(args.min_clearance)
    stack = stack_from_root(work, root)
    emit("summary", {
        "prim_count": sum(1 for _ in stage.Traverse()),
        "placed_assets": [{k: a[k] for k in ("path", "catalog_id", "x", "y", "yaw_deg")} for a in s["placed_assets"]],
        "agent_prims": s["agent_prims"],
        "precheck_violations": s["violations"], "site_volumes": s["site_volumes"], "docks": s["docks"],
        "layer_stack": stack, "layer_stack_hash": layer_stack_hash([(x["step"], x["sha256"]) for x in stack]),
        "flatten_sha256": flatten_sha256(stage),
    }, ok=True, started=started)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="sceneops.apply")
    ap.add_argument("--work", default="/sandbox")
    ap.add_argument("--site", default="/opt/site")
    ap.add_argument("--asset-root", default="/opt/assets")
    ap.add_argument("--min-clearance", type=float, default=0.6)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("apply")
    a.add_argument("--step", type=int, required=True)
    a.add_argument("--meta", action="append", default=[], help="k=v stamped into the layer's customLayerData")
    sub.add_parser("rehydrate")
    sub.add_parser("summary")
    args = ap.parse_args(argv)
    started = time.time()
    try:
        {"apply": cmd_apply, "rehydrate": cmd_rehydrate, "summary": cmd_summary}[args.cmd](args, started)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        emit(args.cmd, None, ok=False, error={"type": "Internal", "msg": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=4)},
             started=started, code=EXIT_INTERNAL)


if __name__ == "__main__":
    main()
