"""Sandbox Activities: build a sandbox generation with the recorded grants and the agent's layers, run the agent's tools in
it, apply a grant while it runs, export its log and delete it; write the manifest at the end. Every side effect lives
here; the Workflow only sees the models. Every Activity heartbeats for its whole duration and is a NeMo Relay tool call."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from pathlib import Path
from typing import Any, AsyncIterator

from temporalio import activity
from temporalio.exceptions import ApplicationError

import relay
from activities.openshell import ExecResult, SandboxLost, backend
from models import (
    ApplyInput,
    ApplyOutput,
    EnsureInput,
    Grant,
    GrantInput,
    LayerRef,
    ManifestInput,
    ReleaseInput,
    ReleaseOutput,
    SandboxesRef,
    SearchInput,
    SearchOutput,
)
from paths import ASSET_ROOT, CATALOG, SITE_LAYER, run_dir
from sceneops.catalog import catalog_search, load_catalog, merge_hits
from sceneops.hashing import layer_stack_hash, sha256_bytes, sha256_file
from validate.judge import judge_layers
from validate.lint import lint_layer

HEARTBEAT_EVERY = 10.0  # seconds; the Activities run with a 60 s heartbeat timeout (workflows/tools.py)


def sandbox_name(workflow_id: str, kind: str, gen: int) -> str:
    """Deterministic, short, DNS-style names (OpenShell 0.1.2 rejects long ones): a retried Activity converges on the same
    sandbox instead of creating another. sim-<hash>-x1 is generation 1's exec sandbox, sim-<hash>-t2 generation 2's tools."""
    return f"sim-{hashlib.sha1(workflow_id.encode()).hexdigest()[:6]}-{'x' if kind == 'exec' else 't'}{gen}"


@contextlib.asynccontextmanager
async def heartbeating() -> AsyncIterator[dict[str, Any]]:
    """Heartbeat for the whole Activity from a background task. OpenShell calls can stay silent for minutes (an image
    pull on a cold host, `policy update --wait`, the delete poll); a heartbeat only between them lets the timeout fire
    and start a second attempt while the first is still running."""
    details: dict[str, Any] = {"phase": "start"}

    async def beat() -> None:
        while True:
            activity.heartbeat(details)
            await asyncio.sleep(HEARTBEAT_EVERY)

    task = asyncio.create_task(beat())
    try:
        yield details
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def _phase(details: dict[str, Any], phase: str) -> None:
    details["phase"] = phase
    activity.heartbeat(details)


def _lost(e: SandboxLost, what: str) -> ApplicationError:
    # Non-retryable: retrying the same call cannot bring the sandbox back; the tool rebuilds a new generation instead.
    return ApplicationError(f"{what}: {e}", type="SandboxLost", non_retryable=True)


def _envelope(r: ExecResult, what: str) -> dict[str, Any]:
    env = r.envelope
    if env is None:
        raise ApplicationError(f"{what}: no sceneops envelope (exit {r.exit_code}): {r.stderr[-300:]}", type="ExecFailed")
    if r.exit_code == 4:
        raise ApplicationError(f"{what}: sceneops internal error: {env.get('error')}", type="ExecFailed")
    return env


def _layer_file(layer: LayerRef, what: str) -> Path:
    """The recorded layer on this host, verified against its hash. Another host cannot rebuild from it: point RUNS_DIR at
    shared storage when Workers run on several hosts."""
    p = Path(layer.path)
    if not p.exists() or sha256_file(p) != layer.sha256:
        raise ApplicationError(f"{what}: layer {layer.step} ({layer.path}) is missing or changed on this host", type="LayerMissing", non_retryable=True)
    return p


@activity.defn
@relay.traced
async def ensure_sandboxes(inp: EnsureInput) -> SandboxesRef:
    """Generation `gen`: an exec sandbox (no network, ever) and a tools sandbox whose policy is the base policy plus the
    approved grants in the Workflow's history. The layers the agent already applied are uploaded back into the exec
    sandbox and recomposed; the layer stack hash must match the Workflow's record, so a rebuilt generation starts exactly
    where the deleted one stopped."""
    sb = backend()
    names = {kind: sandbox_name(inp.workflow_id, kind, inp.gen) for kind in ("exec", "tools")}
    logs_dir = run_dir(inp.run_id) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    # Check the layers before touching any sandbox: a generation that cannot be rehydrated is never created, and the
    # previous generation's survivors are not retired for nothing.
    sources = {layer.step: _layer_file(layer, "rehydrate") for layer in inp.layers}
    async with heartbeating() as hb:
        retired: list[str] = []
        if inp.gen > 1:  # whatever survives of the previous generation: keep its gateway log, then delete it
            for kind in ("exec", "tools"):
                old = sandbox_name(inp.workflow_id, kind, inp.gen - 1)
                if await sb.exists(old):
                    _phase(hb, f"retire:{old}")
                    (logs_dir / f"g{inp.gen - 1}-{kind}.log").write_text(await sb.logs(old))
                    await sb.delete(old)
                    retired.append(old)
        created: list[str] = []
        for kind, name in names.items():
            _phase(hb, f"ensure:{kind}")
            if await sb.ensure(name, kind, inp.grants, heartbeat=lambda phase, k=kind: _phase(hb, f"{k}:{phase}")):
                created.append(name)
        pairs = [(layer.step, layer.sha256) for layer in inp.layers]
        stack_hash = layer_stack_hash([])
        try:
            for layer in inp.layers:
                _phase(hb, f"rehydrate:{layer.step}")
                await sb.upload_layer(names["exec"], sources[layer.step], layer.step)
            if inp.layers:
                env = _envelope(await sb.rehydrate(names["exec"], pairs), "rehydrate")
                if not env.get("ok"):
                    raise ApplicationError(f"rehydrate: {env.get('error')}", type="RehydrateMismatch", non_retryable=True)
                stack_hash = env["data"]["layer_stack_hash"]
                if stack_hash != layer_stack_hash(pairs):
                    raise ApplicationError(f"rehydrate produced {stack_hash[:12]}, expected {layer_stack_hash(pairs)[:12]}", type="RehydrateMismatch", non_retryable=True)
        except SandboxLost as e:
            raise _lost(e, "rehydrate")
    return SandboxesRef(gen=inp.gen, exec_sandbox=names["exec"], tools_sandbox=names["tools"], created=created, rehydrated_steps=[s for s, _ in pairs],
                        layer_stack_hash=stack_hash, retired=retired)


@activity.defn
@relay.traced
async def sandbox_search(inp: SearchInput) -> SearchOutput:
    """The agent's asset search, from inside the tools sandbox. A denied egress is a result, not an error: the tool
    decides what to do about it (ask a person). Hits are mapped to catalog ids here, with the trusted catalog; when the
    search is denied, the catalog answers."""
    catalog = load_catalog(CATALOG)
    async with heartbeating() as hb:
        _phase(hb, "search")
        try:
            out = await backend().search(inp.tools_sandbox, inp.host, inp.query)
        except SandboxLost as e:
            raise _lost(e, "search")
    if out.get("denied"):
        return SearchOutput(denied=True, source="catalog", results=catalog_search(catalog, inp.query), detail=str(out.get("error") or "")[:200])
    hits = [h for h in (out.get("hits") or []) if isinstance(h, dict)]
    return SearchOutput(denied=False, source="usd_search", results=merge_hits(hits, catalog, inp.query), detail="")


@activity.defn
@relay.traced
async def sandbox_apply(inp: ApplyInput) -> ApplyOutput:
    """Model-written Python edits the scene through the helper API inside the exec sandbox (no network) and leaves one
    OpenUSD sublayer for the step. The layer is copied out at once, linted before anything else opens it, judged against
    the trusted site layer and catalog, and recorded with its hash. Idempotent by (step, code hash): the sandbox returns
    its cached result when an attempt is repeated, so Temporal may retry this Activity."""
    sb = backend()
    info = activity.info()
    code_sha = sha256_bytes(inp.code.encode())
    meta = {"workflow_id": inp.workflow_id, "activity_id": info.activity_id, "attempt": str(info.attempt), "model": inp.model,
            "sandbox": inp.exec_sandbox, "code_sha256": code_sha}
    prior_hash = layer_stack_hash([(layer.step, layer.sha256) for layer in inp.prior_layers])

    def rejected(error: dict[str, Any], data: dict[str, Any], warnings: list[str]) -> ApplyOutput:
        return ApplyOutput(ok=False, cached=False, layer=None, error=error, warnings=warnings, violations=data.get("precheck_violations", []), asset_overlaps=[],
                           placed_assets=data.get("placed_assets", []), agent_prims=data.get("agent_prims", []), stdout_tail=data.get("stdout_tail", ""),
                           layer_stack_hash=prior_hash)

    async with heartbeating() as hb:
        _phase(hb, f"apply:{inp.step}")
        try:
            env = _envelope(await sb.apply(inp.exec_sandbox, inp.step, inp.code, meta, inp.min_clearance_m), "apply")
            if not env.get("ok"):
                return rejected(env.get("error") or {"type": "ApplyFailed"}, {"stdout_tail": env.get("stdout_tail", "")}, env.get("warnings", []))
            data = env["data"]
            layers_dir = run_dir(inp.run_id) / "layers"
            layers_dir.mkdir(parents=True, exist_ok=True)
            final, tmp = layers_dir / f"{inp.step:03d}.usda", layers_dir / f".{inp.step:03d}.part.usda"  # keep a .usda suffix: Sdf infers the format
            _phase(hb, "download")
            await sb.download_layer(inp.exec_sandbox, inp.step, tmp)
        except SandboxLost as e:
            raise _lost(e, "apply")
        sha = sha256_file(tmp)
        if sha != data["sha256"]:
            tmp.unlink()
            raise ApplicationError("downloaded layer sha256 differs from the sandbox's report", type="ExecFailed")
        _phase(hb, "lint")
        report = lint_layer(tmp, asset_root=ASSET_ROOT)
        if not report.ok:
            tmp.unlink()
            return rejected({"type": "LintRejected", "msg": "; ".join(report.errors)[:800]}, data, report.warnings)
        _phase(hb, "judge")
        prior = [_layer_file(layer, "apply") for layer in inp.prior_layers]
        verdict = judge_layers([*prior, tmp], site_layer=SITE_LAYER, catalog=CATALOG, asset_root=ASSET_ROOT, min_clearance_m=inp.min_clearance_m)
        tmp.replace(final)
    summary = f"step {inp.step}: {inp.rationale[:120]}"
    layer = LayerRef(step=inp.step, path=str(final), sha256=sha, bytes=final.stat().st_size, summary=summary)
    warnings = list(env.get("warnings", []))
    warnings += [f"trusted check: {v['asset']} {v['status']} {v['volume']} ({v['clearance_m']} m < {v['required_m']} m)" for v in verdict["violations"]]
    warnings += [f"assets overlap: {o['a']} and {o['b']}" for o in verdict["asset_overlaps"]]
    return ApplyOutput(ok=True, cached=bool(env.get("cached")), layer=layer, error=None, warnings=warnings, violations=verdict["violations"],
                       asset_overlaps=verdict["asset_overlaps"], placed_assets=verdict["placed_assets"], agent_prims=verdict["agent_prims"],
                       stdout_tail=data.get("stdout_tail", ""), layer_stack_hash=layer_stack_hash([(layer.step, layer.sha256) for layer in inp.prior_layers] + [(inp.step, sha)]))


@activity.defn
@relay.traced
async def apply_grant(inp: GrantInput) -> Grant:
    """Apply an approved grant to the running tools sandbox (a live policy update). Rejected grants are recorded only."""
    if inp.grant.decision != "approved":
        return inp.grant
    async with heartbeating() as hb:
        _phase(hb, "policy_update")
        try:
            await backend().apply_grant(inp.tools_sandbox, inp.grant)
        except SandboxLost as e:
            raise _lost(e, "apply_grant")
    return inp.grant.model_copy(update={"applied": True})


@activity.defn
@relay.traced
async def release_sandboxes(inp: ReleaseInput) -> ReleaseOutput:
    """Export each sandbox's gateway log (the record of what it tried to reach), then delete it. Runs before every human
    review and whenever a round ends another way, so no sandbox outlives its round."""
    sb = backend()
    logs_dir = run_dir(inp.run_id) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    deleted, logs = [], []
    async with heartbeating() as hb:
        for kind, name in (("exec", inp.exec_sandbox), ("tools", inp.tools_sandbox)):
            if not name or not await sb.exists(name):
                continue
            _phase(hb, f"logs:{kind}")
            text = await sb.logs(name)
            if text.strip():
                p = logs_dir / f"g{inp.gen}-{kind}.log"
                p.write_text(text)
                logs.append(str(p))
            _phase(hb, f"delete:{kind}")
            await sb.delete(name)
            deleted.append(name)
    return ReleaseOutput(deleted=deleted, logs=logs)


@activity.defn
async def write_manifest(inp: ManifestInput) -> str:
    """The run's provenance manifest, written when the run ends: what the agent was asked, which model answered, every
    layer and the layer stack hash, every render and lane check, every grant, where the gateway logs and the Relay
    trajectories are. This is the agent Worker's last Activity for the run, so it also closes the Worker's Relay scope
    (not a tool span itself: the scope it closes would have to record it)."""
    relay_files = await relay.close_here(reason=inp.status)
    d = run_dir(inp.run_id)
    st = inp.state
    doc = {
        "schema": "durable-simulation-agent.manifest/1", "run_id": inp.run_id, "workflow_id": inp.workflow_id, "status": inp.status,
        "temporal": {"namespace": activity.info().workflow_namespace, "note": "temporal workflow show -w <workflow-id> --output json exports the history"},
        "brief": inp.brief, "model": inp.model, "reviewer_notes": inp.notes, "decisions": inp.decisions,
        "layers": [layer.model_dump() for layer in st.layers], "layer_stack_hash": layer_stack_hash(st.layer_pairs()),
        "renders": [r.model_dump() for r in st.renders], "lane_checks": [c.model_dump() for c in st.lane_checks], "grants": [g.model_dump() for g in st.grants],
        "openshell": {"sandbox_generations": st.gen, "logs": sorted(str(p) for p in (d / "logs").glob("*.log")) if (d / "logs").exists() else []},
        "relay": {"runtime": relay.version(), "files": relay_files},
    }
    path = d / "manifest.json"
    d.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return str(path)
