"""Sandbox activities: build a sandbox generation with the recorded grants, run the agent's tools in it, apply a grant
while it runs, export its log and delete it. Every side effect lives here; the workflow only sees the models."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import shutil
from pathlib import Path
from typing import Any, AsyncIterator

from temporalio import activity
from temporalio.exceptions import ApplicationError

from activities.openshell import SandboxLost, backend, runs_dir
from models import (
    Artifact,
    EnsureInput,
    Grant,
    GrantInput,
    ReleaseInput,
    ReleaseOutput,
    RunCodeInput,
    RunCodeOutput,
    SandboxesRef,
    SearchInput,
    SearchOutput,
)

HEARTBEAT_EVERY = 10.0  # seconds; the activities run with a 60 s heartbeat timeout (workflows/tools.py)


def sandbox_name(workflow_id: str, kind: str, gen: int) -> str:
    """Deterministic, DNS-style names: a retried activity converges on the same sandbox instead of creating another."""
    return f"cookbook-{hashlib.sha1(workflow_id.encode()).hexdigest()[:8]}-{kind}-g{gen}"


@contextlib.asynccontextmanager
async def heartbeating() -> AsyncIterator[dict[str, Any]]:
    """Heartbeat for the whole activity from a background task. OpenShell calls can stay silent for minutes (an image
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


@activity.defn
async def ensure_sandboxes(inp: EnsureInput) -> SandboxesRef:
    """Generation `gen`: an exec sandbox (no network, ever) and a tools sandbox whose policy is the base policy plus the
    approved grants in the workflow's history. Artifacts the agent already produced are uploaded back into the exec
    sandbox, so a rebuilt generation starts where the deleted one stopped."""
    sb = backend()
    names = {kind: sandbox_name(inp.workflow_id, kind, inp.gen) for kind in ("exec", "tools")}
    logs_dir = runs_dir() / inp.run_id / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    # Check the artifacts before touching any sandbox: a generation that cannot be rehydrated is never created, and the
    # previous generation's survivors are not retired for nothing.
    for a in inp.artifacts:
        src = Path(a.path)
        if not src.exists() or hashlib.sha256(src.read_bytes()).hexdigest() != a.sha256:
            raise ApplicationError(f"artifact {a.name} is missing or changed on this host; generation {inp.gen} cannot be rehydrated here",
                                   type="ArtifactMissing", non_retryable=True)
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
        rehydrated: list[str] = []
        for a in inp.artifacts:
            _phase(hb, f"rehydrate:{a.name}")
            try:
                await sb.upload(names["exec"], Path(a.path), a.name)
            except SandboxLost as e:
                raise _lost(e, "rehydrate")
            rehydrated.append(a.name)
    return SandboxesRef(gen=inp.gen, exec_sandbox=names["exec"], tools_sandbox=names["tools"], created=created, rehydrated=rehydrated, retired=retired)


@activity.defn
async def sandbox_search(inp: SearchInput) -> SearchOutput:
    """The agent's catalog search, from inside the tools sandbox. A denied egress is a result, not an error: the tool
    decides what to do about it (ask a human)."""
    async with heartbeating() as hb:
        _phase(hb, "search")
        try:
            out = await backend().search(inp.tools_sandbox, inp.host, inp.query)
        except SandboxLost as e:
            raise _lost(e, "search")
    results = out.get("results") or []
    return SearchOutput(denied=bool(out.get("denied")), results=[r for r in results if isinstance(r, dict)][:8], detail=str(out.get("error") or "")[:200])


@activity.defn
async def sandbox_run_code(inp: RunCodeInput) -> RunCodeOutput:
    """Model-written Python runs in the exec sandbox (no network). Every file it leaves in the work directory is copied to
    the host right away, with its hash: the artifact outlives the sandbox. The workflow runs this activity with a single
    attempt: model-written code is not safe to repeat on a transient failure."""
    sb = backend()
    async with heartbeating() as hb:
        _phase(hb, "run")
        try:
            r = await sb.run_code(inp.exec_sandbox, inp.code)
            out_dir = runs_dir() / inp.run_id / "artifacts"
            out_dir.mkdir(parents=True, exist_ok=True)
            artifacts: list[Artifact] = []
            for name in await sb.list_files(inp.exec_sandbox):
                dst, tmp = out_dir / name, out_dir / f".{name}.part"
                _phase(hb, f"download:{name}")
                await sb.download(inp.exec_sandbox, name, tmp)
                digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
                if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() == digest:
                    tmp.unlink()  # a file the code did not change (for example a rehydrated input) is not a new artifact
                    continue
                tmp.replace(dst)
                artifacts.append(Artifact(name=name, path=str(dst), sha256=digest, step=inp.step))
        except SandboxLost as e:
            raise _lost(e, "run_code")
    return RunCodeOutput(ok=r.exit_code == 0, exit_code=r.exit_code, stdout_tail=r.stdout[-1500:], stderr_tail=r.stderr[-1500:], artifacts=artifacts)


@activity.defn
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
async def release_sandboxes(inp: ReleaseInput) -> ReleaseOutput:
    """Export each sandbox's gateway log (the record of what it tried to reach), then delete it. Runs before every human
    review and whenever a round ends another way, so no sandbox outlives its round."""
    sb = backend()
    logs_dir = runs_dir() / inp.run_id / "logs"
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


def clean_fake_sandboxes() -> None:
    """Test helper: wipe the fake backend's directory."""
    shutil.rmtree(runs_dir() / "_fake_sandboxes", ignore_errors=True)
