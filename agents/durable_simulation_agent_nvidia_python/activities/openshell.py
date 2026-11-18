"""Two sandbox backends behind one small interface.

`OpenShellCLI` drives NVIDIA OpenShell through its CLI: create a sandbox from a policy, run a command in it, move files in
and out, update its network policy while it runs, read its gateway log, delete it. `FakeSandboxes` reproduces only the
control flow in plain directories, with no isolation at all, so the recipe and its tests run on a laptop without OpenShell:
the same `sceneops` entrypoints run with the host interpreter.

Pick one with SANDBOX_BACKEND=openshell|fake. The default is `openshell` when the `openshell` binary is on PATH; the fake
must be asked for by name.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from models import Grant
from paths import CATALOG, RECIPE_DIR, SITE_DIR, runs_dir
from sceneops.catalog import S3_ROOT, catalog_search, load_catalog

NOT_FOUND = re.compile(r"not found|NOT_FOUND|no such sandbox", re.I)
POLICIES = Path(__file__).parent / "policies"
WORKDIR = "/sandbox"  # the sandbox work directory: layers/NNN.usda and the composed scene.usda
SITE = "/opt/site"  # the site layer and the catalog, shipped read-only in the sandbox image


class SandboxLost(Exception):
    """The sandbox no longer exists: somebody deleted it while the run was in progress."""


class SandboxError(Exception):
    pass


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def envelope(self) -> dict[str, Any] | None:
        """The sceneops JSON envelope: the last non-empty stdout line, when the command spoke sceneops."""
        lines = [line for line in self.stdout.splitlines() if line.strip()]
        if not lines:
            return None
        try:
            obj = json.loads(lines[-1])
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) and "ok" in obj else None


def backend():
    """SANDBOX_BACKEND=openshell (the default when the `openshell` binary is on PATH) or SANDBOX_BACKEND=fake. The fake must
    be asked for by name: a recipe about sandboxing model-written code does not fall back to no isolation on its own."""
    choice = os.environ.get("SANDBOX_BACKEND")
    if choice is None:
        if not shutil.which(os.environ.get("OPENSHELL_BIN", "openshell")):
            raise RuntimeError("no `openshell` binary on PATH. Install NVIDIA OpenShell, or set SANDBOX_BACKEND=fake to run the control flow "
                               "with no sandbox at all (the fake runs model-written code on this host).")
        choice = "openshell"
    if choice == "openshell":
        return OpenShellCLI()
    if choice == "fake":
        return FakeSandboxes()
    raise RuntimeError(f"SANDBOX_BACKEND={choice!r}: expected 'openshell' or 'fake'")


def _meta_args(meta: dict[str, str]) -> list[str]:
    return [arg for k, v in meta.items() for arg in ("--meta", f"{k}={v}")]


# ---------------------------------------------------------------- NVIDIA OpenShell, through its CLI


class OpenShellCLI:
    def __init__(self) -> None:
        self.binary = os.environ.get("OPENSHELL_BIN", "openshell")
        self.image = os.environ.get("SANDBOX_IMAGE", "durable-simulation-agent-sandbox:0.1")  # sandbox-image/Dockerfile
        # Network rules match the REAL path of the connecting executable (symlinks do not count): the image's interpreter.
        self.python = os.environ.get("SANDBOX_PYTHON", "/usr/local/bin/python3.12")

    async def _run(self, *args: str, stdin: bytes | None = None, timeout: float = 120) -> ExecResult:
        proc = await asyncio.create_subprocess_exec(self.binary, *args, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
        try:
            out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise SandboxError(f"openshell {' '.join(args[:3])} took longer than {timeout:.0f}s")
        return ExecResult(proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace"))

    async def exists(self, name: str) -> bool:
        r = await self._run("sandbox", "get", name, "-o", "json", timeout=30)
        if r.exit_code == 0:
            return True
        if NOT_FOUND.search(r.stderr + r.stdout):
            return False
        raise SandboxError(f"sandbox get {name}: {r.stderr.strip()[:300]}")

    # @@@SNIPSTART policy-file
    def policy_file(self, kind: str, grants: list[Grant]) -> Path:
        """The base policy for this kind of sandbox plus the approved grants recorded in the Workflow (OpenShell policy v1).
        The exec sandbox never gets network rules; grants only ever extend the tools sandbox."""
        text = (POLICIES / f"{kind}.policy.yaml").read_text()
        approved = [g for g in grants if g.decision == "approved"] if kind == "tools" else []
        if not approved:
            return POLICIES / f"{kind}.policy.yaml"
        lines = [text.replace("network_policies: {}", "network_policies:").rstrip("\n"), "# --- grants recorded in the Temporal workflow, re-applied on every rebuild ---"]
        for g in approved:
            lines += [f"  {g.rule_name}:", f"    name: {g.rule_name}", "    endpoints:", f"      - host: {g.host}", f"        port: {g.port}",
                      "        protocol: rest", "        access: read-write", "        enforcement: enforce", "    binaries:", f"      - path: {self.python}"]
        fd, path = tempfile.mkstemp(prefix=f"{kind}-policy-", suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines) + "\n")
        return Path(path)
    # @@@SNIPEND

    async def ensure(self, name: str, kind: str, grants: list[Grant], heartbeat=lambda phase: None) -> bool:
        """Idempotent: get, create if missing, wait until ready. Returns whether this call created it."""
        created = False
        if not await self.exists(name):
            heartbeat("create")
            r = await self._run("sandbox", "create", "--name", name, "--from", self.image, "--policy", str(self.policy_file(kind, grants)),
                                "--no-auto-providers", "--detach", "-o", "json", "--", "sleep", "infinity", timeout=600)
            if r.exit_code != 0:
                raise SandboxError(f"sandbox create {name}: {r.stderr.strip()[:300]}")
            created = True
        deadline = time.monotonic() + 300
        while True:
            heartbeat("wait_ready")
            r = await self._run("sandbox", "get", name, "-o", "json", timeout=30)
            info = json.loads(r.stdout or "{}") if r.exit_code == 0 else {}
            phase = (info.get("status") or {}).get("phase") or info.get("phase")
            if phase in (None, "ready", "running", "Ready", "Running"):
                return created
            if phase in ("error", "failed", "Error", "Failed") or time.monotonic() > deadline:
                raise SandboxError(f"sandbox {name} is in phase {phase}")
            await asyncio.sleep(2)

    async def delete(self, name: str) -> None:
        r = await self._run("sandbox", "delete", name, timeout=60)
        if r.exit_code != 0 and not NOT_FOUND.search(r.stderr + r.stdout):
            raise SandboxError(f"sandbox delete {name}: {r.stderr.strip()[:300]}")
        for _ in range(60):  # delete is asynchronous
            if not await self.exists(name):
                return
            await asyncio.sleep(1)
        raise SandboxError(f"sandbox {name} still present 60s after delete")

    async def _gone(self, name: str, r: ExecResult) -> bool:
        """A failed call whose stderr says "not found" is a lost sandbox only if the sandbox really is gone: the stderr of the
        program running inside the sandbox (model-written code included) travels the same way and may say the same words."""
        return bool(NOT_FOUND.search(r.stderr)) and not await self.exists(name)

    async def exec(self, name: str, argv: Sequence[str], *, stdin: bytes | None = None, timeout: float = 120) -> ExecResult:
        r = await self._run("sandbox", "exec", "-n", name, "--timeout", str(int(timeout)), "--no-login-shell", "--", *argv, stdin=stdin, timeout=timeout + 15)
        if r.exit_code != 0 and await self._gone(name, r):
            raise SandboxLost(f"sandbox {name}: {r.stderr.strip()[:200]}")
        if r.exit_code == 124:
            raise SandboxError(f"command in {name} hit its {timeout:.0f}s timeout")
        return r

    async def apply(self, name: str, step: int, code: str, meta: dict[str, str], min_clearance_m: float) -> ExecResult:
        """`sceneops.apply` in the exec sandbox: the model's code on stdin, the step's layer in /sandbox/layers/."""
        argv = [self.python, "-I", "-m", "sceneops.apply", "--work", WORKDIR, "--site", SITE, "--min-clearance", str(min_clearance_m),
                "apply", "--step", str(step), *_meta_args(meta)]
        return await self.exec(name, argv, stdin=code.encode(), timeout=240)

    async def rehydrate(self, name: str, layers: list[tuple[int, str]]) -> ExecResult:
        """Recompose the uploaded layers and report the layer stack hash (checked against the Workflow's record)."""
        manifest = json.dumps({"layers": [{"step": s, "sha256": sha} for s, sha in layers]}).encode()
        return await self.exec(name, [self.python, "-I", "-m", "sceneops.apply", "--work", WORKDIR, "--site", SITE, "rehydrate"], stdin=manifest, timeout=300)

    async def search(self, name: str, host: str, query: str) -> dict[str, Any]:
        """`sceneops.search` in the tools sandbox. A denied egress comes back as a result ({"denied": true}), not an error."""
        r = await self.exec(name, [self.python, "-I", "-m", "sceneops.search", "--host", host, "--query", query], timeout=60)
        try:
            return json.loads(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise SandboxError(f"search in {name} produced no result: {r.stderr.strip()[:200]}")

    async def download_layer(self, name: str, step: int, dst: Path) -> None:
        r = await self._run("sandbox", "download", name, f"{WORKDIR}/layers/{step:03d}.usda", str(dst), timeout=120)
        if r.exit_code != 0:
            raise (SandboxLost if await self._gone(name, r) else SandboxError)(f"download {name} layer {step}: {r.stderr.strip()[:200]}")

    async def upload_layer(self, name: str, src: Path, step: int) -> None:
        await self.exec(name, [self.python, "-I", "-c", f"import os; os.makedirs('{WORKDIR}/layers', exist_ok=True)"], timeout=30)
        r = await self._run("sandbox", "upload", name, str(src), f"{WORKDIR}/layers/{step:03d}.usda", timeout=120)
        if r.exit_code != 0:
            raise (SandboxLost if await self._gone(name, r) else SandboxError)(f"upload {name} layer {step}: {r.stderr.strip()[:200]}")

    # @@@SNIPSTART apply-grant
    async def apply_grant(self, name: str, grant: Grant) -> None:
        """The documented live policy update: `openshell policy update <sandbox> --rule-name … --binary … --add-endpoint … --wait`.
        The sandbox logs `CONFIG:LOADED Policy reloaded`; re-adding the same endpoint is accepted, so retries are safe."""
        r = await self._run("policy", "update", name, "--rule-name", grant.rule_name, "--binary", self.python,
                            "--add-endpoint", f"{grant.host}:{grant.port}:read-write:rest:enforce", "--wait", timeout=120)
        if r.exit_code != 0:
            raise (SandboxLost if await self._gone(name, r) else SandboxError)(f"policy update {name}: {r.stderr.strip()[:300]}")
    # @@@SNIPEND

    async def logs(self, name: str) -> str:
        r = await self._run("logs", name, "--source", "sandbox", "-n", "20000", timeout=60)
        return r.stdout if r.exit_code == 0 else ""


# ---------------------------------------------------------------- the fake: directories, no isolation

# The fake runs the sceneops entrypoints with the host interpreter in isolated mode; the recipe directory goes on sys.path by hand.
_BOOTSTRAP = "import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module(sys.argv.pop(1), run_name='__main__', alter_sys=True)"


class FakeSandboxes:
    """Each sandbox is a directory under RUNS_DIR/_fake_sandboxes/<name> with a `policy.json` (granted hosts), a `work/`
    directory and a `gateway.log`. A search to a host that is not granted is denied; `apply_grant` allows it; `apply` runs
    `sceneops.apply` on the host. Use it to try the flow, never for containment."""

    def __init__(self) -> None:
        self.root = runs_dir() / "_fake_sandboxes"

    def _dir(self, name: str) -> Path:
        d = self.root / name
        if not d.exists():
            raise SandboxLost(f"sandbox {name} not found")
        return d

    def _log(self, name: str, line: str) -> None:
        with (self._dir(name) / "gateway.log").open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {line}\n")

    async def exists(self, name: str) -> bool:
        return (self.root / name).exists()

    async def ensure(self, name: str, kind: str, grants: list[Grant], heartbeat=lambda phase: None) -> bool:
        if await self.exists(name):
            return False
        heartbeat("create")
        d = self.root / name
        (d / "work" / "layers").mkdir(parents=True)
        hosts = [g.host for g in grants if g.decision == "approved"] if kind == "tools" else []
        (d / "policy.json").write_text(json.dumps({"kind": kind, "hosts": hosts}))
        self._log(name, f"CONFIG:LOADED policy for {kind} sandbox ({len(hosts)} granted host(s): {', '.join(hosts) or 'none'})")
        return True

    async def delete(self, name: str) -> None:
        shutil.rmtree(self.root / name, ignore_errors=True)

    async def _sceneops(self, name: str, module: str, argv: list[str], *, stdin: bytes | None, timeout: float = 120) -> ExecResult:
        work = self._dir(name) / "work"
        proc = await asyncio.create_subprocess_exec(sys.executable, "-I", "-c", _BOOTSTRAP, str(RECIPE_DIR), module, *argv, cwd=work,
                                                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise SandboxError(f"{module} hit its {timeout:.0f}s timeout")
        return ExecResult(proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace"))

    def apply_calls(self) -> int:
        """How many times `apply` ran on this fake (the tests check what a retry does)."""
        p = self.root / ".apply_calls"
        return int(p.read_text()) if p.exists() else 0

    async def apply(self, name: str, step: int, code: str, meta: dict[str, str], min_clearance_m: float) -> ExecResult:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / ".apply_calls").write_text(str(self.apply_calls() + 1))
        if os.environ.get("FAKE_APPLY_FAIL_ONCE") == "1" and self.apply_calls() == 1:
            raise SandboxError("fake: transient CLI failure on the first apply")  # chaos for the tests
        if os.environ.get("FAKE_LOSE_ONCE") == "exec" and not (self.root / ".lost-once").exists():
            # Chaos for the tests: the sandbox disappears under the first apply, as if an operator had deleted it.
            (self.root / ".lost-once").write_text(name)
            shutil.rmtree(self.root / name, ignore_errors=True)
            raise SandboxLost(f"sandbox {name} not found")
        work = self._dir(name) / "work"
        argv = ["--work", str(work), "--site", str(SITE_DIR), "--min-clearance", str(min_clearance_m), "apply", "--step", str(step), *_meta_args(meta)]
        return await self._sceneops(name, "sceneops.apply", argv, stdin=code.encode())

    async def rehydrate(self, name: str, layers: list[tuple[int, str]]) -> ExecResult:
        work = self._dir(name) / "work"
        manifest = json.dumps({"layers": [{"step": s, "sha256": sha} for s, sha in layers]}).encode()
        return await self._sceneops(name, "sceneops.apply", ["--work", str(work), "--site", str(SITE_DIR), "rehydrate"], stdin=manifest)

    async def search(self, name: str, host: str, query: str) -> dict[str, Any]:
        policy = json.loads((self._dir(name) / "policy.json").read_text())
        if host not in policy["hosts"]:
            self._log(name, f"NET:OPEN tcp {host}:443 DENIED policy_denied")
            return {"denied": True, "error": "[Errno 13] Permission denied"}
        self._log(name, f"NET:OPEN tcp {host}:443 ALLOWED")
        self._log(name, f"HTTP:POST ALLOWED https://{host}/search")
        hits = [{"url": S3_ROOT + a["usd"], "score": round(0.9 - 0.05 * i, 2)} for i, a in enumerate(catalog_search(load_catalog(CATALOG), query))]
        return {"denied": False, "hits": hits}

    async def download_layer(self, name: str, step: int, dst: Path) -> None:
        shutil.copyfile(self._dir(name) / "work" / "layers" / f"{step:03d}.usda", dst)

    async def upload_layer(self, name: str, src: Path, step: int) -> None:
        d = self._dir(name) / "work" / "layers"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, d / f"{step:03d}.usda")

    async def apply_grant(self, name: str, grant: Grant) -> None:
        d = self._dir(name)
        policy = json.loads((d / "policy.json").read_text())
        if grant.host not in policy["hosts"]:
            policy["hosts"].append(grant.host)
        (d / "policy.json").write_text(json.dumps(policy))
        self._log(name, f"CONFIG:LOADED Policy reloaded (rule {grant.rule_name}: {grant.host}:{grant.port})")

    async def logs(self, name: str) -> str:
        p = self.root / name / "gateway.log"
        return p.read_text() if p.exists() else ""
