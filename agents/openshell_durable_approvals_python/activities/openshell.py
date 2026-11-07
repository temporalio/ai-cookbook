"""Two sandbox backends behind one small interface.

`OpenShellCLI` drives NVIDIA OpenShell through its CLI: create a sandbox from a policy, run a command in it, update its
network policy while it runs, read its gateway log, delete it. `FakeSandboxes` reproduces only the control flow in plain
directories, with no isolation at all, so the recipe and its tests run on a laptop without OpenShell.

Pick one with SANDBOX_BACKEND=openshell|fake. The default is `openshell` when the `openshell` binary is on PATH and
`fake` otherwise.
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
from typing import Sequence

from models import Grant

NOT_FOUND = re.compile(r"not found|NOT_FOUND|no such sandbox", re.I)
POLICIES = Path(__file__).parent / "policies"
WORKDIR = "/sandbox"

# Runs inside the tools sandbox. OpenShell denies an egress at the socket, so the request fails with an OSError
# (Permission denied, or a connection error when the policy DNS answers): the script reports `denied`, it does not raise.
SEARCH_SCRIPT = r"""
import json, sys, urllib.request
host, query = sys.argv[1], sys.argv[2]
req = urllib.request.Request(f"https://{host}/search", data=json.dumps({"description": query, "limit": 5}).encode(),
                             headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=20) as r:
        print(json.dumps({"denied": False, "results": json.load(r)}))
except OSError as e:
    print(json.dumps({"denied": True, "error": str(e)[:200]}))
"""

CATALOG = [  # what the fake backend answers once the host is granted (a few SimReady-style warehouse assets)
    {"name": "packing_table", "url": "Industrial/Warehouse/Equipment/Packing_Table.usd", "size_m": [1.8, 0.9, 0.9]},
    {"name": "conveyor_belt", "url": "Industrial/Warehouse/Equipment/Conveyor_Belt_Short.usd", "size_m": [3.0, 0.8, 0.9]},
    {"name": "pallet", "url": "Props/Pallet/Pallet_Wood.usd", "size_m": [1.2, 0.8, 0.15]},
    {"name": "task_light", "url": "Industrial/Lighting/Task_Light.usd", "size_m": [0.4, 0.4, 2.2]},
    {"name": "forklift", "url": "Props/Forklift/Forklift.usd", "size_m": [2.6, 1.2, 2.1]},
]


class SandboxLost(Exception):
    """The sandbox no longer exists: somebody deleted it while the run was in progress."""


class SandboxError(Exception):
    pass


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str


def runs_dir() -> Path:
    return Path(os.environ.get("RUNS_DIR", "runs")).resolve()


def backend():
    choice = os.environ.get("SANDBOX_BACKEND") or ("openshell" if shutil.which(os.environ.get("OPENSHELL_BIN", "openshell")) else "fake")
    return OpenShellCLI() if choice == "openshell" else FakeSandboxes()


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- NVIDIA OpenShell, through its CLI


class OpenShellCLI:
    def __init__(self) -> None:
        self.binary = os.environ.get("OPENSHELL_BIN", "openshell")
        self.image = os.environ.get("SANDBOX_IMAGE", "python:3.12-slim")
        # Network rules match the REAL path of the connecting executable (symlinks do not count): the interpreter in python:3.12-slim.
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

    def policy_file(self, kind: str, grants: list[Grant]) -> Path:
        """The base policy for this kind of sandbox plus the approved grants recorded in the workflow (OpenShell policy v1).
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

    async def exec(self, name: str, argv: Sequence[str], *, stdin: bytes | None = None, timeout: float = 120) -> ExecResult:
        r = await self._run("sandbox", "exec", "-n", name, "--timeout", str(int(timeout)), "--no-login-shell", "--", *argv, stdin=stdin, timeout=timeout + 15)
        if r.exit_code != 0 and NOT_FOUND.search(r.stderr):
            raise SandboxLost(f"sandbox {name}: {r.stderr.strip()[:200]}")
        if r.exit_code == 124:
            raise SandboxError(f"command in {name} hit its {timeout:.0f}s timeout")
        return r

    async def search(self, name: str, host: str, query: str) -> dict:
        r = await self.exec(name, [self.python, "-I", "-c", SEARCH_SCRIPT, host, query], timeout=60)
        try:
            return json.loads(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise SandboxError(f"search in {name} produced no result: {r.stderr.strip()[:200]}")

    async def run_code(self, name: str, code: str) -> ExecResult:
        return await self.exec(name, [self.python, "-I", "-"], stdin=code.encode(), timeout=120)

    async def list_files(self, name: str) -> list[str]:
        r = await self.exec(name, [self.python, "-I", "-c", f"import json,os; print(json.dumps(sorted(f for f in os.listdir('{WORKDIR}') if os.path.isfile(os.path.join('{WORKDIR}', f)))))"], timeout=30)
        return json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else []

    async def download(self, name: str, filename: str, dst: Path) -> None:
        r = await self._run("sandbox", "download", name, f"{WORKDIR}/{filename}", str(dst), timeout=120)
        if r.exit_code != 0:
            raise (SandboxLost if NOT_FOUND.search(r.stderr) else SandboxError)(f"download {name}:{filename}: {r.stderr.strip()[:200]}")

    async def upload(self, name: str, src: Path, filename: str) -> None:
        r = await self._run("sandbox", "upload", name, str(src), f"{WORKDIR}/{filename}", timeout=120)
        if r.exit_code != 0:
            raise (SandboxLost if NOT_FOUND.search(r.stderr) else SandboxError)(f"upload {name}:{filename}: {r.stderr.strip()[:200]}")

    async def apply_grant(self, name: str, grant: Grant) -> None:
        """The documented live policy update: `openshell policy update <sandbox> --rule-name … --binary … --add-endpoint … --wait`.
        The sandbox logs `CONFIG:LOADED Policy reloaded`; re-adding the same endpoint is accepted, so retries are safe."""
        r = await self._run("policy", "update", name, "--rule-name", grant.rule_name, "--binary", self.python,
                            "--add-endpoint", f"{grant.host}:{grant.port}:read-write:rest:enforce", "--wait", timeout=120)
        if r.exit_code != 0:
            raise (SandboxLost if NOT_FOUND.search(r.stderr) else SandboxError)(f"policy update {name}: {r.stderr.strip()[:300]}")

    async def logs(self, name: str) -> str:
        r = await self._run("logs", name, "--source", "sandbox", "-n", "20000", timeout=60)
        return r.stdout if r.exit_code == 0 else ""


# ---------------------------------------------------------------- the fake: directories, no isolation


class FakeSandboxes:
    """Each sandbox is a directory under RUNS_DIR/_fake_sandboxes/<name> with a `policy.json` (granted hosts) and a
    `gateway.log`. A search to a host that is not granted is denied; `apply_grant` allows it; `run_code` runs Python in the
    directory with the host interpreter. Use it to try the flow, never for containment."""

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
        (d / "work").mkdir(parents=True)
        hosts = [g.host for g in grants if g.decision == "approved"] if kind == "tools" else []
        (d / "policy.json").write_text(json.dumps({"kind": kind, "hosts": hosts}))
        self._log(name, f"CONFIG:LOADED policy for {kind} sandbox ({len(hosts)} granted host(s): {', '.join(hosts) or 'none'})")
        return True

    async def delete(self, name: str) -> None:
        shutil.rmtree(self.root / name, ignore_errors=True)

    async def search(self, name: str, host: str, query: str) -> dict:
        policy = json.loads((self._dir(name) / "policy.json").read_text())
        if host not in policy["hosts"]:
            self._log(name, f"NET:OPEN tcp {host}:443 DENIED policy_denied")
            return {"denied": True, "error": "[Errno 13] Permission denied"}
        self._log(name, f"NET:OPEN tcp {host}:443 ALLOWED")
        words = {w for w in re.split(r"[\W_]+", query.lower()) if w}
        hits = [a for a in CATALOG if words & set(re.split(r"[\W_]+", a["name"]))]
        return {"denied": False, "results": hits or CATALOG}

    async def run_code(self, name: str, code: str) -> ExecResult:
        if os.environ.get("FAKE_LOSE_ONCE") == "exec" and not (self.root / ".lost-once").exists():
            # Chaos for the tests: the sandbox disappears under the first run_code, as if an operator had deleted it.
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / ".lost-once").write_text(name)
            shutil.rmtree(self.root / name, ignore_errors=True)
            raise SandboxLost(f"sandbox {name} not found")
        work = self._dir(name) / "work"
        proc = await asyncio.create_subprocess_exec(sys.executable, "-I", "-", cwd=work, stdin=asyncio.subprocess.PIPE,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(code.encode()), 60)
        except asyncio.TimeoutError:
            proc.kill()
            raise SandboxError("code hit its 60s timeout")
        return ExecResult(proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace"))

    async def list_files(self, name: str) -> list[str]:
        return sorted(p.name for p in (self._dir(name) / "work").iterdir() if p.is_file())

    async def download(self, name: str, filename: str, dst: Path) -> None:
        shutil.copyfile(self._dir(name) / "work" / filename, dst)

    async def upload(self, name: str, src: Path, filename: str) -> None:
        shutil.copyfile(src, self._dir(name) / "work" / filename)

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
