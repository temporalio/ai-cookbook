"""The OpenShell CLI adapter against a stub `openshell` binary: "not found" in a program's own stderr is not a lost
sandbox; a sandbox that is really gone is."""

from __future__ import annotations

import stat
import sys

import pytest

from activities.openshell import OpenShellCLI, SandboxLost

STUB = r"""#!/usr/bin/env bash
# A stub openshell: `sandbox get` answers from a state directory; `sandbox exec` runs the command on the host.
set -u
STATE="$OPENSHELL_STUB_STATE"
case "$1 $2" in
  "sandbox get")
    name="$3"
    if [ -e "$STATE/$name.deleted" ]; then echo "Error: sandbox $name not found" >&2; exit 1; fi
    echo "{\"name\": \"$name\", \"status\": {\"phase\": \"ready\"}}"; exit 0;;
  "sandbox exec")
    shift 2
    name=""
    while [ $# -gt 0 ]; do
      case "$1" in -n) name="$2"; shift 2;; --timeout) shift 2;; --no-login-shell) shift;; --) shift; break;; *) shift;; esac
    done
    if [ -e "$STATE/$name.deleted" ]; then echo "Error: sandbox $name not found" >&2; exit 1; fi
    exec "$@";;
  *) echo "stub: unsupported: $*" >&2; exit 2;;
esac
"""


@pytest.fixture
def cli(tmp_path, monkeypatch):
    stub = tmp_path / "openshell"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("OPENSHELL_BIN", str(stub))
    monkeypatch.setenv("OPENSHELL_STUB_STATE", str(state))
    monkeypatch.setenv("SANDBOX_PYTHON", sys.executable)
    return OpenShellCLI(), state


async def test_a_program_that_prints_not_found_is_not_a_lost_sandbox(cli):
    osh, _ = cli
    r = await osh.exec("live-sandbox", [sys.executable, "-c", "import sys; print('asset pallet_xl not found in catalog', file=sys.stderr); sys.exit(1)"])
    assert r.exit_code == 1 and "not found" in r.stderr and r.envelope is None


async def test_a_deleted_sandbox_is_lost(cli):
    osh, state = cli
    (state / "gone-sandbox.deleted").write_text("")
    with pytest.raises(SandboxLost):
        await osh.search("gone-sandbox", "search.example", "pallet")
    assert await osh.exists("gone-sandbox") is False and await osh.exists("live-sandbox") is True


def test_the_apply_command_and_the_grant_policy(cli):
    osh, _ = cli
    from models import Grant

    g = Grant(host="search.example", port=443, rule_name="usd_search", decision="approved", approver="t", at="now", applied=False)
    text = osh.policy_file("tools", [g]).read_text()
    assert "usd_search:" in text and "host: search.example" in text and f"path: {sys.executable}" in text
    assert osh.policy_file("exec", [g]).name == "exec.policy.yaml", "the exec sandbox never gets network rules"
