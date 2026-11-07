"""The fake sandbox backend reproduces the control flow the recipe depends on: a search to a host without a grant is
denied, a grant allows it, model-written code runs and leaves artifacts, a deleted sandbox is SandboxLost."""

from __future__ import annotations

import json

import pytest

from activities.openshell import FakeSandboxes, SandboxLost
from models import Grant


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("SANDBOX_BACKEND", "fake")
    return FakeSandboxes()


def _grant(host: str) -> Grant:
    return Grant(host=host, port=443, rule_name="usd_search", decision="approved", approver="test", at="2026-01-01T00:00:00", applied=False)


async def test_search_is_denied_until_granted_and_the_log_shows_both(fake):
    assert await fake.ensure("t-tools", "tools", []) is True
    assert await fake.ensure("t-tools", "tools", []) is False  # idempotent
    denied = await fake.search("t-tools", "search.example", "packing table")
    assert denied["denied"] is True and "results" not in denied
    await fake.apply_grant("t-tools", _grant("search.example"))
    allowed = await fake.search("t-tools", "search.example", "packing table")
    assert allowed["denied"] is False and [r["name"] for r in allowed["results"]] == ["packing_table"]
    log = await fake.logs("t-tools")
    assert "DENIED" in log and "Policy reloaded" in log and "ALLOWED" in log


async def test_a_rebuilt_tools_sandbox_starts_with_the_recorded_grants(fake):
    await fake.ensure("t-tools-g2", "tools", [_grant("search.example")])
    assert (await fake.search("t-tools-g2", "search.example", "pallet"))["denied"] is False
    await fake.ensure("t-exec-g2", "exec", [_grant("search.example")])  # the exec sandbox never gets network rules
    assert json.loads((fake.root / "t-exec-g2" / "policy.json").read_text())["hosts"] == []


async def test_run_code_leaves_artifacts_and_a_deleted_sandbox_is_lost(fake, tmp_path):
    await fake.ensure("t-exec", "exec", [])
    r = await fake.run_code("t-exec", "import json\njson.dump({'a': 1}, open('layout.json', 'w'))\nprint('ok')")
    assert r.exit_code == 0 and r.stdout.strip() == "ok"
    assert await fake.list_files("t-exec") == ["layout.json"]
    await fake.download("t-exec", "layout.json", tmp_path / "copy.json")
    assert json.loads((tmp_path / "copy.json").read_text()) == {"a": 1}
    await fake.delete("t-exec")
    assert await fake.exists("t-exec") is False
    with pytest.raises(SandboxLost):
        await fake.run_code("t-exec", "print(1)")
