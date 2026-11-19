"""The fake sandbox backend reproduces the control flow the recipe depends on: a search to a host without a grant is
denied, a grant allows it, the sceneops entrypoints run on the host and leave a layer, a deleted sandbox is SandboxLost."""

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
    assert denied["denied"] is True and "hits" not in denied
    await fake.apply_grant("t-tools", _grant("search.example"))
    allowed = await fake.search("t-tools", "search.example", "packing table")
    assert allowed["denied"] is False and allowed["hits"][0]["url"].endswith("packing_table.usd")
    log = await fake.logs("t-tools")
    assert log.index("DENIED") < log.index("Policy reloaded") < log.index("ALLOWED")


async def test_a_rebuilt_tools_sandbox_starts_with_the_recorded_grants(fake):
    await fake.ensure("t-tools-g2", "tools", [_grant("search.example")])
    assert (await fake.search("t-tools-g2", "search.example", "pallet"))["denied"] is False
    await fake.ensure("t-exec-g2", "exec", [_grant("search.example")])  # the exec sandbox never gets network rules
    assert json.loads((fake.root / "t-exec-g2" / "policy.json").read_text())["hosts"] == []


async def test_apply_leaves_a_layer_and_a_deleted_sandbox_is_lost(fake, tmp_path):
    await fake.ensure("t-exec", "exec", [])
    r = await fake.apply("t-exec", 1, "h.place_asset('pallet', -2.0, -2.0, name='p1')\nprint('ok')\n", {"model": "test"}, 0.6)
    assert r.exit_code == 0 and r.envelope["ok"] and r.envelope["data"]["placed_assets"][0]["catalog_id"] == "pallet"
    await fake.download_layer("t-exec", 1, tmp_path / "copy.usda")
    assert (tmp_path / "copy.usda").read_text().startswith("#usda 1.0")
    await fake.delete("t-exec")
    assert await fake.exists("t-exec") is False
    with pytest.raises(SandboxLost):
        await fake.apply("t-exec", 2, "print(1)", {}, 0.6)
