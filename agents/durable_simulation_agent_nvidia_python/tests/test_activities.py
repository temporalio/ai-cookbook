"""The sandbox Activities on the fake backend, with the ActivityEnvironment recording heartbeats."""

from __future__ import annotations

from temporalio.testing import ActivityEnvironment

from activities import sandbox as sb
from models import EnsureInput, ReleaseInput, SearchInput


async def test_activities_heartbeat_with_phases_and_release_deletes_both_sandboxes(fake_env):
    env = ActivityEnvironment()
    beats: list[dict] = []
    env.on_heartbeat = lambda *details: beats.append(dict(details[0])) if details and isinstance(details[0], dict) else None
    ref = await env.run(sb.ensure_sandboxes, EnsureInput(run_id="r1", workflow_id="wf-1", temporal_run_id="run-1", gen=1, grants=[], layers=[]))
    assert ref.gen == 1 and len(ref.created) == 2 and ref.exec_sandbox.endswith("-x1") and ref.tools_sandbox.endswith("-t1")
    out = await env.run(sb.sandbox_search, SearchInput(run_id="r1", tools_sandbox=ref.tools_sandbox, host="search.example", query="pallet"))
    assert out.denied is True and out.source == "catalog" and "pallet" in [r["id"] for r in out.results]
    rel = await env.run(sb.release_sandboxes, ReleaseInput(run_id="r1", gen=1, exec_sandbox=ref.exec_sandbox, tools_sandbox=ref.tools_sandbox))
    assert sorted(rel.deleted) == sorted([ref.exec_sandbox, ref.tools_sandbox]) and (fake_env / "r1" / "logs" / "g1-tools.log").exists()
    phases = {b["phase"] for b in beats}
    assert {"ensure:exec", "ensure:tools", "search", "logs:exec", "delete:tools"} <= phases


def _grant(host: str):
    from models import Grant

    return Grant(host=host, port=443, rule_name="usd_search", decision="approved", approver="t", at="2026-01-01T00:00:00", applied=True)


async def test_a_later_execution_retires_what_an_earlier_one_left_and_inherits_no_grant(fake_env):
    """An execution that timed out or was terminated cannot release its sandboxes. The next execution with the same
    Workflow Id gets its own names, retires the leftovers (keeping their logs), and starts without their grants."""
    env = ActivityEnvironment()
    first = await env.run(sb.ensure_sandboxes, EnsureInput(run_id="wf", workflow_id="wf", temporal_run_id="run-a", gen=1, grants=[_grant("search.example")], layers=[]))
    allowed = await env.run(sb.sandbox_search, SearchInput(run_id="wf", tools_sandbox=first.tools_sandbox, host="search.example", query="pallet"))
    assert allowed.denied is False
    second = await env.run(sb.ensure_sandboxes, EnsureInput(run_id="wf", workflow_id="wf", temporal_run_id="run-b", gen=1, grants=[], layers=[]))
    assert {second.exec_sandbox, second.tools_sandbox}.isdisjoint({first.exec_sandbox, first.tools_sandbox})
    assert sorted(second.retired) == sorted([first.exec_sandbox, first.tools_sandbox])
    assert "NET:OPEN tcp search.example:443 ALLOWED" in (fake_env / "wf" / "logs" / f"orphan-{first.tools_sandbox}.log").read_text()
    denied = await env.run(sb.sandbox_search, SearchInput(run_id="wf", tools_sandbox=second.tools_sandbox, host="search.example", query="pallet"))
    assert denied.denied is True, "the grant was the earlier execution's, not this one's"


async def test_release_deletes_both_sandboxes_even_when_a_log_or_a_delete_fails(fake_env, monkeypatch):
    from temporalio.exceptions import ApplicationError

    from activities.openshell import FakeSandboxes, SandboxError

    env = ActivityEnvironment()
    ref = await env.run(sb.ensure_sandboxes, EnsureInput(run_id="r2", workflow_id="wf-2", temporal_run_id="run-1", gen=1, grants=[], layers=[]))

    async def no_logs(self, name):
        raise SandboxError("logs timed out")

    monkeypatch.setattr(FakeSandboxes, "logs", no_logs)
    rel = await env.run(sb.release_sandboxes, ReleaseInput(run_id="r2", gen=1, exec_sandbox=ref.exec_sandbox, tools_sandbox=ref.tools_sandbox))
    assert sorted(rel.deleted) == sorted([ref.exec_sandbox, ref.tools_sandbox]) and rel.logs == []

    ref = await env.run(sb.ensure_sandboxes, EnsureInput(run_id="r2", workflow_id="wf-2", temporal_run_id="run-2", gen=1, grants=[], layers=[]))
    real_delete = FakeSandboxes.delete

    async def exec_delete_fails(self, name):
        if name == ref.exec_sandbox:
            raise SandboxError("delete timed out")
        await real_delete(self, name)

    monkeypatch.setattr(FakeSandboxes, "delete", exec_delete_fails)
    try:
        await env.run(sb.release_sandboxes, ReleaseInput(run_id="r2", gen=1, exec_sandbox=ref.exec_sandbox, tools_sandbox=ref.tools_sandbox))
        raise AssertionError("a failed delete must fail the Activity so Temporal retries it")
    except ApplicationError as e:
        assert e.type == "ReleaseFailed" and ref.exec_sandbox in e.message
    assert not await FakeSandboxes().exists(ref.tools_sandbox), "the other sandbox was still deleted"
