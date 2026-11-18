"""The sandbox activities on the fake backend, with the ActivityEnvironment recording heartbeats."""

from __future__ import annotations

from temporalio.testing import ActivityEnvironment

from activities import sandbox as sb
from models import EnsureInput, ReleaseInput, SearchInput


async def test_activities_heartbeat_with_phases_and_release_deletes_both_sandboxes(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("SANDBOX_BACKEND", "fake")
    env = ActivityEnvironment()
    beats: list[dict] = []
    env.on_heartbeat = lambda *details: beats.append(dict(details[0])) if details and isinstance(details[0], dict) else None
    ref = await env.run(sb.ensure_sandboxes, EnsureInput(run_id="r1", workflow_id="wf-1", gen=1, grants=[], artifacts=[]))
    assert ref.gen == 1 and len(ref.created) == 2
    out = await env.run(sb.sandbox_search, SearchInput(run_id="r1", tools_sandbox=ref.tools_sandbox, host="search.example", query="pallet"))
    assert out.denied is True
    rel = await env.run(sb.release_sandboxes, ReleaseInput(run_id="r1", gen=1, exec_sandbox=ref.exec_sandbox, tools_sandbox=ref.tools_sandbox))
    assert sorted(rel.deleted) == sorted([ref.exec_sandbox, ref.tools_sandbox]) and (tmp_path / "r1" / "logs" / "g1-tools.log").exists()
    phases = {b["phase"] for b in beats}
    assert {"ensure:exec", "ensure:tools", "search", "logs:exec", "delete:tools"} <= phases
