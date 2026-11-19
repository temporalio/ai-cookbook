"""NeMo Relay: one agent scope per run per process, every Activity a tool span, a second process resuming the same trace
from the stored propagation context, and the model proxy's fallback recorded as a mark."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any

import pytest
from agents.models.interface import Model
from temporalio import activity
from temporalio.testing import ActivityEnvironment

import relay
from provider import RelayModel


def _env(workflow_id: str, activity_type: str, attempt: int = 1) -> ActivityEnvironment:
    env = ActivityEnvironment()
    env.info = dataclasses.replace(env.info, workflow_id=workflow_id, activity_type=activity_type, attempt=attempt)
    return env


@activity.defn
@relay.traced
async def ping(x: dict) -> dict:
    return {"ok": True, **x}


@activity.defn
@relay.traced
async def finish(reason: str) -> list[str]:
    return await relay.close_here(reason=reason)


async def test_activities_are_tool_spans_and_a_second_process_resumes_the_trace(fake_env):
    relay.configure(identity="w1")
    assert (await _env("wf-relay", "ping").run(ping, {"a": 1}))["ok"]
    assert (await _env("wf-relay", "ping", attempt=2).run(ping, {"a": 2}))["a"] == 2
    files = await _env("wf-relay", "finish").run(finish, "approved")
    d = fake_env / "wf-relay" / "relay"
    assert sorted(p.name for p in d.iterdir()) == sorted(p.rsplit("/", 1)[-1] for p in files)
    prop = json.loads((d / "propagation.json").read_text())
    atif = json.loads(next(d.glob("atif-w1-*.json")).read_text())
    results = [r for s in atif["steps"] for r in (s.get("observation") or {}).get("results", [])]
    assert [r["extra"]["event_name"] for r in results] == ["ping", "ping"] and results[1]["extra"]["tool_result"] == {"ok": True, "a": 2}
    events = [json.loads(line) for line in next(d.glob("events-w1-*.jsonl")).read_text().splitlines()]
    names = [e.get("name") for e in events]
    assert "temporal_retry" in names and names[-1] == "run:wf-relay" and "run_closed" in names
    retry = next(e for e in events if e.get("name") == "temporal_retry")
    assert retry["data"] == {"activity": "ping", "attempt": 2}
    spans = [e for e in events if e.get("name") == "ping" and e.get("kind") == "scope"]
    assert spans[2]["data"]["attempt"] == 2 and spans[2]["data"]["input"] == {"a": 2}, "the span start carries the Activity's attempt and input"

    # the gpu Worker, or the agent Worker after a kill: a new process continues the same trace
    relay.configure(identity="w2")
    await _env("wf-relay", "ping").run(ping, {"a": 3})
    await relay.drain()  # the trajectory is written in the background
    run = relay._runs["wf-relay"]
    assert run.resumed and run.root_uuid == prop["root_uuid"]
    assert len(list(d.glob("atif-*.json"))) == 2 and json.loads(next(d.glob("atif-w2-*.json")).read_text())["session_id"] == "wf-relay/w2"


class _FakeModel(Model):
    def __init__(self, name: str, fail: bool = False) -> None:
        self.name, self.fail, self.calls = name, fail, 0

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return type("Resp", (), {"usage": None, "output": [], "model": self.name})()

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


async def test_a_provider_fault_is_answered_by_the_fallback_and_recorded(fake_env, monkeypatch):
    monkeypatch.setenv("FAULT_MODEL_CALLS", "1")
    relay.configure(identity="w3")
    primary, fallback = _FakeModel("primary"), _FakeModel("fallback")
    model = RelayModel(primary, "nvidia/primary", fallback, "nvidia/fallback")

    @activity.defn
    @relay.traced
    async def model_turn(x: dict) -> str:
        resp = await model.get_response(system_instructions="s", input="hi", model_settings=None, tools=[], output_schema=None, handoffs=[], tracing=None,
                                        previous_response_id=None, conversation_id=None, prompt=None)
        return resp.model

    env = _env("wf-fallback", "invoke_model_activity")
    assert await env.run(model_turn, {}) == "fallback" and primary.calls == 0 and fallback.calls == 1
    assert await env.run(model_turn, {}) == "primary", "only the first call was faulted"
    await _env("wf-fallback", "finish").run(finish, "done")
    d = fake_env / "wf-fallback" / "relay"
    events = [json.loads(line) for line in next(d.glob("events-w3-*.jsonl")).read_text().splitlines()]
    mark = next(e for e in events if e.get("name") == "model_fallback")
    assert mark["data"]["primary"] == "nvidia/primary" and mark["data"]["fallback"] == "nvidia/fallback" and "InjectedFault" in mark["data"]["error"]
    atif = json.loads(next(d.glob("atif-w3-*.json")).read_text())
    llm = [s for s in atif["steps"] if s.get("source") == "agent"]
    assert [s["model_name"] for s in llm] == ["nvidia/fallback", "nvidia/primary"], "the trajectory names the model that answered"
    assert llm[0]["extra"]["llm_response"]["fallback"] is True and llm[0]["extra"]["llm_response"]["primary"] == "nvidia/primary"
    assert llm[1]["extra"]["llm_response"]["fallback"] is False


def _results(d, worker: str) -> list[dict]:
    atif = json.loads(next(d.glob(f"atif-{worker}-*.json")).read_text())
    return [res["extra"]["tool_result"] for s in atif["steps"] for res in (s.get("observation") or {}).get("results", [])]


async def test_two_runs_in_one_process_keep_separate_trajectories(fake_env):
    relay.configure(identity="w4")
    await _env("wf-A", "ping").run(ping, {"run": "A"})
    await _env("wf-B", "ping").run(ping, {"run": "B"})
    await _env("wf-A", "ping").run(ping, {"run": "A2"})
    await _env("wf-A", "finish").run(finish, "done")
    await _env("wf-B", "finish").run(finish, "done")
    assert [r["run"] for r in _results(fake_env / "wf-A" / "relay", "w4")] == ["A", "A2"]
    assert [r["run"] for r in _results(fake_env / "wf-B" / "relay", "w4")] == ["B"]


@pytest.mark.timeout(10, method="thread")
async def test_an_activity_does_not_wait_for_another_runs_model_call(fake_env):
    """Relay's exporter waits for every model call it has seen, other runs' included. Run B's call stays open until run
    A's Activity has returned, so an Activity that waited for its export would never return."""
    relay.configure(identity="concurrent")
    provider_started, complete_provider = asyncio.Event(), asyncio.Event()

    async def provider(headers, request):
        provider_started.set()
        await complete_provider.wait()
        return "response", {"output": "response"}

    @relay.traced
    async def tool():
        return {"run": "A"}

    model = asyncio.create_task(_env("wf-B", "model").run(relay.model_call, "test-model", {}, provider))
    await provider_started.wait()
    assert await asyncio.wait_for(_env("wf-A", "ping").run(tool), 2) == {"run": "A"}
    complete_provider.set()
    assert await model == "response"
    await relay.drain()
    assert _results(fake_env / "wf-A" / "relay", "concurrent") == [{"run": "A"}]
    assert _results(fake_env / "wf-B" / "relay", "concurrent") == []


async def test_truncated_propagation_is_replaced(fake_env):
    path = fake_env / "wf-damaged" / "relay" / "propagation.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"root_uuid":')
    assert await _env("wf-damaged", "ping").run(ping, {"n": 1}) == {"ok": True, "n": 1}
    assert json.loads(path.read_text())["root_uuid"] == relay._runs["wf-damaged"].root_uuid
    assert not relay._runs["wf-damaged"].resumed


async def test_relay_setup_failure_does_not_repeat_or_fail_the_call(fake_env, monkeypatch):
    def unavailable(*args):
        raise OSError("Relay storage unavailable")

    calls = []

    async def provider(headers, request):
        calls.append(request)
        return "response", {}

    monkeypatch.setattr(relay, "_event_writer", unavailable)
    assert await _env("wf-a", "ping").run(ping, {"n": 1}) == {"ok": True, "n": 1}
    assert await _env("wf-b", "model").run(relay.model_call, "test-model", {"n": 2}, provider) == "response"
    assert calls == [{"n": 2}]


async def test_an_idle_run_is_closed_and_its_next_activity_resumes_the_trace(fake_env, monkeypatch):
    monkeypatch.setenv("NEMO_RELAY_IDLE_CLOSE_S", "0")
    relay.configure(identity="w5")
    await _env("wf-idle", "ping").run(ping, {"n": 1})
    await _env("wf-other", "ping").run(ping, {"n": 2})  # closes wf-idle: nothing of it is running here
    assert "wf-idle" not in relay._runs
    await relay.drain()
    d = fake_env / "wf-idle" / "relay"
    events = [json.loads(line) for line in next(d.glob("events-w5-*.jsonl")).read_text().splitlines()]
    assert next(e for e in events if e.get("name") == "run_closed")["data"]["reason"] == "idle"
    await _env("wf-idle", "ping").run(ping, {"n": 3})
    await relay.drain()
    run = relay._runs["wf-idle"]
    assert run.resumed and run.root_uuid == json.loads((d / "propagation.json").read_text())["root_uuid"]
    assert len(list(d.glob("atif-w5-*.json"))) == 2, "the resumed scope writes its own trajectory next to the first"


async def test_the_idle_sweep_never_closes_a_run_that_is_working(fake_env, monkeypatch):
    monkeypatch.setenv("NEMO_RELAY_IDLE_CLOSE_S", "0")
    relay.configure(identity="w6")
    await _env("wf-busy", "ping").run(ping, {"n": 1})
    relay._runs["wf-busy"].in_flight = 1  # an Activity of wf-busy is running in this process
    await _env("wf-sweeper", "ping").run(ping, {"n": 2})
    assert "wf-busy" in relay._runs, "a run with an Activity in flight is not idle"
    relay._runs["wf-busy"].in_flight = 0
    await _env("wf-sweeper", "ping").run(ping, {"n": 3})
    assert "wf-busy" not in relay._runs
