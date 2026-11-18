"""NeMo Relay: one agent scope per run per process, every Activity a tool span, a second process resuming the same trace
from the stored propagation context, and the model proxy's fallback recorded as a mark."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

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
