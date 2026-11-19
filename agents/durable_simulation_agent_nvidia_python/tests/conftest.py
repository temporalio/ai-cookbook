"""Shared fixtures: the fake sandbox backend in a temporary runs directory, NeMo Relay on with a per-test process identity."""

from __future__ import annotations

import json

import pytest
from temporalio.openai_agents.testing import ResponseBuilders, TestModel

import relay


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("SANDBOX_BACKEND", "fake")
    monkeypatch.setenv("NEMO_RELAY", "1")
    for var in ("FAKE_LOSE_ONCE", "FAKE_APPLY_FAIL_ONCE", "RENDERER_URL", "RENDERER_SHARED_DIR", "FAULT_MODEL_CALLS"):
        monkeypatch.delenv(var, raising=False)
    relay.configure(identity=f"test-{tmp_path.name[-6:]}")
    yield tmp_path
    relay.configure()


def scripted_model(script: list[tuple[str, dict]]) -> TestModel:
    """A model that answers each turn with the next scripted tool call."""
    counter = iter(range(1, 1000))

    def tool_call(name: str, args: dict):
        i = next(counter)
        return ResponseBuilders.tool_call(json.dumps(args), name, call_id=f"call_{i}", item_id=f"fc_{i}")

    return TestModel.returning_responses([tool_call(n, a) for n, a in script])
