"""Tests for start_workflow.py — runs the actual submission script end-to-end
against a scripted model provider, so a wiring mistake (e.g. a bad
WorkflowIDConflictPolicy member, or a client/plugin misconfiguration) fails
the way it would for a real user running `uv run start_workflow.py`."""

from datetime import timedelta

import pytest
from fakes import ScriptedModel, ScriptedModelProvider, text_response
from temporalio.client import Client
from temporalio.contrib.openai_agents import ModelActivityParameters, OpenAIAgentsPlugin
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

import start_workflow
from activities.tools import calculate_circle_area, get_weather
from workflows.hello_world_workflow import HelloWorldAgent

TASK_QUEUE = "hello-world-openai-agent-task-queue"


class TestStartWorkflowScript:
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_main_submits_workflow_and_prints_result(self, monkeypatch, capsys):
        plugin = OpenAIAgentsPlugin(
            model_params=ModelActivityParameters(start_to_close_timeout=timedelta(seconds=10)),
            model_provider=ScriptedModelProvider(ScriptedModel([text_response("Hello there!")])),
        )
        async with await WorkflowEnvironment.start_time_skipping(plugins=[plugin]) as env:

            async def fake_connect(*args, **kwargs):
                return env.client

            monkeypatch.setattr(Client, "connect", fake_connect)
            monkeypatch.setattr("builtins.input", lambda _: "Say hello")

            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[HelloWorldAgent],
                activities=[get_weather, calculate_circle_area],
            ):
                await start_workflow.main()

        assert "Result: Hello there!" in capsys.readouterr().out
