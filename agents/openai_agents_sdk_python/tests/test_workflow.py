"""Tests for HelloWorldAgent with a scripted model provider — no real OpenAI calls."""

import json
from datetime import timedelta

import pytest
from temporalio.contrib.openai_agents import ModelActivityParameters, OpenAIAgentsPlugin
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities.tools import calculate_circle_area, get_weather
from fakes import ScriptedModel, ScriptedModelProvider, text_response, tool_call_response
from workflows.hello_world_workflow import HelloWorldAgent

TASK_QUEUE = "test-hello-world-agent"


async def _run_workflow(prompt: str, responses, workflow_id: str) -> str:
    plugin = OpenAIAgentsPlugin(
        model_params=ModelActivityParameters(start_to_close_timeout=timedelta(seconds=10)),
        model_provider=ScriptedModelProvider(ScriptedModel(responses)),
    )
    async with await WorkflowEnvironment.start_time_skipping(plugins=[plugin]) as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[HelloWorldAgent],
            activities=[get_weather, calculate_circle_area],
        ):
            return await env.client.execute_workflow(
                HelloWorldAgent.run,
                prompt,
                id=workflow_id,
                task_queue=TASK_QUEUE,
                run_timeout=timedelta(seconds=30),
            )


class TestHelloWorldAgentTextOnly:
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_returns_text_when_no_tool_needed(self):
        result = await _run_workflow(
            "Say hello",
            [text_response("Hello there!")],
            "test-text-only",
        )
        assert result == "Hello there!"


class TestHelloWorldAgentWeatherTool:
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_uses_get_weather_tool(self):
        result = await _run_workflow(
            "What's the weather in Tokyo?",
            [
                tool_call_response("get_weather", "call_1", json.dumps({"city": "Tokyo"})),
                text_response("It's sunny in Tokyo, 14-20C."),
            ],
            "test-weather-tool",
        )
        assert "Tokyo" in result


class TestHelloWorldAgentCircleAreaTool:
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_uses_calculate_circle_area_tool(self):
        result = await _run_workflow(
            "Calculate the area of a circle with radius 5",
            [
                tool_call_response("calculate_circle_area", "call_2", json.dumps({"radius": 5})),
                text_response("The area is approximately 78.54 square units."),
            ],
            "test-circle-area-tool",
        )
        assert "78.54" in result
