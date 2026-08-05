"""Tests for ToolCallingWorkflow with mocked activities."""

import json

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from datetime import timedelta
from dataclasses import dataclass
from typing import Any

from openai.types.responses import (
    Response,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseFunctionToolCall,
)

from activities.get_weather_alerts import GetWeatherAlertsRequest
from workflows.get_weather_workflow import ToolCallingWorkflow

TASK_QUEUE = "test-tool-calling"


@dataclass
class OpenAIResponsesRequest:
    model: str
    instructions: str
    input: list
    tools: list


def _make_message_response(text: str) -> Response:
    return Response(
        id="resp_test",
        created_at=1700000000,
        model="gpt-4o-mini",
        object="response",
        output=[ResponseOutputMessage(
            id="msg_test",
            type="message",
            role="assistant",
            status="completed",
            content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
        )],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_function_call_response(name: str, call_id: str, arguments: str) -> Response:
    return Response(
        id="resp_test",
        created_at=1700000000,
        model="gpt-4o-mini",
        object="response",
        output=[ResponseFunctionToolCall(
            id="fc_test",
            type="function_call",
            name=name,
            call_id=call_id,
            arguments=arguments,
            status="completed",
        )],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


_create_call_count = 0


@activity.defn(name="create")
async def mock_create_no_tool(request: OpenAIResponsesRequest) -> Response:
    return _make_message_response("Code calls itself\nBase case breaks the loop at last\nStack unwinds with grace")


@activity.defn(name="create")
async def mock_create_with_weather(request: OpenAIResponsesRequest) -> Response:
    global _create_call_count
    _create_call_count += 1
    if _create_call_count == 1:
        return _make_function_call_response(
            "get_weather_alerts",
            "call_weather_123",
            json.dumps({"state": "CA"}),
        )
    return _make_message_response("There is a Flood Watch in California.")


@activity.defn(name="get_weather_alerts")
async def mock_get_weather(request: GetWeatherAlertsRequest) -> str:
    return json.dumps({"type": "FeatureCollection", "features": [{"properties": {"event": "Flood Watch"}}]})


class TestToolCallingWorkflowNoTool:
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_returns_text_when_no_tool_needed(self):
        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter,
        ) as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[ToolCallingWorkflow],
                activities=[mock_create_no_tool, mock_get_weather],
            ):
                result = await env.client.execute_workflow(
                    ToolCallingWorkflow.run,
                    "Tell me about recursion",
                    id="test-no-tool",
                    task_queue=TASK_QUEUE,
                    run_timeout=timedelta(seconds=30),
                )

        assert "Code calls itself" in result


class TestToolCallingWorkflowWithWeather:
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_handles_weather_tool_call(self):
        global _create_call_count
        _create_call_count = 0

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter,
        ) as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[ToolCallingWorkflow],
                activities=[mock_create_with_weather, mock_get_weather],
            ):
                result = await env.client.execute_workflow(
                    ToolCallingWorkflow.run,
                    "Are there weather alerts in California?",
                    id="test-weather-tool",
                    task_queue=TASK_QUEUE,
                    run_timeout=timedelta(seconds=30),
                )

        assert "Flood Watch" in result
