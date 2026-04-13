"""Tests for AgentWorkflow with mocked OpenAI activities."""

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio import activity
from temporalio.common import RawValue
from temporalio.contrib.pydantic import pydantic_data_converter
from datetime import timedelta
from typing import Sequence
from dataclasses import dataclass

from openai.types.responses import (
    Response,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseFunctionToolCall,
)

from workflows.agent import AgentWorkflow

TASK_QUEUE = "test-openai-agent"


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
async def mock_create_text_only(request: OpenAIResponsesRequest) -> Response:
    return _make_message_response("Recursion calls itself\nUntil the base case is met\nStack frames come and go")


@activity.defn(name="create")
async def mock_create_with_tool(request: OpenAIResponsesRequest) -> Response:
    global _create_call_count
    _create_call_count += 1
    if _create_call_count == 1:
        return _make_function_call_response("get_ip_address", "call_abc", "{}")
    return _make_message_response("Your IP is 1.2.3.4")


@activity.defn(dynamic=True)
async def mock_dynamic_tool(args: Sequence[RawValue]) -> str:
    return "1.2.3.4"


class TestAgentWorkflowTextOnly:
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_returns_text_when_no_tools_needed(self):
        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter,
        ) as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[AgentWorkflow],
                activities=[mock_create_text_only, mock_dynamic_tool],
            ):
                result = await env.client.execute_workflow(
                    AgentWorkflow.run,
                    "Tell me about recursion",
                    id="test-openai-text-only",
                    task_queue=TASK_QUEUE,
                    run_timeout=timedelta(seconds=30),
                )

        assert "Recursion" in result


class TestAgentWorkflowWithToolCall:
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_handles_tool_call_and_returns_final_text(self):
        global _create_call_count
        _create_call_count = 0

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter,
        ) as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[AgentWorkflow],
                activities=[mock_create_with_tool, mock_dynamic_tool],
            ):
                result = await env.client.execute_workflow(
                    AgentWorkflow.run,
                    "What is my IP address?",
                    id="test-openai-tool-call",
                    task_queue=TASK_QUEUE,
                    run_timeout=timedelta(seconds=30),
                )

        assert "1.2.3.4" in result
