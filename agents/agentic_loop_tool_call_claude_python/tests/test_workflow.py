"""Tests for AgentWorkflow with mocked Claude activities."""

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio import activity
from temporalio.common import RawValue
from temporalio.contrib.pydantic import pydantic_data_converter
from datetime import timedelta
from typing import Sequence
from dataclasses import dataclass

from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from workflows.agent import AgentWorkflow

TASK_QUEUE = "test-claude-agent"


@dataclass
class ClaudeResponsesRequest:
    model: str
    system: str
    messages: list
    tools: list
    max_tokens: int = 4096


def _make_text_message(text: str) -> Message:
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-sonnet-4-20250514",
        content=[TextBlock(type="text", text=text)],
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=20, cache_creation_input_tokens=0, cache_read_input_tokens=0),
    )


def _make_tool_use_message(tool_name: str, tool_input: dict) -> Message:
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-sonnet-4-20250514",
        content=[ToolUseBlock(type="tool_use", id="toolu_123", name=tool_name, input=tool_input)],
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=20, cache_creation_input_tokens=0, cache_read_input_tokens=0),
    )


_create_call_count = 0


@activity.defn(name="create")
async def mock_create_text_only(request: ClaudeResponsesRequest) -> Message:
    return _make_text_message("A haiku response\nAbout recursion and code\nStack frames come and go")


@activity.defn(name="create")
async def mock_create_with_tool(request: ClaudeResponsesRequest) -> Message:
    global _create_call_count
    _create_call_count += 1
    if _create_call_count == 1:
        return _make_tool_use_message("get_ip_address", {})
    return _make_text_message("Your IP is 1.2.3.4")


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
                    id="test-claude-text-only",
                    task_queue=TASK_QUEUE,
                    run_timeout=timedelta(seconds=30),
                )

        assert "haiku" in result.lower() or "recursion" in result.lower()


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
                    id="test-claude-tool-call",
                    task_queue=TASK_QUEUE,
                    run_timeout=timedelta(seconds=30),
                )

        assert "1.2.3.4" in result
