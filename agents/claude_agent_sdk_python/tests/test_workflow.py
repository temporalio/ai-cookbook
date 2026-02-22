"""
Tests for AgentExecutionWorkflow

Uses Temporal's test environment to run the workflow with mocked activities.
No real Claude Agent SDK or API calls are made.

Usage:
    uv run pytest tests/test_workflow.py -v
"""

import uuid
from datetime import timedelta
from unittest.mock import AsyncMock, patch

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio import activity

from models import AgentInput, AgentOutput
from workflows.agent import AgentExecutionWorkflow


# ---------------------------------------------------------------------------
# Mock activities — replace real SDK calls with deterministic stubs
# ---------------------------------------------------------------------------

@activity.defn(name="execute_agent_activity")
async def mock_execute_agent_activity(input_data: AgentInput) -> AgentOutput:
    """Mock agent execution that returns a canned response."""
    return AgentOutput(
        status="success",
        response=f"Mock response to: {input_data.prompt}",
        total_tokens=150,
        num_events=5,
        processing_time_seconds=1.23,
    )


@activity.defn(name="execute_agent_activity")
async def mock_execute_agent_activity_error(input_data: AgentInput) -> AgentOutput:
    """Mock agent execution that returns an error."""
    return AgentOutput(
        status="error",
        response="",
        error_message="Model overloaded, please try again",
        processing_time_seconds=0.5,
    )


@activity.defn(name="log_result_activity")
async def mock_log_result_activity(output: AgentOutput) -> None:
    """Mock logging activity — no-op."""
    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_workflow_success():
    """Test that a successful agent execution flows through both steps."""
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env:
        async with Worker(
            env.client,
            task_queue="test-queue",
            workflows=[AgentExecutionWorkflow],
            activities=[mock_execute_agent_activity, mock_log_result_activity],
        ):
            input_data = AgentInput(prompt="What is 2 + 2?")

            result = await env.client.execute_workflow(
                AgentExecutionWorkflow.run,
                input_data,
                id=f"test-{uuid.uuid4()}",
                task_queue="test-queue",
            )

            assert result.status == "success"
            assert "Mock response" in result.response
            assert result.total_tokens == 150
            assert result.num_events == 5
            assert result.processing_time_seconds == 1.23


@pytest.mark.asyncio
async def test_workflow_error_response():
    """Test that an agent error is returned (not raised) by the workflow."""
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env:
        async with Worker(
            env.client,
            task_queue="test-queue",
            workflows=[AgentExecutionWorkflow],
            activities=[mock_execute_agent_activity_error, mock_log_result_activity],
        ):
            input_data = AgentInput(prompt="This will fail")

            result = await env.client.execute_workflow(
                AgentExecutionWorkflow.run,
                input_data,
                id=f"test-{uuid.uuid4()}",
                task_queue="test-queue",
            )

            # Errors are modeled as completions, not exceptions
            assert result.status == "error"
            assert result.error_message == "Model overloaded, please try again"
            assert result.response == ""


@pytest.mark.asyncio
async def test_workflow_with_system_prompt():
    """Test that system_prompt is passed through to the activity."""
    captured_inputs = []

    @activity.defn(name="execute_agent_activity")
    async def capture_input_activity(input_data: AgentInput) -> AgentOutput:
        captured_inputs.append(input_data)
        return AgentOutput(
            status="success",
            response="Response with system prompt",
            processing_time_seconds=0.5,
        )

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env:
        async with Worker(
            env.client,
            task_queue="test-queue",
            workflows=[AgentExecutionWorkflow],
            activities=[capture_input_activity, mock_log_result_activity],
        ):
            input_data = AgentInput(
                prompt="Hello",
                system_prompt="You are a helpful assistant.",
                model="claude-sonnet-4-5-20250929",
            )

            result = await env.client.execute_workflow(
                AgentExecutionWorkflow.run,
                input_data,
                id=f"test-{uuid.uuid4()}",
                task_queue="test-queue",
            )

            assert result.status == "success"
            assert len(captured_inputs) == 1
            assert captured_inputs[0].system_prompt == "You are a helpful assistant."
            assert captured_inputs[0].model == "claude-sonnet-4-5-20250929"
