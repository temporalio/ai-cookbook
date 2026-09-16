"""Tests for the local ``activity_tool`` wrapper's failure handling.

``test_workflow.py`` mocks ``invoke_model`` with canned text responses that
never actually call the activity-backed tools, so the wrapper's headline
behavior — turning a failed activity into an ``ERROR:`` string for the LLM,
while still letting a genuine programming bug surface — has no coverage
elsewhere. These tests exercise ``activity_tool`` directly against small
throwaway activities, without involving ADK at all.
"""

from datetime import timedelta

import pytest
from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from workflows._activity_tool import activity_tool

TASK_QUEUE = "test-activity-tool-queue"


@activity.defn
async def _greet(name: str) -> str:
    return f"hello {name}"


@activity.defn
async def _always_fails() -> str:
    raise ApplicationError("boom", non_retryable=True)


_greet_tool = activity_tool(
    _greet,
    task_queue=TASK_QUEUE,
    start_to_close_timeout=timedelta(seconds=5),
)
_failing_tool = activity_tool(
    _always_fails,
    task_queue=TASK_QUEUE,
    start_to_close_timeout=timedelta(seconds=5),
    retry_policy=RetryPolicy(maximum_attempts=1),
)


@workflow.defn
class _ActivityToolProbeWorkflow:
    """Minimal workflow that calls a wrapped tool directly (no ADK)."""

    @workflow.run
    async def run(self, mode: str) -> str:
        if mode == "success":
            return await _greet_tool("world")
        return await _failing_tool()


async def _run_probe(mode: str, workflow_id: str) -> str:
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[_ActivityToolProbeWorkflow],
            activities=[_greet, _always_fails],
        ):
            return await env.client.execute_workflow(
                _ActivityToolProbeWorkflow.run,
                mode,
                id=workflow_id,
                task_queue=TASK_QUEUE,
            )


class TestActivityToolWrapper:
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_successful_call_returns_activity_result(self):
        result = await _run_probe("success", "test-activity-tool-success")
        assert result == "hello world"

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_activity_failure_becomes_error_string(self):
        result = await _run_probe("failure", "test-activity-tool-failure")
        assert result.startswith("ERROR: Tool _always_fails failed:")

    @pytest.mark.asyncio
    async def test_non_activity_error_propagates(self):
        """A bad call (argument-binding failure) is not swallowed as a tool error."""
        with pytest.raises(TypeError):
            await _greet_tool(unexpected_kwarg="oops")
