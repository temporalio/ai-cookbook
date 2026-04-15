import pytest
from unittest.mock import patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from workflows.agent import StrandsAgentWorkflow


@pytest.fixture
async def env():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        yield env


class TestStrandsAgentWorkflow:

    @pytest.mark.asyncio
    async def test_returns_final_answer_directly(self, env):
        from temporalio import activity

        @activity.defn(name="agent_activity")
        async def mock_agent_activity(request):
            return {
                "tool_calls": [],
                "final_answer": "Hello! How can I help you?",
                "reasoning": "Simple greeting"
            }

        async with Worker(
            env.client,
            task_queue="test-queue",
            workflows=[StrandsAgentWorkflow],
            activities=[mock_agent_activity],
        ):
            result = await env.client.execute_workflow(
                StrandsAgentWorkflow.run,
                "Hello",
                id="test-direct-answer",
                task_queue="test-queue",
            )
            assert result == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_executes_tool_and_returns_answer(self, env):
        call_count = 0

        async def mock_agent_activity(request):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return {
                    "tool_calls": [{"tool_name": "get_time", "parameters": {}}],
                    "final_answer": None,
                    "reasoning": "Checking time"
                }
            else:
                return {
                    "tool_calls": [],
                    "final_answer": "The current time is 10:30 AM",
                    "reasoning": "Got the time"
                }

        async def mock_get_time():
            return "2024-01-15 10:30:00"

        from temporalio import activity

        @activity.defn(name="agent_activity")
        async def agent_activity_wrapper(request):
            return await mock_agent_activity(request)

        @activity.defn(name="get_time_activity")
        async def get_time_wrapper():
            return await mock_get_time()

        async with Worker(
            env.client,
            task_queue="test-queue",
            workflows=[StrandsAgentWorkflow],
            activities=[agent_activity_wrapper, get_time_wrapper],
        ):
            result = await env.client.execute_workflow(
                StrandsAgentWorkflow.run,
                "What time is it?",
                id="test-tool-call",
                task_queue="test-queue",
            )
            assert "10:30" in result
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_max_iterations_exceeded(self, env):
        async def mock_agent_activity_loop(request):
            return {
                "tool_calls": [{"tool_name": "get_time", "parameters": {}}],
                "final_answer": None,
                "reasoning": "Still checking"
            }

        async def mock_get_time():
            return "2024-01-15 10:30:00"

        from temporalio import activity

        @activity.defn(name="agent_activity")
        async def agent_activity_wrapper(request):
            return await mock_agent_activity_loop(request)

        @activity.defn(name="get_time_activity")
        async def get_time_wrapper():
            return await mock_get_time()

        async with Worker(
            env.client,
            task_queue="test-queue",
            workflows=[StrandsAgentWorkflow],
            activities=[agent_activity_wrapper, get_time_wrapper],
        ):
            result = await env.client.execute_workflow(
                StrandsAgentWorkflow.run,
                "Keep looping",
                id="test-max-iterations",
                task_queue="test-queue",
            )
            assert "exceeded maximum iterations" in result

    @pytest.mark.asyncio
    async def test_handles_unknown_tool(self, env):
        call_count = 0

        async def mock_agent_activity(request):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return {
                    "tool_calls": [{"tool_name": "unknown_tool", "parameters": {}}],
                    "final_answer": None,
                    "reasoning": "Trying unknown tool"
                }
            else:
                return {
                    "tool_calls": [],
                    "final_answer": "Done after unknown tool",
                    "reasoning": "Finished"
                }

        from temporalio import activity

        @activity.defn(name="agent_activity")
        async def agent_activity_wrapper(request):
            return await mock_agent_activity(request)

        async with Worker(
            env.client,
            task_queue="test-queue",
            workflows=[StrandsAgentWorkflow],
            activities=[agent_activity_wrapper],
        ):
            result = await env.client.execute_workflow(
                StrandsAgentWorkflow.run,
                "Use unknown tool",
                id="test-unknown-tool",
                task_queue="test-queue",
            )
            assert result == "Done after unknown tool"

    @pytest.mark.asyncio
    async def test_no_response_returns_failure_message(self, env):
        async def mock_agent_activity_empty(request):
            return {
                "tool_calls": [],
                "final_answer": None,
                "reasoning": None
            }

        from temporalio import activity

        @activity.defn(name="agent_activity")
        async def agent_activity_wrapper(request):
            return await mock_agent_activity_empty(request)

        async with Worker(
            env.client,
            task_queue="test-queue",
            workflows=[StrandsAgentWorkflow],
            activities=[agent_activity_wrapper],
        ):
            result = await env.client.execute_workflow(
                StrandsAgentWorkflow.run,
                "Get empty response",
                id="test-empty-response",
                task_queue="test-queue",
            )
            assert "failed to provide" in result
