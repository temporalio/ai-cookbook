import asyncio

import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities.llm_call import CallLlmRequest
from activities.tools import GetTimeRequest, GetWeatherRequest
from workflows.recipe_workflow import ParallelToolAgentWorkflow

TASK_QUEUE = "parallel-tool-calls-task-queue"


def _two_tool_turn() -> Message:
    return Message(
        id="msg_tools",
        type="message",
        role="assistant",
        model="claude-sonnet-4-6",
        content=[
            ToolUseBlock(type="tool_use", id="toolu_w", name="get_weather", input={"city": "Seattle"}),
            ToolUseBlock(
                type="tool_use",
                id="toolu_t",
                name="get_time",
                input={"timezone": "America/Los_Angeles"},
            ),
        ],
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=20),
    )


def _final_turn() -> Message:
    return Message(
        id="msg_final",
        type="message",
        role="assistant",
        model="claude-sonnet-4-6",
        content=[TextBlock(type="text", text="Seattle is rainy and it is 09:00.")],
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=20),
    )


# State the test inspects: every call_llm request the workflow sent, and a record of
# which tool activities were running at the same instant.
_llm_requests: list[CallLlmRequest] = []
_concurrent_peak = 0


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    _llm_requests.clear()
    global _concurrent_peak
    _concurrent_peak = 0


@activity.defn(name="call_llm")
async def mock_call_llm(request: CallLlmRequest) -> Message:
    _llm_requests.append(request)
    # Turn one: ask for two tools. Turn two: the workflow has sent tool results back,
    # so return the final answer.
    if len(_llm_requests) == 1:
        return _two_tool_turn()
    return _final_turn()


# Wrap the real demo tools so the test can observe overlap. Each tool holds a small
# barrier: if the two run concurrently both arrive before either proceeds, driving the
# observed peak to 2. Serial execution would never exceed 1.
_running = 0
_both_running = asyncio.Event()


@activity.defn(name="get_weather")
async def spy_get_weather(request: GetWeatherRequest) -> str:
    return await _tracked("rainy, 54F")


@activity.defn(name="get_time")
async def spy_get_time(request: GetTimeRequest) -> str:
    return await _tracked("09:00")


async def _tracked(result: str) -> str:
    global _running, _concurrent_peak
    _running += 1
    _concurrent_peak = max(_concurrent_peak, _running)
    if _running >= 2:
        _both_running.set()
    try:
        # Wait until both tools are in-flight (or give up quickly if running serially).
        await asyncio.wait_for(_both_running.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        pass
    finally:
        _running -= 1
    return result


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_parallel_tools_run_concurrently_and_both_return() -> None:
    _both_running.clear()
    async with (
        await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env,
        Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[ParallelToolAgentWorkflow],
            activities=[mock_call_llm, spy_get_weather, spy_get_time],
        ),
    ):
        result = await env.client.execute_workflow(
            ParallelToolAgentWorkflow.run,
            "weather in Seattle and time in LA?",
            id="test-parallel-tool-calls",
            task_queue=TASK_QUEUE,
        )

    # The model produced its final answer.
    assert "09:00" in result

    # Both tools were in-flight at the same time: the fan-out is concurrent, not serial.
    assert _concurrent_peak == 2

    # The workflow looped exactly twice: tool turn, then final turn.
    assert len(_llm_requests) == 2

    # Turn two carried exactly one tool_result per requested tool, order preserved.
    second_turn_content = _llm_requests[1].messages[-1]["content"]
    assert [r["tool_use_id"] for r in second_turn_content] == ["toolu_w", "toolu_t"]
    assert second_turn_content[0]["content"] == "rainy, 54F"
    assert second_turn_content[1]["content"] == "09:00"
