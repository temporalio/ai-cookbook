import asyncio

import pytest
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities.llm_call import CallLlmRequest
from workflows.agent_steering_workflow import SteerableAgentWorkflow

TASK_QUEUE = "agent-steering-task-queue"

# Every call_llm request the mock receives, in order, so tests can inspect context.
_captured_requests: list[CallLlmRequest] = []


@pytest.fixture(autouse=True)
def _clear_captured() -> None:
    _captured_requests.clear()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_steer_guidance_reaches_next_call() -> None:
    """A steer signal sent during a turn shows up in the next call's messages."""
    first_call_started = asyncio.Event()
    release_first_call = asyncio.Event()

    @activity.defn(name="call_llm")
    async def mock_call_llm(request: CallLlmRequest) -> str:
        _captured_requests.append(request)
        contents = [m.content for m in request.messages]
        if "Prefer the safer remediation plan." not in contents:
            # Hold the first turn open until the test has sent its steer signal,
            # so the guidance is queued before this turn finishes.
            first_call_started.set()
            await release_first_call.wait()
        return f"reply to {len(request.messages)} messages"

    async with (
        await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env,
        Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[SteerableAgentWorkflow],
            activities=[mock_call_llm],
        ),
    ):
        handle = await env.client.start_workflow(
            SteerableAgentWorkflow.run,
            "Plan the migration.",
            id="test-steer",
            task_queue=TASK_QUEUE,
        )

        # Steer while the first turn is still in flight, then let it finish.
        await asyncio.wait_for(first_call_started.wait(), timeout=10)
        await handle.signal(SteerableAgentWorkflow.steer, "Prefer the safer remediation plan.")
        release_first_call.set()

        await handle.result()

    # The first call saw only the original prompt; the second saw the guidance.
    assert len(_captured_requests) >= 2
    first_contents = [m.content for m in _captured_requests[0].messages]
    second_contents = [m.content for m in _captured_requests[1].messages]
    assert "Prefer the safer remediation plan." not in first_contents
    assert "Prefer the safer remediation plan." in second_contents


# Records whether the first (interrupted) call actually received CancelledError.
_first_call_was_cancelled: list[bool] = []


@pytest.fixture(autouse=True)
def _clear_cancelled_flag() -> None:
    _first_call_was_cancelled.clear()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_interrupt_cancels_call_and_replaces_prompt() -> None:
    """An interrupt genuinely cancels the in-flight call, drops its reply, swaps the prompt."""
    first_call_started = asyncio.Event()

    @activity.defn(name="call_llm")
    async def mock_call_llm(request: CallLlmRequest) -> str:
        _captured_requests.append(request)
        contents = [m.content for m in request.messages]
        if "Check the customer's latest order first." not in contents:
            # The first turn stands in for an in-flight model call: heartbeat so the
            # worker polls for cancellation, and await so CancelledError can be raised
            # into us. A real call_llm heartbeats and awaits messages.create the same way.
            first_call_started.set()
            try:
                while True:
                    activity.heartbeat()
                    await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                # Cancellation was delivered and interrupted the await: the model call
                # was genuinely cancelled, not left running. Re-raise so Temporal marks
                # the Activity cancelled rather than completed.
                _first_call_was_cancelled.append(True)
                raise
        return "reply after interrupt"

    async with (
        await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env,
        Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[SteerableAgentWorkflow],
            activities=[mock_call_llm],
        ),
    ):
        handle = await env.client.start_workflow(
            SteerableAgentWorkflow.run,
            "Start down the risky path.",
            id="test-interrupt",
            task_queue=TASK_QUEUE,
        )

        await asyncio.wait_for(first_call_started.wait(), timeout=10)
        await handle.signal(
            SteerableAgentWorkflow.interrupt,
            "Check the customer's latest order first.",
        )

        result = await handle.result()

    # The in-flight call was genuinely cancelled mid-await, not just abandoned.
    assert _first_call_was_cancelled == [True]
    # The replacement prompt drove the next call, and no partial reply was kept.
    assert result == "reply after interrupt"
    last_contents = [m.content for m in _captured_requests[-1].messages]
    assert "Check the customer's latest order first." in last_contents
    assert all(m.content != "reply after interrupt" for m in _captured_requests[-1].messages)
