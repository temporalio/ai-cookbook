import asyncio
import uuid

import pytest
from temporalio import activity
from temporalio.client import WorkflowExecutionStatus, WorkflowFailureError
from temporalio.exceptions import CancelledError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities import agent
from models import RunRequest, RunResult
from workflows.agent import ClaudeRun


@activity.defn(name="run_claude")
async def fake_run_claude(request: RunRequest) -> RunResult:
    if request.effect_delay_seconds:
        activity.heartbeat()
        await activity.wait_for_cancelled()
        raise asyncio.CancelledError
    return RunResult(
        response=request.prompt,
        session_id="session",
        activity_attempt=activity.info().attempt,
    )


@pytest.mark.asyncio
async def test_detached_result_and_cancel() -> None:
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-queue",
            workflows=[ClaudeRun],
            activities=[fake_run_claude],
        ):
            workflow_id = f"test-{uuid.uuid4()}"
            await env.client.start_workflow(
                ClaudeRun.run,
                RunRequest(prompt="done", effect="effect"),
                id=workflow_id,
                task_queue="test-queue",
            )
            handle = env.client.get_workflow_handle_for(ClaudeRun.run, workflow_id)
            assert (await handle.result()).response == "done"

            cancel_id = f"test-{uuid.uuid4()}"
            cancel_handle = await env.client.start_workflow(
                ClaudeRun.run,
                RunRequest(
                    prompt="cancel",
                    effect="effect",
                    effect_delay_seconds=1,
                ),
                id=cancel_id,
                task_queue="test-queue",
            )
            assert (await cancel_handle.describe()).status == WorkflowExecutionStatus.RUNNING
            await cancel_handle.cancel()
            with pytest.raises(WorkflowFailureError) as error:
                await cancel_handle.result()
            assert isinstance(error.value.cause, CancelledError)


def test_effect_is_idempotent(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(agent, "EFFECT_DB", tmp_path / "effects.db")

    assert agent._record_once("operation", "value")
    assert not agent._record_once("operation", "value")
