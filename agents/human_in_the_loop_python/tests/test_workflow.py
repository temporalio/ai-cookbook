"""Tests for HumanInTheLoopWorkflow with mocked activities."""

import asyncio

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from datetime import timedelta
from dataclasses import dataclass

from models.models import (
    WorkflowInput,
    ProposedAction,
    ApprovalRequest,
    ApprovalDecision,
)
from workflows.human_in_the_loop_workflow import HumanInTheLoopWorkflow

TASK_QUEUE = "test-human-in-the-loop"


# Capture approval requests so we can extract the request_id for signals
_captured_approval_requests: list[ApprovalRequest] = []


def _make_safe_action() -> str:
    return ProposedAction(
        action_type="list_files",
        description="List directory contents",
        reasoning="Safe read-only operation",
        risky_action=False,
    ).model_dump_json()


def _make_risky_action() -> str:
    return ProposedAction(
        action_type="delete_data",
        description="Delete production data",
        reasoning="User requested deletion",
        risky_action=True,
    ).model_dump_json()


def _make_mock_create_safe():
    @dataclass
    class _Req:
        model: str
        instructions: str
        input: str

    @activity.defn(name="create")
    async def mock_create(request: _Req) -> str:
        return _make_safe_action()

    return mock_create


def _make_mock_create_risky():
    @dataclass
    class _Req:
        model: str
        instructions: str
        input: str

    @activity.defn(name="create")
    async def mock_create(request: _Req) -> str:
        return _make_risky_action()

    return mock_create


@activity.defn(name="notify_approval_needed")
async def mock_notify(request: ApprovalRequest) -> None:
    _captured_approval_requests.append(request)


@activity.defn(name="execute_action")
async def mock_execute(action: ProposedAction) -> str:
    return f"Executed: {action.action_type}"


class TestSafeActionAutoApproved:
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_safe_action_auto_approved(self):
        mock_create = _make_mock_create_safe()
        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter,
        ) as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[HumanInTheLoopWorkflow],
                activities=[mock_create, mock_notify, mock_execute],
            ):
                result = await env.client.execute_workflow(
                    HumanInTheLoopWorkflow.run,
                    WorkflowInput(user_request="List files"),
                    id="test-safe-action",
                    task_queue=TASK_QUEUE,
                    run_timeout=timedelta(seconds=30),
                )

        assert "auto-approved" in result


class TestRiskyActionApproved:
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_risky_action_approved(self):
        _captured_approval_requests.clear()
        mock_create = _make_mock_create_risky()

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter,
        ) as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[HumanInTheLoopWorkflow],
                activities=[mock_create, mock_notify, mock_execute],
            ):
                handle = await env.client.start_workflow(
                    HumanInTheLoopWorkflow.run,
                    WorkflowInput(
                        user_request="Delete data",
                        approval_timeout_seconds=30,
                    ),
                    id="test-risky-approved",
                    task_queue=TASK_QUEUE,
                    run_timeout=timedelta(seconds=30),
                )

                # Wait for the notify activity to capture the request_id
                for _ in range(50):
                    if _captured_approval_requests:
                        break
                    await asyncio.sleep(0.1)

                assert len(_captured_approval_requests) > 0
                request_id = _captured_approval_requests[-1].request_id

                await handle.signal(
                    HumanInTheLoopWorkflow.approval_decision,
                    ApprovalDecision(
                        request_id=request_id,
                        approved=True,
                        reviewer_notes="Approved",
                        decided_at="2025-01-01T00:00:00",
                    ),
                )

                result = await handle.result()

        assert "completed successfully" in result.lower()


class TestRiskyActionRejected:
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_risky_action_rejected(self):
        _captured_approval_requests.clear()
        mock_create = _make_mock_create_risky()

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter,
        ) as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[HumanInTheLoopWorkflow],
                activities=[mock_create, mock_notify, mock_execute],
            ):
                handle = await env.client.start_workflow(
                    HumanInTheLoopWorkflow.run,
                    WorkflowInput(
                        user_request="Delete data",
                        approval_timeout_seconds=30,
                    ),
                    id="test-risky-rejected",
                    task_queue=TASK_QUEUE,
                    run_timeout=timedelta(seconds=30),
                )

                for _ in range(50):
                    if _captured_approval_requests:
                        break
                    await asyncio.sleep(0.1)

                request_id = _captured_approval_requests[-1].request_id

                await handle.signal(
                    HumanInTheLoopWorkflow.approval_decision,
                    ApprovalDecision(
                        request_id=request_id,
                        approved=False,
                        reviewer_notes="Too risky",
                        decided_at="2025-01-01T00:00:00",
                    ),
                )

                result = await handle.result()

        assert "rejected" in result.lower()
        assert "Too risky" in result


class TestRiskyActionTimeout:
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_risky_action_times_out(self):
        _captured_approval_requests.clear()
        mock_create = _make_mock_create_risky()

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter,
        ) as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[HumanInTheLoopWorkflow],
                activities=[mock_create, mock_notify, mock_execute],
            ):
                result = await env.client.execute_workflow(
                    HumanInTheLoopWorkflow.run,
                    WorkflowInput(
                        user_request="Delete data",
                        approval_timeout_seconds=1,
                    ),
                    id="test-risky-timeout",
                    task_queue=TASK_QUEUE,
                    run_timeout=timedelta(seconds=30),
                )

        assert "timed out" in result.lower()
