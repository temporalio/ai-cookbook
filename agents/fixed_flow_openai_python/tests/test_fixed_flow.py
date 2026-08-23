from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities.run_agent_stage import run_agent_stage
from models.flow import AgentStageRequest, AgentStageResult, FixedFlowInput
from workflows.fixed_flow_workflow import FixedFlowWorkflow

TASK_QUEUE = "test-fixed-flow-openai"
_stage_requests: list[AgentStageRequest] = []


@activity.defn(name="run_agent_stage")
async def mock_run_agent_stage(request: AgentStageRequest) -> AgentStageResult:
    _stage_requests.append(request)
    outputs = {
        "analysis": "The pilot has demand and cost evidence.",
        "critique": "The source has no Sunday staffing plan.",
        "recommendation": "Run a Saturday-only pilot and measure attendance.",
    }
    return AgentStageResult(stage=request.stage, output=outputs[request.stage])


class TestRunAgentStageActivity:
    @pytest.mark.asyncio
    async def test_returns_trimmed_output_and_closes_client(self):
        response = MagicMock(output_text="  useful analysis  ")
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=response)
        mock_client.close = AsyncMock()

        with patch(
            "activities.run_agent_stage.AsyncOpenAI",
            return_value=mock_client,
        ) as client_class:
            result = await run_agent_stage(
                AgentStageRequest(
                    stage="analysis",
                    model="gpt-4o-mini",
                    instructions="Analyze the source.",
                    input="Source text",
                )
            )

        assert result == AgentStageResult(stage="analysis", output="useful analysis")
        client_class.assert_called_once_with(max_retries=0)
        mock_client.responses.create.assert_awaited_once_with(
            model="gpt-4o-mini",
            instructions="Analyze the source.",
            input="Source text",
            timeout=30,
        )
        mock_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_response_is_retryable(self):
        response = MagicMock(output_text="   ")
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=response)
        mock_client.close = AsyncMock()

        with patch(
            "activities.run_agent_stage.AsyncOpenAI",
            return_value=mock_client,
        ):
            with pytest.raises(ApplicationError) as exc_info:
                await run_agent_stage(
                    AgentStageRequest(
                        stage="critique",
                        model="gpt-4o-mini",
                        instructions="Review the analysis.",
                        input="Analysis text",
                    )
                )

        assert exc_info.value.type == "EmptyModelResponse"
        assert exc_info.value.non_retryable is False
        mock_client.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_workflow_runs_all_stages_in_order_and_passes_context():
    _stage_requests.clear()

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[FixedFlowWorkflow],
            activities=[mock_run_agent_stage],
        ):
            result = await env.client.execute_workflow(
                FixedFlowWorkflow.run,
                FixedFlowInput(
                    topic="Library hours",
                    source_material="Visits are up. Sunday staffing is unknown.",
                ),
                id="test-fixed-flow",
                task_queue=TASK_QUEUE,
            )

    assert [request.stage for request in _stage_requests] == [
        "analysis",
        "critique",
        "recommendation",
    ]
    assert _stage_requests[0].model == "gpt-4o-mini"
    assert "The pilot has demand and cost evidence." in _stage_requests[1].input
    assert "The source has no Sunday staffing plan." in _stage_requests[2].input
    assert result.analysis == "The pilot has demand and cost evidence."
    assert result.critique == "The source has no Sunday staffing plan."
    assert result.recommendation == "Run a Saturday-only pilot and measure attendance."
