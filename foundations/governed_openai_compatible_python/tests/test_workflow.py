import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities.openai_compatible import ModelRequest
from workflows.governed_model_workflow import GovernedModelWorkflow


TASK_QUEUE = "test-governed-openai-compatible"
last_request: ModelRequest | None = None


@activity.defn(name="invoke_model")
async def mock_invoke_model(request: ModelRequest) -> str:
    global last_request
    last_request = request
    return "Temporal owns durable retries."


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_workflow_propagates_stable_correlation_ids():
    global last_request
    last_request = None

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[GovernedModelWorkflow],
            activities=[mock_invoke_model],
        ):
            result = await env.client.execute_workflow(
                GovernedModelWorkflow.run,
                {"prompt": "Why should retries be durable?", "model": "gpt-4o"},
                id="governed-workflow-123",
                task_queue=TASK_QUEUE,
            )

    assert result == "Temporal owns durable retries."
    assert last_request is not None
    assert last_request.model == "gpt-4o"
    assert last_request.run_id == "governed-workflow-123"
    assert last_request.request_id == "req_governed-workflow-123_model"
