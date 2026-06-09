import pytest
from activities.llm_call import CallLLMRequest
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from workflows.recipe_workflow import RecipeWorkflow

TASK_QUEUE = "RECIPE_SLUG-task-queue"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_workflow_returns_activity_result() -> None:
    @activity.defn(name="call_llm")
    async def mock_call_llm(request: CallLLMRequest) -> str:
        return "mocked response"

    async with (
        await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env,
        Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[RecipeWorkflow],
            activities=[mock_call_llm],
        ),
    ):
        result = await env.client.execute_workflow(
            RecipeWorkflow.run,
            "hi",
            id="test-recipe",
            task_queue=TASK_QUEUE,
        )
    assert result == "mocked response"
