import pytest
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities.llm_call import CallLlmRequest
from helpers.context_window import window_messages
from workflows.recipe_workflow import (
    AgentInput,
    AgentResult,
    SummarizationConfig,
    SummarizingAgentWorkflow,
)

TASK_QUEUE = "context-summarization-continue-as-new-task-queue"


def _conversation(turns: int) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for i in range(turns):
        messages.append({"role": "user", "content": f"user message {i}"})
        messages.append({"role": "assistant", "content": f"assistant reply {i}"})
    return messages


def test_window_pins_initial_keeps_recent_and_alternates() -> None:
    messages = _conversation(10)  # 20 messages
    selected, dropped = window_messages(messages, max_recent=6, max_context_tokens=10_000)

    assert dropped > 0
    assert len(selected) <= 6
    assert selected[0] == messages[0]  # initial user message is pinned
    assert selected[-1] == messages[-1]  # most recent turn is preserved
    assert selected[0]["role"] == "user"  # valid provider request
    roles = [m["role"] for m in selected]
    assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))


def test_window_shrinks_to_token_budget() -> None:
    messages = _conversation(10)
    # A tiny budget forces compaction even though the count is under max_recent.
    selected, dropped = window_messages(messages, max_recent=50, max_context_tokens=5)

    assert dropped > 0
    assert len(selected) < len(messages)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_continue_as_new_compacts_and_resumes() -> None:
    seen_sizes: list[int] = []

    @activity.defn(name="call_llm")
    async def mock_call_llm(request: CallLlmRequest) -> str:
        seen_sizes.append(len(request.messages))
        return f"reply over {len(request.messages)} messages"

    async with (
        await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env,
        Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[SummarizingAgentWorkflow],
            activities=[mock_call_llm],
        ),
    ):
        result = await env.client.execute_workflow(
            SummarizingAgentWorkflow.run,
            AgentInput(
                prompts=[f"turn {i}" for i in range(3)],
                config=SummarizationConfig(
                    max_recent_messages=4,
                    max_context_tokens=10_000,
                    continue_as_new_after_turns=1,
                ),
            ),
            id="test-context-summarization-continue-as-new",
            task_queue=TASK_QUEUE,
        )

    assert isinstance(result, AgentResult)
    assert result.total_turns == 3
    # threshold=1 with 3 prompts forces a handoff after each of the first two turns
    assert result.compactions == 2
    # the model never saw more than the configured window, even as turns accrued
    assert max(seen_sizes) <= 4


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_empty_prompts_fails_fast() -> None:
    @activity.defn(name="call_llm")
    async def mock_call_llm(request: CallLlmRequest) -> str:
        return "unused"

    async with (
        await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env,
        Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[SummarizingAgentWorkflow],
            activities=[mock_call_llm],
        ),
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await env.client.execute_workflow(
                SummarizingAgentWorkflow.run,
                AgentInput(prompts=[]),
                id="test-context-summarization-empty-prompts",
                task_queue=TASK_QUEUE,
            )

    assert isinstance(exc_info.value.cause, ApplicationError)
