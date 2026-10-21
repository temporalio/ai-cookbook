import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities.llm_call import CallLlmRequest
from workflows.recipe_workflow import SubagentWorkflow, SupervisorAgentWorkflow

TASK_QUEUE = "subagent-supervisor-task-queue"


def _usage() -> Usage:
    return Usage(
        input_tokens=10,
        output_tokens=20,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )


def _text(text: str) -> Message:
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-sonnet-4-6",
        content=[TextBlock(type="text", text=text)],
        stop_reason="end_turn",
        usage=_usage(),
    )


def _tool_use(name: str, tool_input: dict) -> Message:
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-sonnet-4-6",
        content=[ToolUseBlock(type="tool_use", id="toolu_1", name=name, input=tool_input)],
        stop_reason="tool_use",
        usage=_usage(),
    )


@pytest.fixture
def reset_calls():
    # Module-level call counter drives the scripted LLM responses; clear it per test.
    global _call_count
    _call_count = 0
    yield


_call_count = 0


@activity.defn(name="call_llm")
async def mock_call_llm(request: CallLlmRequest) -> Message:
    # The supervisor exposes the delegate tool; the subagent does not. We route on whether
    # the request's toolset contains delegate_to_subagent to script each agent's turn.
    global _call_count
    _call_count += 1
    tool_names = {tool["name"] for tool in request.tools}
    is_supervisor = "delegate_to_subagent" in tool_names

    if is_supervisor and _call_count == 1:
        # Supervisor's first turn: delegate a word-count sub-task.
        return _tool_use(
            "delegate_to_subagent",
            {"task": "count the words in 'durable execution'", "tool_names": ["word_count"]},
        )
    if not is_supervisor:
        # Subagent answers directly with its result.
        return _text("The phrase has 2 words.")
    # Supervisor's final turn, after folding in the subagent's tool_result.
    return _text("The subagent reports the phrase has 2 words.")


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_supervisor_delegates_to_subagent(reset_calls) -> None:
    async with (
        await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env,
        Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[SupervisorAgentWorkflow, SubagentWorkflow],
            activities=[mock_call_llm],
        ),
    ):
        result = await env.client.execute_workflow(
            SupervisorAgentWorkflow.run,
            "How many words are in 'durable execution'?",
            id="test-subagent-supervisor",
            task_queue=TASK_QUEUE,
        )

    # The subagent ran and its "2 words" result reached the supervisor's final answer.
    assert "2 words" in result


@activity.defn(name="call_llm")
async def mock_call_llm_subagent_fails(request: CallLlmRequest) -> Message:
    # Same routing as the happy-path mock, but the subagent's LLM call fails permanently.
    # The non-retryable ApplicationError fails the subagent workflow, which surfaces to the
    # supervisor as a ChildWorkflowError that _delegate catches.
    global _call_count
    _call_count += 1
    tool_names = {tool["name"] for tool in request.tools}
    is_supervisor = "delegate_to_subagent" in tool_names

    if is_supervisor and _call_count == 1:
        return _tool_use(
            "delegate_to_subagent",
            {"task": "count the words in 'durable execution'", "tool_names": ["word_count"]},
        )
    if not is_supervisor:
        raise ApplicationError("subagent boom", type="BadRequestError", non_retryable=True)
    # Supervisor's final turn, after folding in the failed subagent's tool_result.
    return _text("The subagent could not finish, so I cannot count the words.")


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_failed_subagent_surfaces_as_tool_error(reset_calls) -> None:
    async with (
        await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env,
        Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[SupervisorAgentWorkflow, SubagentWorkflow],
            activities=[mock_call_llm_subagent_fails],
        ),
    ):
        # The supervisor must not crash: a failed subagent comes back as a tool error,
        # so the supervisor still produces a final answer.
        result = await env.client.execute_workflow(
            SupervisorAgentWorkflow.run,
            "How many words are in 'durable execution'?",
            id="test-subagent-supervisor-failure",
            task_queue=TASK_QUEUE,
        )

    assert "could not finish" in result
