"""Workflow-level tests for MultiAgentAssignmentWorkflow.

These run the real ADK pipeline (Parallel(Fleet, Customer) → Dispatch) inside a
Temporal ``WorkflowEnvironment``, but replace the ``invoke_model`` activity with
a deterministic mock so no real LLM is needed. The mock inspects the calling
agent (via ``llm_request.config.labels``) and returns canned ``LlmResponse``
objects, letting us assert the two workflow paths:

* the Dispatch agent calls ``tool_submit_assignment`` → workflow returns the
  submitted ``driver_id``;
* the Dispatch agent never calls it → workflow returns the empty-decision
  fallback.

The mock is swapped in via a ``GoogleAdkPlugin`` subclass so the sandbox
passthroughs, deterministic runtime, and ADK data converter all still apply —
only the model activity changes.
"""

import pytest
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from temporalio import activity
from temporalio.contrib.google_adk_agents import GoogleAdkPlugin
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities.tools import (
    tool_get_fleet_status,
    tool_get_order_priorities,
    tool_get_route_info,
)
from models.models import AssignmentInput
from workflows.assignment_workflow import (
    TASK_QUEUE,
    MultiAgentAssignmentWorkflow,
)

SAMPLE_ORDER = AssignmentInput(
    order_id="order-001",
    hotel="Caesars Palace",
    priority="VIP",
    servings=50,
    deadline_minutes=25,
    delivery_lat=36.1162,
    delivery_lng=-115.1745,
)


# --- LlmResponse builders ---


def _text_response(text: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def _submit_call(driver_id: str, reasoning: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="tool_submit_assignment",
                        args={"driver_id": driver_id, "reasoning_summary": reasoning},
                    )
                )
            ],
        )
    )


def _declared_tools(llm_request) -> set[str]:
    """Names of the tools declared on this request.

    Used to tell the agents apart from inside the mocked ``invoke_model``
    activity: unlike ``config.labels['adk_agent_name']``, the tool
    declarations survive serialization across the activity boundary.
    """
    names: set[str] = set()
    config = getattr(llm_request, "config", None)
    for tool in getattr(config, "tools", None) or []:
        for declaration in getattr(tool, "function_declarations", None) or []:
            if getattr(declaration, "name", None):
                names.add(declaration.name)
    return names


def _submission_already_ran(llm_request) -> bool:
    """True once the Dispatch agent's tool_submit_assignment result is in history."""
    for content in getattr(llm_request, "contents", None) or []:
        for part in getattr(content, "parts", None) or []:
            fr = getattr(part, "function_response", None)
            if fr is not None and fr.name == "tool_submit_assignment":
                return True
    return False


# --- Mock invoke_model activities (one per scenario) ---


@activity.defn(name="invoke_model")
async def mock_invoke_model_submits(llm_request: LlmRequest) -> list[LlmResponse]:
    tools = _declared_tools(llm_request)
    if "tool_submit_assignment" in tools:  # dispatch agent
        if _submission_already_ran(llm_request):
            return [_text_response("Assignment submitted.")]
        return [_submit_call("driver-b", "Closest driver with capacity for a VIP order.")]
    if "tool_get_fleet_status" in tools:  # fleet agent
        return [_text_response("driver-b — 4min ETA, closest with capacity.")]
    if "tool_get_order_priorities" in tools:  # customer agent
        return [_text_response("VIP, tight deadline (25min), large order.")]
    return [_text_response("ok")]


@activity.defn(name="invoke_model")
async def mock_invoke_model_no_submit(llm_request: LlmRequest) -> list[LlmResponse]:
    tools = _declared_tools(llm_request)
    if "tool_get_fleet_status" in tools:  # fleet agent
        return [_text_response("driver-b — 4min ETA, closest with capacity.")]
    if "tool_get_order_priorities" in tools:  # customer agent
        return [_text_response("VIP, tight deadline (25min), large order.")]
    # Dispatch agent reasons aloud but never calls tool_submit_assignment.
    return [_text_response("I'd lean towards driver-b, but I'm not submitting a decision.")]


class _MockModelAdkPlugin(GoogleAdkPlugin):
    """GoogleAdkPlugin with its invoke_model/invoke_model_streaming activities
    replaced by a single deterministic mock, keeping all other ADK config."""

    def __init__(self, mock_activity):
        super().__init__()
        self.activities = [mock_activity]


async def _run(mock_activity, order: AssignmentInput, workflow_id: str):
    plugin = _MockModelAdkPlugin(mock_activity)
    async with await WorkflowEnvironment.start_time_skipping(
        plugins=[plugin],
    ) as env:
        # The plugin is inherited from the client, so it also configures the
        # worker (mock activity + sandbox passthroughs + deterministic runtime).
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[MultiAgentAssignmentWorkflow],
            activities=[
                tool_get_fleet_status,
                tool_get_order_priorities,
                tool_get_route_info,
            ],
        ):
            return await env.client.execute_workflow(
                MultiAgentAssignmentWorkflow.run,
                order,
                id=workflow_id,
                task_queue=TASK_QUEUE,
            )


class TestAssignmentWorkflow:
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_returns_submitted_driver(self):
        result = await _run(
            mock_invoke_model_submits, SAMPLE_ORDER, "test-adk-submits"
        )
        assert result.driver_id == "driver-b"
        assert result.reasoning_summary == "Closest driver with capacity for a VIP order."

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_fallback_when_dispatch_does_not_submit(self):
        result = await _run(
            mock_invoke_model_no_submit, SAMPLE_ORDER, "test-adk-no-submit"
        )
        assert result.driver_id == ""
        assert "did not submit" in result.reasoning_summary
