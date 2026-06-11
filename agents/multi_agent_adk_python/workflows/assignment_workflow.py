"""Multi-agent ADK orchestration inside a Temporal workflow.

Pipeline:
    ParallelAgent(Fleet, Customer)  →  Dispatch
       └─ run concurrently            └─ synthesizes both,
                                          submits structured output

Every LLM call routes through ``TemporalModel`` (an ``invoke_model``
activity), and every tool call routes through ``activity_tool`` (one
activity per call). That gives per-call durability and a complete trace
of the agent reasoning in workflow history.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.workflow import ActivityConfig

with workflow.unsafe.imports_passed_through():
    from google.adk.agents import Agent, ParallelAgent, SequentialAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools import ToolContext
    from google.genai.types import Content, Part
    from temporalio.contrib.google_adk_agents import TemporalModel

    from workflows._activity_tool import activity_tool
    from activities.tools import (
        tool_get_fleet_status,
        tool_get_order_priorities,
        tool_get_route_info,
    )
    from models.models import AssignmentInput, AssignmentOutput

TASK_QUEUE = "multi-agent-adk-task-queue"
DEFAULT_MODEL = "gemini-2.5-flash"
APP_NAME = "multi_agent_adk_recipe"

_TOOL_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=5,
)


# --- Activity-backed tools (one Temporal activity per tool call) ---

_fleet_status_tool = activity_tool(
    tool_get_fleet_status,
    task_queue=TASK_QUEUE,
    summary="Fleet Agent — get fleet status",
    start_to_close_timeout=timedelta(seconds=10),
    retry_policy=_TOOL_RETRY,
)
_order_priorities_tool = activity_tool(
    tool_get_order_priorities,
    task_queue=TASK_QUEUE,
    summary="Customer Agent — get order priorities",
    start_to_close_timeout=timedelta(seconds=10),
    retry_policy=_TOOL_RETRY,
)
_route_info_tool = activity_tool(
    tool_get_route_info,
    task_queue=TASK_QUEUE,
    summary="Fleet Agent — assess ETA",
    start_to_close_timeout=timedelta(seconds=15),
    retry_policy=_TOOL_RETRY,
)


# --- Structured output tool (plain Python — writes to ADK session state) ---


async def tool_submit_assignment(
    tool_context: ToolContext,
    driver_id: str,
    reasoning_summary: str,
) -> str:
    """Submit the final order assignment. You MUST call this tool with your decision.

    Args:
        driver_id: The driver to assign the order to (e.g. "driver-a").
        reasoning_summary: One-sentence explanation of the choice.
    """
    tool_context.state["assignment"] = {
        "driver_id": driver_id,
        "reasoning_summary": reasoning_summary,
    }
    return "Assignment submitted."


# --- Agent factories ---


def _fleet_agent() -> Agent:
    return Agent(
        name="fleet_agent",
        model=TemporalModel(
            DEFAULT_MODEL,
            activity_config=ActivityConfig(
                task_queue=TASK_QUEUE,
                summary="Fleet Agent — LLM reasoning",
            ),
        ),
        description=(
            "Fleet specialist. Assesses driver positions, capacity, and ETAs "
            "to recommend the best driver for a new order."
        ),
        instruction=(
            "You are the Fleet Operations agent. A new order has arrived — "
            "assess which driver should handle it.\n\n"
            "Step 1: Call tool_get_fleet_status to get each driver's position, "
            "capacity, and status. Identify the 1–3 closest drivers with "
            "available capacity (skip DISCONNECTED and FULL drivers).\n\n"
            "Step 2: Call tool_get_route_info for each top candidate to get "
            "driving ETAs. Do NOT call it for every driver.\n\n"
            "Rules:\n"
            "- NEVER recommend a DISCONNECTED driver.\n"
            "- Skip drivers at capacity.\n"
            "- Prefer the closest driver with capacity.\n\n"
            "Respond with ONLY: the recommended driver ID and ETA. "
            "Example: 'driver-b — 4min ETA, closest with capacity.'"
        ),
        tools=[_fleet_status_tool, _route_info_tool],
        output_key="fleet_assessment",
    )


def _customer_agent() -> Agent:
    return Agent(
        name="customer_agent",
        model=TemporalModel(
            DEFAULT_MODEL,
            activity_config=ActivityConfig(
                task_queue=TASK_QUEUE,
                summary="Customer Agent — LLM reasoning",
            ),
        ),
        description=(
            "Customer priority specialist. Evaluates order priority, urgency, "
            "and deadline pressure."
        ),
        instruction=(
            "You are the Customer Relations agent. A new order has arrived — "
            "assess its priority and urgency.\n\n"
            "Call tool_get_order_priorities for the priority context.\n\n"
            "Assess: VIP or standard? Deadline tight? How many servings?\n\n"
            "Respond with ONLY: priority level and the key urgency factor. "
            "Example: 'VIP, tight deadline (25min), large order.'"
        ),
        tools=[_order_priorities_tool],
        output_key="customer_assessment",
    )


def _dispatch_agent() -> Agent:
    return Agent(
        name="dispatch_agent",
        model=TemporalModel(
            DEFAULT_MODEL,
            activity_config=ActivityConfig(
                task_queue=TASK_QUEUE,
                summary="Dispatch Agent — LLM reasoning",
            ),
        ),
        description=(
            "Dispatch coordinator. Synthesizes fleet and customer assessments "
            "and submits the final driver assignment."
        ),
        instruction=(
            "You are the Dispatch Coordinator. The Fleet Agent and Customer "
            "Agent have each posted an assessment of a new order.\n\n"
            "Rules:\n"
            "- NEVER assign to a DISCONNECTED driver.\n"
            "- If an upstream agent reports a tool failure, decide with the "
            "  data you have.\n\n"
            "You MUST call tool_submit_assignment with:\n"
            "- driver_id: the driver that should get this order\n"
            "- reasoning_summary: one sentence explaining the choice "
            "  (under 20 words)"
        ),
        tools=[tool_submit_assignment],
    )


def build_assignment_pipeline() -> SequentialAgent:
    """Compose the full pipeline: Parallel(Fleet, Customer) → Dispatch."""
    return SequentialAgent(
        name="order_assignment",
        sub_agents=[
            ParallelAgent(
                name="assessment_parallel",
                sub_agents=[_fleet_agent(), _customer_agent()],
            ),
            _dispatch_agent(),
        ],
    )


# --- Workflow ---


@workflow.defn
class MultiAgentAssignmentWorkflow:
    """Run the multi-agent pipeline for a single order assignment."""

    @workflow.run
    async def run(self, order: AssignmentInput) -> AssignmentOutput:
        workflow.logger.info(f"Assigning {order.order_id} for {order.hotel}")

        agent = build_assignment_pipeline()
        session_service = InMemorySessionService()
        runner = Runner(
            agent=agent,
            app_name=APP_NAME,
            session_service=session_service,
        )

        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id="workflow",
        )

        prompt = (
            f"NEW ORDER — assign to the best driver:\n"
            f"Order ID: {order.order_id}\n"
            f"Hotel: {order.hotel}\n"
            f"Priority: {order.priority}\n"
            f"Servings: {order.servings}\n"
            f"Deadline: {order.deadline_minutes} minutes\n"
            f"Coordinates: ({order.delivery_lat}, {order.delivery_lng})\n\n"
            f"After Fleet and Customer agents have assessed, the Dispatch "
            f"agent MUST call tool_submit_assignment with the driver_id and "
            f"reasoning."
        )

        async for _ in runner.run_async(
            user_id="workflow",
            session_id=session.id,
            new_message=Content(parts=[Part(text=prompt)]),
        ):
            pass

        updated = await session_service.get_session(
            app_name=APP_NAME,
            user_id="workflow",
            session_id=session.id,
        )
        state = (updated.state if updated else None) or {}
        assignment = state.get("assignment") or {}

        return AssignmentOutput(
            driver_id=assignment.get("driver_id", ""),
            reasoning_summary=assignment.get(
                "reasoning_summary", "Dispatch agent did not submit a decision."
            ),
        )
