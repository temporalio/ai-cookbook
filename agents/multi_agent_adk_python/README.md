<!--
description: Build a multi-agent pipeline (parallel + sequential) with Google ADK on Temporal — every LLM call and every tool call is a durable activity.
tags: [agents, python, gemini, google-adk]
priority: 750
-->

# Multi-Agent Orchestration — Google ADK + Temporal

This recipe builds a **multi-agent dispatch pipeline** using
[Google ADK](https://google.github.io/adk-docs/) and the
[Google ADK integration for Temporal](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/google_adk_agents).
Two agents reason in parallel, a third synthesizes their output and submits a
structured decision — and every LLM call and every tool call is a durable
Temporal Activity.

[![Watch the 60-second walkthrough](https://img.youtube.com/vi/Wq7hiN2KYnk/maxresdefault.jpg)](https://youtube.com/shorts/Wq7hiN2KYnk?feature=share)

## ADK ↔ Temporal mapping

| ADK concept              | Temporal mapping                                            |
| ------------------------ | ----------------------------------------------------------- |
| Orchestrator agent       | Pure Python inside the workflow (`Runner.run_async`)        |
| LLM call (`BaseLlm`)     | `invoke_model` activity (via `TemporalModel`)               |
| Tool call (`@activity`)  | A named activity (via `activity_tool`)                      |
| Session state            | Read back from the in-memory `SessionService` after the run |
| Agent reasoning          | Lives in the workflow; *durable, replayable*                |
| Anything that does I/O   | Lives in an activity; *retried, timed out, observed*        |

## Pipeline

```
                      ┌──────────────┐
                      │ Order input  │
                      └──────┬───────┘
                             │
            ┌────────────────┴────────────────┐
            │       ParallelAgent             │
            │ ┌─────────────┐ ┌─────────────┐ │
            │ │ Fleet Agent │ │ Customer    │ │
            │ │             │ │ Agent       │ │
            │ └─────────────┘ └─────────────┘ │
            └────────────────┬────────────────┘
                             │
                      ┌──────▼───────┐
                      │ Dispatch     │
                      │ Agent        │
                      └──────┬───────┘
                             │
                  tool_submit_assignment
                             │
                      ┌──────▼───────┐
                      │ Workflow     │
                      │ returns      │
                      │ AssignmentOut│
                      └──────────────┘
```

- **Fleet Agent** — checks driver positions, capacity, ETAs.
- **Customer Agent** — checks order priority, deadline, urgency.
  *(Runs in parallel with Fleet Agent.)*
- **Dispatch Agent** — reads both assessments from session state, picks a
  driver, submits the structured decision via `tool_submit_assignment`.

This recipe highlights:

- **Multi-agent composition with ADK** — `ParallelAgent` + `SequentialAgent`
  driven by a single `Runner` invocation.
- **LLM-as-activity** — `TemporalModel` routes every LLM call through an
  `invoke_model` activity. Each call appears as a separate event in
  workflow history with retries, timeouts, and a Temporal-UI summary.
- **Tool-as-activity** — `activity_tool` wraps a Temporal Activity so the
  agent can call it. Each tool invocation is its own activity event.
- **Structured output via session state** — the final agent calls a Python
  tool that writes to `tool_context.state`. The workflow reads that key
  back after the runner completes.
- **Sandbox-safe ADK imports** — ADK and `google.genai` are imported under
  `workflow.unsafe.imports_passed_through()` so the workflow sandbox does
  not reject them.

## Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- A running Temporal Dev Server (`temporal server start-dev`)
- A Google API key with access to Gemini (`GOOGLE_API_KEY`)

## Setup

```bash
uv sync
export GOOGLE_API_KEY='your-api-key-here'
```

## Running

In one terminal, start the Temporal Dev Server:

```bash
temporal server start-dev
```

In a second terminal, start the worker:

```bash
uv run python worker.py
```

In a third terminal, kick off an assignment:

```bash
uv run python start_workflow.py
```

You should see a final assignment printed, e.g.:

```
Assigned driver: driver-a
Reasoning:       Closest available driver with capacity for VIP order
```

Open the Temporal UI at http://localhost:8233 to see each LLM call and
tool call recorded as its own activity in the workflow history.

## Architecture

```
multi_agent_adk_python/
├── activities/
│   └── tools.py             # @activity.defn — fleet status, priorities, route info
├── models/
│   └── models.py            # AssignmentInput / AssignmentOutput (pydantic)
├── workflows/
│   └── assignment_workflow.py  # agents + workflow inline
├── _activity_tool.py        # ADK ↔ Temporal tool adapter
├── worker.py
├── start_workflow.py
└── tests/
    └── test_activities.py
```

### Activities as agent tools

Each tool the agents can call is a Temporal Activity:

```python
# activities/tools.py
@activity.defn
async def tool_get_fleet_status() -> str:
    """Return current fleet state: driver positions, capacity, and status."""
    return (
        "Fleet status:\n"
        "- driver-a: pos=(36.1147, -115.1728)  capacity=2/3  status=AVAILABLE\n"
        ...
    )
```

In a real system the body would query a fleet database or hit an internal
service. Here we return canned strings so the recipe runs without any
backing infrastructure.

### TemporalModel — every LLM call is an activity

`TemporalModel` (from
`temporalio.contrib.google_adk_agents`) is an ADK `BaseLlm` whose
`generate_content_async` runs through a Temporal Activity. The plugin
registers that activity (`invoke_model`) on your worker for you.

```python
# workflows/assignment_workflow.py
agent = Agent(
    name="fleet_agent",
    model=TemporalModel(
        DEFAULT_MODEL,
        activity_config=ActivityConfig(
            task_queue=TASK_QUEUE,
            summary="Fleet Agent — LLM reasoning",
        ),
    ),
    instruction="...",
    tools=[_fleet_status_tool, _route_info_tool],
    output_key="fleet_assessment",
)
```

### activity_tool — every tool call is an activity

`activity_tool` wraps a `@activity.defn` so it presents to ADK as a regular
Python tool, but the call body executes via `workflow.execute_activity`:

```python
# workflows/assignment_workflow.py
_fleet_status_tool = activity_tool(
    tool_get_fleet_status,
    task_queue=TASK_QUEUE,
    summary="Fleet Agent — get fleet status",
    start_to_close_timeout=timedelta(seconds=10),
    retry_policy=_TOOL_RETRY,
)
```

The local `_activity_tool.py` adds **graceful failure** on top of the
upstream `temporalio.contrib.google_adk_agents.workflow.activity_tool`:
when an activity execution fails (retry policy exhausted, non-retryable
application error, timeout), the wrapper catches the `ActivityError`
and returns it to the LLM as a string so the agent can adapt instead of
crashing the pipeline. (The retry attempts still appear in workflow
history.) Programming bugs — e.g. argument-binding errors — are not
caught and propagate normally. Upstream `activity_tool` (temporalio>=1.25)
already handles multi-arg activities and local non-workflow ADK runs.

### Composing the pipeline

```python
# workflows/assignment_workflow.py
def build_assignment_pipeline() -> SequentialAgent:
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
```

Fleet and Customer agents run concurrently inside `ParallelAgent`. When
both finish, the Dispatch agent runs. Each sub-agent's `output_key`
(`fleet_assessment`, `customer_assessment`) writes its final response into
session state — that's how the Dispatch agent gets the upstream context
without passing it explicitly.

### Structured output via session state

The final agent calls a plain Python tool (not a Temporal activity) that
writes the decision into ADK session state:

```python
async def tool_submit_assignment(
    tool_context: ToolContext,
    driver_id: str,
    reasoning_summary: str,
) -> str:
    tool_context.state["assignment"] = {
        "driver_id": driver_id,
        "reasoning_summary": reasoning_summary,
    }
    return "Assignment submitted."
```

The workflow runs the pipeline to exhaustion, then reads that key back:

```python
# workflows/assignment_workflow.py
async for _ in runner.run_async(
    user_id="workflow",
    session_id=session.id,
    new_message=Content(parts=[Part(text=prompt)]),
):
    pass

updated = await session_service.get_session(...)
assignment = (updated.state or {}).get("assignment") or {}
return AssignmentOutput(
    driver_id=assignment.get("driver_id", ""),
    reasoning_summary=assignment.get(
        "reasoning_summary", "Dispatch agent did not submit a decision."
    ),
)
```

This pattern — a tool call that writes structured output into session
state — is how you reliably extract a typed decision from a multi-agent
pipeline that emits many intermediate events.

### Sandbox-safe ADK imports

ADK and `google.genai` aren't safe under Temporal's workflow sandbox by
default, so they're imported under
`workflow.unsafe.imports_passed_through()`:

```python
# workflows/assignment_workflow.py
with workflow.unsafe.imports_passed_through():
    from google.adk.agents import Agent, ParallelAgent, SequentialAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools import ToolContext
    from google.genai.types import Content, Part
    from temporalio.contrib.google_adk_agents import TemporalModel
```

The `GoogleAdkPlugin` registered on the worker handles the rest of the
sandbox passthroughs and deterministic-runtime overrides ADK needs (UUIDs,
clocks).

### Worker — one queue, one plugin

```python
# worker.py
client = await Client.connect("localhost:7233")
worker = Worker(
    client,
    task_queue=TASK_QUEUE,
    workflows=[MultiAgentAssignmentWorkflow],
    activities=[
        tool_get_fleet_status,
        tool_get_order_priorities,
        tool_get_route_info,
    ],
    plugins=[GoogleAdkPlugin()],
)
await worker.run()
```

`GoogleAdkPlugin` registers the `invoke_model` activity that
`TemporalModel` routes LLM calls to — you don't need to register it
yourself.

## Extensions

This pipeline is the minimal multi-agent shape. Natural ways to extend it:

- **Real backing services** — replace the canned tool bodies with real
  database queries, route APIs, or internal microservices.
- **More parallel branches** — add agents to the `ParallelAgent` (an
  inventory agent, a credit-check agent, a fraud-screen agent) — each
  posts its assessment via `output_key` for the synthesizer to read.
- **Multiple sequential stages** — add a validation or post-processing
  stage after Dispatch by extending the outer `SequentialAgent`.
- **Human-in-the-loop on the synthesizer** — pause before the final
  decision via a Temporal Signal. See the
  [`human_in_the_loop_python`](../human_in_the_loop_python) recipe.
- **Graceful degradation** — fail-fast retry policies on a sub-agent's
  tools so the synthesizer can decide with partial data when an upstream
  service is unavailable.
