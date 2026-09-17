<!--
description: Build a multi-agent pipeline (parallel + sequential) with Google ADK on Temporal — every LLM call and every I/O tool call runs as a durable activity.
tags: [agents, python, gemini, google-adk]
priority: 750
-->

# Multi-Agent Orchestration — Google ADK + Temporal

This recipe builds a **multi-agent dispatch pipeline** using
[Google ADK](https://google.github.io/adk-docs/) and the
[Google ADK integration for Temporal](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/google_adk_agents).
Two agents reason in parallel, a third synthesizes their output and submits a
structured decision — and every LLM call and every I/O tool call is a durable
Temporal Activity. (The final decision write is a plain in-workflow tool —
see [Structured output via session state](#structured-output-via-session-state).)

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
  agent can call it. Each of these tool invocations is its own activity
  event. (The Dispatch agent's final `tool_submit_assignment` call is a
  plain in-workflow tool, not an activity — it only writes to local
  session state.)
- **Structured output via session state** — the final agent calls a Python
  tool that writes to `tool_context.state`. The workflow reads that key
  back after the runner completes.
- **Sandbox-safe ADK imports** — ADK and `google.genai` are imported under
  `workflow.unsafe.imports_passed_through()` so the workflow sandbox does
  not reject them.

## Prerequisites

- Python 3.10 to 3.13
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
│   ├── assignment_workflow.py  # agents + workflow inline
│   └── _activity_tool.py       # ADK ↔ Temporal tool adapter
├── worker.py
├── start_workflow.py
└── tests/
    ├── test_activities.py
    └── test_workflow.py        # workflow logic with a mocked invoke_model
```

### Activities as agent tools

Each tool the agents can call is a Temporal Activity:

*File: activities/tools.py*
<!--SNIPSTART activities/tools.py {"startPattern": "^async def tool_get_fleet_status\\(\\) -> str:$", "endPattern": "^    \\)$"}-->
```python
async def tool_get_fleet_status() -> str:
    """Return current fleet state: driver positions, capacity, and status."""
    return (
        "Fleet status:\n"
        "- driver-a: pos=(36.1147, -115.1728)  capacity=2/3  status=AVAILABLE\n"
        "- driver-b: pos=(36.1099, -115.1750)  capacity=0/3  status=AVAILABLE\n"
        "- driver-c: pos=(36.1162, -115.1745)  capacity=3/3  status=FULL\n"
        "- driver-d: pos=(36.1213, -115.1700)  capacity=1/3  status=AVAILABLE\n"
        "- driver-e: pos=(36.1080, -115.1760)  capacity=2/3  status=DISCONNECTED"
    )
```
<!--SNIPEND-->

In a real system the body would query a fleet database or hit an internal
service. Here we return canned strings so the recipe runs without any
backing infrastructure.

### TemporalModel — every LLM call is an activity

`TemporalModel` (from
`temporalio.contrib.google_adk_agents`) is an ADK `BaseLlm` whose
`generate_content_async` runs through a Temporal Activity. The plugin
registers that activity (`invoke_model`) on your worker for you.

*File: workflows/assignment_workflow.py*
<!--SNIPSTART workflows/assignment_workflow.py {"startPattern": "^def _fleet_agent\\(\\) -> Agent:$", "endPattern": "^    \\)$", "selectedLines": ["1-10", "30-32"]}-->
```python
def _fleet_agent() -> Agent:
    return Agent(
        name="fleet_agent",
        model=TemporalModel(
            DEFAULT_MODEL,
            activity_config=ActivityConfig(
                task_queue=TASK_QUEUE,
                summary="Fleet Agent — LLM reasoning",
                retry_policy=_LLM_RETRY,
            ),
        ...
        ),
        tools=[_fleet_status_tool, _route_info_tool],
        output_key="fleet_assessment",
        ...
```
<!--SNIPEND-->

### activity_tool — every tool call is an activity

`activity_tool` wraps a `@activity.defn` so it presents to ADK as a regular
Python tool, but the call body executes via `workflow.execute_activity`:

*File: workflows/assignment_workflow.py*
<!--SNIPSTART workflows/assignment_workflow.py {"startPattern": "^_fleet_status_tool = activity_tool\\($", "endPattern": "^\\)$"}-->
```python
_fleet_status_tool = activity_tool(
    tool_get_fleet_status,
    task_queue=TASK_QUEUE,
    summary="Fleet Agent — get fleet status",
    start_to_close_timeout=timedelta(seconds=10),
    retry_policy=_TOOL_RETRY,
)
```
<!--SNIPEND-->

The local `workflows/_activity_tool.py` adds **graceful failure** on top of the
upstream `temporalio.contrib.google_adk_agents.workflow.activity_tool`:
when an activity execution fails (retry policy exhausted, non-retryable
application error, timeout), the wrapper catches the `ActivityError`
and returns it to the LLM as a string so the agent can adapt instead of
crashing the pipeline. (The retry attempts still appear in workflow
history.) Programming bugs — e.g. argument-binding errors — are not
caught and propagate normally. Upstream `activity_tool` (temporalio>=1.25)
already handles multi-arg activities and local non-workflow ADK runs.

### Composing the pipeline

*File: workflows/assignment_workflow.py*
<!--SNIPSTART workflows/assignment_workflow.py {"startPattern": "^def build_assignment_pipeline\\(\\) -> SequentialAgent:$", "endPattern": "^    \\)$"}-->
```python
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
```
<!--SNIPEND-->

Fleet and Customer agents run concurrently inside `ParallelAgent`. When
both finish, the Dispatch agent runs. Each sub-agent's `output_key`
(`fleet_assessment`, `customer_assessment`) writes its final response into
session state — that's how the Dispatch agent gets the upstream context
without passing it explicitly.

### Structured output via session state

The final agent calls a plain Python tool (not a Temporal activity) that
writes the decision into ADK session state:

*File: workflows/assignment_workflow.py*
<!--SNIPSTART workflows/assignment_workflow.py {"startPattern": "^async def tool_submit_assignment\\($", "endPattern": "Assignment submitted"}-->
```python
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
```
<!--SNIPEND-->

The workflow runs the pipeline to exhaustion, then reads that key back:

*File: workflows/assignment_workflow.py*
<!--SNIPSTART workflows/assignment_workflow.py:workflow-run-tail-->
```python
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
```
<!--SNIPEND-->

This pattern — a tool call that writes structured output into session
state — is how you reliably extract a typed decision from a multi-agent
pipeline that emits many intermediate events.

### Sandbox-safe ADK imports

ADK and `google.genai` aren't safe under Temporal's workflow sandbox by
default, so they're imported under
`workflow.unsafe.imports_passed_through()`:

*File: workflows/assignment_workflow.py*
<!--SNIPSTART workflows/assignment_workflow.py {"startPattern": "^with workflow\\.unsafe\\.imports_passed_through\\(\\):$", "endPattern": "^    from temporalio\\.contrib\\.google_adk_agents import TemporalModel$"}-->
```python
with workflow.unsafe.imports_passed_through():
    from google.adk.agents import Agent, ParallelAgent, SequentialAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools import ToolContext
    from google.genai.types import Content, Part
    from temporalio.contrib.google_adk_agents import TemporalModel
```
<!--SNIPEND-->

The `GoogleAdkPlugin` registered on the worker handles the rest of the
sandbox passthroughs and deterministic-runtime overrides ADK needs (UUIDs,
clocks).

### Worker — one queue, one plugin

*File: worker.py*
<!--SNIPSTART worker.py {"startPattern": "^    client = await Client\\.connect\\($", "endPattern": "^    await worker\\.run\\(\\)$"}-->
```python
client = await Client.connect(
    "localhost:7233",
    data_converter=pydantic_data_converter,
)

# GoogleAdkPlugin registers the `invoke_model` activity (used by
# TemporalModel for LLM calls) and provides the workflow-sandbox
# passthroughs and deterministic runtime overrides ADK needs.
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
<!--SNIPEND-->

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
