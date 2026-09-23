<!--
description: Durable agent execution using Claude Agent SDK (claude-agent-sdk) with streaming, heartbeats, and session management.
tags: [agents, python, claude, claude-agent-sdk, streaming]
priority: 800
-->
# Claude Agent SDK with Temporal

This example demonstrates how to run the [Claude Agent SDK](https://docs.anthropic.com/en/docs/claude-code/sdk) (`claude-agent-sdk`) inside a Temporal workflow for durable, observable agent execution.

Unlike the [basic agentic loop example](../agentic_loop_tool_call_claude_python/) that builds a manual tool-calling loop using Claude's Messages API, this example delegates the entire agentic loop to the Claude Agent SDK. The SDK handles tool selection, execution, multi-turn conversation, and streaming internally — the Temporal activity simply streams events and collects the final response.

This recipe highlights the following key design decisions:

- **SDK-managed agentic loop**: The Claude Agent SDK runs the entire agent loop (prompt → tool calls → responses) internally. The Temporal activity wraps this as a single long-running operation rather than breaking each LLM call into a separate activity.
- **Background heartbeats**: An `asyncio.create_task()` sends Temporal heartbeats every 60 seconds, independent of the SDK event stream. This is critical because the SDK may execute long-running tools (e.g., git operations, file reads) that emit no events for extended periods.
- **Staleness guard**: If no SDK events arrive for 15 minutes, the heartbeat loop exits. Temporal's `heartbeat_timeout` (10 min) then fires, killing a truly hung agent instead of letting it block the full 30-minute `start_to_close_timeout`.
- **Response deduplication**: The Claude Agent SDK emits both `StreamEvent` (incremental text chunks) and `AssistantMessage` (complete text blocks). Both contain the same text, so we only accumulate from `AssistantMessage` to avoid doubling the response.
- **Session resumption**: The SDK stores sessions as JSONL files on disk. The activity captures the `session_id` from the `SystemMessage(init)` event and returns it in the output. Subsequent calls can pass this ID to resume from where the previous session left off.
- **Errors modeled as completions**: Agent failures are returned as `AgentOutput(status="error", ...)` rather than raising exceptions, giving the workflow full control over retry/notification logic.

## Application Components

This example includes the following components:
- [Pydantic Models](#pydantic-models) for workflow input/output.
- The [Activity](#create-the-activity) that wraps the Claude Agent SDK with heartbeats and staleness detection.
- The [Workflow](#create-the-workflow) that orchestrates execution and result logging.
- The [Worker](#create-the-worker) that hosts the Workflow and Activities.
- A [CLI script](#initiate-an-interaction) to submit prompts to the agent.

## Pydantic Models

The input and output models define the contract between the workflow and activity.

*File: models.py*

```python
from typing import Literal, Optional
from pydantic import BaseModel, Field


class AgentInput(BaseModel):
    """Input for the agent execution activity."""
    prompt: str = Field(..., description="User message to send to the agent")
    model: str = Field(default="claude-sonnet-4-5-20250929")
    system_prompt: Optional[str] = Field(default=None)
    max_turns: int = Field(default=30)
    permission_mode: str = Field(default="bypassPermissions")
    resume_session_id: Optional[str] = Field(default=None)


class AgentOutput(BaseModel):
    """Output from the agent execution activity."""
    status: Literal["success", "error"]
    response: str = Field(default="")
    total_tokens: int = Field(default=0)
    num_events: int = Field(default=0)
    processing_time_seconds: Optional[float] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
```

## Create the Activity

### Agent execution with background heartbeats

The core activity wraps `query()` from the Claude Agent SDK and streams events. Three production patterns make this robust:

1. **Background heartbeat task** — An `asyncio.create_task()` sends heartbeats every 60 seconds regardless of SDK event flow. Without this, long tool executions (no events for 10+ minutes) cause Temporal to kill the activity.

2. **Staleness guard** — If no events arrive for 15 minutes, the heartbeat loop exits. This prevents a truly hung agent from blocking the full 30-minute `start_to_close_timeout`.

3. **Response deduplication** — Only `AssistantMessage` events are used for response text. `StreamEvent` contains the same text as incremental chunks and must be skipped.

4. **Session resumption** — The SDK emits a `SystemMessage(subtype='init')` with a `session_id` at the start of each session. We capture this and return it in the output. To resume, pass the session_id as `resume_session_id` in the next request — the SDK picks up from where it left off.

*File: activities/agent_executor.py*

```python
import asyncio
import time
from datetime import datetime, timezone
from temporalio import activity
from models import AgentInput, AgentOutput

HEARTBEAT_INTERVAL = 60
MAX_IDLE_SECONDS = 15 * 60  # 15 minutes


@activity.defn
async def execute_agent_activity(input_data: AgentInput) -> AgentOutput:
    from claude_agent_sdk import query
    from claude_agent_sdk.types import (
        ClaudeAgentOptions, AssistantMessage, ResultMessage, SystemMessage,
    )

    start_time = datetime.now(timezone.utc)

    try:
        options = ClaudeAgentOptions(
            model=input_data.model,
            max_turns=input_data.max_turns,
            permission_mode=input_data.permission_mode,
        )
        if input_data.system_prompt:
            options.system_prompt = input_data.system_prompt

        # Session resumption: tell the SDK to resume from a previous session
        if input_data.resume_session_id:
            options.resume = input_data.resume_session_id

        # Shared state between event loop and heartbeat task
        heartbeat_state = {
            "event_count": 0,
            "last_event_time": time.time(),
            "done": False,
        }

        async def _heartbeat_loop():
            """Background task: sends heartbeats independent of event stream."""
            while not heartbeat_state["done"]:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                if heartbeat_state["done"]:
                    break

                idle_seconds = time.time() - heartbeat_state["last_event_time"]

                # Staleness guard
                if idle_seconds > MAX_IDLE_SECONDS:
                    activity.logger.warning(
                        f"No events for {idle_seconds:.0f}s — stopping heartbeat"
                    )
                    break

                activity.heartbeat(
                    f"events={heartbeat_state['event_count']}, idle={idle_seconds:.0f}s"
                )

        heartbeat_task = asyncio.create_task(_heartbeat_loop())

        response_text = ""
        total_tokens = 0
        event_count = 0
        session_id = None

        try:
            async for event in query(
                prompt=input_data.prompt,
                options=options,
            ):
                event_count += 1
                heartbeat_state["event_count"] = event_count
                heartbeat_state["last_event_time"] = time.time()

                # Capture session_id from the init SystemMessage
                if isinstance(event, SystemMessage):
                    if getattr(event, "subtype", None) == "init":
                        data = getattr(event, "data", None)
                        if isinstance(data, dict):
                            session_id = data.get("session_id") or session_id

                # DEDUP: Only use AssistantMessage for response text.
                # StreamEvent contains the same text as incremental chunks.
                if isinstance(event, AssistantMessage):
                    for block in event.content:
                        if hasattr(block, "text"):
                            response_text += block.text

                if isinstance(event, ResultMessage):
                    total_tokens = getattr(event, "total_tokens", 0) or 0

        finally:
            heartbeat_state["done"] = True
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        return AgentOutput(
            status="success",
            response=response_text,
            total_tokens=total_tokens,
            num_events=event_count,
            processing_time_seconds=processing_time,
            session_id=session_id,
        )

    except Exception as e:
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        return AgentOutput(
            status="error",
            error_message=str(e),
            processing_time_seconds=processing_time,
        )
```

### Result logging activity

A lightweight activity for logging (or persisting) results. In production, this would write to a database.

```python
@activity.defn
async def log_result_activity(output: AgentOutput) -> None:
    if output.status == "success":
        activity.logger.info(
            f"Agent succeeded: {len(output.response)} chars, "
            f"{output.total_tokens} tokens"
        )
    else:
        activity.logger.error(f"Agent failed: {output.error_message}")
```

## Create the Workflow

The workflow orchestrates two steps: execute the agent, then log the result. String-based activity names are used to avoid importing activity modules in deterministic workflow code.

*File: workflows/agent.py*

```python
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from models import AgentInput, AgentOutput


@workflow.defn
class AgentExecutionWorkflow:

    @workflow.run
    async def run(self, input_data: AgentInput) -> AgentOutput:

        # Step 1: Execute agent
        # - 30 min timeout: agents can run complex multi-step tasks
        # - 10 min heartbeat: activity heartbeats every 60s; 10 min buffer
        # - Staleness guard stops heartbeating after 15 min idle
        execution_retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=2),
            maximum_interval=timedelta(seconds=60),
            backoff_coefficient=2.0,
            maximum_attempts=3,
            non_retryable_error_types=["ValueError", "PermissionError"],
        )

        output: AgentOutput = await workflow.execute_activity(
            "execute_agent_activity",
            input_data,
            start_to_close_timeout=timedelta(minutes=30),
            heartbeat_timeout=timedelta(minutes=10),
            retry_policy=execution_retry_policy,
            result_type=AgentOutput,
        )

        # Step 2: Log result (in production, persist to database)
        await workflow.execute_activity(
            "log_result_activity",
            output,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=5),
        )

        return output
```

## Create the Worker

The worker hosts both the workflow and the activities.

*File: worker.py*

```python
import asyncio
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.contrib.pydantic import pydantic_data_converter
from workflows.agent import AgentExecutionWorkflow
from activities.agent_executor import execute_agent_activity, log_result_activity


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    worker = Worker(
        client,
        task_queue="claude-agent-sdk-task-queue",
        workflows=[AgentExecutionWorkflow],
        activities=[execute_agent_activity, log_result_activity],
    )

    print("Worker started, listening on claude-agent-sdk-task-queue")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
```

## Initiate an Interaction

Submit a prompt to the agent via Temporal.

*File: start_workflow.py*

```python
import asyncio
import sys
import uuid
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from workflows.agent import AgentExecutionWorkflow
from models import AgentInput


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    # Parse arguments: --resume SESSION_ID is optional
    args = sys.argv[1:]
    resume_session_id = None

    if len(args) >= 2 and args[0] == "--resume":
        resume_session_id = args[1]
        args = args[2:]

    prompt = args[0] if args else "Tell me about recursion"
    input_data = AgentInput(prompt=prompt, resume_session_id=resume_session_id)

    result = await client.execute_workflow(
        AgentExecutionWorkflow.run,
        input_data,
        id=f"claude-agent-sdk-{uuid.uuid4()}",
        task_queue="claude-agent-sdk-task-queue",
    )

    print(f"\nStatus:     {result.status}")
    print(f"Tokens:     {result.total_tokens}")
    print(f"Events:     {result.num_events}")
    print(f"Time:       {result.processing_time_seconds:.2f}s")
    if result.session_id:
        print(f"Session ID: {result.session_id}")
    print(f"\nResponse:\n{result.response}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Running the app

### Prerequisites

- [Temporal CLI](https://docs.temporal.io/cli) or a running Temporal server on `localhost:7233`
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Start Temporal

```bash
temporal server start-dev
```

### Set up the environment

```bash
cd agents/claude_agent_sdk_python
uv sync
```

### Start the worker

```bash
uv run python -m worker
```

### Send a prompt

```bash
uv run python -m start_workflow "explain how binary search works"
```

Try different prompts:

```bash
uv run python -m start_workflow "what files are in the current directory?"
uv run python -m start_workflow "write a Python function to check if a number is prime"
uv run python -m start_workflow "tell me about recursion"
```

### Resume a session

The agent returns a `session_id` in the output. Pass it back with `--resume` to continue the conversation:

```bash
# First interaction — note the session_id in the output
uv run python -m start_workflow "explain binary search"
# Output: Session ID: sess-abc-123

# Follow-up — the agent remembers the previous context
uv run python -m start_workflow --resume sess-abc-123 "now implement it in Python"
```

The SDK stores sessions as JSONL files on the worker's filesystem. When `--resume` is passed, the SDK loads the previous session and continues from where it left off, preserving the full conversation history including tool calls and results.

## Testing

The example includes tests for both the workflow (with mocked activities) and the activity (with mocked SDK events). Tests use Temporal's `WorkflowEnvironment` and `ActivityEnvironment` — no real Temporal server or Claude API calls needed.

```bash
# Install test dependencies
uv sync --extra test

# Run all tests
uv run pytest tests/ -v
```

### What the tests cover

- **`test_workflow.py`** — Workflow-level tests with mock activities:
  - Successful execution flows through both steps (execute + log)
  - Agent errors are returned as completions (not exceptions)
  - Input parameters (system_prompt, model) pass through correctly
  - Session resumption: `resume_session_id` passes through, `session_id` is returned

- **`test_activity.py`** — Activity-level tests with mock SDK events:
  - Response deduplication: `StreamEvent` text is skipped, only `AssistantMessage` is used
  - Error handling: SDK exceptions are caught and returned as `AgentOutput(status="error")`
  - Token usage captured from `ResultMessage`
  - Session ID captured from `SystemMessage(subtype='init')`
  - Resume option set on SDK options when `resume_session_id` is provided

## Comparison with Basic Agentic Loop

| Aspect | Basic Agentic Loop | Claude Agent SDK |
|--------|-------------------|-----------------|
| Agentic loop | Hand-built in workflow | SDK-managed internally |
| Tool execution | Dynamic activities | SDK handles tools |
| Streaming | No (request/response) | Yes (async event stream) |
| Heartbeats | Not needed (30s timeout) | Background task (60s interval) |
| Timeout | 30 seconds | 30 minutes |
| Staleness detection | N/A | 15-min idle guard |
| Response dedup | N/A | AssistantMessage only |
| Session resumption | Not supported | Built-in via session_id |
| Best for | Simple tool calling | Long-running autonomous agents |
