<!--
description: Stream a durable OpenAI Agents SDK tool-calling agent's progress to external subscribers in real time using Temporal Workflow Streams.
tags: [agents, python, openai]
priority: 740
-->

# Streaming Durable Agent with Tools - OpenAI Agents SDK + Workflow Streams

This recipe extends [`agents/openai_agents_sdk_python`](../openai_agents_sdk_python/) with real-time observability: instead of blocking on `client.execute_workflow(...)` for a single final answer, it uses [Workflow Streams](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/workflow_streams) so an external process can watch the agent's tool-calling loop as it happens — text tokens streaming in, which tool the model decided to call, and that tool's result — all while the workflow itself stays durable.

> **⚠️ Public Preview.** Both `temporalio.contrib.workflow_streams` and the streaming support in `temporalio.contrib.openai_agents` (`Runner.run_streamed`) are experimental and may change before General Availability. Don't build production systems on this API shape yet.

This recipe highlights:

- **`Runner.run_streamed` inside a durable workflow**: the same agentic tool-calling loop as the base recipe, but the model activity streams its raw response events instead of returning only a final result.
- **`WorkflowStream`**: the workflow hosts a durable, offset-addressed event log. The streaming model activity publishes every native OpenAI response-stream event to it as they're produced — not just at the end.
- **`WorkflowStreamClient`**: a separate process (`subscribe.py`) subscribes to that log from outside the workflow, entirely decoupled from the worker.
- **Two-terminal pattern**: `start_workflow.py` kicks the agent off and returns immediately; `subscribe.py` (run separately, even on a different machine) watches its progress.

## Create the Activities

The tool activities are the same as the base recipe (`get_weather`, `calculate_circle_area`). Each one also publishes to a second topic, `"tool-events"`, using the same Workflow Streams mechanism — demonstrating that *any* activity can publish to a stream, not just the model activity. This is what lets a subscriber print a tool's actual result, since the model's own raw stream only carries the *request* to call a tool (its name and arguments), not what the tool returned.

*File: activities/tools.py*

```python
from contextlib import asynccontextmanager
from dataclasses import dataclass

import math

from temporalio import activity
from temporalio.contrib.workflow_streams import WorkflowStreamClient

MODEL_EVENTS_TOPIC = "model-events"
TOOL_EVENTS_TOPIC = "tool-events"


@dataclass
class Weather:
    city: str
    temperature_range: str
    conditions: str


@asynccontextmanager
async def _tool_events():
    """Best-effort publisher for the tool-events topic. Falls back to a
    no-op outside of a live activity-with-client context (e.g. unit tests)."""
    try:
        stream = WorkflowStreamClient.from_within_activity()
    except RuntimeError:
        yield None
        return
    async with stream:
        yield stream.topic(TOOL_EVENTS_TOPIC)


@activity.defn
async def get_weather(city: str) -> Weather:
    """Get the weather for a given city."""
    async with _tool_events() as tool_events:
        if tool_events:
            tool_events.publish(f"-> calling get_weather(city={city!r})")
        result = Weather(city=city, temperature_range="14-20C", conditions="Sunny with wind.")
        if tool_events:
            tool_events.publish(f"<- get_weather returned: {result}")
    return result


@activity.defn
async def calculate_circle_area(radius: float) -> float:
    """Calculate the area of a circle given its radius."""
    async with _tool_events() as tool_events:
        if tool_events:
            tool_events.publish(f"-> calling calculate_circle_area(radius={radius})")
        result = math.pi * radius**2
        if tool_events:
            tool_events.publish(f"<- calculate_circle_area returned: {result}")
    return result
```

## Create the Workflow

`WorkflowStream` must be constructed from `@workflow.init` — its constructor registers the signal/update/query handlers that the streaming model activity publishes into. `@workflow.init`'s parameters must match `@workflow.run`'s, so `__init__` takes `prompt` too even though it only needs it to satisfy that constraint.

The agent code itself is otherwise identical to the base recipe, swapping `Runner.run` for `Runner.run_streamed`.

*File: workflows/streaming_agent_workflow.py*

```python
from datetime import timedelta

from agents import Agent, Runner
from temporalio import workflow
from temporalio.contrib import openai_agents
from temporalio.contrib.workflow_streams import WorkflowStream

from activities.tools import calculate_circle_area, get_weather


@workflow.defn
class StreamingAgent:
    @workflow.init
    def __init__(self, prompt: str) -> None:
        self._stream = WorkflowStream()

    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(
            name="Streaming Hello World Agent",
            instructions="You are a helpful assistant that determines what tool to use based on the user's question.",
            tools=[
                openai_agents.workflow.activity_as_tool(
                    get_weather,
                    start_to_close_timeout=timedelta(seconds=10)
                ),
                openai_agents.workflow.activity_as_tool(
                    calculate_circle_area,
                    start_to_close_timeout=timedelta(seconds=10)
                )
            ]
        )

        result = Runner.run_streamed(agent, prompt)
        async for _ in result.stream_events():
            pass  # events are already published to "model-events" by the streaming activity
        return result.final_output
```

## Create the Worker

`ModelActivityParameters.streaming_topic` is what turns on streaming: without it, `Runner.run_streamed` raises before scheduling any activity.

*File: worker.py*

```python
import asyncio
from datetime import timedelta

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.contrib.openai_agents import OpenAIAgentsPlugin, ModelActivityParameters

from workflows.streaming_agent_workflow import StreamingAgent
from activities.tools import MODEL_EVENTS_TOPIC, get_weather, calculate_circle_area

async def worker_main():
    client = await Client.connect(
        "localhost:7233",
        plugins=[
            OpenAIAgentsPlugin(
                model_params=ModelActivityParameters(
                    start_to_close_timeout=timedelta(seconds=30),
                    streaming_topic=MODEL_EVENTS_TOPIC,
                )
            ),
        ],
    )

    worker = Worker(
        client,
        task_queue="streaming-openai-agent-task-queue",
        workflows=[StreamingAgent],
        activities=[get_weather, calculate_circle_area],
    )
    await worker.run()

if __name__ == "__main__":
    asyncio.run(worker_main())
```

## Start the Workflow

Unlike the base recipe, `start_workflow.py` calls `client.start_workflow(...)` and returns immediately — it doesn't block waiting for a result, since the whole point is to watch progress from a second process.

*File: start_workflow.py* — connects, starts `StreamingAgent`, and prints the workflow ID for `subscribe.py`.

## Subscribe to the Stream

*File: subscribe.py* — connects a `WorkflowStreamClient` to the running workflow and subscribes to both `"model-events"` (raw OpenAI stream events, decoded as plain dicts since neither side uses `pydantic_data_converter`) and `"tool-events"` (the strings published by the tool activities above):

- `response.output_text.delta` events print incrementally, giving live streaming text.
- `response.output_item.done` events for a `function_call` item print `-> model requested tool call: name(args)`.
- `tool-events` items print the tool's actual call and result as the activity reports them.

## Running

Start the Temporal Dev Server:

```bash
temporal server start-dev
```

Set an OpenAI API key:

```bash
export OPENAI_API_KEY=sk...
```

Run the worker (terminal 1):

```bash
uv run python -m worker
```

Start the agent (terminal 2) — it prints a workflow ID and exits immediately:

```bash
uv run python -m start_workflow
```

Subscribe to it (terminal 2 or a third terminal), using the printed workflow ID:

```bash
uv run python -m subscribe <workflow-id>
```

## Example Interactions

- "What's the weather in London?"
- "Calculate the area of a circle with radius 5"
- "What's the weather in Tokyo and calculate the area of a circle with radius 3"

## Durability and Retries

Workflow Streams' delivery semantics matter here: if the streaming model activity retries mid-response (e.g. a transient OpenAI API failure after some tokens already streamed), the retried attempt is a *new publisher* on the same topic. Events already published by the failed attempt stay on the stream — they aren't rolled back — and the retry's events are appended after them as a second sequence. `RunResultStreaming.stream_events()` inside the workflow doesn't see this at all, since it only reads the activity's final return value (the last successful attempt's collected events). Subscribers watching the raw `"model-events"` topic, however, see both attempts' events back to back.

A subscriber that wants to reconstruct a clean transcript across a retry needs to detect the transition and discard (or clearly mark) the partial output from the failed attempt — this recipe's `subscribe.py` does not do this (it just prints everything it sees) since demonstrating that reconciliation logic is outside this recipe's scope. See [Workflow Streams' delivery semantics](https://docs.temporal.io/develop/python/workflows/workflow-streams) for the documented `RETRY` event convention some subscribers use to signal this transition explicitly.
