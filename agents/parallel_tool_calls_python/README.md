<!--
description: When the model requests several tools in one turn, the workflow runs them concurrently and returns all results together on the next turn.
tags: [agents, python, anthropic]
priority: 720
-->

# Parallel Tool Calls in an Agentic Loop

Claude can request several tools in a single turn. Running them one after another makes the turn as slow as the sum of its calls, even though the calls are independent. This recipe fans the tools out concurrently with `asyncio.gather` inside the workflow, so the turn takes as long as the slowest single tool, while still returning exactly one `tool_result` block per requested tool so the provider contract holds. Each tool runs in its own Temporal Activity, so each is retried and made durable independently.

## Create the tool Activities

The agent has two demo tools: `get_weather` and `get_time`. Each takes a typed Pydantic request and returns a deterministic string with no network I/O, so the recipe (and its tests) run without external services. In a real recipe each tool would call a separate API; here they stay independent and side-effect-light so the concurrency is the point, not the payload.

*File: activities/tools.py*

```python
class GetWeatherRequest(BaseModel):
    city: str = Field(description="City name, e.g. 'Seattle'.")


class GetTimeRequest(BaseModel):
    timezone: str = Field(description="IANA timezone name, e.g. 'America/Los_Angeles'.")


@activity.defn
async def get_weather(request: GetWeatherRequest) -> str:
    return _WEATHER_BY_CITY.get(request.city.lower(), "clear, 65F")


@activity.defn
async def get_time(request: GetTimeRequest) -> str:
    return _TIME_BY_ZONE.get(request.timezone.lower(), "12:00")
```

Keeping each tool in its own Activity is what makes the fan-out durable: Temporal schedules, retries, and records each call separately.

## Create the tool registry

The registry declares the tools once. The workflow uses it both to build the Claude tool definitions (Claude wants an `input_schema` JSON Schema per tool) and to map a requested tool name back to the Activity and request model that run it. Deriving the schema from each tool's Pydantic model keeps the schema and the Activity signature in sync.

*File: tools/registry.py*

```python
TOOLS: list[ToolSpec] = [
    ToolSpec("get_weather", "Get the current weather for a city.", get_weather, GetWeatherRequest),
    ToolSpec("get_time", "Get the current local time for an IANA timezone.", get_time, GetTimeRequest),
]


def get_tool(name: str) -> ToolSpec:
    return _TOOLS_BY_NAME[name]


def claude_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.request_model.model_json_schema(),
        }
        for spec in TOOLS
    ]
```

## Create the LLM Activity

A generic wrapper over the Anthropic Messages API. The workflow passes the model, system prompt, messages, and tool definitions, so the workflow controls behavior without re-registering the Activity.

*File: activities/llm_call.py*

```python
@activity.defn
async def call_llm(request: CallLlmRequest) -> Message:
    # Disable client-side retries so Temporal owns retry behavior via the Activity
    # retry policy. Client retries would double-retry and fight Temporal's backoff.
    client = AsyncAnthropic(max_retries=0)
    try:
        return await client.messages.create(
            model=request.model,
            system=request.system,
            messages=request.messages,
            tools=request.tools,
            max_tokens=request.max_tokens,
        )
    except (
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.BadRequestError,
    ) as exc:
        raise ApplicationError(
            str(exc), type=type(exc).__name__, non_retryable=True
        ) from exc
    finally:
        await client.close()
```

`max_retries=0` hands retries to Temporal. Permanent errors (auth, permission, bad request) re-raise as a non-retryable `ApplicationError` so Temporal stops; transient errors (rate limits, 5xx) propagate and let the Activity retry policy retry them.

## Create the Workflow

The workflow runs the tool-calling loop. It calls `call_llm`, collects the `tool_use` blocks, and when the model requests tools it schedules one `asyncio` task per block, then awaits them together with `asyncio.gather`. The results go back as a single user turn, and the loop repeats until the model returns a text answer.

*File: workflows/recipe_workflow.py*

```python
@workflow.defn
class ParallelToolAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

        while True:
            message = await workflow.execute_activity(
                call_llm,
                CallLlmRequest(
                    model=MODEL,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    tools=claude_tool_definitions(),
                ),
                start_to_close_timeout=timedelta(seconds=30),
            )

            tool_use_blocks = [b for b in message.content if b.type == "tool_use"]

            if not tool_use_blocks:
                text_blocks = [b.text for b in message.content if b.type == "text"]
                return "\n".join(text_blocks)

            messages.append({"role": "assistant", "content": _serialize(message.content)})

            tool_results = await asyncio.gather(
                *(self._run_tool(block) for block in tool_use_blocks)
            )

            messages.append({"role": "user", "content": tool_results})

    async def _run_tool(self, block: "ToolUseBlock") -> dict[str, Any]:
        spec = get_tool(block.name)
        result = await workflow.execute_activity(
            spec.activity,
            spec.request_model(**block.input),
            start_to_close_timeout=timedelta(seconds=30),
        )
        return {"type": "tool_result", "tool_use_id": block.id, "content": str(result)}
```

`asyncio.gather` is the whole pattern. It starts every tool Activity, lets them run concurrently, and returns their results in the order the tasks were created. That ordering is deterministic under Temporal regardless of which Activity finishes first, so the workflow replays identically and Claude gets one `tool_result` per requested tool in the same order it asked. Each `execute_activity` sets `start_to_close_timeout`, and because each tool is its own Activity, a failure in one retries on its own without re-running the others.

This recipe keeps the loop deliberately simple. A production agent might also support interrupting or cancelling tools mid-flight (for example with `workflow.wait_condition`); that machinery is omitted here so `asyncio.gather` stays the clear teaching shape.

## Create the Worker

The worker registers the workflow and all three activities. The `pydantic_data_converter` lets Pydantic models (the request types and the Anthropic `Message`) round-trip as workflow and activity payloads.

*File: worker.py*

```python
async def main() -> None:
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    worker = Worker(
        client,
        task_queue="parallel-tool-calls-task-queue",
        workflows=[ParallelToolAgentWorkflow],
        activities=[call_llm, get_weather, get_time],
    )
    await worker.run()
```

## Create the Workflow Starter

The starter connects with the same converter and submits a prompt that needs both tools, so the model requests them together and the workflow fans them out.

*File: start_workflow.py*

```python
async def main() -> None:
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    result = await client.execute_workflow(
        ParallelToolAgentWorkflow.run,
        "What is the weather in Seattle and the current time in America/Los_Angeles?",
        id="parallel-tool-calls-example",
        task_queue="parallel-tool-calls-task-queue",
    )
    print(f"Result: {result}")
```

## Running

```bash
temporal server start-dev
export ANTHROPIC_API_KEY=sk-ant-...
uv sync
uv run python -m worker              # terminal 1
uv run python -m start_workflow      # terminal 2
```
