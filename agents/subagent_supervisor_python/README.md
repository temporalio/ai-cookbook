<!--
description: A supervisor agent delegates a focused task to a child-workflow subagent through a tool call, then folds the subagent's result back into its own loop.
tags: [agents, python, anthropic]
priority: 400
-->

# Supervisor Agent with Subagent Delegation

A single agent loop with one giant toolset and one growing context is hard to steer and expensive: every turn re-reasons over every tool. This recipe keeps each agent's context small by giving a supervisor a single `delegate_to_subagent` tool that hands a focused sub-task to a child-workflow subagent with a restricted toolset. The child workflow is the durable boundary, so a crash mid-delegation does not lose the subagent's work, and the supervisor always has a reliable handle to its result. If a subagent fails outright, the supervisor catches that failure and feeds it back to the model as a tool error rather than crashing. Every LLM call happens in a Temporal Activity, so Temporal owns retries and durability.

## Application Components

- A shared `call_llm` Activity wraps the Anthropic Messages API.
- `SupervisorAgentWorkflow` runs an agentic loop whose only tool is `delegate_to_subagent`.
- `SubagentWorkflow` runs a smaller bounded loop with a restricted toolset and the delegate tool removed, so it cannot recurse.
- Both workflows share one task queue and the one Activity.

## Create the Tools

The supervisor gets one tool: `delegate_to_subagent`. The subagent gets a tiny set of trivial, deterministic string tools. Because those handlers take a string and return a string with no I/O, the subagent runs them inline; a real subagent would call them through Activities.

*File: tools/registry.py*

```python
DELEGATE_TO_SUBAGENT = "delegate_to_subagent"


def word_count(text: str) -> str:
    return str(len(text.split()))


def to_upper(text: str) -> str:
    return text.upper()


SUBAGENT_TOOLS: dict[str, Callable[[str], str]] = {
    "word_count": word_count,
    "to_upper": to_upper,
}


def subagent_tool_schemas(tool_names: list[str]) -> list[dict[str, Any]]:
    # Only expose the granted tools to the subagent. The delegate tool is never
    # in this set, so a subagent cannot recursively spawn more subagents.
    schemas: list[dict[str, Any]] = []
    for name in tool_names:
        if name in SUBAGENT_TOOLS:
            schemas.append(
                {
                    "name": name,
                    "description": f"Apply {name} to a string.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                }
            )
    return schemas
```

`subagent_tool_schemas` is what enforces scoping: the supervisor passes a list of tool names when it delegates, and only those tools are ever shown to the subagent. The delegate tool is not in `SUBAGENT_TOOLS`, so it can never leak into a subagent and cause recursion.

## Create the Activity

`call_llm` wraps the Anthropic Messages API. It is generic: the workflow passes the model, system prompt, messages, and toolset, so the same Activity serves both the supervisor and the subagent.

*File: activities/llm_call.py*

```python
@activity.defn
async def call_llm(request: CallLlmRequest) -> Message:
    # Disable client-side retries so Temporal's Activity retry policy owns retries.
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
        # These can never succeed on retry, so mark them non-retryable and stop Temporal retrying.
        raise ApplicationError(str(exc), type=type(exc).__name__, non_retryable=True) from exc
    finally:
        await client.close()
```

We set `max_retries=0` so the Anthropic client does not retry; that is Temporal's job via the Activity retry policy. Permanent errors (auth, permission, bad request) re-raise as a non-retryable `ApplicationError` so Temporal stops retrying a call that can never succeed. Transient errors propagate and stay retryable.

## Create the Workflows

Both workflows run the same agentic-loop shape: call the model, run any tool calls, append the results, repeat until the model answers with text. The difference is the toolset and what a tool call does.

`SubagentWorkflow` is the smaller loop. It only ever sees the tools it was granted, and it runs them inline because they are pure string functions.

*File: workflows/recipe_workflow.py*

```python
@workflow.defn
class SubagentWorkflow:
    @workflow.run
    async def run(self, request: SubagentRequest) -> SubagentResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": request.task}]
        tools = subagent_tool_schemas(request.tool_names)

        for _ in range(MAX_TURNS):
            message = await workflow.execute_activity(
                call_llm,
                CallLlmRequest(
                    model=MODEL, system=SUBAGENT_SYSTEM, messages=messages, tools=tools
                ),
                start_to_close_timeout=timedelta(seconds=30),
            )
            tool_uses = [b for b in message.content if b.type == "tool_use"]
            if not tool_uses:
                return SubagentResult(result=_final_text(message))

            messages.append({"role": "assistant", "content": _assistant_content(message)})
            results: list[dict[str, Any]] = []
            for block in tool_uses:
                handler = SUBAGENT_TOOLS[block.name]
                output = handler(block.input["text"])
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
            messages.append({"role": "user", "content": results})

        return SubagentResult(result="Subagent stopped: turn limit reached.")
```

`SupervisorAgentWorkflow` runs the same loop with one tool, `delegate_to_subagent`. When the model calls it, `_delegate` starts a child workflow and the child's result comes back as a `tool_result` block.

*File: workflows/recipe_workflow.py*

```python
@workflow.defn
class SupervisorAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        tools = [delegate_tool_schema()]

        for _ in range(MAX_TURNS):
            message = await workflow.execute_activity(
                call_llm,
                CallLlmRequest(
                    model=MODEL, system=SUPERVISOR_SYSTEM, messages=messages, tools=tools
                ),
                start_to_close_timeout=timedelta(seconds=30),
            )
            tool_uses = [b for b in message.content if b.type == "tool_use"]
            if not tool_uses:
                return _final_text(message)

            messages.append({"role": "assistant", "content": _assistant_content(message)})
            results: list[dict[str, Any]] = []
            for block in tool_uses:
                output = await self._delegate(block.input)
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
            messages.append({"role": "user", "content": results})

        return "Supervisor stopped: turn limit reached."

    async def _delegate(self, tool_input: dict[str, Any]) -> str:
        granted = [n for n in tool_input.get("tool_names", []) if n != DELEGATE_TO_SUBAGENT]
        child_id = f"{workflow.info().workflow_id}-subagent-{workflow.uuid4()}"
        try:
            result = await workflow.execute_child_workflow(
                SubagentWorkflow.run,
                SubagentRequest(task=tool_input["task"], tool_names=granted),
                id=child_id,
                task_queue=workflow.info().task_queue,
                static_summary="delegate_to_subagent:run",
            )
        except ChildWorkflowError as exc:
            return f"Subagent failed: {exc.cause or exc}"
        return result.result
```

`_delegate` is where the two-level topology comes together. It strips the delegate tool from the granted set so a subagent can never recurse, derives a deterministic child id from the parent id plus `workflow.uuid4()` (deterministic inside a workflow, and traceable in history), and sets a `static_summary` so the child shows up cleanly in Temporal's UI. `workflow.execute_child_workflow` is the durable boundary: the child runs to completion as its own workflow, and its `SubagentResult` flows back into the supervisor's loop as a tool result.

The child gets no `RetryPolicy`. Workflows do not retry by default, and that is what we want here; the subagent's retries belong to its `call_llm` Activity. If the subagent does fail outright (for example, its Activity exhausts retries on a permanent error), Temporal raises a `ChildWorkflowError` in the supervisor, with the underlying cause in `exc.cause`. We catch it and return the failure as the tool's result string, so a failed delegation reaches the model as a tool error it can react to instead of crashing the supervisor.

## Create the Worker

The Worker registers both workflows and the single shared Activity on one task queue.

*File: worker.py*

```python
worker = Worker(
    client,
    task_queue="subagent-supervisor-task-queue",
    # Both workflows share one task queue and the single call_llm Activity.
    workflows=[SupervisorAgentWorkflow, SubagentWorkflow],
    activities=[call_llm],
)
```

The client connects with `pydantic_data_converter` so the Pydantic request and result models round-trip as workflow and activity payloads. The child workflow is dispatched onto the same task queue, so this one Worker picks up both the supervisor and its subagents.

## Create the Workflow Starter

The starter submits a supervisor run with a prompt that invites delegation.

*File: start_workflow.py*

```python
result = await client.execute_workflow(
    SupervisorAgentWorkflow.run,
    "How many words are in the phrase 'durable execution is delightful'? "
    "Delegate the counting to a subagent.",
    id="subagent-supervisor-example",
    task_queue="subagent-supervisor-task-queue",
)
print(f"Result: {result}")
```

The starter also connects with `pydantic_data_converter` so its payloads match the Worker's.

## Running

```bash
temporal server start-dev
uv sync
export ANTHROPIC_API_KEY=...           # required to run live; tests do not need it
uv run python -m worker                # terminal 1
uv run python -m start_workflow        # terminal 2
```

Run the tests (no API key needed; the `call_llm` Activity is mocked):

```bash
uv run pytest tests/ --timeout=30
```
