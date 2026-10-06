<!--
description: An agentic loop that compacts its conversation history and continues-as-new when workflow history grows large, carrying a bounded context snapshot into the next run.
tags: [agents, python, anthropic]
priority: 450
-->

# Context Summarization with Continue-as-New

A long-running agent accumulates an ever-growing message history. Two things break: the
model's context window overflows, and the Temporal workflow history grows without bound.
This recipe keeps both bounded. Before every model call a deterministic context window trims
the conversation to a token budget, and when the workflow history gets large the agent
compacts its context and continues-as-new, carrying a small snapshot into a fresh run. The
model call happens in a Temporal Activity, so retries and durability are handled for you.

## Key design decisions

- The **context window** is pure, deterministic workflow code (no clocks, no I/O), so it can
  run inside the workflow and replay safely.
- The full conversation lives in workflow state; only a **windowed view** is sent to the
  model each turn.
- Continue-as-new carries a **compacted snapshot** as a normal workflow payload, so the next
  run resumes with a clean, bounded history instead of replaying everything.

## Create the Activity

The Activity is the only place that touches the network. It takes the already-windowed
messages and returns Claude's text reply. The client is built with `max_retries=0` so
Temporal's Activity retry policy is the single source of retry behavior, and permanent API
errors are re-raised as a non-retryable `ApplicationError` so Temporal stops retrying a call
that can never succeed.

*File: activities/llm_call.py*

```python
@activity.defn
async def call_llm(request: CallLlmRequest) -> str:
    client = AsyncAnthropic(max_retries=0)
    try:
        response = await client.messages.create(
            model=request.model,
            system=request.system,
            messages=request.messages,
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

    return "".join(block.text for block in response.content if block.type == "text")
```

## Create the context window

This is the heart of the recipe. `window_messages` decides what the model actually sees: it
keeps at most `max_recent` messages, pins the initial user message (it usually states the
task), and then sheds the oldest kept turns until the estimate fits the token budget. It
keeps the user/assistant turns alternating and starting with a user turn, so the result is a
valid provider request, and it reports how many messages were dropped so the caller can note
the compaction. Because it runs in workflow code, token cost is estimated from character
length rather than a tokenizer call.

*File: helpers/context_window.py*

```python
def window_messages(
    messages: list[dict[str, Any]],
    *,
    max_recent: int,
    max_context_tokens: int,
    preserve_initial: bool = True,
    chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
) -> tuple[list[dict[str, Any]], int]:
    if not messages:
        return [], 0
    within_count = len(messages) <= max_recent
    within_budget = estimate_tokens(messages, chars_per_token) <= max_context_tokens
    if within_count and within_budget:
        return list(messages), 0

    head = [messages[0]] if preserve_initial else []
    tail_start = max(len(head), len(messages) - (max_recent - len(head)))
    tail = messages[tail_start:]
    tail = _trim_front_for_valid_request(head, tail)

    while tail and estimate_tokens(head + tail, chars_per_token) > max_context_tokens:
        tail = _trim_front_for_valid_request(head, tail[1:])

    selected = head + tail
    return selected, len(messages) - len(selected)
```

When any turns are dropped, the workflow appends a short note to the system prompt so the
model knows earlier context was summarized away.

## Create the Workflow

The workflow runs the agentic loop: record the user turn, window the conversation, call the
Activity, record the reply. The full history stays in `messages`; only the windowed view is
sent to the model. When the run should hand off (Temporal suggests continue-as-new, or a
configured turn threshold is hit) and prompts remain, it compacts the context and calls
`workflow.continue_as_new` with a snapshot. The next run restores that snapshot and keeps
going with a clean history.

*File: workflows/recipe_workflow.py*

```python
while remaining:
    messages.append({"role": "user", "content": remaining.pop(0)})

    model_messages, dropped = window_messages(
        messages,
        max_recent=config.max_recent_messages,
        max_context_tokens=config.max_context_tokens,
    )
    system = config.system
    if dropped:
        system = f"{config.system}\n\n[Context note] {COMPACTION_NOTE}"

    reply = await workflow.execute_activity(
        call_llm,
        CallLlmRequest(model=config.model, system=system, messages=model_messages,
                       max_tokens=config.max_tokens),
        start_to_close_timeout=timedelta(seconds=30),
    )
    messages.append({"role": "assistant", "content": reply})
    turns_completed += 1

    if remaining and self._should_continue_as_new(turns_this_run, config):
        compacted, _ = window_messages(
            messages,
            max_recent=config.max_recent_messages,
            max_context_tokens=config.max_context_tokens,
        )
        workflow.continue_as_new(
            AgentInput(
                prompts=remaining,
                config=config,
                agent_state=AgentState(
                    messages=compacted,
                    turns_completed=turns_completed,
                    compactions=compactions + 1,
                ),
            )
        )
```

`_should_continue_as_new` returns `True` when `workflow.info().is_continue_as_new_suggested()`
fires (the production trigger) or when the optional `continue_as_new_after_turns` threshold is
reached, which makes the handoff easy to demonstrate. The snapshot is a plain Pydantic model,
so it serializes through `pydantic_data_converter` like any other workflow payload.

## Create the Worker

The Worker registers the workflow and the Activity, and connects with the Pydantic data
converter so the request and snapshot models round-trip.

*File: worker.py*

```python
async def main() -> None:
    client = await Client.connect(
        "localhost:7233", data_converter=pydantic_data_converter
    )
    worker = Worker(
        client,
        task_queue="context-summarization-continue-as-new-task-queue",
        workflows=[SummarizingAgentWorkflow],
        activities=[call_llm],
    )
    print("Worker started, ctrl+c to exit.")
    await worker.run()
```

## Create the Workflow Starter

The starter submits a multi-turn conversation. It sets `continue_as_new_after_turns=2` so you
can watch the handoffs in the Temporal UI; leave it at `0` in production and let Temporal
decide when history is large enough.

*File: start_workflow.py*

```python
result = await client.execute_workflow(
    SummarizingAgentWorkflow.run,
    AgentInput(
        prompts=[
            "I'm planning a 3-day trip to Tokyo. Suggest a theme for each day.",
            "Expand day 1 into a morning, afternoon, and evening plan.",
            "What should I pack given those plans?",
            "Summarize the whole trip in three sentences.",
        ],
        config=SummarizationConfig(continue_as_new_after_turns=2),
    ),
    id="context-summarization-continue-as-new-example",
    task_queue="context-summarization-continue-as-new-task-queue",
)
```

## Running

```bash
temporal server start-dev
uv sync
export ANTHROPIC_API_KEY=sk-ant-...  # the Activity reads this from the environment
uv run python -m worker              # terminal 1
uv run python -m start_workflow      # terminal 2
```

Run the tests (no API key needed; the LLM call is mocked) with:

```bash
uv run pytest tests/
```
