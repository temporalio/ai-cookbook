<!--
description: An agentic loop that accepts out-of-band user guidance mid-run. A steer signal folds new context into the next model call; an interrupt signal cancels the in-flight call and replaces the prompt.
tags: [agents, python, anthropic]
priority: 700
-->

# Steering a Running Agent

Once an agent starts a multi-turn task, a user often wants to nudge it ("prefer the safer plan", "actually check the latest order first") without throwing away the run and starting over. This recipe builds a steerable agent: an agentic loop you can guide while it is still thinking. Two signals carry the guidance, the model call lives in a Temporal Activity, and Temporal's durability keeps the conversation intact across the interruption.

The agent supports two kinds of steering:

- **Steer** queues guidance and folds it into the context before the next model call. The current turn finishes; the guidance shapes the turn after it.
- **Interrupt** cancels the in-flight model call, drops its partial reply, and replaces the prompt with new instructions right away.

The signal handlers only mutate state (append to a queue, set a flag). Every context change and the model call itself happen in the deterministic `run` method, so the workflow replays correctly.

This recipe steers a plain conversational loop with no tools, which is enough to teach the pattern. Steering an agent mid-tool-call adds bookkeeping (every requested tool still needs a result block) without changing the lesson.

## Create the Activity

The Activity sends the running transcript to Claude and returns the assistant's text. Because the workflow folds steering guidance into `messages` before calling, the model already sees the guidance as part of the conversation. The request is a Pydantic model so it round-trips through Temporal's Pydantic data converter.

We set `max_retries=0` on the client: Temporal owns retries through the Activity retry policy, so client-side retries would double up. Permanent API errors (auth, permission, bad request) can never succeed on retry, so we re-raise them as a non-retryable `ApplicationError` to stop Temporal from retrying; transient errors propagate and stay retryable.

The Activity also heartbeats while the request is in flight. The server only relays a cancellation request to the worker in the response to a heartbeat, so an Activity that never heartbeats can never be cancelled mid-run. With the heartbeat in place, when the workflow cancels this Activity the SDK raises `asyncio.CancelledError` into the awaited `messages.create` call, which cancels the in-flight HTTP request instead of letting it run to completion. That is what makes the interrupt cancel the model call, not just discard its result.

*File: activities/llm_call.py*

```python
async def _heartbeat_forever(interval_seconds: float) -> None:
    """Heartbeat on a fixed interval so the worker keeps polling for cancellation."""
    while True:
        activity.heartbeat()
        await asyncio.sleep(interval_seconds)


@activity.defn
async def call_llm(request: CallLlmRequest) -> str:
    # Temporal owns retries via the Activity retry policy, so disable client retries.
    client = AsyncAnthropic(max_retries=0)
    # Heartbeat well inside the heartbeat timeout so cancellation lands promptly.
    heartbeat = asyncio.create_task(_heartbeat_forever(interval_seconds=1.0))
    try:
        messages: list[MessageParam] = [
            {"role": m.role, "content": m.content} for m in request.messages
        ]
        response = await client.messages.create(
            model=request.model,
            system=request.system,
            messages=messages,
            max_tokens=request.max_tokens,
        )
        return "".join(block.text for block in response.content if block.type == "text")
    except (AuthenticationError, PermissionDeniedError, BadRequestError) as exc:
        # These can never succeed on retry, so stop Temporal from retrying them.
        raise ApplicationError(
            str(exc), type=type(exc).__name__, non_retryable=True
        ) from exc
    finally:
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat
        await client.close()
```

## Create the Workflow

The Workflow runs the agentic loop and exposes the two signal handlers. Each turn it drains any queued guidance into the conversation, then runs one model call that races against the interrupt flag. The handlers do nothing but mutate state, which is what keeps the loop deterministic.

*File: workflows/agent_steering_workflow.py*

```python
@workflow.defn
class SteerableAgentWorkflow:
    def __init__(self) -> None:
        self._pending_guidance: list[str] = []
        self._interrupt_prompt: str | None = None

    @workflow.run
    async def run(self, prompt: str, max_turns: int = 4) -> str:
        messages = [Message(role="user", content=prompt)]
        last_reply = ""

        for _ in range(max_turns):
            # Fold any guidance that arrived since the last turn into the context.
            self._drain_guidance(messages)

            reply = await self._call_with_interrupt(messages)
            if reply is None:
                # The turn was interrupted: discard the partial reply and replace
                # the prompt with the interrupt text, then start a fresh turn.
                replacement = self._interrupt_prompt
                self._interrupt_prompt = None
                messages.append(Message(role="user", content=replacement or ""))
                continue

            last_reply = reply
            messages.append(Message(role="assistant", content=reply))

            # Give any in-flight signal a chance to land before deciding to stop.
            # The timeout is a durable Temporal timer, replayed from history, not a
            # wall-clock sleep, so this stays deterministic across replays.
            try:
                await workflow.wait_condition(
                    lambda: bool(self._pending_guidance)
                    or self._interrupt_prompt is not None,
                    timeout=timedelta(seconds=1),
                )
            except asyncio.TimeoutError:
                # No further steering arrived; the task is done.
                break

        return last_reply

    async def _call_with_interrupt(self, messages: list[Message]) -> str | None:
        # heartbeat_timeout is required for the model call to be cancellable: the
        # Activity heartbeats, so a cancel reaches the worker and stops the call.
        call_task = workflow.start_activity(
            call_llm,
            CallLlmRequest(messages=messages, system=SYSTEM_PROMPT),
            start_to_close_timeout=timedelta(seconds=30),
            heartbeat_timeout=timedelta(seconds=5),
        )
        interrupt_task = asyncio.create_task(
            workflow.wait_condition(lambda: self._interrupt_prompt is not None)
        )

        await workflow.wait(
            [call_task, interrupt_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if call_task.done():
            # The model answered first; stop waiting on the interrupt flag.
            interrupt_task.cancel()
            return call_task.result()

        # Interrupt won the race: cancel the in-flight model call and drop its output.
        call_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, CancelledError, ActivityError):
            await call_task
        return None

    @workflow.signal
    async def steer(self, guidance: str) -> None:
        """Queue guidance for the loop to fold in before its next model call."""
        self._pending_guidance.append(guidance)

    @workflow.signal
    async def interrupt(self, replacement_prompt: str) -> None:
        """Request cancellation of the in-flight call and a replacement prompt."""
        self._interrupt_prompt = replacement_prompt
```

The interrupt is a race. `_call_with_interrupt` starts the model call with `workflow.start_activity` and a `wait_condition` on the interrupt flag, then waits for whichever finishes first with `workflow.wait` (the deterministic, replay-safe version of `asyncio.wait`). If the model answers first, the loop keeps its reply. If an interrupt lands first, the loop cancels the activity, swallows the resulting cancellation, returns `None`, and the run method appends the replacement prompt instead. Because the Activity heartbeats and sets a `heartbeat_timeout`, the cancel is delivered to the worker and the in-flight model call is actually cancelled, so its tokens stop being generated rather than completing unused.

## Create the Worker

The Worker registers the workflow and the Activity on the task queue. It connects with the Pydantic data converter so the `CallLlmRequest` and `Message` models serialize correctly.

*File: worker.py*

```python
async def main() -> None:
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    worker = Worker(
        client,
        task_queue="agent-steering-task-queue",
        workflows=[SteerableAgentWorkflow],
        activities=[call_llm],
    )
    print("Worker started, ctrl+c to exit.")
    await worker.run()
```

## Create the Workflow Starter

The starter kicks off the agent and immediately sends a `steer` signal, so the running agent picks up the guidance on its next model call. Sending a signal needs a workflow handle, so we use `start_workflow` (not `execute_workflow`) and wait for the result afterward.

*File: start_workflow.py*

```python
async def main() -> None:
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)

    handle = await client.start_workflow(
        SteerableAgentWorkflow.run,
        "Draft a plan to migrate our service to a new datacenter.",
        id="agent-steering-example",
        task_queue="agent-steering-task-queue",
    )

    # Nudge the running agent mid-task. The loop folds this into the next model call.
    await handle.signal(SteerableAgentWorkflow.steer, "Prefer the lowest-downtime approach.")

    result = await handle.result()
    print(f"Result: {result}")
```

## Running

```bash
temporal server start-dev
uv sync
uv run python -m worker              # terminal 1
uv run python -m start_workflow      # terminal 2
```

To steer or interrupt a running agent yourself, send signals from the CLI against the workflow ID:

```bash
temporal workflow signal --workflow-id agent-steering-example \
  --name steer --input '"Prefer the safer remediation plan."'

temporal workflow signal --workflow-id agent-steering-example \
  --name interrupt --input '"Stop that path. Check the customer'\''s latest order first."'
```
