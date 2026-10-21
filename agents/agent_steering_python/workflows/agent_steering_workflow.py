import asyncio
import contextlib
from datetime import timedelta

from temporalio import workflow
from temporalio.exceptions import ActivityError, CancelledError

with workflow.unsafe.imports_passed_through():
    from activities.llm_call import CallLlmRequest, Message, call_llm

SYSTEM_PROMPT = (
    "You are a helpful agent working through a multi-step task. "
    "When the user sends new guidance mid-task, incorporate it into your next step."
)


@workflow.defn
class SteerableAgentWorkflow:
    """An agentic loop that takes user guidance while it is still running.

    Two signals drive the steering. ``steer`` queues guidance that the loop folds
    into the context before the next model call. ``interrupt`` cancels the in-flight
    model call, drops its partial response, and replaces the prompt. Both handlers
    only mutate state; every context change and the model call itself happen in the
    deterministic ``run`` method so replay stays correct.
    """

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
            try:
                await workflow.wait_condition(
                    lambda: bool(self._pending_guidance) or self._interrupt_prompt is not None,
                    timeout=timedelta(seconds=1),
                )
            except asyncio.TimeoutError:
                # No further steering arrived; the task is done.
                break

        return last_reply

    async def _call_with_interrupt(self, messages: list[Message]) -> str | None:
        """Run one model call, racing it against the interrupt flag.

        Returns the reply text, or ``None`` if an interrupt won the race and the
        in-flight call was cancelled. The Activity sets a ``heartbeat_timeout`` and
        heartbeats while the request is in flight, so ``cancel`` is delivered to the
        worker and genuinely cancels the model call (not merely abandons its result).
        """
        call_task = workflow.start_activity(
            call_llm,
            CallLlmRequest(messages=messages, system=SYSTEM_PROMPT),
            start_to_close_timeout=timedelta(seconds=30),
            heartbeat_timeout=timedelta(seconds=5),
        )
        interrupt_task = asyncio.create_task(workflow.wait_condition(lambda: self._interrupt_prompt is not None))

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

    def _drain_guidance(self, messages: list[Message]) -> None:
        while self._pending_guidance:
            guidance = self._pending_guidance.pop(0)
            messages.append(Message(role="user", content=guidance))

    @workflow.signal
    async def steer(self, guidance: str) -> None:
        """Queue guidance for the loop to fold in before its next model call."""
        self._pending_guidance.append(guidance)

    @workflow.signal
    async def interrupt(self, replacement_prompt: str) -> None:
        """Request cancellation of the in-flight call and a replacement prompt."""
        self._interrupt_prompt = replacement_prompt
