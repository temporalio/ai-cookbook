import asyncio
import contextlib
from typing import Literal

from anthropic import (
    AsyncAnthropic,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
)
from anthropic.types import MessageParam
from pydantic import BaseModel
from temporalio import activity
from temporalio.exceptions import ApplicationError


class Message(BaseModel):
    """One turn of the conversation in Anthropic's message shape."""

    role: Literal["user", "assistant"]
    content: str


class CallLlmRequest(BaseModel):
    """Everything the model needs for one turn: the running transcript plus config."""

    messages: list[Message]
    system: str
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024


async def _heartbeat_forever(interval_seconds: float) -> None:
    """Heartbeat on a fixed interval so the worker keeps polling for cancellation.

    The server only relays a cancellation request to the worker in the response to a
    heartbeat, so an Activity that never heartbeats can never be cancelled mid-run.
    """
    while True:
        activity.heartbeat()
        await asyncio.sleep(interval_seconds)


@activity.defn
async def call_llm(request: CallLlmRequest) -> str:
    """Send the conversation to Claude and return the assistant's text reply.

    The whole transcript is sent every turn, so any steering guidance the workflow
    folded into ``messages`` is already part of the context the model sees.

    A background task heartbeats while the request is in flight. When the workflow
    cancels this Activity, the SDK raises ``asyncio.CancelledError`` into the awaited
    ``messages.create`` call, which cancels the in-flight HTTP request rather than
    letting it run to completion.
    """
    # Temporal owns retries via the Activity retry policy, so disable client retries.
    client = AsyncAnthropic(max_retries=0)
    # Heartbeat well inside the heartbeat timeout so cancellation lands promptly.
    heartbeat = asyncio.create_task(_heartbeat_forever(interval_seconds=1.0))
    try:
        messages: list[MessageParam] = [{"role": m.role, "content": m.content} for m in request.messages]
        response = await client.messages.create(
            model=request.model,
            system=request.system,
            messages=messages,
            max_tokens=request.max_tokens,
        )
        return "".join(block.text for block in response.content if block.type == "text")
    except (AuthenticationError, PermissionDeniedError, BadRequestError) as exc:
        # These can never succeed on retry, so stop Temporal from retrying them.
        raise ApplicationError(str(exc), type=type(exc).__name__, non_retryable=True) from exc
    finally:
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat
        await client.close()
