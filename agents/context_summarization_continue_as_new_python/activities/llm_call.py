from __future__ import annotations

from typing import Any

import anthropic
from anthropic import AsyncAnthropic
from pydantic import BaseModel
from temporalio import activity
from temporalio.exceptions import ApplicationError


class CallLlmRequest(BaseModel):
    """The model-visible request: the windowed messages plus generation options."""

    model: str
    system: str
    messages: list[dict[str, Any]]
    max_tokens: int = 1024


@activity.defn
async def call_llm(request: CallLlmRequest) -> str:
    """Call Claude with the already-windowed messages and return its text reply."""
    # Temporal best practice: disable the client's own retry loop so the Activity
    # retry policy is the single source of retry behavior.
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
        # Permanent failures (bad key, malformed request) can never succeed on
        # retry, so surface them as non-retryable. Transient errors propagate
        # unchanged and the Activity retry policy handles them.
        raise ApplicationError(str(exc), type=type(exc).__name__, non_retryable=True) from exc
    finally:
        await client.close()

    return "".join(block.text for block in response.content if block.type == "text")
