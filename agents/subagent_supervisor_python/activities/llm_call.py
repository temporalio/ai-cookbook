from typing import Any

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import Message
from pydantic import BaseModel
from temporalio import activity
from temporalio.exceptions import ApplicationError


# Temporal best practice: hold the request parameters in a data structure so the
# workflow controls the model, system prompt, and toolset without re-registering the Activity.
class CallLlmRequest(BaseModel):
    model: str
    system: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    max_tokens: int = 1024


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
