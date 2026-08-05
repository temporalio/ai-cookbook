from temporalio import activity
from temporalio.exceptions import ApplicationError
import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import Message
from dataclasses import dataclass
from typing import Any

# Temporal best practice: Create a data structure to hold the request parameters.
@dataclass
class ClaudeResponsesRequest:
    model: str
    system: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    max_tokens: int = 4096

@activity.defn
async def create(request: ClaudeResponsesRequest) -> Message:
    # We disable retry logic in Anthropic API client library so that Temporal can handle retries.
    client = AsyncAnthropic(max_retries=0)

    try:
        resp = await client.messages.create(
            model=request.model,
            system=request.system,
            messages=request.messages,
            tools=request.tools,
            max_tokens=request.max_tokens,
        )
        return resp
    except (
        anthropic.BadRequestError,
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.NotFoundError,
        anthropic.UnprocessableEntityError,
    ) as exc:
        # These errors will never succeed on a retry: a retired model identifier, for
        # example, returns 404. Retrying them under the default policy would loop
        # forever instead of surfacing the problem. Everything else, such as rate
        # limits and 5xx responses, propagates so Temporal can retry it.
        raise ApplicationError(
            str(exc),
            type=exc.__class__.__name__,
            non_retryable=True,
        ) from exc
    finally:
        await client.close()

