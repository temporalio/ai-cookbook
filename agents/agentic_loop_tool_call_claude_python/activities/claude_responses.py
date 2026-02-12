from temporalio import activity
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
    # In a real setting, you would need to handle any errors coming back from the Anthropic API,
    # so that Temporal can appropriately retry in the manner that Anthropic API would.
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
    finally:
        await client.close()

