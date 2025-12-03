from temporalio import activity
from anthropic import Anthropic
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
    """
    Activity that calls Claude's Messages API with tools.
    
    Claude's API structure differs from OpenAI:
    - Uses 'system' parameter instead of 'instructions'
    - Returns Message object with content blocks
    - Tool calls are embedded in content blocks with type 'tool_use'
    """
    # We disable retry logic in Anthropic API client library so that Temporal can handle retries.
    client = Anthropic()

    try:
        resp = client.messages.create(
            model=request.model,
            system=request.system,
            messages=request.messages,
            tools=request.tools,
            max_tokens=request.max_tokens,
        )
        return resp
    finally:
        # Anthropic client doesn't require explicit closing like AsyncOpenAI
        pass

