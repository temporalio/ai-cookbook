from temporalio import activity
from openai import AsyncOpenAI
from openai.types.responses import Response
from dataclasses import dataclass
from typing import Any

# Temporal best practice: Create a data structure to hold the request parameters.
@dataclass
class OpenAIResponsesRequest:
    model: str
    instructions: str
    input: object
    tools: list[dict[str, Any]]

@activity.defn
async def create(request: OpenAIResponsesRequest) -> Response:
    # We disable retry logic in OpenAI API client library so that Temporal can handle retries.
    # In a real setting, you would need to handle any errors coming back from the OpenAI API,
    # so that Temporal can appropriately retry in the manner that OpenAI API would.
    # See the `http_retry_enhancement_python` example for inspiration.
    client = AsyncOpenAI(max_retries=0)

    try:
        resp = await client.responses.create(
            model=request.model,
            instructions=request.instructions,
            input=request.input,
            tools=request.tools,
            timeout=30,
        )
        return resp
    finally:
        await client.close()