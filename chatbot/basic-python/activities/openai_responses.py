from temporalio import activity
from openai import AsyncOpenAI
from dataclasses import dataclass

# Temporal best practice: Create a data structure to hold the request parameters.
@dataclass
class OpenAIResponsesRequest:
    model: str
    instructions: str
    input: str

@activity.defn
async def create(request: OpenAIResponsesRequest) -> str:
    # Temporal best practice: Disable retry logic in OpenAI API client library.
    client = AsyncOpenAI(max_retries=0)

    resp = await client.responses.create(
        model=request.model,
        instructions=request.instructions,
        input=request.input,
        timeout=15,
    )

    return resp.output_text