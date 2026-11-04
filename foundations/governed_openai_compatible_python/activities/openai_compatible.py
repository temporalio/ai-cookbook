import os
from dataclasses import dataclass

from openai import AsyncOpenAI
from temporalio import activity


@dataclass
class ModelRequest:
    model: str
    instructions: str
    input: str
    run_id: str
    request_id: str


@activity.defn
async def invoke_model(request: ModelRequest) -> str:
    """Call an OpenAI-compatible model endpoint from a retryable Activity."""
    base_url = os.environ.get("OPENAI_BASE_URL")
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=base_url,
        max_retries=0,
        default_headers={
            "X-Request-ID": request.request_id,
            "X-TE-Run-ID": request.run_id,
        },
    )

    response = await client.responses.create(
        model=request.model,
        instructions=request.instructions,
        input=request.input,
        timeout=30,
    )
    return response.output_text
