from temporalio import activity
from openai import AsyncOpenAI
from dataclasses import dataclass

@dataclass
class OpenAIResponsesRequest:
    instructions: str
    input: str

@activity.defn
async def create(request: OpenAIResponsesRequest) -> str:
    client = AsyncOpenAI()

    resp = await client.responses.create(
        model="gpt-5-mini",
        instructions=request.instructions,
        input=request.input,
    )

    return resp.output_text