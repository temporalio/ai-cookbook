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
    print(f"Creating response for with instructions: {request.instructions} and input: {request.input}")
    resp = await client.responses.create(
        model="gpt-4o-mini",
        instructions=request.instructions,
        input=request.input,
        timeout=15,
    )
    print(f"Response: {resp.output_text}")
    return resp.output_text