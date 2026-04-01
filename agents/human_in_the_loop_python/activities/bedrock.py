import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import boto3
from temporalio import activity


@dataclass
class BedrockRequest:
    model_id: str
    instructions: str
    input: str


@activity.defn
async def create(request: BedrockRequest) -> str:
    # Temporal best practice: run blocking SDK calls in a thread executor
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return await loop.run_in_executor(executor, _invoke, request)


def _invoke(request: BedrockRequest) -> str:
    client = boto3.client(service_name="bedrock-runtime")

    response = client.converse(
        modelId=request.model_id,
        system=[{"text": request.instructions}],
        messages=[{"role": "user", "content": [{"text": request.input}]}],
    )

    content = response["output"]["message"]["content"]
    return next(block["text"] for block in content if "text" in block)
