from dataclasses import dataclass

from openai import AsyncOpenAI
from temporalio import activity
from temporalio.exceptions import ApplicationError


@dataclass
class CallLLMRequest:
    prompt: str
    model: str = "gpt-4o-mini"


@activity.defn
async def call_llm(request: CallLLMRequest) -> str:
    # max_retries=0 so Temporal owns retries, not the client.
    client = AsyncOpenAI(max_retries=0)
    try:
        response = await client.responses.create(model=request.model, input=request.prompt)
    except Exception as exc:
        # TODO: catch provider-specific PERMANENT errors (auth, bad request) here and mark
        # them non-retryable; let transient errors propagate so Temporal retries them.
        raise ApplicationError(str(exc), type="LLMError", non_retryable=True) from exc
    return response.output_text
