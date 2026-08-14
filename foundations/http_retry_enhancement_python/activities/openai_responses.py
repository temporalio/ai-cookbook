from dataclasses import dataclass

from openai import APIStatusError, AsyncOpenAI
from openai.types.responses import Response
from temporalio import activity

from util.translate_http_errors import http_response_to_application_error


# Temporal best practice: Create a data structure to hold the request parameters.
@dataclass
class OpenAIResponsesRequest:
    model: str
    instructions: str
    input: str

@activity.defn
async def create(request: OpenAIResponsesRequest) -> Response:
    # Temporal best practice: Disable retry logic in OpenAI API client library.
    client = AsyncOpenAI(max_retries=0)

    try:
        resp = await client.responses.create(
            model=request.model,
            instructions=request.instructions,
            input=request.input,
            timeout=15,
        )
        return resp
    except APIStatusError as e:
        raise http_response_to_application_error(e.response) from e