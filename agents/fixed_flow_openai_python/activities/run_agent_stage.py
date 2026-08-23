import openai
from openai import AsyncOpenAI
from temporalio import activity
from temporalio.exceptions import ApplicationError

from models.flow import AgentStageRequest, AgentStageResult


@activity.defn
async def run_agent_stage(request: AgentStageRequest) -> AgentStageResult:
    """Run one specialist role as a retryable Temporal Activity."""

    client = AsyncOpenAI(max_retries=0)
    try:
        response = await client.responses.create(
            model=request.model,
            instructions=request.instructions,
            input=request.input,
            timeout=30,
        )
    except (
        openai.BadRequestError,
        openai.AuthenticationError,
        openai.PermissionDeniedError,
        openai.NotFoundError,
        openai.UnprocessableEntityError,
    ) as exc:
        raise ApplicationError(
            str(exc),
            type=exc.__class__.__name__,
            non_retryable=True,
        ) from exc
    finally:
        await client.close()

    output = response.output_text.strip()
    if not output:
        # An empty response may be transient, so let the Workflow's Activity retry
        # policy decide whether to try the stage again.
        raise ApplicationError(
            f"The {request.stage} stage returned no text.",
            type="EmptyModelResponse",
        )

    return AgentStageResult(stage=request.stage, output=output)
