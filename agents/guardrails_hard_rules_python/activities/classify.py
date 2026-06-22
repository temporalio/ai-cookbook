import anthropic
from temporalio import activity
from temporalio.exceptions import ApplicationError
from pydantic import BaseModel

from models.signals import ContentSignals
from models.verdict import LLMVerdict, Verdict
from guardrails.hard_rules import apply_hard_rules

_SYSTEM = """You are a content moderation assistant. Classify the submitted text as:
- safe: acceptable content with no policy concerns
- review: borderline content that a human should check
- block: clear policy violation (hate speech, harassment, explicit content, obvious spam)

When uncertain, use 'review' — it's better to flag for human review than to miss a violation."""

_SUBMIT_VERDICT_TOOL = {
    "name": "submit_verdict",
    "description": "Submit your content moderation classification.",
    "input_schema": LLMVerdict.model_json_schema(),
}


class ClassifyRequest(BaseModel):
    signals: ContentSignals
    model: str = "claude-sonnet-4-6"


@activity.defn
async def classify(request: ClassifyRequest) -> Verdict:
    client = anthropic.AsyncAnthropic(max_retries=0)

    try:
        response = await client.messages.create(
            model=request.model,
            max_tokens=512,
            system=_SYSTEM,
            messages=[
                {"role": "user", "content": f"Classify this content:\n\n{request.signals.text}"}
            ],
            tools=[_SUBMIT_VERDICT_TOOL],
            tool_choice={"type": "tool", "name": "submit_verdict"},
        )
    except anthropic.AuthenticationError as exc:
        raise ApplicationError(str(exc), type="AuthenticationError", non_retryable=True) from exc
    except anthropic.BadRequestError as exc:
        raise ApplicationError(str(exc), type="BadRequestError", non_retryable=True) from exc

    tool_block = next(b for b in response.content if b.type == "tool_use")
    llm_verdict = Verdict.model_validate(tool_block.input)

    return apply_hard_rules(request.signals, llm_verdict)
