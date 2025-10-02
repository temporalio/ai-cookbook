from typing import Any, Dict

import litellm
from temporalio import activity
from temporalio.exceptions import ApplicationError

from activities.models import LiteLLMRequest


@activity.defn(name="activities.litellm_completion.create")
async def create(request: LiteLLMRequest) -> Dict[str, Any]:
    # Temporal best practice: disable LiteLLM retries and let Temporal handle them.
    kwargs = request.to_acompletion_kwargs()
    kwargs["num_retries"] = 0

    try:
        response = await litellm.acompletion(**kwargs)
    except (
        litellm.AuthenticationError,
        litellm.BadRequestError,
        litellm.InvalidRequestError,
        litellm.UnsupportedParamsError,
        litellm.JSONSchemaValidationError,
        litellm.ContentPolicyViolationError,
        litellm.NotFoundError,
    ) as exc:
        raise ApplicationError(
            str(exc),
            type=exc.__class__.__name__,
            non_retryable=True,
        ) from exc
    except litellm.APIError:
        raise

    return response
