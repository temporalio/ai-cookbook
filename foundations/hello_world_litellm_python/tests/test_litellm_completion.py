# ABOUTME: Tests for LiteLLM completion activity and request model.
# Covers LiteLLMRequest.to_acompletion_kwargs and create activity error handling.

from unittest.mock import AsyncMock, patch

import litellm
import pytest
from activities.litellm_completion import create
from activities.models import LiteLLMRequest
from temporalio.exceptions import ApplicationError


class TestCreateActivity:
    """Tests for the create activity (mocked litellm)."""

    @pytest.mark.asyncio
    async def test_create_success(self):
        """Successful completion returns the response."""
        mock_response = {"choices": [{"message": {"content": "Hi there"}}]}
        request = LiteLLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )
        with patch(
            "activities.litellm_completion.litellm.acompletion", new_callable=AsyncMock
        ) as mock_acompletion:
            mock_acompletion.return_value = mock_response
            result = await create(request)

        assert result == mock_response

    @pytest.mark.asyncio
    async def test_create_disables_litellm_retries(self):
        """Verify num_retries=0 is passed to acompletion."""
        request = LiteLLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )
        with patch(
            "activities.litellm_completion.litellm.acompletion", new_callable=AsyncMock
        ) as mock_acompletion:
            mock_acompletion.return_value = {"choices": []}
            await create(request)

            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs["num_retries"] == 0

    @pytest.mark.parametrize(
        "exception_class",
        [
            litellm.AuthenticationError,
            litellm.BadRequestError,
            litellm.InvalidRequestError,
            litellm.UnsupportedParamsError,
            litellm.JSONSchemaValidationError,
            litellm.ContentPolicyViolationError,
            litellm.NotFoundError,
        ],
    )
    @pytest.mark.asyncio
    async def test_create_non_retryable_errors(self, exception_class):
        """All 7 non-retryable exception types raise ApplicationError with non_retryable=True."""
        request = LiteLLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )
        # Most litellm exceptions take message/model/llm_provider, but
        # JSONSchemaValidationError has a different signature.
        if exception_class is litellm.JSONSchemaValidationError:
            exc = exception_class(
                model="gpt-4",
                llm_provider="openai",
                raw_response="{}",
                schema='{"type": "object"}',
            )
        else:
            exc = exception_class(
                message="test error",
                model="gpt-4",
                llm_provider="openai",
            )
        with patch(
            "activities.litellm_completion.litellm.acompletion", new_callable=AsyncMock
        ) as mock_acompletion:
            mock_acompletion.side_effect = exc
            with pytest.raises(ApplicationError) as exc_info:
                await create(request)

            assert exc_info.value.non_retryable is True
            assert exc_info.value.type == exception_class.__name__
