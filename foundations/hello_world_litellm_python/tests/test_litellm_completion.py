# ABOUTME: Tests for LiteLLM completion activity and request model.
# Covers LiteLLMRequest.to_acompletion_kwargs and create activity error handling.

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import litellm
from temporalio.exceptions import ApplicationError

from activities.models import LiteLLMRequest
from activities.litellm_completion import create


class TestLiteLLMRequestToKwargs:
    """Tests for LiteLLMRequest.to_acompletion_kwargs conversion."""

    def test_to_acompletion_kwargs_required_only(self):
        """Only model and messages when no optionals are set."""
        request = LiteLLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )
        kwargs = request.to_acompletion_kwargs()
        assert kwargs == {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }

    def test_to_acompletion_kwargs_with_optionals(self):
        """Optional fields are included when set."""
        request = LiteLLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
            timeout=30.0,
        )
        kwargs = request.to_acompletion_kwargs()
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 100
        assert kwargs["timeout"] == 30.0

    def test_to_acompletion_kwargs_extra_options(self):
        """Extra options are merged into kwargs."""
        request = LiteLLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            extra_options={"top_p": 0.9, "presence_penalty": 0.5},
        )
        kwargs = request.to_acompletion_kwargs()
        assert kwargs["top_p"] == 0.9
        assert kwargs["presence_penalty"] == 0.5
        assert kwargs["model"] == "gpt-4"


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
        with patch("activities.litellm_completion.litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
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
        with patch("activities.litellm_completion.litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
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
        with patch("activities.litellm_completion.litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.side_effect = exc
            with pytest.raises(ApplicationError) as exc_info:
                await create(request)

            assert exc_info.value.non_retryable is True
            assert exc_info.value.type == exception_class.__name__

    @pytest.mark.asyncio
    async def test_create_api_error_retryable(self):
        """APIError propagates directly (Temporal handles retries)."""
        request = LiteLLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )
        exc = litellm.APIError(
            message="server error",
            model="gpt-4",
            llm_provider="openai",
            status_code=500,
        )
        with patch("activities.litellm_completion.litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.side_effect = exc
            with pytest.raises(litellm.APIError):
                await create(request)
