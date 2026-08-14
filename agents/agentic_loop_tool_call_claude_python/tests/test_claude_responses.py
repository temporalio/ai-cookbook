"""Tests for the Claude invocation Activity's error classification."""

from unittest.mock import AsyncMock, patch

import anthropic
import httpx
import pytest
from temporalio.exceptions import ApplicationError

from activities.claude_responses import ClaudeResponsesRequest, create


def _request() -> ClaudeResponsesRequest:
    return ClaudeResponsesRequest(
        model="claude-sonnet-4-5-20250929",
        system="You are helpful.",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[],
    )


def _api_error(exception_class, status_code: int):
    response = httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    return exception_class("test error", response=response, body=None)


def _patched_client(side_effect):
    patcher = patch("activities.claude_responses.AsyncAnthropic")
    mock_cls = patcher.start()
    client = mock_cls.return_value
    client.messages.create = AsyncMock(side_effect=side_effect)
    client.close = AsyncMock()
    return patcher, client


@pytest.mark.parametrize(
    "exception_class,status_code",
    [
        (anthropic.BadRequestError, 400),
        (anthropic.AuthenticationError, 401),
        (anthropic.PermissionDeniedError, 403),
        (anthropic.NotFoundError, 404),
        (anthropic.UnprocessableEntityError, 422),
    ],
)
@pytest.mark.asyncio
async def test_permanent_errors_are_non_retryable(exception_class, status_code):
    exc = _api_error(exception_class, status_code)
    patcher, client = _patched_client(exc)
    try:
        with pytest.raises(ApplicationError) as exc_info:
            await create(_request())

        assert exc_info.value.non_retryable is True
        assert exc_info.value.type == exception_class.__name__
        client.close.assert_awaited_once()
    finally:
        patcher.stop()


@pytest.mark.parametrize(
    "exception_class,status_code",
    [
        (anthropic.RateLimitError, 429),
        (anthropic.InternalServerError, 500),
    ],
)
@pytest.mark.asyncio
async def test_transient_errors_propagate_for_temporal_to_retry(
    exception_class, status_code
):
    exc = _api_error(exception_class, status_code)
    patcher, client = _patched_client(exc)
    try:
        with pytest.raises(exception_class):
            await create(_request())

        client.close.assert_awaited_once()
    finally:
        patcher.stop()
