"""Test suite for http_retry.py

Tests generic HTTP retry logic based on status codes and headers.
The behavior matches the original OpenAI Python client but works with any HTTP API.
"""

import pytest
from datetime import timedelta
from unittest.mock import Mock, patch
from temporalio.exceptions import ApplicationError
import httpx

from util.translate_http_errors import http_response_to_application_error


def create_mock_http_response(status_code: int, headers: dict = None) -> httpx.Response:
    """Create a mock httpx.Response with specified status code and headers."""
    response = Mock(spec=httpx.Response)
    response.status_code = status_code
    response.headers = httpx.Headers(headers or {})
    # Add a mock request for HTTPStatusError creation
    response.request = Mock(spec=httpx.Request)
    response.request.method = "POST"
    response.request.url = "https://api.example.com/test"

    return response


class TestXShouldRetryHeader:
    """Test x-should-retry header handling (highest priority)."""

    def test_x_should_retry_true_allows_retry(self):
        """When x-should-retry=true, should allow retry (non_retryable=False), regardless of status code."""
        response = create_mock_http_response(400, {"x-should-retry": "true"})

        # Should return ApplicationError with non_retryable=False to allow Temporal to retry
        app_error = http_response_to_application_error(response)

        assert isinstance(app_error, ApplicationError)
        assert app_error.non_retryable is False
        assert "x-should-retry=true" in str(app_error)
        assert "HTTP 400" in str(app_error)

    def test_x_should_retry_false_prevents_retry(self):
        """When x-should-retry=false, should prevent retry (non_retryable=True), regardless of status code."""
        response = create_mock_http_response(500, {"x-should-retry": "false"})

        app_error = http_response_to_application_error(response)

        assert isinstance(app_error, ApplicationError)
        assert app_error.non_retryable is True
        # Verify the error message is descriptive and actionable
        assert "x-should-retry=false" in str(app_error)
        assert "HTTP 500" in str(app_error)

    def test_x_should_retry_with_retry_after(self):
        """x-should-retry=false should not include retry delay info since we're not retrying."""
        response = create_mock_http_response(429, {
            "x-should-retry": "false",
            "retry-after": "30"
        })
        # Header precedence: x-should-retry overrides status codes, retry-after is ignored when not retrying
        app_error = http_response_to_application_error(response)

        assert isinstance(app_error, ApplicationError)
        assert app_error.non_retryable is True
        assert app_error.next_retry_delay is None


class TestStatusCodeRetryLogic:
    """Test status code based retry decisions (when no x-should-retry header)."""

    @pytest.mark.parametrize("status_code,should_retry", [
        # Retryable status codes
        (408, True),  # Request Timeout
        (409, True),  # Conflict
        (429, True),  # Too Many Requests
        (500, True),  # Internal Server Error
        (501, True),  # Not Implemented
        (502, True),  # Bad Gateway
        (503, True),  # Service Unavailable
        (504, True),  # Gateway Timeout
        (599, True),  # Any 5xx error

        # Non-retryable status codes
        (400, False), # Bad Request
        (401, False), # Unauthorized
        (403, False), # Forbidden
        (404, False), # Not Found
        (422, False), # Unprocessable Entity
        (499, False), # Any other 4xx
    ])
    def test_status_code_retry_decisions(self, status_code: int, should_retry: bool):
        """Test that status codes map to correct retry decisions."""
        response = create_mock_http_response(status_code)

        app_error = http_response_to_application_error(response)

        assert isinstance(app_error, ApplicationError)
        assert app_error.non_retryable is not should_retry

        if should_retry:
            # Check for specific retryable error patterns
            error_message = str(app_error)
            retryable_patterns = [
                "will retry with backoff",
                "request timeout",
                "conflict/lock timeout",
                "rate limit exceeded",
                "server error"
            ]
            assert any(pattern in error_message for pattern in retryable_patterns), f"Expected retryable error pattern in: {error_message}"
        else:
            # Check for non-retryable error patterns
            error_message = str(app_error)
            assert "client error" in error_message and "not retrying" in error_message, f"Expected non-retryable error pattern in: {error_message}"


class TestRetryAfterHeaderParsing:
    """Test retry-after header parsing (ported from OpenAI client tests)."""

    # Complete test matrix from original OpenAI client tests
    # Reference: https://github.com/openai/openai-python/blob/main/tests/test_client.py
    # Uses mock time.time() = 1696004797 (Wed, 29 Sep 2023 16:26:37 GMT)
    @pytest.mark.parametrize("retry_after,expected_delay", [
        # Valid numeric seconds (original validates 0 < retry_after <= 60)
        ("20", timedelta(seconds=20)),
        ("60", timedelta(seconds=60)),

        # Invalid numeric values → should fallback to None (exponential backoff)
        ("0", None),      # Zero is invalid per original client
        ("-10", None),    # Negative values invalid
        ("61", None),     # Above 60 second limit
        ("99999999999999999999999999999999999", None),  # Overflow protection
        ("", None),       # Empty string
        ("invalid", None), # Non-numeric string

        # HTTP date parsing (using mocked time: Wed, 29 Sep 2023 16:26:37 GMT)
        # Valid dates within 60 second window
        ("Fri, 29 Sep 2023 16:26:57 GMT", timedelta(seconds=20)),  # +20 seconds from mock time
        ("Fri, 29 Sep 2023 16:27:37 GMT", timedelta(seconds=60)),  # +60 seconds from mock time

        # Invalid dates → should fallback to None
        ("Fri, 29 Sep 2023 16:26:37 GMT", None),   # Same as current time (0 seconds)
        ("Fri, 29 Sep 2023 16:26:27 GMT", None),   # Past time (-10 seconds)
        ("Fri, 29 Sep 2023 16:27:38 GMT", None),   # Beyond 60 second limit (+61 seconds)
        ("Zun, 29 Sep 2023 16:26:27 GMT", None),   # Invalid day name "Zun"
    ])
    @patch("time.time", return_value=1696004797)  # Mock time: Wed, 29 Sep 2023 16:26:37 GMT
    def test_retry_after_header_parsing(self, mock_time, retry_after: str, expected_delay: timedelta):
        """Test parsing of retry-after header values (including HTTP dates)."""
        response = create_mock_http_response(429, {"retry-after": retry_after})

        app_error = http_response_to_application_error(response)

        assert isinstance(app_error, ApplicationError)
        assert app_error.non_retryable is False  # 429 is retryable

        if expected_delay is not None:
            assert app_error.next_retry_delay == expected_delay
        else:
            assert app_error.next_retry_delay is None

    @pytest.mark.parametrize("retry_after_ms,expected_delay", [
        # Valid milliseconds
        ("1000", timedelta(milliseconds=1000)),
        ("500", timedelta(milliseconds=500)),
        ("2000", timedelta(milliseconds=2000)),

        # Invalid values
        ("invalid", None),
        ("", None),
    ])
    def test_retry_after_ms_header_parsing(self, retry_after_ms: str, expected_delay: timedelta):
        """Test parsing of retry-after-ms header (takes precedence over retry-after)."""
        response = create_mock_http_response(429, {
            "retry-after-ms": retry_after_ms,
            "retry-after": "30"  # Should be ignored if retry-after-ms is present
        })

        app_error = http_response_to_application_error(response)

        assert isinstance(app_error, ApplicationError)
        assert app_error.non_retryable is False  # 429 is retryable

        if expected_delay is not None:
            assert app_error.next_retry_delay == expected_delay
        else:
            # Should fallback to retry-after if retry-after-ms is invalid
            assert app_error.next_retry_delay == timedelta(seconds=30)

    def test_retry_after_precedence(self):
        """retry-after-ms should take precedence over retry-after."""
        response = create_mock_http_response(429, {
            "retry-after-ms": "2000",  # 2 seconds
            "retry-after": "30"        # 30 seconds - should be ignored
        })

        app_error = http_response_to_application_error(response)

        assert isinstance(app_error, ApplicationError)
        assert app_error.non_retryable is False  # 429 is retryable
        assert app_error.next_retry_delay == timedelta(milliseconds=2000)


class TestEdgeCases:
    """Test edge cases and malformed responses."""

    def test_no_headers(self):
        """Test behavior with no retry-related headers."""
        response = create_mock_http_response(500)

        app_error = http_response_to_application_error(response)

        assert isinstance(app_error, ApplicationError)
        assert app_error.non_retryable is False  # 500 is retryable
        assert app_error.next_retry_delay is None  # No server hint

    def test_multiple_conflicting_headers(self):
        """Test behavior with conflicting retry headers."""
        response = create_mock_http_response(500, {
            "x-should-retry": "false",  # Should take precedence
            "retry-after": "30"
        })

        app_error = http_response_to_application_error(response)

        assert isinstance(app_error, ApplicationError)
        assert app_error.non_retryable is True  # x-should-retry=false wins
        assert app_error.next_retry_delay is None  # Delay ignored when not retrying

    def test_missing_response_headers(self):
        """Test handling of response with empty headers."""
        response = create_mock_http_response(429)  # Already creates empty headers

        # Should not crash, should treat as no headers
        app_error = http_response_to_application_error(response)

        assert isinstance(app_error, ApplicationError)
        assert app_error.non_retryable is False  # 429 is retryable
        assert app_error.next_retry_delay is None


class TestApplicationErrorMapping:
    """Test that ApplicationError properties correctly map to original retry logic."""

    def test_retryable_error_properties(self):
        """Retryable errors should have non_retryable=False."""
        response = create_mock_http_response(500)

        app_error = http_response_to_application_error(response)

        assert isinstance(app_error, ApplicationError)
        assert app_error.non_retryable is False

    def test_non_retryable_error_properties(self):
        """Non-retryable errors should have non_retryable=True."""
        response = create_mock_http_response(400)

        app_error = http_response_to_application_error(response)

        assert isinstance(app_error, ApplicationError)
        assert app_error.non_retryable is True

    def test_error_message_content(self):
        """Error messages should be descriptive and actionable."""
        # Test server error message (retryable)
        response = create_mock_http_response(500)
        app_error = http_response_to_application_error(response)
        assert "server error" in str(app_error) and "500" in str(app_error)

        # Test client error message (non-retryable)
        response = create_mock_http_response(400)
        app_error = http_response_to_application_error(response)
        assert "client error" in str(app_error) and "check your request" in str(app_error)

        # Test rate limit with server delay hint
        response = create_mock_http_response(429, {"retry-after": "30"})
        app_error = http_response_to_application_error(response)
        error_message = str(app_error)
        assert "rate limit exceeded" in error_message
        assert "30.0s delay" in error_message

        # Test x-should-retry=false message
        response = create_mock_http_response(200, {"x-should-retry": "false"})
        app_error = http_response_to_application_error(response)
        assert "x-should-retry=false" in str(app_error)