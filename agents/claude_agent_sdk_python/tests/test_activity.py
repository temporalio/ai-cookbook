"""
Tests for agent_executor activity

Tests the heartbeat, staleness, and deduplication logic using mocked
Claude Agent SDK responses.

Usage:
    uv run pytest tests/test_activity.py -v
"""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from models import AgentInput, AgentOutput
from activities.agent_executor import execute_agent_activity, log_result_activity


# ---------------------------------------------------------------------------
# Mock SDK event types
# ---------------------------------------------------------------------------

@dataclass
class MockTextBlock:
    text: str
    type: str = "text"


@dataclass
class MockAssistantMessage:
    """Simulates claude_agent_sdk.types.AssistantMessage"""
    content: list


@dataclass
class MockStreamEvent:
    """Simulates claude_agent_sdk.types.StreamEvent (incremental text chunk)"""
    text: str


@dataclass
class MockResultMessage:
    """Simulates claude_agent_sdk.types.ResultMessage"""
    total_tokens: int = 200


# ---------------------------------------------------------------------------
# Mock SDK query generators
# ---------------------------------------------------------------------------

async def mock_sdk_query_success(prompt, options):
    """Mock SDK that yields assistant message + result."""
    yield MockAssistantMessage(content=[MockTextBlock(text="Hello, ")])
    yield MockAssistantMessage(content=[MockTextBlock(text="world!")])
    yield MockResultMessage(total_tokens=200)


async def mock_sdk_query_with_stream_events(prompt, options):
    """Mock SDK that yields BOTH stream events AND assistant messages.

    This tests the deduplication logic — only AssistantMessage text
    should be accumulated, not StreamEvent text.
    """
    # StreamEvent comes first (incremental chunk) — should be SKIPPED
    yield MockStreamEvent(text="Hello, ")
    # AssistantMessage comes after (complete block) — should be USED
    yield MockAssistantMessage(content=[MockTextBlock(text="Hello, world!")])
    yield MockResultMessage(total_tokens=100)


async def mock_sdk_query_error(prompt, options):
    """Mock SDK that raises an exception."""
    raise RuntimeError("SDK connection failed")
    yield  # make it an async generator  # noqa: E305


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_sdk_mock(query_fn):
    """
    Create mock modules for claude_agent_sdk and claude_agent_sdk.types.

    The activity imports:
        from claude_agent_sdk import query
        from claude_agent_sdk.types import ClaudeAgentOptions, AssistantMessage, ResultMessage
    """
    mock_types = MagicMock()
    mock_types.ClaudeAgentOptions = lambda **kwargs: MagicMock()
    mock_types.AssistantMessage = MockAssistantMessage
    mock_types.ResultMessage = MockResultMessage

    mock_sdk = MagicMock()
    mock_sdk.query = query_fn
    mock_sdk.types = mock_types

    return {
        "claude_agent_sdk": mock_sdk,
        "claude_agent_sdk.types": mock_types,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_agent_success():
    """Test successful agent execution with response collection."""
    env = ActivityEnvironment()

    with patch.dict("sys.modules", _create_sdk_mock(mock_sdk_query_success)):
        input_data = AgentInput(prompt="Say hello")
        result = await env.run(execute_agent_activity, input_data)

    assert result.status == "success"
    assert result.response == "Hello, world!"
    assert result.total_tokens == 200
    assert result.num_events == 3
    assert result.processing_time_seconds is not None
    assert result.processing_time_seconds > 0


@pytest.mark.asyncio
async def test_response_deduplication():
    """Test that StreamEvent text is NOT accumulated (only AssistantMessage)."""
    env = ActivityEnvironment()

    with patch.dict("sys.modules", _create_sdk_mock(mock_sdk_query_with_stream_events)):
        input_data = AgentInput(prompt="Test dedup")
        result = await env.run(execute_agent_activity, input_data)

    assert result.status == "success"
    # Should be "Hello, world!" (from AssistantMessage only)
    # NOT "Hello, Hello, world!" (which would happen if StreamEvent was also used)
    assert result.response == "Hello, world!"


@pytest.mark.asyncio
async def test_execute_agent_error():
    """Test that SDK errors are caught and returned as AgentOutput."""
    env = ActivityEnvironment()

    with patch.dict("sys.modules", _create_sdk_mock(mock_sdk_query_error)):
        input_data = AgentInput(prompt="This will fail")
        result = await env.run(execute_agent_activity, input_data)

    # Errors modeled as completions, not exceptions
    assert result.status == "error"
    assert "SDK connection failed" in result.error_message
    assert result.response == ""


@pytest.mark.asyncio
async def test_log_result_activity_success():
    """Test that log_result_activity handles success output."""
    env = ActivityEnvironment()

    output = AgentOutput(
        status="success",
        response="Hello world",
        total_tokens=100,
        processing_time_seconds=1.5,
    )

    # Should not raise
    await env.run(log_result_activity, output)


@pytest.mark.asyncio
async def test_log_result_activity_error():
    """Test that log_result_activity handles error output."""
    env = ActivityEnvironment()

    output = AgentOutput(
        status="error",
        response="",
        error_message="Something broke",
        processing_time_seconds=0.1,
    )

    # Should not raise
    await env.run(log_result_activity, output)
