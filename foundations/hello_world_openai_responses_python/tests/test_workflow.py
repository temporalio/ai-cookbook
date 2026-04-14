"""Tests for HelloWorld workflow with mocked OpenAI activity."""

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from datetime import timedelta
from dataclasses import dataclass

from openai.types.responses import Response, ResponseOutputMessage, ResponseOutputText

from activities.openai_responses import OpenAIResponsesRequest
from workflows.hello_world_workflow import HelloWorld

TASK_QUEUE = "test-hello-world"

_last_request: OpenAIResponsesRequest | None = None


def _make_response(text: str) -> Response:
    return Response(
        id="resp_test",
        created_at=1700000000,
        model="gpt-4o-mini",
        object="response",
        output=[ResponseOutputMessage(
            id="msg_test",
            type="message",
            role="assistant",
            status="completed",
            content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
        )],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


@activity.defn(name="create")
async def mock_create(request: OpenAIResponsesRequest) -> Response:
    global _last_request
    _last_request = request
    return _make_response("Recursive calls descend\nBase case stops the endless loop\nStack unwinds with grace")


class TestHelloWorldWorkflow:
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_returns_haiku_response(self):
        global _last_request
        _last_request = None

        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter,
        ) as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[HelloWorld],
                activities=[mock_create],
            ):
                result = await env.client.execute_workflow(
                    HelloWorld.run,
                    "Tell me about recursion in programming.",
                    id="test-hello-world-wf",
                    task_queue=TASK_QUEUE,
                    run_timeout=timedelta(seconds=30),
                )

        assert "Recursive" in result
        assert _last_request is not None
        assert _last_request.model == "gpt-4o-mini"
        assert "haiku" in _last_request.instructions.lower()
        assert _last_request.input == "Tell me about recursion in programming."


class TestOpenAIResponsesRequest:
    def test_construction(self):
        req = OpenAIResponsesRequest(
            model="gpt-4o-mini",
            instructions="Be helpful.",
            input="Hello",
        )
        assert req.model == "gpt-4o-mini"
        assert req.instructions == "Be helpful."
        assert req.input == "Hello"
