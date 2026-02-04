import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.requests import AgentRequest, WeatherRequest
from models.orchestrator import AgentResponse, ToolCall


class TestAgentRequest:

    def test_default_model_id(self):
        request = AgentRequest(messages=[{"role": "user", "content": "hello"}])
        assert "claude" in request.model_id.lower()

    def test_custom_model_id(self):
        request = AgentRequest(
            messages=[{"role": "user", "content": "hello"}],
            model_id="custom-model-123"
        )
        assert request.model_id == "custom-model-123"

    def test_messages_preserved(self):
        messages = [
            {"role": "user", "content": "What time is it?"},
            {"role": "assistant", "content": "Let me check."}
        ]
        request = AgentRequest(messages=messages)
        assert len(request.messages) == 2
        assert request.messages[0]["role"] == "user"


class TestWeatherRequest:

    def test_city_required(self):
        request = WeatherRequest(city="London")
        assert request.city == "London"

    def test_missing_city_raises(self):
        with pytest.raises(Exception):
            WeatherRequest()


class TestToolCall:

    def test_with_parameters(self):
        tool = ToolCall(tool_name="get_weather", parameters={"city": "Paris"})
        assert tool.tool_name == "get_weather"
        assert tool.parameters["city"] == "Paris"

    def test_default_empty_parameters(self):
        tool = ToolCall(tool_name="get_time")
        assert tool.parameters == {}


class TestAgentResponse:

    def test_with_tool_calls(self):
        response = AgentResponse(
            tool_calls=[ToolCall(tool_name="get_time")],
            final_answer=None,
            reasoning="Need to check time"
        )
        assert len(response.tool_calls) == 1
        assert response.final_answer is None

    def test_with_final_answer(self):
        response = AgentResponse(
            tool_calls=[],
            final_answer="The time is 10:30 AM",
            reasoning="Got the time"
        )
        assert response.tool_calls == []
        assert response.final_answer == "The time is 10:30 AM"

    def test_defaults(self):
        response = AgentResponse()
        assert response.tool_calls == []
        assert response.final_answer is None
        assert response.reasoning is None

    def test_serialization(self):
        response = AgentResponse(
            tool_calls=[ToolCall(tool_name="list_files", parameters={})],
            final_answer=None,
            reasoning="Listing files"
        )
        data = response.model_dump()
        assert data["tool_calls"][0]["tool_name"] == "list_files"
