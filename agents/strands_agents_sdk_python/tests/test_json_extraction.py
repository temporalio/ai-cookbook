import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from activities.strands_agent import extract_json


class TestExtractJson:

    def test_valid_json_direct(self):
        text = '{"tool_calls": [], "final_answer": "Hello", "reasoning": "test"}'
        result = extract_json(text)
        assert result == {"tool_calls": [], "final_answer": "Hello", "reasoning": "test"}

    def test_valid_json_with_whitespace(self):
        text = '  \n  {"tool_calls": [], "final_answer": "test"}  \n  '
        result = extract_json(text)
        assert result["final_answer"] == "test"

    def test_json_embedded_in_text(self):
        text = '''Here is my response:
        {"tool_calls": [{"tool_name": "get_time", "parameters": {}}], "final_answer": null, "reasoning": "Need time"}
        That's my answer.'''
        result = extract_json(text)
        assert result["tool_calls"][0]["tool_name"] == "get_time"
        assert result["final_answer"] is None

    def test_json_with_tool_calls(self):
        text = '{"tool_calls": [{"tool_name": "get_weather", "parameters": {"city": "London"}}], "final_answer": null, "reasoning": "Checking weather"}'
        result = extract_json(text)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["tool_name"] == "get_weather"
        assert result["tool_calls"][0]["parameters"]["city"] == "London"

    def test_json_with_final_answer(self):
        text = '{"tool_calls": [], "final_answer": "The current time is 2024-01-15 10:30:00", "reasoning": "Got the time"}'
        result = extract_json(text)
        assert result["tool_calls"] == []
        assert "current time" in result["final_answer"]

    def test_invalid_json_raises_error(self):
        text = "This is not JSON at all"
        with pytest.raises(ValueError, match="No valid JSON found"):
            extract_json(text)

    def test_malformed_json_raises_error(self):
        text = '{"tool_calls": [}'
        with pytest.raises(ValueError, match="No valid JSON found"):
            extract_json(text)

    def test_empty_string_raises_error(self):
        with pytest.raises(ValueError, match="No valid JSON found"):
            extract_json("")

    def test_nested_json_objects(self):
        text = '''{"tool_calls": [
            {"tool_name": "get_weather", "parameters": {"city": "New York", "units": "metric"}}
        ], "final_answer": null, "reasoning": "User wants weather"}'''
        result = extract_json(text)
        assert result["tool_calls"][0]["parameters"]["units"] == "metric"

    def test_json_with_special_characters(self):
        text = '{"tool_calls": [], "final_answer": "Temperature: 25\\u00b0C", "reasoning": "Done"}'
        result = extract_json(text)
        assert "Temperature" in result["final_answer"]
