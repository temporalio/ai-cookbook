"""Tests for helpers/tool_helpers.py — claude_tool_from_model and system instructions."""

from pydantic import BaseModel, Field

from helpers.tool_helpers import claude_tool_from_model, HELPFUL_AGENT_SYSTEM_INSTRUCTIONS


class _SampleModel(BaseModel):
    name: str = Field(description="A name")
    count: int = Field(description="A count")


class TestClaudeToolFromModel:
    def test_with_pydantic_model(self):
        result = claude_tool_from_model("my_tool", "Does something", _SampleModel)
        assert result["name"] == "my_tool"
        assert result["description"] == "Does something"
        schema = result["input_schema"]
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "count" in schema["properties"]

    def test_with_none_model(self):
        result = claude_tool_from_model("no_params", "No parameters needed", None)
        assert result["name"] == "no_params"
        assert result["description"] == "No parameters needed"
        assert result["input_schema"]["type"] == "object"
        assert result["input_schema"]["properties"] == {}
        assert result["input_schema"]["required"] == []


class TestSystemInstructions:
    def test_is_nonempty_string(self):
        assert isinstance(HELPFUL_AGENT_SYSTEM_INSTRUCTIONS, str)
        assert len(HELPFUL_AGENT_SYSTEM_INSTRUCTIONS.strip()) > 0
