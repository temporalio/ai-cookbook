import pytest
import json
from activities.invoke_model import InvokeModelRequest, InvokeModelResponse
from pydantic import BaseModel, Field, ValidationError


class MockResponse(BaseModel):
    result: str = Field(description="Your message")


class MockComplexResponse(BaseModel):
    title: str
    items: list[str]
    count: int


def test_invoke_model_request_basic_serialization():
    """Test basic InvokeModelRequest serialization without response_format"""
    request = InvokeModelRequest(
        model="gpt-4o-mini",
        instructions="Generate a cheerful message",
        input="hello world",
    )

    # Test serialization
    json_str = request.model_dump_json()
    parsed_json = json.loads(json_str)

    assert parsed_json["model"] == "gpt-4o-mini"
    assert parsed_json["instructions"] == "Generate a cheerful message"
    assert parsed_json["input"] == "hello world"
    assert parsed_json["response_format"] is None
    assert parsed_json["tools"] is None

    # Test deserialization
    restored = InvokeModelRequest.model_validate_json(json_str)
    assert restored.model == request.model
    assert restored.instructions == request.instructions
    assert restored.input == request.input
    assert restored.response_format is None
    assert restored.tools is None


def test_invoke_model_request_with_response_format():
    """Test InvokeModelRequest serialization with response_format"""
    request = InvokeModelRequest(
        model="gpt-4o-mini",
        instructions="Generate a structured message",
        input="hello world",
        response_format=MockResponse,
    )

    # Test serialization
    json_str = request.model_dump_json()
    parsed_json = json.loads(json_str)

    assert parsed_json["model"] == "gpt-4o-mini"
    assert (
        parsed_json["response_format"] == "test_invoke_model_serialization:MockResponse"
    )

    # Test deserialization
    restored = InvokeModelRequest.model_validate_json(json_str)
    assert restored.model == request.model
    assert restored.instructions == request.instructions
    assert restored.input == request.input
    assert restored.response_format == MockResponse


def test_invoke_model_request_with_tools():
    """Test InvokeModelRequest serialization with tools"""
    tools = [{"type": "function", "name": "test_func"}]
    request = InvokeModelRequest(
        model="gpt-4o-mini", instructions="Use tools", input="hello", tools=tools
    )

    json_str = request.model_dump_json()
    parsed_json = json.loads(json_str)

    assert parsed_json["tools"] == tools

    restored = InvokeModelRequest.model_validate_json(json_str)
    assert restored.tools == tools


def test_invoke_model_request_roundtrip():
    """Test that serialization->deserialization->serialization is consistent"""
    request = InvokeModelRequest(
        model="gpt-4o-mini",
        instructions="Test roundtrip",
        input="data",
        response_format=MockComplexResponse,
        tools=[{"name": "func1"}, {"name": "func2"}],
    )

    # First serialization
    json1 = request.model_dump_json()

    # Deserialize
    restored = InvokeModelRequest.model_validate_json(json1)

    # Second serialization
    json2 = restored.model_dump_json()

    # Should be identical
    assert json1 == json2


def test_invoke_model_request_validation_errors():
    """Test that invalid data raises appropriate validation errors"""
    with pytest.raises(ValidationError):
        InvokeModelRequest()  # type: ignore

    with pytest.raises(ValidationError):
        InvokeModelRequest(model="gpt-4")  # type: ignore  # missing required fields

    # Test invalid JSON deserialization
    with pytest.raises(ValidationError):
        InvokeModelRequest.model_validate_json('{"invalid": "json"}')


def test_invoke_model_response_basic():
    """Test InvokeModelResponse without response_format"""
    response = InvokeModelResponse(
        response_model="Simple text response", response_format=None
    )

    json_str = response.model_dump_json()
    parsed_json = json.loads(json_str)

    assert parsed_json["response_model"] == "Simple text response"
    assert parsed_json["response_format"] is None

    restored = InvokeModelResponse.model_validate_json(json_str)
    assert restored.response_model == "Simple text response"
    assert restored.response_format is None
    assert restored.response == "Simple text response"


def test_invoke_model_response_with_format():
    """Test InvokeModelResponse with response_format"""
    mock_data = {"result": "test message"}
    response = InvokeModelResponse(
        response_model=mock_data, response_format=MockResponse
    )

    json_str = response.model_dump_json()
    parsed_json = json.loads(json_str)

    assert parsed_json["response_model"] == mock_data
    assert (
        parsed_json["response_format"] == "test_invoke_model_serialization:MockResponse"
    )

    restored = InvokeModelResponse.model_validate_json(json_str)
    assert restored.response_model == mock_data
    assert restored.response_format == MockResponse

    # Test the response property reconstruction
    reconstructed = restored.response
    assert isinstance(reconstructed, MockResponse)
    assert reconstructed.result == "test message"


def test_invoke_model_response_roundtrip():
    """Test InvokeModelResponse serialization roundtrip"""
    complex_data = {"title": "Test Title", "items": ["a", "b", "c"], "count": 3}
    response = InvokeModelResponse(
        response_model=complex_data, response_format=MockComplexResponse
    )

    json1 = response.model_dump_json()
    restored = InvokeModelResponse.model_validate_json(json1)
    json2 = restored.model_dump_json()

    assert json1 == json2

    # Verify response property works
    reconstructed = restored.response
    assert isinstance(reconstructed, MockComplexResponse)
    assert reconstructed.title == "Test Title"
    assert reconstructed.items == ["a", "b", "c"]
    assert reconstructed.count == 3


@pytest.mark.parametrize(
    "model,instructions,input_text",
    [
        ("gpt-4o-mini", "Be helpful", "Hello"),
        ("gpt-4", "Be creative", "Write a story"),
        ("claude-3-sonnet", "Analyze this", "Complex data"),
    ],
)
def test_invoke_model_request_parametrized(model, instructions, input_text):
    """Test InvokeModelRequest with various parameter combinations"""
    request = InvokeModelRequest(
        model=model, instructions=instructions, input=input_text
    )

    json_str = request.model_dump_json()
    restored = InvokeModelRequest.model_validate_json(json_str)

    assert restored.model == model
    assert restored.instructions == instructions
    assert restored.input == input_text


if __name__ == "__main__":
    pytest.main([__file__])
