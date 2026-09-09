"""Shared fake Model/ModelProvider helpers for testing the OpenAI Agents SDK
integration without making real network calls to OpenAI."""

from agents.items import ModelResponse
from agents.models.interface import Model, ModelProvider
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)


def text_response(text: str, response_id: str = "resp_text") -> ModelResponse:
    return ModelResponse(
        output=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
            )
        ],
        usage=Usage(),
        response_id=response_id,
    )


def tool_call_response(
    name: str, call_id: str, arguments: str, response_id: str = "resp_tool"
) -> ModelResponse:
    return ModelResponse(
        output=[
            ResponseFunctionToolCall(
                id="fc_1",
                type="function_call",
                call_id=call_id,
                name=name,
                arguments=arguments,
                status="completed",
            )
        ],
        usage=Usage(),
        response_id=response_id,
    )


class ScriptedModel(Model):
    """Returns one canned ModelResponse per call, in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._index = 0

    async def get_response(self, *args, **kwargs):
        response = self._responses[self._index]
        self._index += 1
        return response

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError("This recipe does not use streaming responses")
        yield


class ScriptedModelProvider(ModelProvider):
    def __init__(self, model: Model):
        self._model = model

    def get_model(self, model_name):
        return self._model
