# ABOUTME: Temporal activity that wraps Gemini API chat completions.
# Handles conversation history using Google SDK native Content/Part abstractions.

import os
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types
from temporalio import activity


@dataclass
class GeminiChatRequest:
    """Request parameters for a Gemini chat completion."""

    model: str
    system_instruction: str
    contents: list[dict[str, Any]]  # Serialized Content objects
    tools: list[dict[str, Any]]  # Serialized Tool objects


@dataclass
class GeminiChatResponse:
    """Response from a Gemini chat completion."""

    text: str | None  # The text response, if any
    function_calls: list[dict[str, Any]]  # List of function calls, if any
    raw_parts: list[dict[str, Any]]  # Raw parts for conversation history


def _deserialize_contents(contents: list[dict[str, Any]]) -> list[types.Content]:
    """Convert serialized content dicts back to Content objects."""
    result = []
    for content_dict in contents:
        parts = []
        for part_dict in content_dict.get("parts", []):
            if "text" in part_dict:
                parts.append(types.Part.from_text(text=part_dict["text"]))
            elif "function_call" in part_dict:
                fc = part_dict["function_call"]
                parts.append(
                    types.Part.from_function_call(name=fc["name"], args=fc.get("args", {}))
                )
            elif "function_response" in part_dict:
                fr = part_dict["function_response"]
                parts.append(
                    types.Part.from_function_response(
                        name=fr["name"], response=fr.get("response", {})
                    )
                )
        result.append(types.Content(role=content_dict.get("role", "user"), parts=parts))
    return result


def _deserialize_tools(tools: list[dict[str, Any]]) -> list[types.Tool]:
    """Convert serialized tool dicts back to Tool objects."""
    if not tools:
        return []

    function_declarations = []
    for tool_dict in tools:
        for fd_dict in tool_dict.get("function_declarations", []):
            params = fd_dict.get("parameters")
            if params:
                properties = {}
                for prop_name, prop_info in params.get("properties", {}).items():
                    prop_type_str = prop_info.get("type", "STRING")
                    # Handle both string type names and Type enum values
                    if isinstance(prop_type_str, str):
                        type_mapping = {
                            "STRING": types.Type.STRING,
                            "INTEGER": types.Type.INTEGER,
                            "NUMBER": types.Type.NUMBER,
                            "BOOLEAN": types.Type.BOOLEAN,
                            "ARRAY": types.Type.ARRAY,
                            "OBJECT": types.Type.OBJECT,
                        }
                        gemini_type = type_mapping.get(prop_type_str.upper(), types.Type.STRING)
                    else:
                        gemini_type = prop_type_str
                    properties[prop_name] = types.Schema(
                        type=gemini_type, description=prop_info.get("description", "")
                    )
                parameters = types.Schema(
                    type=types.Type.OBJECT,
                    properties=properties,
                    required=params.get("required", []),
                )
            else:
                parameters = types.Schema(type=types.Type.OBJECT, properties={}, required=[])

            function_declarations.append(
                types.FunctionDeclaration(
                    name=fd_dict["name"],
                    description=fd_dict.get("description", ""),
                    parameters=parameters,
                )
            )

    return [types.Tool(function_declarations=function_declarations)]


def _serialize_part(part: types.Part) -> dict[str, Any]:
    """Serialize a Part object to a dict for storage."""
    if part.text is not None:
        return {"text": part.text}
    elif part.function_call is not None:
        return {
            "function_call": {
                "name": part.function_call.name,
                "args": dict(part.function_call.args) if part.function_call.args else {},
            }
        }
    elif part.function_response is not None:
        return {
            "function_response": {
                "name": part.function_response.name,
                "response": (
                    dict(part.function_response.response)
                    if part.function_response.response
                    else {}
                ),
            }
        }
    return {}


@activity.defn
async def generate_content(request: GeminiChatRequest) -> GeminiChatResponse:
    """Execute a Gemini chat completion with tool support.

    This activity wraps the Gemini API call and handles serialization/deserialization
    of Content objects for Temporal workflow compatibility.
    """
    # Create the Gemini client with explicit API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    client = genai.Client(api_key=api_key)

    # Deserialize inputs
    contents = _deserialize_contents(request.contents)
    tools = _deserialize_tools(request.tools)

    # Configure the request with automatic function calling disabled
    # (Temporal handles tool execution, not the SDK)
    config = types.GenerateContentConfig(
        system_instruction=request.system_instruction,
        tools=tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # Make the API call
    response = await client.aio.models.generate_content(
        model=request.model,
        contents=contents,
        config=config,
    )

    # Extract function calls if present
    function_calls = []
    raw_parts = []

    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts:
            raw_parts.append(_serialize_part(part))
            if part.function_call:
                function_calls.append(
                    {
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args) if part.function_call.args else {},
                    }
                )

    return GeminiChatResponse(
        text=response.text if hasattr(response, "text") else None,
        function_calls=function_calls,
        raw_parts=raw_parts,
    )
