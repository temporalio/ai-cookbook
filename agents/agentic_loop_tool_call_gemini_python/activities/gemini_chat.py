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
    contents: list[types.Content]
    tools: list[types.Tool]


@dataclass
class GeminiChatResponse:
    """Response from a Gemini chat completion."""

    text: str | None  # The text response, if any
    function_calls: list[dict[str, Any]]  # List of function calls (name and args)
    raw_parts: list[types.Part]  # Raw parts for conversation history


@activity.defn
async def generate_content(request: GeminiChatRequest) -> GeminiChatResponse:
    """Execute a Gemini chat completion with tool support."""
    # Create the Gemini client with explicit API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )

    # Configure the request with automatic function calling disabled
    # (Temporal handles tool execution, not the SDK)
    config = types.GenerateContentConfig(
        system_instruction=request.system_instruction,
        tools=request.tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # Make the API call
    response = await client.aio.models.generate_content(
        model=request.model,
        contents=request.contents,
        config=config,
    )

    # Extract function calls and text from response parts
    function_calls = []
    raw_parts = []
    text_parts = []

    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts:
            raw_parts.append(part)
            if part.function_call:
                function_calls.append(
                    {
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args) if part.function_call.args else {},
                    }
                )
            elif part.text:
                text_parts.append(part.text)

    # Only include text if there are no function calls (avoids SDK warning)
    text = "".join(text_parts) if text_parts and not function_calls else None

    return GeminiChatResponse(
        text=text,
        function_calls=function_calls,
        raw_parts=raw_parts,
    )
