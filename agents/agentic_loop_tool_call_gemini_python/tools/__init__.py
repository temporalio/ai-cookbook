# ABOUTME: Tool registry for the Gemini-based agentic loop.
# Defines which tools are available and maps tool names to handlers.

import os
from typing import Any, Awaitable, Callable

from google import genai
from google.genai import types

from .get_location import get_location_info, get_ip_address
from .get_weather import get_weather_alerts

ToolHandler = Callable[..., Awaitable[Any]]

# Cache for the generated Tool object
_tools_cache: types.Tool | None = None


def get_handler(tool_name: str) -> ToolHandler:
    """Get the handler function for a given tool name."""
    if tool_name == "get_location_info":
        return get_location_info
    if tool_name == "get_ip_address":
        return get_ip_address
    if tool_name == "get_weather_alerts":
        return get_weather_alerts
    raise ValueError(f"Unknown tool name: {tool_name}")


def get_tools() -> types.Tool:
    """Get the Tool object containing all available function declarations.

    Uses FunctionDeclaration.from_callable() from the Google GenAI SDK to generate 
    tool definitions from the handler functions. The result is cached to avoid repeated
    client creation.
    """
    global _tools_cache
    if _tools_cache is not None:
        return _tools_cache

    # Create client to generate FunctionDeclarations
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    client = genai.Client(api_key=api_key)

    # Generate FunctionDeclarations from callables
    _tools_cache = types.Tool(
        function_declarations=[
            types.FunctionDeclaration.from_callable(
                client=client, callable=get_weather_alerts
            ),
            types.FunctionDeclaration.from_callable(
                client=client, callable=get_location_info
            ),
            types.FunctionDeclaration.from_callable(
                client=client, callable=get_ip_address
            ),
        ]
    )
    return _tools_cache
