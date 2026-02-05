# ABOUTME: Tool registry for the Gemini-based agentic loop.
# Defines which tools are available and maps tool names to handlers.

from typing import Any, Awaitable, Callable
from google.genai import types

from .get_location import (
    get_location_info,
    get_ip_address,
    GET_LOCATION_TOOL_GEMINI,
    GET_IP_ADDRESS_TOOL_GEMINI,
)
from .get_weather import get_weather_alerts, WEATHER_ALERTS_TOOL_GEMINI

ToolHandler = Callable[..., Awaitable[Any]]


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
    """Get the Tool object containing all available function declarations."""
    return types.Tool(
        function_declarations=[
            WEATHER_ALERTS_TOOL_GEMINI,
            GET_LOCATION_TOOL_GEMINI,
            GET_IP_ADDRESS_TOOL_GEMINI,
        ]
    )
