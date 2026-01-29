from typing import Any, Awaitable, Callable

# Location and weather related tools
from .get_location import get_location_info, get_ip_address
from .get_weather import get_weather_alerts
from . import get_weather
from . import get_location

ToolHandler = Callable[..., Awaitable[Any]]

def get_handler(tool_name: str) -> ToolHandler:
    if tool_name == "get_location_info":
        return get_location_info
    if tool_name == "get_ip_address":
        return get_ip_address
    if tool_name == "get_weather_alerts":
        return get_weather_alerts
    raise ValueError(f"Unknown tool name: {tool_name}")

def get_tools() -> list[dict[str, Any]]:
    return [
        get_weather.WEATHER_ALERTS_TOOL_CLAUDE,
        get_location.GET_LOCATION_TOOL_CLAUDE,
        get_location.GET_IP_ADDRESS_TOOL_CLAUDE
    ]

