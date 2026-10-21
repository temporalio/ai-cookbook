from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from activities.tools import (
    GetTimeRequest,
    GetWeatherRequest,
    get_time,
    get_weather,
)


@dataclass
class ToolSpec:
    name: str
    description: str
    activity: Callable[..., Any]
    request_model: type[BaseModel]


# The registry is the single place tools are declared. The workflow uses it both to
# build the Claude tool definitions and to map a requested tool name back to the
# Activity (and request model) that runs it.
TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="get_weather",
        description="Get the current weather for a city.",
        activity=get_weather,
        request_model=GetWeatherRequest,
    ),
    ToolSpec(
        name="get_time",
        description="Get the current local time for an IANA timezone.",
        activity=get_time,
        request_model=GetTimeRequest,
    ),
]

_TOOLS_BY_NAME = {spec.name: spec for spec in TOOLS}


def get_tool(name: str) -> ToolSpec:
    return _TOOLS_BY_NAME[name]


def claude_tool_definitions() -> list[dict[str, Any]]:
    # Claude expects an `input_schema` JSON Schema per tool. We derive it from each
    # tool's Pydantic request model so the schema and the Activity signature stay in sync.
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.request_model.model_json_schema(),
        }
        for spec in TOOLS
    ]
