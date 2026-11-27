from contextlib import asynccontextmanager
from dataclasses import dataclass

import math

from temporalio import activity
from temporalio.contrib.workflow_streams import WorkflowStreamClient

MODEL_EVENTS_TOPIC = "model-events"
TOOL_EVENTS_TOPIC = "tool-events"


# Temporal best practice: Create a data structure to hold the request parameters.
@dataclass
class Weather:
    city: str
    temperature_range: str
    conditions: str


@asynccontextmanager
async def _tool_events():
    """Best-effort publisher for the tool-events topic.

    Tool activities are the source of truth for their own results, so we
    publish here (rather than relying on the model-events topic, which only
    carries the model's raw request/response events) to show a subscriber
    the tool call and its result as they actually happen. Falls back to a
    no-op outside of a live activity-with-client context (e.g. when a test
    calls these functions directly via ActivityEnvironment) so the tools
    stay unit-testable without a running Temporal server.
    """
    try:
        stream = WorkflowStreamClient.from_within_activity()
    except RuntimeError:
        yield None
        return
    async with stream:
        yield stream.topic(TOOL_EVENTS_TOPIC)


@activity.defn
async def get_weather(city: str) -> Weather:
    """Get the weather for a given city."""
    async with _tool_events() as tool_events:
        if tool_events:
            tool_events.publish(f"-> calling get_weather(city={city!r})")
        result = Weather(city=city, temperature_range="14-20C", conditions="Sunny with wind.")
        if tool_events:
            tool_events.publish(f"<- get_weather returned: {result}")
    return result


@activity.defn
async def calculate_circle_area(radius: float) -> float:
    """Calculate the area of a circle given its radius."""
    async with _tool_events() as tool_events:
        if tool_events:
            tool_events.publish(f"-> calling calculate_circle_area(radius={radius})")
        result = math.pi * radius**2
        if tool_events:
            tool_events.publish(f"<- calculate_circle_area returned: {result}")
    return result
