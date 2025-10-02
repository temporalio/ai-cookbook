# weather_activities.py

from typing import Any
from temporalio import activity
import httpx
import json
from pydantic import BaseModel
import openai
from helpers import tool_helpers
from pydantic import Field

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

def _alerts_url(state: str) -> str:
    return f"{NWS_API_BASE}/alerts/active/area/{state}"

# External calls happen via activities now
async def _make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }

    async with httpx.AsyncClient() as client:

        response = await client.get(url, headers=headers, timeout=5.0)
        response.raise_for_status()
        return response.json()

# Build the tool for the OpenAI Responses API. We use Pydantic to create a structure
# that encapsulates the input parameters for both the weather alerts activity and the
# tool definition that is passed to the OpenAI Responses API.
class GetWeatherAlertsRequest(BaseModel):
    state: str = Field(description="Two-letter US state code (e.g. CA, NY)")

WEATHER_ALERTS_TOOL_OAI: dict[str, Any] = tool_helpers.oai_responses_tool_from_model(
    "get_weather_alerts",
    "Get weather alerts for a US state.",
    GetWeatherAlertsRequest)

@activity.defn
async def get_weather_alerts(weather_alerts_request: GetWeatherAlertsRequest) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    data = await _make_nws_request(_alerts_url(weather_alerts_request.state))
    return json.dumps(data)
