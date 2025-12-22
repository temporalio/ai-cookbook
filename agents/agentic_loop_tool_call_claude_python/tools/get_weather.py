# get_weather.py

from typing import Any
import json
from pydantic import BaseModel, Field
from helpers import tool_helpers
import httpx

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

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

# Build the tool for the Claude API. We use Pydantic to create a structure
# that encapsulates the input parameters for both the weather alerts activity and the
# tool definition that is passed to Claude.

class GetWeatherAlertsRequest(BaseModel):
    state: str = Field(description="Two-letter US state code (e.g. CA, NY)")

# Build the tool definition for Claude
WEATHER_ALERTS_TOOL_CLAUDE: dict[str, Any] = tool_helpers.claude_tool_from_model(
    "get_weather_alerts",
    "Get weather alerts for a US state.",
    GetWeatherAlertsRequest)

# The function
async def get_weather_alerts(weather_alerts_request: GetWeatherAlertsRequest) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{weather_alerts_request.state}"
    data = await _make_nws_request(url)
    return json.dumps(data)

