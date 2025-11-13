from typing import Any
import json
from pydantic import BaseModel, Field
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

# Build the tool for the Gemini API. We use Pydantic to create a structure
# that encapsulates the input parameters for both the weather alerts activity and the
# tool definition that is passed to the Gemini API.

# For the weather alerts tool we use Pydantic to create a structure that encapsulates
# the input parameter (a US state code).
# This is used for both the weather alerts function and to craft the tool definition
# that is passed to the Gemini API.
class GetWeatherAlertsRequest(BaseModel):
    state: str = Field(description="Two-letter US state code (e.g. CA, NY)")

# Build the tool definition for the Gemini API.
GET_WEATHER_ALERTS_TOOL_GEMINI = {
    "function_declarations": [
        {
            "name": "get_weather_alerts",
            "description": "Get weather alerts for a US state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "description": "Two-letter US state code (e.g. CA, NY)"
                    }
                },
                "required": ["state"],
            },
        }
    ]
}

# The function
async def get_weather_alerts(weather_alerts_request: GetWeatherAlertsRequest) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{weather_alerts_request.state}"
    data = await _make_nws_request(url)
    return json.dumps(data)
