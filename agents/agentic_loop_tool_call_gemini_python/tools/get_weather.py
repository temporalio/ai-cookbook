# ABOUTME: Weather alert tool using the National Weather Service API.
# Provides a tool to get active weather alerts for a US state.

import json
from typing import Any

import httpx
from pydantic import BaseModel, Field

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"


async def _make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, timeout=5.0)
        response.raise_for_status()
        return response.json()


class GetWeatherAlertsRequest(BaseModel):
    """Request model for getting weather alerts."""

    state: str = Field(description="Two-letter US state code (e.g. CA, NY)")


async def get_weather_alerts(request: GetWeatherAlertsRequest) -> str:
    """Get weather alerts for a US state.

    Args:
        request: The request object containing:
            - state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{request.state}"
    data = await _make_nws_request(url)
    return json.dumps(data)
