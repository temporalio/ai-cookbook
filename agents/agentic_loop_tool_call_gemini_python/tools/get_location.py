# ABOUTME: Location-related tools for IP geolocation.
# Provides tools to get the current IP address and look up location info for an IP.

import httpx
from pydantic import BaseModel, Field

from helpers import tool_helpers


class GetLocationRequest(BaseModel):
    """Request model for getting location info from an IP address."""

    ipaddress: str = Field(description="An IP address")


# Build the tool definitions for the Gemini API
GET_LOCATION_TOOL_GEMINI = tool_helpers.gemini_tool_from_model(
    "get_location_info",
    "Get the location information for an IP address. This includes the city, state, and country.",
    GetLocationRequest,
)

GET_IP_ADDRESS_TOOL_GEMINI = tool_helpers.gemini_tool_from_model(
    "get_ip_address",
    "Get the IP address of the current machine.",
    None,
)


async def get_ip_address() -> str:
    """Get the public IP address of the current machine."""
    async with httpx.AsyncClient() as client:
        response = await client.get("https://icanhazip.com")
        response.raise_for_status()
        return response.text.strip()


async def get_location_info(req: GetLocationRequest) -> str:
    """Get the location information for an IP address."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://ip-api.com/json/{req.ipaddress}")
        response.raise_for_status()
        result = response.json()
        return f"{result['city']}, {result['regionName']}, {result['country']}"
