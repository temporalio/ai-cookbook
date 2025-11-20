from typing import Any
import httpx
from pydantic import BaseModel, Field

# For the location finder we use Pydantic to create a structure that encapsulates the input parameter
# (an IP address).
# This is used for both the location finding function and to craft the tool definitions that
# are passed to the Gemini API.
class GetLocationRequest(BaseModel):
    ipaddress: str = Field(description="An IP address")

# Build the tool definitions for the Gemini API.
GET_LOCATION_TOOL_GEMINI = {
    "function_declarations": [
        {
            "name": "get_location_info",
            "description": "Get the location information for an IP address. This includes the city, state, and country. For user location queries, first call get_ip_address to obtain the IP, then use this tool with that IP address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ipaddress": {
                        "type": "string",
                        "description": "An IP address"
                    }
                },
                "required": ["ipaddress"],
            },
        }
    ]
}

GET_IP_ADDRESS_TOOL_GEMINI = {
    "function_declarations": [
        {
            "name": "get_ip_address",
            "description": "Get the IP address of the current machine. Use this tool first when answering location-related questions (like 'where am I?').",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }
    ]
}

# The functions
async def get_ip_address() -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get("https://icanhazip.com")
        response.raise_for_status()
        return response.text.strip()

async def get_location_info(req: GetLocationRequest) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://ip-api.com/json/{req.ipaddress}")
        response.raise_for_status()
        result = response.json()
        return f"{result['city']}, {result['regionName']}, {result['country']}"
