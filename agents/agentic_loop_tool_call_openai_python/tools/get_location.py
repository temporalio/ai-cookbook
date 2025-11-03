# get_location.py

from typing import Any
import requests
from pydantic import BaseModel, Field
from helpers import tool_helpers

# For the location finder we use Pydantic to create a structure that encapsulates the input parameter 
# (an IP address). 
# This is used for both the location finding function and to craft the tool definitions that 
# are passed to the OpenAI Responses API.
class GetLocationRequest(BaseModel):
    ipaddress: str = Field(description="An IP address")

# Build the tool definitions for the OpenAI Responses API. 
GET_LOCATION_TOOL_OAI: dict[str, Any] = tool_helpers.oai_responses_tool_from_model(
    "get_location_info",
    "Get the location information for an IP address. This includes the city, state, and country.",
    GetLocationRequest)

GET_IP_ADDRESS_TOOL_OAI: dict[str, Any] = tool_helpers.oai_responses_tool_from_model(
    "get_ip_address",
    "Get the IP address of the current machine.",
    None)

# The functions
def get_ip_address() -> str:
    response = requests.get("https://icanhazip.com")
    response.raise_for_status()
    return response.text.strip()

def get_location_info(req: GetLocationRequest) -> str:
    response = requests.get(f"http://ip-api.com/json/{req.ipaddress}")
    response.raise_for_status()
    result = response.json()
    return f"{result['city']}, {result['regionName']}, {result['country']}"
