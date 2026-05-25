"""PydanticAI agent with tools for documentation Q&A."""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import math
import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass

# ============================================================================
# Dependencies for Tools
# ============================================================================

@dataclass
class DocsContext:
    """Context passed to agent tools."""
    docs: dict[str, str]


# ============================================================================
# Pydantic Models for Tool Results
# ============================================================================

class SearchResult(BaseModel):
    """Result from searching documentation."""
    matching_docs: list[str] = Field(description="List of document titles that match")
    total_matches: int = Field(description="Total number of matching documents")


# ============================================================================
# Agent Configuration
# ============================================================================

def _select_model() -> str:
    """Select model based on available API keys.

    Returns:
        Model string for PydanticAI
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic:claude-sonnet-4-5"
    elif os.getenv("OPENAI_API_KEY"):
        return "openai:gpt-4o"
    elif os.getenv("GOOGLE_API_KEY"):
        return "google:gemini-2.0-flash-exp"
    else:
        raise ValueError(
            "No API key found. Set one of:\n"
            "  OPENAI_API_KEY=sk-...\n"
            "  ANTHROPIC_API_KEY=sk-ant-...\n"
            "  GOOGLE_API_KEY=..."
        )


# ============================================================================
# PydanticAI Agent with Tools
# ============================================================================

documentation_agent = Agent(
    _select_model(),
    deps_type=DocsContext,
    name='documentation_agent',  # Required for Temporal
    system_prompt="""You are a helpful documentation assistant that can answer questions about technical documentation.

You have access to several tools:
- search_documentation: Search through available documentation by keywords
- list_available_docs: See what documentation is available
- get_ip_address: Get the public IP address of the current machine
- get_location_info: Get city, state, and country for an IP address
- get_weather_alerts: Get active weather alerts for a US state (e.g. CA, NY)
- calculate_circle_area: Calculate the area of a circle (for demonstration)

Use these tools to help answer questions. You can call multiple tools in sequence if needed.
For documentation questions, start by searching or listing available docs.
Be clear, concise, and helpful in your answers.""",
)


@documentation_agent.tool
async def search_documentation(
    ctx: RunContext[DocsContext],
    keywords: list[str]
) -> SearchResult:
    """Search documentation by keywords."""
    matching_docs = []
    for title, content in ctx.deps.docs.items():
        if any(keyword.lower() in content.lower() for keyword in keywords):
            matching_docs.append(title)

    return SearchResult(
        matching_docs=matching_docs,
        total_matches=len(matching_docs)
    )


@documentation_agent.tool
async def list_available_docs(ctx: RunContext[DocsContext]) -> list[str]:
    """List all available documentation files."""
    return list(ctx.deps.docs.keys())


@documentation_agent.tool
async def get_ip_address(_ctx: RunContext[DocsContext]) -> str:
    """Get the public IP address of the current machine."""
    async with httpx.AsyncClient() as client:
        response = await client.get("https://icanhazip.com")
        response.raise_for_status()
        return response.text.strip()


@documentation_agent.tool
async def get_location_info(_ctx: RunContext[DocsContext], ipaddress: str) -> str:
    """Get the location information for an IP address, including city, state, and country.

    Args:
        ipaddress: An IP address
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://ip-api.com/json/{ipaddress}")
        response.raise_for_status()
        result = response.json()
        return f"{result['city']}, {result['regionName']}, {result['country']}"


NWS_API_BASE = "https://api.weather.gov"
NWS_USER_AGENT = "weather-app/1.0"


@documentation_agent.tool
async def get_weather_alerts(_ctx: RunContext[DocsContext], state: str) -> str:
    """Get active weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    headers = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, timeout=5.0)
        response.raise_for_status()
        return json.dumps(response.json())


@documentation_agent.tool
async def calculate_circle_area(_ctx: RunContext[DocsContext], radius: float) -> float:
    """Calculate the area of a circle given its radius."""
    return math.pi * radius ** 2


