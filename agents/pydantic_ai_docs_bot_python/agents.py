"""PydanticAI agent with tools for documentation Q&A."""

from dotenv import load_dotenv
load_dotenv()

import os
import math
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


class WeatherInfo(BaseModel):
    """Weather information for a city."""
    city: str
    temperature_range: str
    conditions: str


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

# Create agent with tools
documentation_agent = Agent(
    _select_model(),
    deps_type=DocsContext,
    name='documentation_agent',  # Required for Temporal
    system_prompt="""You are a helpful documentation assistant that can answer questions about technical documentation.

You have access to several tools:
- search_documentation: Search through available documentation by keywords
- list_available_docs: See what documentation is available
- get_weather: Get weather information for a city (for demonstration)
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
    """Search documentation by keywords.

    Args:
        ctx: Context containing documentation
        keywords: List of keywords to search for

    Returns:
        SearchResult with matching document titles
    """
    print(f"🔍 Tool called: search_documentation(keywords={keywords})")

    matching_docs = []

    for title, content in ctx.deps.docs.items():
        # Check if any keyword appears in the doc
        if any(keyword.lower() in content.lower() for keyword in keywords):
            matching_docs.append(title)

    result = SearchResult(
        matching_docs=matching_docs,
        total_matches=len(matching_docs)
    )

    print(f"   ✓ Found {len(matching_docs)} matching documents")
    return result


@documentation_agent.tool
async def list_available_docs(ctx: RunContext[DocsContext]) -> list[str]:
    """List all available documentation files.

    Args:
        ctx: Context containing documentation

    Returns:
        List of document titles
    """
    print(f"📋 Tool called: list_available_docs()")

    docs = list(ctx.deps.docs.keys())
    print(f"   ✓ Found {len(docs)} documents: {', '.join(docs)}")

    return docs


@documentation_agent.tool
async def get_weather(city: str) -> WeatherInfo:
    """Get weather information for a city.

    This is a demonstration tool showing how agents can call multiple different tools.

    Args:
        city: City name

    Returns:
        Weather information
    """
    print(f"🌤️  Tool called: get_weather(city='{city}')")

    # Mock weather data
    weather = WeatherInfo(
        city=city,
        temperature_range="14-20C",
        conditions="Sunny with wind"
    )

    print(f"   ✓ Weather: {weather.temperature_range}, {weather.conditions}")
    return weather


@documentation_agent.tool
async def calculate_circle_area(radius: float) -> float:
    """Calculate the area of a circle given its radius.

    This is a demonstration tool showing how agents can call mathematical tools.

    Args:
        radius: Circle radius

    Returns:
        Circle area
    """
    print(f"🔢 Tool called: calculate_circle_area(radius={radius})")

    area = math.pi * radius ** 2
    print(f"   ✓ Area: {area:.2f}")

    return area


