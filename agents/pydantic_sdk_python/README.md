<!--
description: Build a durable AI agent with PydanticAI and Temporal that can intelligently choose tools to answer user questions
tags: [agents, python, pydantic-ai]
priority: 750
-->

# Durable Agent with Tools - PydanticAI

This example shows how to build a durable AI agent using [PydanticAI](https://ai.pydantic.dev/) and [Temporal](https://temporal.io/). The agent has access to tools (Temporal Activities) that it can intelligently choose to answer user questions. The agent determines which tools to use based on the user's input and executes them as needed.

This recipe highlights key implementation patterns:

- **Agent-based architecture**: Uses PydanticAI to create an intelligent agent that can reason about which tools to use
- **Model flexibility**: PydanticAI supports multiple LLM providers - use any model supported by the framework (OpenAI, Anthropic, Google, and more)
- **Tool integration**: Tools are registered with the `@agent.tool` decorator and automatically converted to Temporal Activities by the `TemporalAgent` wrapper
- **Durable execution**: The agent's state and execution are managed by Temporal, providing reliability and observability
- **Plugin configuration**: Uses the `PydanticAIPlugin` to configure Temporal for PydanticAI integration
- **Context injection**: Tools receive context through PydanticAI's `RunContext` parameter - prefix with underscore for tools that don't use it

## Prerequisites

- Python 3.11+
- Temporal dev server running ([install guide](https://docs.temporal.io/cli/#install))
- API key for one of: OpenAI, Anthropic, or Google

## Defining Tools with @agent.tool

Tools are registered using PydanticAI's `@agent.tool` decorator. The agent is configured with a `deps_type` that can be passed to tools that need access to shared context.

*File: agents.py*

```python
import httpx
import json
import math
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass

# DocsContext is passed to tools that need access to loaded documentation
@dataclass
class DocsContext:
    docs: dict[str, str]

# SearchResult is a Pydantic model - PydanticAI serializes it automatically
# across Temporal activity boundaries
class SearchResult(BaseModel):
    matching_docs: list[str] = Field(description="List of document titles that match")
    total_matches: int = Field(description="Total number of matching documents")

documentation_agent = Agent(
    model_name,
    deps_type=DocsContext,
    name='documentation_agent',
    system_prompt="...",
)

# Tools that use context receive it as the first argument
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
    return SearchResult(matching_docs=matching_docs, total_matches=len(matching_docs))

# Tools that don't use context still require the parameter - prefix with _ to
# signal it's intentionally unused
@documentation_agent.tool
async def get_ip_address(_ctx: RunContext[DocsContext]) -> str:
    """Get the public IP address of the current machine."""
    async with httpx.AsyncClient() as client:
        response = await client.get("https://icanhazip.com")
        response.raise_for_status()
        return response.text.strip()

@documentation_agent.tool
async def get_location_info(_ctx: RunContext[DocsContext], ipaddress: str) -> str:
    """Get the location information for an IP address, including city, state, and country."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://ip-api.com/json/{ipaddress}")
        response.raise_for_status()
        result = response.json()
        return f"{result['city']}, {result['regionName']}, {result['country']}"

@documentation_agent.tool
async def get_weather_alerts(_ctx: RunContext[DocsContext], state: str) -> str:
    """Get active weather alerts for a US state."""
    url = f"https://api.weather.gov/alerts/active/area/{state}"
    headers = {"User-Agent": "weather-app/1.0", "Accept": "application/geo+json"}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, timeout=5.0)
        response.raise_for_status()
        return json.dumps(response.json())

@documentation_agent.tool
async def calculate_circle_area(_ctx: RunContext[DocsContext], radius: float) -> float:
    """Calculate the area of a circle given its radius."""
    return math.pi * radius ** 2
```

The agent has access to six tools:
- **`search_documentation(keywords: list[str])`** - Search through available documentation
- **`list_available_docs()`** - List all available documentation files
- **`get_ip_address()`** - Get the public IP address of the current machine
- **`get_location_info(ipaddress: str)`** - Get city, state, and country for an IP address
- **`get_weather_alerts(state: str)`** - Get active NWS weather alerts for a US state (e.g. CA, NY)
- **`calculate_circle_area(radius: float)`** - Calculate circle area (demonstration tool)

## How the Agentic Loop Works

The agentic loop is an autonomous cycle where the AI agent makes decisions about what to do next based on its goal:

**User gives goal → Agent analyzes → Agent calls tool → Agent receives result → Agent decides → Calls another tool OR returns final answer**

The key is **autonomy** - you don't tell the agent which tools to call or when. You provide a goal and the agent figures out the rest.

Here's what happens when you ask "Are there any weather alerts where I am?":

```
┌───────────────────────────────────────────────────────┐
│ User: "Are there any weather alerts where I am?"      │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
                ┌───────────────┐
                │  LLM Call #1  │
                │ Analyzes query│
                │ Returns: TOOL │ ────┐
                └───────────────┘     │
                        ▲             │
                        │             ▼
                        │     ┌──────────────────────────┐
                        │     │ Execute Tool Call        │
                        │     │ (Temporal Activity)      │
                        │     │                          │
                        │     │ get_ip_address()         │
                        │     │ → GET icanhazip.com      │
                        │     │ Returns: "203.0.113.42"  │
                        │     └──────────┬───────────────┘
                        │                │
                        │                ▼
                ┌───────┴────────────────────┐
                │      LLM Call #2           │
                │ Receives IP address        │
                │ Returns: TOOL              │ ────┐
                └────────────────────────────┘     │
                        ▲                          │
                        │                          ▼
                        │     ┌──────────────────────────────────┐
                        │     │ Execute Tool Call                │
                        │     │ (Temporal Activity)              │
                        │     │                                  │
                        │     │ get_location_info("203.0.113.42")│
                        │     │ → GET ip-api.com/json/...        │
                        │     │ Returns: "San Francisco,         │
                        │     │  California, United States"      │
                        │     └──────────┬───────────────────────┘
                        │                │
                        │                ▼
                ┌───────┴────────────────────┐
                │      LLM Call #3           │
                │ Receives location          │
                │ Returns: TOOL              │ ────┐
                └────────────────────────────┘     │
                        ▲                          │
                        │                          ▼
                        │          ┌──────────────────────────┐
                        │          │ Execute Tool Call        │
                        │          │ (Temporal Activity)      │
                        │          │                          │
                        │          │ get_weather_alerts("CA") │
                        │          │ → GET api.weather.gov/   │
                        │          │   alerts/active/area/CA  │
                        │          │ Returns: live NWS alert  │
                        │          │ JSON for California      │
                        │          └──────────┬───────────────┘
                        │                     │
                        │                     ▼
                ┌───────┴────────────────────────┐
                │      LLM Call #4               │
                │ Receives alert data            │
                │ Returns: TEXT                  │
                └───────────┬────────────────────┘
                            │
                            ▼
            ┌────────────────────────────────────────┐
            │ "Based on your location in San         │
            │ Francisco, CA, there are currently     │
            │ flood warnings in several Northern     │
            │ California counties and a wind         │
            │ advisory near Lake Tahoe."             │
            └────────────────────────────────────────┘
```

### What Happens Inside `temporal_agent.run()`

When you call `temporal_agent.run(prompt, deps=DocsContext(docs))`, this entire loop happens in a single function call. PydanticAI handles the tool registration, decision making, and execution:

**Tool Registration**: When you use `@agent.tool`, PydanticAI extracts the function signature and docstring to create a schema the LLM understands. The agent knows what each tool does and what arguments it requires.

**Loop Iteration 1:**
- Agent receives: "Are there any weather alerts where I am?"
- Agent thinks: "I need to find their location. First, get their IP address."
- LLM calls `get_ip_address()` — converted to a Temporal activity by `TemporalAgent`
- Tool calls `icanhazip.com` and returns the public IP

**Loop Iteration 2:**
- Agent receives the IP address
- Agent thinks: "Now I can resolve this IP to a location."
- Agent calls `get_location_info("203.0.113.42")`
- Tool calls `ip-api.com` and returns `"San Francisco, California, United States"`

**Loop Iteration 3:**
- Agent receives the location
- Agent thinks: "Now I can look up weather alerts for California."
- Agent calls `get_weather_alerts("CA")`
- Tool calls `api.weather.gov` and returns live NWS alert JSON

**Loop Iteration 4:**
- Agent receives the alert data
- Agent thinks: "I have everything I need to answer."
- LLM returns TEXT instead of TOOL — loop ends
- Agent returns a natural language summary of the alerts for the user's location

### Why This Matters

Write `agent.run(prompt)` instead of orchestration logic with if/else statements - the LLM decides which tools to call and when to stop. The same code handles questions needing zero tools, one tool, or multiple tools chained together. PydanticAI's structured outputs mean tools return type-safe Pydantic models instead of dictionaries, and you can swap between OpenAI, Anthropic, or Google without changing code.

Temporal makes each tool call a durable activity with retry logic and full observability in the Web UI. The PydanticAI integration handles serialization across activity boundaries automatically, so your Pydantic models just work across the distributed execution.

## Create the Workflow

The workflow wraps the PydanticAI agent with `TemporalAgent` to enable durable execution. The agent processes user input and autonomously decides which tools to use.

*File: workflow.py*

```python
from temporalio import workflow
from datetime import timedelta
from pydantic_ai.durable_exec.temporal import PydanticAIWorkflow, TemporalAgent

from agents import documentation_agent, DocsContext

temporal_agent = TemporalAgent(documentation_agent)

@workflow.defn
class DocumentationAgent(PydanticAIWorkflow):
    """Documentation Q&A agent demonstrating autonomous tool calling."""

    __pydantic_ai_agents__ = [temporal_agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        # Load documentation
        docs = await workflow.execute_activity(
            load_docs,
            start_to_close_timeout=timedelta(seconds=30),
        )

        # Run agent - it will autonomously call tools as needed
        result = await temporal_agent.run(
            prompt,
            deps=DocsContext(docs=docs)
        )

        return result.output
```

## Create the Worker

The worker is configured with the `PydanticAIPlugin` to handle agent tool activities.

*File: worker.py*

```python
import asyncio
from temporalio.client import Client
from temporalio.worker import Worker
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin

from workflow import DocumentationAgent, load_docs

async def main():
    client = await Client.connect(
        "localhost:7233",
        plugins=[PydanticAIPlugin()],
    )

    # Tool activities are auto-registered by PydanticAIPlugin
    worker = Worker(
        client,
        task_queue="docs-agent-queue",
        workflows=[DocumentationAgent],
        activities=[load_docs],
    )

    print("Worker started. Waiting for workflows...")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
```

## Create the Workflow Starter

The starter script submits the agent workflow to Temporal for execution, then waits for the result.

*File: start_workflow.py*

```python
import asyncio
from temporalio.client import Client
from temporalio.common import WorkflowIDReusePolicy
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin
from rich.console import Console
from rich.panel import Panel

from workflow import DocumentationAgent

console = Console()

async def main():
    console.print("\n[cyan]Connecting to Temporal...[/cyan]")
    client = await Client.connect(
        "localhost:7233",
        plugins=[PydanticAIPlugin()],
    )

    user_input = console.input("\n[bold yellow]Enter a question:[/bold yellow] ")

    console.print("\n[dim]Starting agent workflow...[/dim]")
    result = await client.execute_workflow(
        DocumentationAgent.run,
        user_input,
        id="docs-agent-workflow",
        task_queue="docs-agent-queue",
        id_reuse_policy=WorkflowIDReusePolicy.TERMINATE_IF_RUNNING,
    )

    console.print()
    console.print(Panel(
        result,
        title="[bold green]Result[/bold green]",
        border_style="green",
    ))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user[/yellow]")
```

## Running

Start the Temporal Dev Server:

```bash
temporal server start-dev
```

Install dependencies:

```bash
cd agents/pydantic_sdk_python
uv sync
```

Create a `docs/` directory and add markdown files for the agent to search:

```bash
mkdir docs
curl -o docs/workflows.md https://raw.githubusercontent.com/temporalio/documentation/main/docs/develop/python/core-application.md
```

Configure an API key:

```bash
cp .env.example .env
# Edit .env and set one of:
#   OPENAI_API_KEY=sk-your-key-here
#   ANTHROPIC_API_KEY=sk-ant-your-key-here
#   GOOGLE_API_KEY=your-key-here
```

Run the worker:

```bash
uv run python worker.py
```

Start execution:

```bash
uv run python start_workflow.py
```

## Example Interactions

Try asking the agent questions like:

- "Are there any weather alerts where I am?"
- "What is my IP address?"
- "What documentation is available?"
- "Search for information about workflows"
- "What are the weather alerts for TX?"
- "Calculate the area of a circle with radius 5"

The agent will determine which tools to use and provide intelligent responses. Watch the worker terminal to see tool execution logs, or visit the [Temporal Web UI](http://localhost:8233) to view the complete workflow execution history.

## Related Recipes

- [OpenAI Agents SDK](../openai_agents_sdk_python/) - Similar agentic loop pattern with OpenAI
- [Tool Calling Agent](../tool_call_openai_python/) - Foundation for basic tool calling patterns

## Resources

- [PydanticAI Documentation](https://ai.pydantic.dev/) - Official docs for PydanticAI framework
- [PydanticAI with Temporal](https://ai.pydantic.dev/agents/#temporalagent) - TemporalAgent integration guide
- [PydanticAI Tools](https://ai.pydantic.dev/tools/) - Guide to defining and using tools
- [VibeCheck: Building a production AI agent system with Pydantic](https://www.youtube.com/watch?v=3rpwaKQXI7A) - Live coding session showcasing Pydantic Temporal support
