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
- **Structured outputs**: Tools return Pydantic models for type-safe, validated responses that automatically serialize across Temporal activity boundaries
- **Tool integration**: Tools are registered with the `@agent.tool` decorator and automatically converted to Temporal Activities by the `TemporalAgent` wrapper
- **Durable execution**: The agent's state and execution are managed by Temporal, providing reliability and observability
- **Plugin configuration**: Uses the `PydanticAIPlugin` to configure Temporal for PydanticAI integration
- **Optional dependencies**: Tools can optionally receive context through PydanticAI's `RunContext` parameter - only include it if the tool needs it

## Prerequisites

- Python 3.11+
- Temporal dev server running ([install guide](https://docs.temporal.io/cli/#install))
- API key for one of: OpenAI, Anthropic, or Google

## Defining Tools with @agent.tool

Tools are registered using PydanticAI's `@agent.tool` decorator. The agent is configured with a `deps_type` that can be passed to tools that need access to shared context.

*File: agents.py*

```python
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass

@dataclass
class DocsContext:
    """Context passed to agent tools."""
    docs: dict[str, str]

documentation_agent = Agent(
    model_name,
    deps_type=DocsContext,
    name='documentation_agent',
    system_prompt="""You are a helpful documentation assistant...

Use these tools to help answer questions. You can call multiple tools
in sequence if needed.""",
)

# Tools that need context include the RunContext parameter
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

# Tools that don't need context can omit the RunContext parameter
@documentation_agent.tool
async def calculate_circle_area(radius: float) -> float:
    """Calculate the area of a circle."""
    return math.pi * radius ** 2
```

The agent has access to four tools:
- **`search_documentation(keywords: list[str])`** - Search through available documentation
- **`list_available_docs()`** - List all available documentation files
- **`get_weather(city: str)`** - Get weather information (demonstration tool)
- **`calculate_circle_area(radius: float)`** - Calculate circle area (demonstration tool)

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

from workflow import DocumentationAgent

async def main():
    client = await Client.connect(
        "localhost:7233",
        plugins=[PydanticAIPlugin()],
    )

    user_input = input("Enter a question: ")

    result = await client.execute_workflow(
        DocumentationAgent.run,
        user_input,
        id="docs-agent-workflow",
        task_queue="docs-agent-queue",
        id_reuse_policy=WorkflowIDReusePolicy.TERMINATE_IF_RUNNING,
    )

    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Running

Start the Temporal Dev Server:

```bash
temporal server start-dev
```

Install dependencies:

```bash
cd agents/pydantic_ai_docs_bot_python
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

- "What documentation is available?"
- "Search for information about workflows"
- "Calculate the area of a circle with radius 5"
- "What's the weather in Tokyo and calculate the area of a circle with radius 3"

The agent will determine which tools to use and provide intelligent responses. Watch the worker terminal to see tool execution logs, or visit the [Temporal Web UI](http://localhost:8233) to view the complete workflow execution history.

## Related Recipes

- [OpenAI Agents SDK](../openai_agents_sdk_python/) - Similar agentic loop pattern with OpenAI
- [Tool Calling Agent](../tool_call_openai_python/) - Foundation for basic tool calling patterns

## Resources

- [PydanticAI Documentation](https://ai.pydantic.dev/) - Official docs for PydanticAI framework
- [PydanticAI with Temporal](https://ai.pydantic.dev/agents/#temporalagent) - TemporalAgent integration guide
- [PydanticAI Tools](https://ai.pydantic.dev/tools/) - Guide to defining and using tools
- [VibeCheck: Building a production AI agent system with Pydantic](https://www.youtube.com/watch?v=3rpwaKQXI7A) - Live coding session showcasing Pydantic Temporal support
