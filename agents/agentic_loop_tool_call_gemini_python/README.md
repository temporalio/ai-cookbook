<!--
description: Agentic loop with tool calling using Google Gemini and Temporal
tags: [agents, python, gemini, google-genai]
priority: 750
-->

# Durable Agent: Agentic Loop with Tool Calling (Gemini/Python)

This example demonstrates how to build a durable AI agent using Google's Gemini model with Temporal for orchestration. The agent can use tools to accomplish tasks, with Temporal providing durability and fault tolerance.

## Features

- **Agentic Loop**: The agent iteratively calls the LLM and executes tools until it has enough information to respond
- **Tool Independence**: The agentic loop is completely decoupled from the tools - add or remove tools without changing agent code
- **Google GenAI SDK**: Uses Google's official `google-genai` Python SDK with native Content/Part abstractions
- **Durable Execution**: Temporal ensures the agent can recover from failures at any point
- **Dynamic Activities**: Tools are executed via Temporal's dynamic activity feature
- **Pydantic Tool Parameters**: Tools use Pydantic models for parameters, following Temporal best practices

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AgentWorkflow                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Agentic Loop                          │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │   │
│  │  │  LLM Call    │───▶│  Tool Check  │───▶│  Execute  │  │   │
│  │  │  (Activity)  │    │              │    │  Tool     │  │   │
│  │  └──────────────┘    └──────────────┘    └───────────┘  │   │
│  │         ▲                                      │         │   │
│  │         └──────────────────────────────────────┘         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Google AI API key (get one at https://aistudio.google.com/apikey)
- Temporal server running locally

## Setup

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Create a `.env` file** with your Google API key:
   ```bash
   echo "GOOGLE_API_KEY=your-api-key-here" > .env
   ```

   Note: Only the worker needs the API key. The client does not require it.

3. **Start the Temporal dev server** (in a separate terminal):
   ```bash
   temporal server start-dev
   ```

## Running the Example

1. **Start the worker** (in one terminal):
   ```bash
   uv run python -m worker
   ```

2. **Run a query** (in another terminal):
   ```bash
   uv run python -m start_workflow "What's the weather like where I am?"
   ```

   Or with a custom query:
   ```bash
   uv run python -m start_workflow "Are there any weather alerts in California?"
   ```

## Available Tools

The agent has access to three tools:

| Tool | Description |
|------|-------------|
| `get_ip_address` | Gets the public IP address of the machine |
| `get_location_info` | Looks up geographic location for an IP address |
| `get_weather_alerts` | Fetches active weather alerts for a US state |

## Project Structure

```
agentic_loop_tool_call_gemini_python/
├── pyproject.toml          # Dependencies
├── .env                    # API key (create this, not checked in)
├── worker.py               # Temporal worker
├── start_workflow.py       # Client to start workflows
├── workflows/
│   └── agent.py            # Agentic loop workflow
├── activities/
│   ├── gemini_chat.py      # Gemini API activity
│   └── tool_invoker.py     # Dynamic tool execution
├── agent_config/
│   └── prompts.py          # System prompts defining agent behavior
└── tools/
    ├── __init__.py         # Tool registry and generation
    ├── get_location.py     # Location tools
    └── get_weather.py      # Weather tool
```

## How It Works

1. **Worker starts** and initializes tool definitions (caches them)
2. **User submits a query** via `start_workflow.py`
3. **Workflow starts** the agentic loop with the user's message in conversation history
4. **LLM activity** sends the conversation to Gemini with available tools
5. **If Gemini requests a tool call**:
   - The tool is executed via a dynamic activity
   - The result is added to conversation history
   - Loop continues
6. **If Gemini returns text** (no tool calls):
   - The response is returned to the user
   - Workflow completes

## Key Design Decisions

### Tool Definition Generation

Tool definitions are generated using `FunctionDeclaration.from_callable()` from the Google GenAI SDK:

```python
types.FunctionDeclaration.from_callable(client=client, callable=get_weather_alerts)
```

This approach:
- Uses the SDK's built-in schema generation
- Extracts function name and description from the callable
- Generates parameter schemas from type annotations

### Pydantic Models for Tool Parameters

Tools use Pydantic models for parameters, following Temporal's best practice of single-parameter activities:

```python
class GetWeatherAlertsRequest(BaseModel):
    state: str = Field(description="Two-letter US state code")

async def get_weather_alerts(request: GetWeatherAlertsRequest) -> str:
    ...
```

Benefits:
- **Backward compatibility**: Add optional fields without breaking existing workflows
- **Clear contracts**: Explicit parameter definitions
- **Validation**: Pydantic validates inputs automatically

### Parameter Descriptions in Docstrings

Since `from_callable()` doesn't extract Pydantic `Field(description=...)` values, parameter descriptions are included in the function docstring:

```python
async def get_weather_alerts(request: GetWeatherAlertsRequest) -> str:
    """Get weather alerts for a US state.

    Args:
        request: The request object containing:
            - state: Two-letter US state code (e.g. CA, NY)
    """
```

The entire docstring is passed to the LLM as the function description.

### Nested LLM Output

When using Pydantic model parameters, the LLM produces nested output:

```python
{"request": {"state": "CA"}}  # Not {"state": "CA"}
```

The `tool_invoker` handles this by extracting the nested dict using the parameter name.

### Tool Caching and Initialization

Tool definitions are generated once at worker startup and cached:

```python
# In worker.py
from tools import get_tools
get_tools()  # Populate cache before workflow import
```

This ensures:
- Tools are generated once, not per-workflow
- The client doesn't need the API key (uses string-based workflow execution)
- Tool generation happens outside the workflow sandbox

### Temporal Serialization

`types.Tool` from the Google GenAI SDK is a Pydantic model, which Temporal's `pydantic_data_converter` can serialize directly. No manual serialization/deserialization needed.

### Automatic Function Calling Disabled

The SDK's automatic function calling is disabled since Temporal handles tool execution:

```python
config = types.GenerateContentConfig(
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)
```

## Adding New Tools

1. Create a new file in `tools/` with:
   - A Pydantic model for parameters (or no parameters)
   - An async handler function with descriptive docstring

   ```python
   class MyToolRequest(BaseModel):
       param1: str = Field(description="Description here")

   async def my_tool(request: MyToolRequest) -> str:
       """Brief description of what the tool does.

       Args:
           request: The request object containing:
               - param1: Description of param1
       """
       # Implementation
       return result
   ```

2. Register in `tools/__init__.py`:
   - Import the handler function
   - Add to `get_handler()` mapping
   - Add `FunctionDeclaration.from_callable()` call in `get_tools()`

No changes to the workflow are needed.

## References

- [Google GenAI Python SDK](https://googleapis.github.io/python-genai/)
- [Temporal Python SDK](https://docs.temporal.io/develop/python)
- [Function Calling with Gemini](https://ai.google.dev/gemini-api/docs/function-calling)
