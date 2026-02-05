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

2. **Set your Google API key**:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

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
├── worker.py               # Temporal worker
├── start_workflow.py       # Client to start workflows
├── workflows/
│   └── agent.py            # Agentic loop workflow
├── activities/
│   ├── gemini_chat.py      # Gemini API activity
│   └── tool_invoker.py     # Dynamic tool execution
├── helpers/
│   └── tool_helpers.py     # Tool definition helpers
└── tools/
    ├── __init__.py         # Tool registry
    ├── get_location.py     # Location tools
    └── get_weather.py      # Weather tool
```

## How It Works

1. **User submits a query** via `start_workflow.py`
2. **Workflow starts** the agentic loop with the user's message in conversation history
3. **LLM activity** sends the conversation to Gemini with available tools
4. **If Gemini requests a tool call**:
   - The tool is executed via a dynamic activity
   - The result is added to conversation history
   - Loop continues
5. **If Gemini returns text** (no tool calls):
   - The response is returned to the user
   - Workflow completes

## Key Implementation Details

### Conversation History

Uses Google's native `Content` and `Part` abstractions (serialized for Temporal):

```python
contents = [
    {"role": "user", "parts": [{"text": "What's the weather?"}]},
    {"role": "model", "parts": [{"function_call": {...}}]},
    {"role": "user", "parts": [{"function_response": {...}}]},
]
```

### Automatic Function Calling Disabled

The SDK's automatic function calling is disabled since Temporal handles tool execution:

```python
config = types.GenerateContentConfig(
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)
```

### Dynamic Activities

Tools are invoked using the tool name as the activity name:

```python
result = await workflow.execute_activity(
    tool_name,  # e.g., "get_weather_alerts"
    tool_args,
    start_to_close_timeout=timedelta(seconds=30),
)
```

## Adding New Tools

1. Create a new file in `tools/` with:
   - A Pydantic model for parameters
   - A `FunctionDeclaration` using `tool_helpers.gemini_tool_from_model()`
   - An async handler function

2. Register in `tools/__init__.py`:
   - Add to `get_tools()` function declarations
   - Add to `get_handler()` mapping

No changes to the workflow are needed.

## References

- [Google GenAI Python SDK](https://googleapis.github.io/python-genai/)
- [Temporal Python SDK](https://docs.temporal.io/develop/python)
- [Function Calling with Gemini](https://ai.google.dev/gemini-api/docs/function-calling)
