# Basic Agentic Loop with Tool Calling - Gemini Version

This is a Gemini API implementation of the agentic loop with tool calling.

## Overview

This example implements a basic agentic loop using Google's Gemini API with function calling capabilities. The agent can:
- Invoke a dynamic set of tools based on user requests
- Respond directly if no tools are needed
- Use Temporal's dynamic activities pattern for loose coupling between agent and tools

## Key Differences from OpenAI Version

- Uses Google's Gemini API (`google-generativeai` library) instead of OpenAI
- Tool definitions use Gemini's `function_declarations` format
- Uses `httpx` for async HTTP requests (replacing `requests`)
- Model: `gemini-2.0-flash-exp`

## Architecture

The implementation uses the following architecture:

- **Dynamic Activities**: Tools are loosely coupled from the agent using Temporal's dynamic activities
- **Conversation History**: The agentic loop maintains full conversation history including user input, LLM responses, and tool outputs
- **Tool Invocation**: The `tool_invoker` activity acts as a broker that routes tool calls by name
- **Retry Handling**: No client-side retries; Temporal handles all retry logic

## Setup

### Prerequisites

- Python 3.10 or higher
- Temporal server running on `localhost:7233`
- Google API key with Gemini API access

### Install Dependencies

```bash
cd gemini_project
export GOOGLE_API_KEY=your-api-key-here
uv sync
```

## Running the Agent

### Start the Worker

In one terminal:

```bash
cd gemini_project
uv run python -m worker
```

### Make Requests

In another terminal:

```bash
cd gemini_project
uv run python -m start_workflow "where am I?"
uv run python -m start_workflow "are there any weather alerts for where I am?"
uv run python -m start_workflow "what is my ip address?"
q
```

## Available Tools

### Location & Weather Tools (default)
- `get_ip_address`: Get the current machine's IP address
- `get_location_info`: Get location information for an IP address
- `get_weather_alerts`: Get weather alerts for a US state

### Random Tools (commented out by default)
- `get_random_number`: Generate a random integer within a range
- `get_random_string`: Generate a random string of specified length

To switch tool sets, edit `tools/__init__.py` and comment/uncomment the appropriate sections.

## Project Structure

```
agentic_loop_tool_call_gemini_python/
├── activities/
│   ├── gemini_responses.py     # Gemini API wrapper activity
│   └── tool_invoker.py         # Dynamic activity for tool routing
├── helpers/
│   └── tool_helpers.py         # System instructions
├── tools/
│   ├── __init__.py            # Tool registry (get_tools, get_handler)
│   ├── get_location.py        # Location tools
│   ├── get_weather.py         # Weather tools
│   └── random_stuff.py        # Random generation tools
├── workflows/
│   └── agent.py               # Agentic loop workflow
├── worker.py                  # Temporal worker
├── start_workflow.py          # Client to start workflow
└── pyproject.toml            # Dependencies
```

## Adding New Tools

1. Create a new tool file in `tools/` (e.g., `my_tool.py`)
2. Define Pydantic models for parameters
3. Create tool definition in Gemini's `function_declarations` format
4. Implement the tool function (regular or async)
5. Update `tools/__init__.py`:
   - Import your tool
   - Add case to `get_handler()`
   - Add tool definition to `get_tools()`

See existing tools for examples.

## Notes

- The agent responds in haikus when no tools are needed (per system instructions)
- All tool invocations happen through Temporal activities for durability
- The implementation uses the same dynamic activities pattern as the OpenAI version for consistency
