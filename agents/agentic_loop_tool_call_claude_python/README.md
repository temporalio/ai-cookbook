# Agentic Loop with Claude (Anthropic) and Tool Calling

This example demonstrates how to build an agentic loop using Claude (Anthropic) with tool calling in Temporal workflows.

## Overview

This cookbook refactors the OpenAI Responses API example to use Anthropic's Claude API instead. The agentic loop allows Claude to:
1. Receive a user query
2. Decide whether to use tools or respond directly
3. Execute tools dynamically if needed
4. Continue the conversation until a final answer is ready

## Key Differences from OpenAI

### API Structure
- **System Instructions**: Claude uses `system` parameter instead of `instructions`
- **Messages Format**: Claude expects an array of messages with `role` and `content`
- **Tool Format**: Claude uses `input_schema` instead of `parameters` for tool definitions

### Tool Calling
- **Tool Calls**: Returned as content blocks with `type: "tool_use"`
- **Tool Results**: Sent back as messages with content blocks of `type: "tool_result"`
- **Multiple Tools**: Claude can call multiple tools in a single response

### Response Structure
```python
# Claude Message object
{
    "id": "msg_...",
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "text", "text": "..."},
        {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
    ]
}
```

## Project Structure

```
agentic_loop_tool_call_claude_python/
├── activities/
│   ├── claude_responses.py    # Claude API activity
│   └── tool_invoker.py        # Dynamic tool execution
├── helpers/
│   └── tool_helpers.py        # Tool definition helpers
├── tools/
│   ├── __init__.py           # Tool registry
│   ├── get_location.py       # IP location tools
│   └── get_weather.py        # Weather alerts tool
├── workflows/
│   └── agent.py              # Agentic loop workflow
├── pyproject.toml            # Dependencies
├── worker.py                 # Temporal worker
└── start_workflow.py         # Workflow starter
```

## Tools Available

1. **get_weather_alerts**: Get weather alerts for a US state
   - Input: `state` (two-letter US state code)
   
2. **get_location_info**: Get location information for an IP address
   - Input: `ipaddress` (IP address string)
   
3. **get_ip_address**: Get the current machine's IP address
   - Input: None

## Setup

### Prerequisites
- Python 3.10+
- Temporal server running on `localhost:7233`
- Anthropic API key

### Installation

1. Install dependencies:
```bash
cd agents/agentic_loop_tool_call_claude_python
uv sync
```

2. Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

## Usage

### Start the Worker

In one terminal:
```bash
uv run python worker.py
```

### Execute a Workflow

In another terminal:
```bash
# Simple query (no tools needed)
uv run python start_workflow.py "Tell me about recursion"

# Query requiring tools
uv run python start_workflow.py "What's the weather like in California?"

# Query with multiple tool calls
uv run python start_workflow.py "Where am I located and what's the weather there?"
```

## How It Works

### 1. Workflow Initialization
The workflow starts with a user query and initializes a messages array.

### 2. Agentic Loop
```python
while True:
    # Call Claude
    result = await claude_responses.create(...)
    
    # Check for tool calls
    if has_tool_use_blocks:
        # Execute tools
        # Add results to messages
        # Continue loop
    else:
        # Return final text response
        return response_text
```

### 3. Tool Execution
Tools are executed as dynamic Temporal activities:
- Workflow calls `execute_activity(tool_name, tool_args)`
- Dynamic activity dispatcher finds and executes the tool
- Results are returned to the workflow

### 4. Context Management
The workflow maintains conversation context by:
- Appending assistant messages (including tool calls)
- Appending user messages (including tool results)
- Passing full message history to each Claude API call

## Claude Models

Available models (as of December 2024):
- `claude-sonnet-4-20250514` (default) - Balanced performance
- `claude-opus-4-20250514` - Highest capability
- `claude-haiku-4-5-20251001` - Fast and efficient
- `claude-3-5-sonnet-20240620` - Previous generation
- `claude-3-haiku-20240307` - Previous generation

## Customization

### Adding New Tools

1. Create a new tool file in `tools/`:
```python
from pydantic import BaseModel, Field
from helpers import tool_helpers

class MyToolRequest(BaseModel):
    param: str = Field(description="Parameter description")

MY_TOOL_CLAUDE = tool_helpers.claude_tool_from_model(
    "my_tool",
    "Tool description",
    MyToolRequest
)

async def my_tool(req: MyToolRequest) -> str:
    # Implementation
    return "result"
```

2. Register in `tools/__init__.py`:
```python
from .my_tool import my_tool, MY_TOOL_CLAUDE

def get_handler(tool_name: str) -> ToolHandler:
    # Add new case
    if tool_name == "my_tool":
        return my_tool
    # ...

def get_tools() -> list[dict[str, Any]]:
    return [
        # Add to list
        MY_TOOL_CLAUDE,
        # ...
    ]
```

### Modifying System Instructions

Edit `HELPFUL_AGENT_SYSTEM_INSTRUCTIONS` in `helpers/tool_helpers.py`:
```python
HELPFUL_AGENT_SYSTEM_INSTRUCTIONS = """
Your custom system prompt here.
"""
```

## Error Handling

The implementation includes:
- **Temporal Retries**: Automatic retries for transient failures
- **Activity Timeouts**: 30-second timeout per activity
- **HTTP Error Handling**: Proper error handling in API calls

## Best Practices

1. **Keep Tools Focused**: Each tool should do one thing well
2. **Use Pydantic Models**: Define clear input schemas
3. **Add Descriptions**: Help Claude understand when to use each tool
4. **Handle Errors**: Implement proper error handling in tool functions
5. **Log Activity**: Use activity logging for debugging

## Comparison with OpenAI Version

| Aspect | OpenAI Responses | Claude (This Version) |
|--------|------------------|----------------------|
| API Client | `AsyncOpenAI` | `Anthropic` |
| System Prompt | `instructions` | `system` |
| Tool Format | `parameters` | `input_schema` |
| Tool Calls | Single item in output | Content blocks |
| Tool Results | `function_call_output` | `tool_result` blocks |
| Response Type | Custom Response object | Message object |

## Resources

- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Claude Tool Use Guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Temporal Python SDK](https://docs.temporal.io/dev-guide/python)
- [Original Claude Quickstarts](https://github.com/anthropics/claude-quickstarts)

## License

MIT

