# Tools Directory

This directory contains tool definitions and implementations for the Claude agentic loop.

## Tool Structure

Each tool consists of three parts:

1. **Pydantic Model** (optional) - Defines the input schema
2. **Tool Definition** - Dictionary in Claude's format
3. **Handler Function** - Async function that implements the tool

## Example Tool

```python
from typing import Any
from pydantic import BaseModel, Field
from helpers import tool_helpers

# 1. Define input schema (optional for tools with no parameters)
class MyToolRequest(BaseModel):
    parameter: str = Field(description="Description of parameter")

# 2. Create tool definition for Claude
MY_TOOL_CLAUDE: dict[str, Any] = tool_helpers.claude_tool_from_model(
    "my_tool",
    "Description of what the tool does",
    MyToolRequest  # or None for tools without parameters
)

# 3. Implement the handler
async def my_tool(req: MyToolRequest) -> str:
    """Implementation of the tool."""
    result = do_something(req.parameter)
    return result
```

## Registering Tools

After creating a tool, register it in `__init__.py`:

```python
from .my_tool import my_tool, MY_TOOL_CLAUDE

def get_handler(tool_name: str) -> ToolHandler:
    if tool_name == "my_tool":
        return my_tool
    # ... other tools

def get_tools() -> list[dict[str, Any]]:
    return [
        MY_TOOL_CLAUDE,
        # ... other tools
    ]
```

## Available Tools

### Weather Tools

**get_weather_alerts**
- Description: Get weather alerts for a US state
- Input: `state` (string) - Two-letter US state code
- Source: National Weather Service API
- Example: `{"state": "CA"}`

### Location Tools

**get_ip_address**
- Description: Get the current machine's IP address
- Input: None
- Source: icanhazip.com
- Example: No parameters needed

**get_location_info**
- Description: Get location information for an IP address
- Input: `ipaddress` (string) - IP address
- Source: ip-api.com
- Example: `{"ipaddress": "8.8.8.8"}`

### Random Tools (Example)

**get_random_number**
- Description: Generate a random number between 1 and 100
- Input: None
- Example: No parameters needed

**generate_random_text**
- Description: Generate random Lorem Ipsum text
- Input: `length` (integer) - Number of words to generate
- Example: `{"length": 10}`

## Claude Tool Format

Claude expects tools in this format:

```json
{
  "name": "tool_name",
  "description": "What the tool does",
  "input_schema": {
    "type": "object",
    "properties": {
      "param_name": {
        "type": "string",
        "description": "Parameter description"
      }
    },
    "required": ["param_name"]
  }
}
```

## Best Practices

### 1. Clear Descriptions
Write clear descriptions that help Claude understand when to use the tool:

```python
# Good
"Get weather alerts for a US state. Use this when users ask about weather conditions, storms, or alerts."

# Less helpful
"Gets weather"
```

### 2. Validate Inputs
Use Pydantic's validation features:

```python
class MyToolRequest(BaseModel):
    count: int = Field(description="Number of items", ge=1, le=100)
    category: str = Field(description="Category name", pattern="^[a-z]+$")
```

### 3. Handle Errors Gracefully
Return error messages as strings:

```python
async def my_tool(req: MyToolRequest) -> str:
    try:
        result = await external_api_call(req.parameter)
        return result
    except Exception as e:
        return f"Error: {str(e)}"
```

### 4. Keep Tools Focused
Each tool should do one thing well:

```python
# Good - focused tool
async def get_weather_alerts(req: GetWeatherAlertsRequest) -> str:
    """Get weather alerts for a state."""
    # ...

# Less good - tool does too many things
async def get_everything(req: GetEverythingRequest) -> str:
    """Get weather, location, and time."""
    # ...
```

### 5. Return Structured Data
Return JSON strings for complex data:

```python
import json

async def my_tool(req: MyToolRequest) -> str:
    data = {
        "result": "value",
        "metadata": {"timestamp": "..."}
    }
    return json.dumps(data)
```

## Testing Tools

Test individual tools before integrating:

```python
# test_my_tool.py
import asyncio
from tools.my_tool import my_tool, MyToolRequest

async def test():
    req = MyToolRequest(parameter="test")
    result = await my_tool(req)
    print(result)

asyncio.run(test())
```

## Switching Between Tool Sets

Comment/uncomment sections in `__init__.py`:

```python
# Weather and location tools
from .get_location import get_location_info, get_ip_address
from .get_weather import get_weather_alerts

def get_tools() -> list[dict[str, Any]]:
    return [
        get_weather.WEATHER_ALERTS_TOOL_CLAUDE,
        get_location.GET_LOCATION_TOOL_CLAUDE,
        get_location.GET_IP_ADDRESS_TOOL_CLAUDE
    ]

# OR use random tools instead
# from .random_stuff import get_random_number, RANDOM_NUMBER_TOOL_CLAUDE
#
# def get_tools() -> list[dict[str, Any]]:
#     return [RANDOM_NUMBER_TOOL_CLAUDE]
```

## Common Issues

### Issue: Tool not being called
- Check tool description is clear and relevant
- Verify tool is registered in `__init__.py`
- Ensure input schema matches what Claude is sending

### Issue: Type errors
- Make sure Pydantic model matches the expected input
- Check that the handler function signature is correct
- Verify async/await usage

### Issue: External API failures
- Add proper error handling
- Consider adding retries for transient failures
- Return meaningful error messages to Claude

## Resources

- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [JSON Schema Reference](https://json-schema.org/)

