# CLAUDE.md - Gemini Agentic Loop Project

## Project Overview

This is a Temporal-based agentic loop implementation using Google's Gemini model. It demonstrates how to build durable AI agents with tool-calling capabilities.

## Project Structure

```
agentic_loop_tool_call_gemini_python/
├── worker.py               # Temporal worker
├── start_workflow.py       # Client (no API key needed)
├── workflows/
│   └── agent.py            # Agentic loop workflow (the "engine")
├── activities/
│   ├── gemini_chat.py      # Gemini API activity
│   └── tool_invoker.py     # Dynamic tool execution
├── agent_config/           # Agent definition (personality)
│   └── prompts.py          # System instructions
└── tools/                  # Agent definition (capabilities)
    ├── __init__.py         # Tool registry and generation
    ├── get_location.py     # Location tools
    └── get_weather.py      # Weather tool
```

## Agent Architecture

The agent is composed of two loosely coupled parts:

1. **Agent Engine** (`workflows/agent.py`): The agentic loop that orchestrates LLM calls and tool execution. This is generic and doesn't change when adding/removing tools.

2. **Agent Definition**: What makes this agent unique:
   - `agent_config/prompts.py`: System instructions defining personality and behavior
   - `tools/`: Available capabilities the agent can use

This separation allows the same engine to power different agents by swapping out the definition.

## Key Implementation Details

### Tool Definition Strategy

We use `FunctionDeclaration.from_callable_with_api_option()` from the Google GenAI SDK to generate tool definitions. This method accepts the API backend as a string (`"GEMINI_API"`), so no client or API key is needed for tool generation.

**Docstrings for parameter descriptions**: The SDK extracts the entire docstring as the function description, but does NOT extract individual parameter descriptions from Pydantic `Field(description=...)`. Put parameter descriptions in the docstring's Args section.

### Why Pydantic Models for Tool Parameters

Temporal best practice is to use 0-1 parameters for activities, typically a data structure. Benefits:
- Backward compatibility when adding optional fields
- Clear activity contracts
- Consistent patterns across the codebase

When using Pydantic models with `from_callable_with_api_option()`, the LLM produces **nested output**:
```python
{"request": {"state": "CA"}}  # Not flat {"state": "CA"}
```

The `tool_invoker` extracts the nested dict using the parameter name.

### Temporal Sandbox Considerations

- `google.genai.Client` uses `threading.local` which is restricted in workflows
- The `get_tools()` function does not create a client, so it can be called safely from within the workflow sandbox

### Serialization

`types.Tool`, `types.Content`, and `types.Part` from Google GenAI SDK are Pydantic models. Temporal's `pydantic_data_converter` handles serialization automatically - no manual serialization/deserialization code is needed.

The workflow uses native SDK types directly:
```python
contents: list[types.Content] = [
    types.Content(role="user", parts=[types.Part(text=input)])
]
```

### String-Based Workflow Execution

The client uses string-based workflow execution to avoid importing the workflow module:
```python
result = await client.execute_workflow(
    "AgentWorkflow",  # String, not AgentWorkflow.run
    query,
    ...
)
```

This avoids importing the workflow module on the client side.

## Common Pitfalls

1. **Forgetting docstring parameter descriptions**: The LLM won't know what parameters mean
2. **Not handling nested LLM output**: Tool invocation will fail
3. **Wrong type for function_calls args**: Use `dict[str, Any]` not `dict[str, str]` since args is a dict
4. **Pydantic sandbox warnings**: Import `pydantic_core` and `annotated_types` early in `imports_passed_through`

## Customizing the Agent

### Changing Agent Behavior

Edit `agent_config/prompts.py` to modify the system instructions. The `SYSTEM_INSTRUCTIONS` constant defines the agent's personality and behavioral guidelines.

### Adding New Tools

1. Create Pydantic model for parameters
2. Create async handler with descriptive docstring (include parameter descriptions!)
3. Register in `tools/__init__.py`:
   - Add to `get_handler()`
   - Add `FunctionDeclaration.from_callable_with_api_option()` in `get_tools()`

No changes to the workflow are needed when adding tools.

## Environment Variables

- `GOOGLE_API_KEY`: Required by worker only (for Gemini API calls)

## Testing

Run the worker:
```bash
uv run python -m worker
```

Run a query (no API key needed):
```bash
uv run python -m start_workflow "What's the weather in California?"
```
