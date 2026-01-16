<!--
description: A basic agentic loop using Claude (Anthropic) with tool calling.
tags: [agents, python, claude]
priority: 775
-->
# Basic Agentic Loop with Claude and Tool Calling

This example implements an agentic loop using Claude (Anthropic) that has a set of tools available. If the agent determines that no tools are needed to satisfy a user request, it will return the response directly. If Claude determines a tool should be used, it will return with the name of the chosen tool and any needed parameters. The agent then invokes the appropriate tool.

Tools are supplied to Claude's Messages API through the `tools` parameter. The `tools` parameter is in JSON format and includes a description of the function as well as descriptions of each of the arguments using Claude's `input_schema` format.

Being external API calls, invoking Claude and invoking any functions/tools are done within a Temporal Activity.

This recipe highlights the following key design decisions:
- We use dynamic Activities to allow the agent to be loosely coupled from specific tools. This sample isolates the tools in the `tools` directory; changing the tools requires NO changes to the agent implementation.
- Because there is an agentic loop, each Claude invocation is passed the accumulated *conversation history* in a structured messages array with role alternation (user/assistant).
- Claude can return multiple tool calls in a single response, and can mix text with tool calls in the same response.
- A generic Activity for invoking Claude's Messages API; instructions and other parameters are passed into the Activity making it appropriate for use in a variety of different use cases.
- Retries are handled by Temporal and not by the Anthropic client library. This is important because client retries can interfere with correct and durable error handling and recovery.

Also see this foundational [recipe for basic tool calling](https://docs.temporal.io/ai-cookbook/tool-calling-python).

## Application Components

This example includes the following components:
- The [Workflow](#create-the-agent-agentic-loop) that contains the agentic loop and tool calling logic; this is the core of the agent implementation.
- The Activities for [invoking Claude](#create-the-activity-for-claude-invocations) and for [invoking tools](#create-the-activity-for-the-tool-invocation).
- A [helper function](#create-the-helper-function) that creates tool definitions in Claude's format.
- Sample [tools](#create-tool-definitions).
- The [Worker](#create-the-worker) that manages the Workflow and the Activities.
- An application that [initiates an interaction](#initiate-an-interaction-with-the-agent) with the agent.

## Create the Agent (Agentic Loop)

### Create the main agentic loop

The agent is implemented as a Temporal Workflow that implements an agentic loop. The loop will continue until the agent responds with no tool calls.

Each time through the loop:
- Claude is called with the accumulated conversation history that is made up of the initial user input and any previous assistant responses and tool outputs.
- The Workflow checks if Claude returned any tool calls (content blocks with `type: "tool_use"`).
- If tool calls are present, the assistant's complete response (including all content blocks) is appended to the messages array, then all tools are executed, and their results are added as a user message.
- If no tool has been called, the text response is returned. 

*File: workflows/agent.py*

```python
from temporalio import workflow
from datetime import timedelta
import json

with workflow.unsafe.imports_passed_through():
    from tools import get_tools
    from helpers import tool_helpers
    from activities import claude_responses

@workflow.defn
class AgentWorkflow:
    @workflow.run
    async def run(self, input: str) -> str:
        
        # Initialize messages list with user input
        messages = [{"role": "user", "content": input}]

        # The agentic loop
        while True:
            print(80 * "=")
                
            # Consult Claude
            result = await workflow.execute_activity(
                claude_responses.create,
                claude_responses.ClaudeResponsesRequest(
                    model="claude-sonnet-4-20250514",
                    system=tool_helpers.HELPFUL_AGENT_SYSTEM_INSTRUCTIONS,
                    messages=messages,
                    tools=get_tools(),
                    max_tokens=4096,
                ),
                start_to_close_timeout=timedelta(seconds=30),
            )

            # Claude returns content blocks - check if any are tool_use
            tool_use_blocks = [block for block in result.content if block.type == "tool_use"]
            
            if tool_use_blocks:
                # We have tool calls to handle
                # First, add the assistant's response to messages
                # Convert content blocks to dictionaries for serialization
                assistant_content = []
                for block in result.content:
                    if block.type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })
                
                messages.append({"role": "assistant", "content": assistant_content})
                
                # Execute all tool calls and collect results
                tool_results = []
                for block in tool_use_blocks:
                    print(f"[Agent] Tool call: {block.name}({block.input})")
                    
                    # Execute the tool
                    tool_result = await self._execute_tool(block.name, block.input)
                    
                    print(f"[Agent] Tool result: {tool_result}")
                    
                    # Add tool result in Claude's expected format
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(tool_result)
                    })
                
                # Add tool results as a user message
                messages.append({"role": "user", "content": tool_results})
            else:
                # No tool calls - extract the text response and return
                text_blocks = [block for block in result.content if block.type == "text"]
                if text_blocks:
                    response_text = text_blocks[0].text
                    print(f"[Agent] Final response: {response_text}")
                    return response_text
                else:
                    return "No text response from Claude"
```

### Create the tool execution handler

The tool execution handler is invoked by the main agentic loop when Claude has chosen tools. Because the Activity implementation is dynamic, the arguments are passed to the Activity as a dictionary. The Activity invocation is the same as any non-dynamic Activity invocation, passing the name of the Activity, the arguments, and any Activity configurations.

*File: workflows/agent.py*

```python
    async def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """
        Execute a tool dynamically.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Dictionary of input parameters
        """
        # Execute dynamic Activity with the tool name and arguments
        result = await workflow.execute_activity(
            tool_name,
            tool_input,
            start_to_close_timeout=timedelta(seconds=30),
        )
        return result
```

## Create the Activity for Claude invocations

We create a wrapper for the `create` method of the `AsyncAnthropic` client object. This is a generic Activity that invokes Claude's Messages API.

We set `max_retries=0` when creating the `AsyncAnthropic` client. This moves the responsibility for retries from the Anthropic client to Temporal. This means that the Activity should interpret any errors coming from Claude's API call and return the appropriate error type so that the Workflow knows if it should retry the Activity or not.

In this implementation, we allow for the model, system instructions, messages, lis6t of tools, and max_tokens (required) to be passed in.

*File: activities/claude_responses.py*
```python
from temporalio import activity
from anthropic import AsyncAnthropic
from anthropic.types import Message
from dataclasses import dataclass
from typing import Any

# Temporal best practice: Create a data structure to hold the request parameters.
@dataclass
class ClaudeResponsesRequest:
    model: str
    system: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    max_tokens: int = 4096

@activity.defn
async def create(request: ClaudeResponsesRequest) -> Message:
    # We disable retry logic in Anthropic API client library so that Temporal can handle retries.
    # In a real setting, you would need to handle any errors coming back from the Anthropic API,
    # so that Temporal can appropriately retry in the manner that Anthropic API would.
    client = AsyncAnthropic(max_retries=0)

    try:
        resp = await client.messages.create(
            model=request.model,
            system=request.system,
            messages=request.messages,
            tools=request.tools,
            max_tokens=request.max_tokens,
        )
        return resp
    finally:
        await client.close()
```

## Create the Activity for the tool invocation

Implement a single tool invocation Activity, as a dynamic Activity (note the `@activity.defn(dynamic=True)` annotation) that acts as a broker to the right tool function. The name of the Activity is drawn from the `activity.info()` and the property bag of arguments from the Activity payload. The `handler` is the function that maps to the `tool_name` (see [Create Tool Definitions](#create-tool-definitions) for more details) and that function is then called with the supplied arguments.

*File: activities/tool_invoker.py*
```python
from temporalio import activity
from typing import Sequence
from temporalio.common import RawValue
import inspect
from pydantic import BaseModel

# We use dynamic activities to allow the agent to be defined independently of the tools it can call.
@activity.defn(dynamic=True)
async def dynamic_tool_activity(args: Sequence[RawValue]) -> dict:
    from tools import get_handler

    # the name of the tool to execute - this is passed in via the execute_activity call in the Workflow
    tool_name = activity.info().activity_type 
    tool_args = activity.payload_converter().from_payload(args[0].payload, dict)
    activity.logger.info(f"Running dynamic tool '{tool_name}' with args: {tool_args}")

    handler = get_handler(tool_name)
    # in dynamic activity
    sig = inspect.signature(handler)
    params = list(sig.parameters.values())

    if len(params) == 0:
        call_args = []
    else:
        ann = params[0].annotation
        if isinstance(tool_args, dict) and isinstance(ann, type) and issubclass(ann, BaseModel):
            call_args = [ann(**tool_args)]  # or ann.model_validate(tool_args) on Pydantic v2
        else:
            call_args = [tool_args]

    if not inspect.iscoroutinefunction(handler):
        raise TypeError("Tool handler must be async (awaitable).")
    result = await handler(*call_args)

    # Optionally log or augment the result
    activity.logger.info(f"Tool '{tool_name}' result: {result}")
    return result
```

## Create the helper function

The `claude_tool_from_model` function accepts a tool name and description, as well as a Pydantic model for the parameters, and returns JSON that is in the format expected for tool definitions in Claude's Messages API.

*File: helpers/tool_helpers.py*
```python
from pydantic import BaseModel
from typing import Any
import json

def claude_tool_from_model(name: str, description: str, model: type[BaseModel] | None) -> dict[str, Any]:
    """
    Convert a Pydantic model to Claude's tool format.
    
    Claude's tool format structure:
    {
        "name": "tool_name",
        "description": "Tool description",
        "input_schema": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }
    """
    if model is None:
        # For tools without parameters
        return {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    
    # Get the JSON schema from the Pydantic model
    schema = model.model_json_schema()
    
    # Claude expects an input_schema field
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", [])
        }
    }
```

This file also holds the system instruction for the agent.
```python
HELPFUL_AGENT_SYSTEM_INSTRUCTIONS = """
You are a helpful agent that can use tools to help the user.
You will be given input from the user and a list of tools to use.
You may or may not need to use the tools to satisfy the user ask.
If no tools are needed, respond in haikus.
"""
```

## Create tool definitions

Tools are defined in the `tools` directory and should be thought of as independent from the agent implementation; as described above, dynamic Activities are leveraged for this loose coupling.

The `__init__.py` file holds tools for providing location (`get_location_info`), IP address (`get_ip_address`), and weather alerts (`get_weather_alerts`).
- The `get_tools` method returns the set of tool definitions that will be passed to Claude.
- The `get_handler` method captures the mapping from tool name to tool function.

*File: tools/__init__.py*
```python
from typing import Any, Awaitable, Callable

# Location and weather related tools
from .get_location import get_location_info, get_ip_address
from .get_weather import get_weather_alerts
from . import get_weather
from . import get_location

ToolHandler = Callable[..., Awaitable[Any]]

def get_handler(tool_name: str) -> ToolHandler:
    if tool_name == "get_location_info":
        return get_location_info
    if tool_name == "get_ip_address":
        return get_ip_address
    if tool_name == "get_weather_alerts":
        return get_weather_alerts
    raise ValueError(f"Unknown tool name: {tool_name}")

def get_tools() -> list[dict[str, Any]]:
    return [
        get_weather.WEATHER_ALERTS_TOOL_CLAUDE, 
        get_location.GET_LOCATION_TOOL_CLAUDE,
        get_location.GET_IP_ADDRESS_TOOL_CLAUDE
    ]
```

The tool descriptions and functions are defined in `tools/get_location.py`, `tools/get_weather.py` and `tools/random_stuff.py` files. Each of these files contains:
- data structures for function arguments
- tool definitions (in JSON form using Claude's `input_schema` format)
- the function definitions.

`tools/get_location.py`
```python
# get_location.py

from typing import Any
import httpx
from pydantic import BaseModel, Field
from helpers import tool_helpers

# For the location finder we use Pydantic to create a structure that encapsulates the input parameter 
# (an IP address). 
# This is used for both the location finding function and to craft the tool definitions that 
# are passed to Claude.
class GetLocationRequest(BaseModel):
    ipaddress: str = Field(description="An IP address")

# Build the tool definitions for Claude
GET_LOCATION_TOOL_CLAUDE: dict[str, Any] = tool_helpers.claude_tool_from_model(
    "get_location_info",
    "Get the location information for an IP address. This includes the city, state, and country.",
    GetLocationRequest)

GET_IP_ADDRESS_TOOL_CLAUDE: dict[str, Any] = tool_helpers.claude_tool_from_model(
    "get_ip_address",
    "Get the IP address of the current machine.",
    None)

# The functions
async def get_ip_address() -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get("https://icanhazip.com")
        response.raise_for_status()
        return response.text.strip()

async def get_location_info(req: GetLocationRequest) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://ip-api.com/json/{req.ipaddress}")
        response.raise_for_status()
        result = response.json()
        return f"{result['city']}, {result['regionName']}, {result['country']}"
```

## Create the Worker

The worker is the process that dispatches work to the various parts of the agent implementation - the orchestrator and the Activities for Claude and tool invocations.

*File: worker.py*

```python
import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from workflows.agent import AgentWorkflow
from activities import claude_responses, tool_invoker
from temporalio.contrib.pydantic import pydantic_data_converter

from concurrent.futures import ThreadPoolExecutor


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    worker = Worker(
        client,
        task_queue="tool-invoking-agent-claude-python-task-queue",
        workflows=[
            AgentWorkflow,
        ],
        activities=[
            claude_responses.create,
            tool_invoker.dynamic_tool_activity,
        ],
        activity_executor=ThreadPoolExecutor(max_workers=10),
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
```

## Initiate an interaction with the agent

In order to interact with this simple AI agent, we create a Temporal client and execute a Workflow.

*File: start_workflow.py*
```python
import asyncio
import sys
import uuid

from temporalio.client import Client

from workflows.agent import AgentWorkflow
from temporalio.contrib.pydantic import pydantic_data_converter


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    query = sys.argv[1] if len(sys.argv) > 1 else "Tell me about recursion"

    # Submit the agent Workflow for execution
    result = await client.execute_workflow(
        AgentWorkflow.run,
        query,
        id=f"agentic-loop-claude-id-{uuid.uuid4()}",
        task_queue="tool-invoking-agent-claude-python-task-queue",
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Running the app

In the terminal where you run the agent Worker, set an Anthropic API key:

```
export ANTHROPIC_API_KEY=sk-ant-...
```

```
uv sync
```

Start the agent Worker:

```bash
uv run python -m worker
```

Make request to the agent:

```bash
uv run python -m start_workflow "are there any weather alerts for where I am?"
```

Try a number of different user prompts:
```bash
uv run python -m start_workflow "where am I?"
uv run python -m start_workflow "what is my ip address?"
uv run python -m start_workflow "tell me about recursion"
```
