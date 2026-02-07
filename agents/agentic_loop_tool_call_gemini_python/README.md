<!--
description: A basic agentic loop that invokes a dynamic set of tools using Google Gemini.
tags: [agents, python, gemini, google-genai]
priority: 775
-->
# Basic Agentic Loop with Tool Calling (Gemini)

This example implements a basic agentic loop that has a set of tools available. If the
agent determines that no tools are needed to satisfy a user request, it will respond
directly. If the LLM determines a tool should be used it will return with the name
of the chosen tool and any needed parameters. The agent then invokes the
appropriate tool.

Tools are supplied to the [`generate_content` API](https://googleapis.github.io/python-genai/) through the `tools` parameter. Tool definitions are generated using `FunctionDeclaration.from_callable()` from the Google GenAI SDK.

> [!NOTE]
> `FunctionDeclaration.from_callable()` extracts the function description from the docstring, but does NOT extract parameter descriptions from Pydantic `Field(description=...)`. Parameter descriptions should be included in the docstring's Args section.

Being external API calls, invoking the LLM and invoking any functions/tools are done within a Temporal Activity.

This recipe highlights the following key design decisions:
- We use dynamic Activities to allow the agent to be loosely coupled from specific
tools. This sample isolates the tools in the `tools` directory; changing the tools
requires NO changes to the agent implementation.
- Because there is an agentic loop, each LLM invocation is passed the accumulated
*conversation history*, that includes the initial user input as well as LLM and tool
calls. This example uses Google GenAI SDK native types (`types.Content`, `types.Part`) for conversation history.
- A generic Activity for invoking an LLM API; that is, instructions and other arguments are passed into the Activity making it appropriate for use in a variety of different use cases. Similarly, the result from the API call is returned out of the Activity so that it is usable in a variety of different use cases.
- Retries are handled by Temporal and not by the underlying libraries. This is important because if you leave the client retries on they can interfere with correct and durable error handling and recovery.

Also see this foundational [recipe for basic tool calling](https://docs.temporal.io/ai-cookbook/tool-calling-python).

## Application Components

This example includes the following components:
- The [workflow](#create-the-agent-agentic-loop) that contains the agentic loop and tool calling logic; this is the core of the agent implementation.
- The activities for [invoking the LLM](#create-the-activity-for-llm-invocations) and for [invoking tools](#create-the-activity-for-the-tool-invocation).
- Sample [tools](#create-tool-definitions).
- The [worker](#create-the-worker) that manages the Workflow and the Activities.
- An application that [initiates an interaction](#initiate-an-interaction-with-the-agent) with the agent.


## Create the Agent (Agentic Loop)

### Create the main agentic loop

The agent is implemented as a Temporal workflow that:
- implements an agentic loop. The loop will continue until the agent responds with
no tool calls.

Each time through the loop:
- the LLM is called with the accumulated conversation history that is made up of
the initial user input and any previous LLM responses and tool outputs.
- the invocation of the function, if the LLM has chosen one
- if a function is called the function result is added to the conversation history
- if no tool has been called, the LLM response is returned. This example
demonstrates a most simple UX where the user provides single shot input. Note
however that the agent is not single shot.

*File: workflows/agent.py*

```python
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    # Import pydantic internals early to avoid sandbox warnings
    import pydantic_core  # noqa: F401
    import annotated_types  # noqa: F401

    from google.genai import types

    from activities import gemini_chat
    from agent_config import prompts
    from tools import get_tools


@workflow.defn
class AgentWorkflow:
    """Agentic loop workflow that uses Gemini for LLM calls and executes tools."""

    @workflow.run
    async def run(self, input: str) -> str:
        # Initialize conversation history with the user's message
        contents: list[types.Content] = [
            types.Content(role="user", parts=[types.Part(text=input)])
        ]

        # Get tools (cached - initialized by worker at startup)
        tools = [get_tools()]

        # The agentic loop
        while True:
            print(80 * "=")

            # Consult the LLM
            result = await workflow.execute_activity(
                gemini_chat.generate_content,
                gemini_chat.GeminiChatRequest(
                    model="gemini-2.5-flash",
                    system_instruction=prompts.SYSTEM_INSTRUCTIONS,
                    contents=contents,
                    tools=tools,
                ),
                start_to_close_timeout=timedelta(seconds=60),
            )

            # Check if there are function calls to handle
            if result.function_calls:
                # Add the model's response (with function calls) to history
                contents.append(types.Content(role="model", parts=result.raw_parts))

                # Process each function call
                for function_call in result.function_calls:
                    tool_result = await self._handle_function_call(function_call)

                    # Add the function response to history
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_function_response(
                                    name=function_call["name"],
                                    response={"result": tool_result},
                                )
                            ],
                        )
                    )

            # If no function calls, we have a final response
            else:
                print(f"No tools chosen, responding with a message: {result.text}")
                return result.text
```

### Create the function call handler

The function call handler is invoked by the main agentic loop when an LLM has chosen
a tool. Because the activity implementation is dynamic, the arguments are passed
to the Activity in a property bag; the `tool_args` variable is appropriately set.
Otherwise, the Activity invocation is the same as any non-dynamic Activity
invocation passing the name of the Activity, the arguments and any Activity
configurations.

*File: workflows/agent.py*

```python
    async def _handle_function_call(self, function_call: dict) -> str:
        """Execute a tool via dynamic activity and return the result."""
        tool_name = function_call["name"]
        tool_args = function_call.get("args", {})

        print(f"Making a tool call to {tool_name} with args: {tool_args}")

        result = await workflow.execute_activity(
            tool_name,
            tool_args,
            start_to_close_timeout=timedelta(seconds=30),
        )

        return result
```

## Create the Activity for LLM invocations

We create a wrapper for the `generate_content` method of the Google GenAI client.
This is a generic Activity that invokes the Gemini LLM.

Automatic function calling is disabled since Temporal handles tool execution durably.
This ensures that tool calls are properly tracked in workflow history and can be replayed if needed.

In this implementation, we allow for the model, system instruction, conversation contents, and tools to be passed in.

*File: activities/gemini_chat.py*
```python
import os
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types
from temporalio import activity


@dataclass
class GeminiChatRequest:
    """Request parameters for a Gemini chat completion."""

    model: str
    system_instruction: str
    contents: list[types.Content]
    tools: list[types.Tool]


@dataclass
class GeminiChatResponse:
    """Response from a Gemini chat completion."""

    text: str | None  # The text response, if any
    function_calls: list[dict[str, Any]]  # List of function calls (name and args)
    raw_parts: list[types.Part]  # Raw parts for conversation history


@activity.defn
async def generate_content(request: GeminiChatRequest) -> GeminiChatResponse:
    """Execute a Gemini chat completion with tool support."""
    # Create the Gemini client with explicit API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    client = genai.Client(api_key=api_key)

    # Configure the request with automatic function calling disabled
    # (Temporal handles tool execution, not the SDK)
    config = types.GenerateContentConfig(
        system_instruction=request.system_instruction,
        tools=request.tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # Make the API call
    response = await client.aio.models.generate_content(
        model=request.model,
        contents=request.contents,
        config=config,
    )

    # Extract function calls and text from response parts
    function_calls = []
    raw_parts = []
    text_parts = []

    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts:
            raw_parts.append(part)
            if part.function_call:
                function_calls.append(
                    {
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args) if part.function_call.args else {},
                    }
                )
            elif part.text:
                text_parts.append(part.text)

    # Only include text if there are no function calls (avoids SDK warning)
    text = "".join(text_parts) if text_parts and not function_calls else None

    return GeminiChatResponse(
        text=text,
        function_calls=function_calls,
        raw_parts=raw_parts,
    )
```

## Create the Activity for the tool invocation

Implement a single tool invocation Activity, as a dynamic Activity (note the
`@activity.defn(dynamic=True)` annotation) that acts as a broker to the right
tool function. The name of the Activity is drawn from the `activity.info()` and the
property bag of arguments from the Activity payload. The `handler` is the function
that maps to the `tool_name`
(see [Create Tool Definitions](#create-tool-definitions) for more details)
and that function is then called with the supplied arguments.

*File: activities/tool_invoker.py*
```python
import inspect
from collections.abc import Sequence

from pydantic import BaseModel
from temporalio import activity
from temporalio.common import RawValue


@activity.defn(dynamic=True)
async def dynamic_tool_activity(args: Sequence[RawValue]) -> dict:
    """Execute a tool dynamically based on the activity name."""
    from tools import get_handler

    # The tool name comes from the activity type (how it was invoked)
    tool_name = activity.info().activity_type
    tool_args = activity.payload_converter().from_payload(args[0].payload, dict)
    activity.logger.info(f"Running dynamic tool '{tool_name}' with args: {tool_args}")

    handler = get_handler(tool_name)

    if not inspect.iscoroutinefunction(handler):
        raise TypeError("Tool handler must be async (awaitable).")

    # Inspect the handler's signature to determine how to pass arguments
    sig = inspect.signature(handler)
    params = list(sig.parameters.values())

    if len(params) == 0:
        # No parameters
        result = await handler()
    else:
        # Get the parameter name and annotation
        param = params[0]
        param_name = param.name
        ann = param.annotation

        if isinstance(ann, type) and issubclass(ann, BaseModel):
            # Handler expects a Pydantic model
            # LLM produces nested output like {'request': {'state': 'CA'}}
            # Extract the nested dict using the parameter name
            nested_args = tool_args.get(param_name, tool_args)
            result = await handler(ann(**nested_args))
        else:
            # Plain parameters - unpack dict as keyword arguments
            result = await handler(**tool_args)

    activity.logger.info(f"Tool '{tool_name}' result: {result}")
    return result
```

## Create tool definitions

Tools are defined in the `tools` directory and should be thought of as independent
from the agent implementation; as described above, dynamic Activities are leveraged
for this loose coupling.

The `__init__.py` file holds the tool registry:
- The `get_tools` method returns the `types.Tool` object containing all function declarations that will be passed to the LLM.
- The `get_handler` method captures the mapping from tool name to tool function.

Tool definitions are generated using `FunctionDeclaration.from_callable()` from the Google GenAI SDK. The result is cached at worker startup to avoid repeated client creation.

*File: tools/__init__.py*
```python
import os
from typing import Any, Awaitable, Callable

from google import genai
from google.genai import types

from .get_location import get_location_info, get_ip_address
from .get_weather import get_weather_alerts

ToolHandler = Callable[..., Awaitable[Any]]

# Cache for the generated Tool object
_tools_cache: types.Tool | None = None


def get_handler(tool_name: str) -> ToolHandler:
    """Get the handler function for a given tool name."""
    if tool_name == "get_location_info":
        return get_location_info
    if tool_name == "get_ip_address":
        return get_ip_address
    if tool_name == "get_weather_alerts":
        return get_weather_alerts
    raise ValueError(f"Unknown tool name: {tool_name}")


def get_tools() -> types.Tool:
    """Get the Tool object containing all available function declarations."""
    global _tools_cache
    if _tools_cache is not None:
        return _tools_cache

    # Create client to generate FunctionDeclarations
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    client = genai.Client(api_key=api_key)

    # Generate FunctionDeclarations from callables
    _tools_cache = types.Tool(
        function_declarations=[
            types.FunctionDeclaration.from_callable(
                client=client, callable=get_weather_alerts
            ),
            types.FunctionDeclaration.from_callable(
                client=client, callable=get_location_info
            ),
            types.FunctionDeclaration.from_callable(
                client=client, callable=get_ip_address
            ),
        ]
    )
    return _tools_cache
```

The tool descriptions and functions are defined in `tools/get_location.py` and
`tools/get_weather.py` files. Each of these files contains:
- data structures for function arguments (Pydantic models)
- the async function definitions with descriptive docstrings

*File: tools/get_location.py*
```python
import httpx
from pydantic import BaseModel, Field


class GetLocationRequest(BaseModel):
    """Request model for getting location info from an IP address."""

    ipaddress: str = Field(description="An IP address")


async def get_ip_address() -> str:
    """Get the public IP address of the current machine."""
    async with httpx.AsyncClient() as client:
        response = await client.get("https://icanhazip.com")
        response.raise_for_status()
        return response.text.strip()


async def get_location_info(request: GetLocationRequest) -> str:
    """Get the location information for an IP address including city, state, and country.

    Args:
        request: The request object containing:
            - ipaddress: An IP address to look up
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://ip-api.com/json/{request.ipaddress}")
        response.raise_for_status()
        result = response.json()
        return f"{result['city']}, {result['regionName']}, {result['country']}"
```

*File: tools/get_weather.py*
```python
import json
from typing import Any

import httpx
from pydantic import BaseModel, Field

NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"


class GetWeatherAlertsRequest(BaseModel):
    """Request model for getting weather alerts."""

    state: str = Field(description="Two-letter US state code (e.g. CA, NY)")


async def get_weather_alerts(request: GetWeatherAlertsRequest) -> str:
    """Get weather alerts for a US state.

    Args:
        request: The request object containing:
            - state: Two-letter US state code (e.g. CA, NY)
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    url = f"{NWS_API_BASE}/alerts/active/area/{request.state}"

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, timeout=5.0)
        response.raise_for_status()
        return json.dumps(response.json())
```

## Create the Worker

The worker is the process that dispatches work to the various parts of the agent implementation - the orchestrator and the activities for the LLM and tool invocations.

> [!IMPORTANT]
> The worker must initialize the tools cache before importing the workflow. This ensures tool generation happens outside the workflow sandbox (which restricts `threading.local` used by the Gemini client).

*File: worker.py*

```python
import asyncio

from dotenv import load_dotenv
load_dotenv()  # Load .env file before anything else

from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

# Initialize tools cache before importing workflow (requires GOOGLE_API_KEY)
from tools import get_tools
get_tools()  # Populate cache

from activities import gemini_chat, tool_invoker
from workflows.agent import AgentWorkflow


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    worker = Worker(
        client,
        task_queue="gemini-agent-python-task-queue",
        workflows=[
            AgentWorkflow,
        ],
        activities=[
            gemini_chat.generate_content,
            tool_invoker.dynamic_tool_activity,
        ],
        activity_executor=ThreadPoolExecutor(max_workers=10),
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
```

## Initiate an interaction with the agent

In order to interact with this simple AI agent, we create a Temporal client and execute a workflow.

> [!NOTE]
> The client uses string-based workflow execution (`"AgentWorkflow"` instead of `AgentWorkflow.run`) to avoid importing the workflow module, which would require the `GOOGLE_API_KEY` environment variable.

*File: start_workflow.py*
```python
import asyncio
import sys
import uuid

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    query = sys.argv[1] if len(sys.argv) > 1 else "Tell me about recursion"

    # Submit the agent workflow for execution
    # Using string-based workflow name to avoid importing workflow module
    # (which requires GOOGLE_API_KEY for tool generation)
    result = await client.execute_workflow(
        "AgentWorkflow",
        query,
        id=f"gemini-agent-id-{uuid.uuid4()}",
        task_queue="gemini-agent-python-task-queue",
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
```


## Running the app

Create a `.env` file with your Google API key (get one at https://aistudio.google.com/apikey):

```
echo "GOOGLE_API_KEY=your-api-key-here" > .env
```

Note: Only the worker needs the API key. The client does not require it.

```
uv sync
```

Start the agent worker:

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
uv run python -m start_workflow "tell me a joke"
```
