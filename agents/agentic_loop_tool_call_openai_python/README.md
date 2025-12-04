<!--
description: A basic agentic loop that invokes a dyanamic set of tools. 
tags: [agents, python]
priority: 775
-->
# Basic Agentic Loop with Tool Calling

This example implements a basic agentic loop that has a set of tools available. If the
agent determines that no tools are needed to satisfy a user request, it will respond
directly. If the LLM determines a tool should be used it will return with the name
of the chosen tool and any needed parameters. The agent then invokes the 
appropriate tool.

Tools are supplied to the [`responses` API](https://platform.openai.com/docs/api-reference/responses/create) through the [`tools` parameter](https://platform.openai.com/docs/api-reference/responses/create#responses-create-tools). The `tools` parameter is in `json` format and includes a description of the function as well as descriptions of each of the arguments.

> [!WARNING]
> The API used to generate the tools `json` is an internal function from the [Open AI API](https://github.com/openai/openai-python) and may therefore change in the future. There currently is no public API to generate the tool definition from a Pydantic model or a function signature.

Being external API calls, invoking the LLM and invoking any functions/tools are done within a Temporal Activity.

This recipe highlights the following key design decisions:
- We use dynamic Activities to allow the agent to be loosely coupled from specific
tools. This sample isolates the tools in the `tools` directory; changing the tools
requires NO changes to the agent implemention.
- Because there is an agentic loop, each LLM invocation is passed the accumulated 
*conversation history*, that includes the initial user input as well as LLM and tool 
calls.
- A generic Activity for invoking an LLM API; that is, instructions and other `responses` arguments are passed into the Activity making it appropriate for use in a variety of different use cases. Similarly, the result from the responses API call is returned out of the Activity so that it is usable in a variety of different use cases.
- Retries are handled by Temporal and not by the underlying libraries such as the OpenAI client. This is important because if you leave the client retries on they can interfere with correct and durable error handling and recovery.


Also see this foundational [recipe for basic tool calling](https://docs.temporal.io/ai-cookbook/tool-calling-python).

## Application Components

This example includes the following components:
- The [workflow](#create-the-agent-agentic-loop) that contains the agentic loop and tool calling logic; this is the core of the agent implementation.
- The activities for [invoking the LLM](#create-the-activity-for-llm-invocations) and for [invoking tools](#create-the-activity-for-the-tool-invocation).
- A [helper function](#create-the-helper-function) that creates tool definitions of the appropriate form.
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

*File:workflows/agent.py*

```python
from temporalio import workflow
from datetime import timedelta

import json

with workflow.unsafe.imports_passed_through():
    from tools import get_tools
    from helpers import tool_helpers
    from activities import openai_responses

@workflow.defn
class AgentWorkflow:
    @workflow.run
    async def run(self, input: str) -> str:

        input_list = [{"type": "message", "role": "user", "content": input}]

        # The agentic loop
        while True:

            print(80 * "=")
                
            # consult the LLM
            result = await workflow.execute_activity(
                openai_responses.create,
                openai_responses.OpenAIResponsesRequest(
                    model="gpt-4o-mini",
                    instructions=tool_helpers.HELPFUL_AGENT_SYSTEM_INSTRUCTIONS,
                    input=input_list,
                    tools=get_tools(),
                ),
                start_to_close_timeout=timedelta(seconds=30),
            )

            # For this simple example, we only have one item in the output list
            # Either the LLM will have chosen a single function call or it will
            # have chosen to respond with a message.
            item = result.output[0]

            # Now process the LLM output to either call a tool or respond with a message.
            
            # if the result is a tool call, call the tool
            if item.type == "function_call":
                result = await self._handle_function_call(item, result, input_list)
                
                # add the tool call result to the input list for context
                input_list.append({"type": "function_call_output",
                                    "call_id": item.call_id,
                                    "output": result})

            # if the result is not a tool call we will just respond with a message
            else:
                print(f"No tools chosen, responding with a message: {result.output_text}")
                return result.output_text
```

### Create the function call handler

The function call handler is invoked by the main agentic loop when an LLM has chosen
a tool. Because the activty implementation is dynamic, the arguments are passed 
to the Activity in a property bag; the `args` variable is appropriately set.
Otherwise, the Activity invocation is the same as any non-dynamic Activity 
invocation passing the name of the Activity, the arguments and any Activity
configurations.

*File: workflows\agent.py*

```python
    async def _handle_function_call(self, item, result, input_list):
        # serialize the LLM output - the decision the LLM made to call a tool
        i = result.output[0]
        input_list += [
            i.model_dump() if hasattr(i, "model_dump") else i
        ]
        # execute dynamic activity with the tool name chosen by the LLM
        # and the arguments crafted by the LLM
        args = json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments

        result = await workflow.execute_activity(
            item.name,
            args,
            start_to_close_timeout=timedelta(seconds=30),
        )

        print(f"Made a tool call to {item.name}")

        return result
```

## Create the Activity for LLM invocations

We create a wrapper for the `create` method of the `AsyncOpenAI` client object.
This is a generic Activity that invokes the OpenAI LLM.

We set `max_retries=0` when creating the `AsyncOpenAI` client.
This moves the responsibility for retries from the OpenAI client to Temporal. This means
that the Activity should interpret any errors coming from the OpenAI API call and return
the appropriate error type so that the workflow knows if it should retry the Activity or not.

In this implementation, we allow for the model, instructions and input to be passed in, and also the list of tools.

*File: activities/openai_responses.py*
```python
from temporalio import activity
from openai import AsyncOpenAI
from openai.types.responses import Response
from dataclasses import dataclass
from typing import Any

# Temporal best practice: Create a data structure to hold the request parameters.
@dataclass
class OpenAIResponsesRequest:
    model: str
    instructions: str
    input: object
    tools: list[dict[str, Any]]

@activity.defn
async def create(request: OpenAIResponsesRequest) -> Response:
    # We disable retry logic in OpenAI API client library so that Temporal can handle retries.
    # In a real setting, you would need to handle any errors coming back from the OpenAI API,
    # so that Temporal can appropriately retry in the manner that OpenAI API would.
    # See the `http_retry_enhancement_python` example for inspiration.
    client = AsyncOpenAI(max_retries=0)

    try:
        resp = await client.responses.create(
            model=request.model,
            instructions=request.instructions,
            input=request.input,
            tools=request.tools,
            timeout=30,
        )
        return resp
    finally:
        await client.close()
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
from temporalio import activity
from typing import Sequence
from temporalio.common import RawValue
import inspect
from pydantic import BaseModel

# We use dynamic activities to allow the agent to be defined independently of the tools it can call.
@activity.defn(dynamic=True)
async def dynamic_tool_activity(args: Sequence[RawValue]) -> dict:
    from tools import get_handler

    # the name of the tool to execute - this is passed in via the execute_activity call in the workflow
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

The `oai_responses_tool_from_model` function accepts a tool name and description, as well as a list of argument name/description pairs and returns json that is in the format expected for tool definitions in the OpenAI responses API.

> [!WARNING]
> The API used to generate the tools json is an interal function from the [Open AI API](https://github.com/openai/openai-python) and may therefore change in the future. There currently is no public API to generate the tool definition from a Pydantic model or a function signature.

*File:helpers/tool_helpers.py*
```python
from openai.lib._pydantic import to_strict_json_schema  # private API; may change
# there currently is no public API to generate the tool definition from a Pydantic model
# or a function signature.
from pydantic import BaseModel

def oai_responses_tool_from_model(name: str, description: str, model: type[BaseModel]):
    return {
        "type": "function",
        "name": name,
        "description": description,
        # OpenAI Responses strict tools require a JSON Schema object where
        # additionalProperties is explicitly false. For tools without
        # parameters, supply an empty object schema.
        "parameters": (
            to_strict_json_schema(model)
            if model
            else {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
        ),
        "strict": True,
    }
```

This file also holds the system instruction for the agent.
```python
HELPFUL_AGENT_SYSTEM_INSTRUCTIONS = """
You are a helpful agent that can use tools to help the user.
You will be given a input from the user and a list of tools to use.
You may or may not need to use the tools to satisfy the user ask.
If no tools are needed, respond in haikus.
"""
```

## Create tool definitions

Tools are defined in the `tools` directory and should be thought of as independent 
from the agent implementation; as described above, dynamic Activities are leveraged 
for this loose coupling. 

The `__init__.py` file holds two examples of tool sets,
one providing location and weather tools, the other a simple random number generating
tool; comment and uncomment sets you would like to include (or combine them by 
updating the `get_tools` and `get_handler` methods). 
- The `get_tools` method returns the set of tool definitions that will be passed to
the LLM.
- The `get_handler` method captures the mapping from tool name to tool function

*File: tools/__init__py*
```python
# Uncomment and comment out the tools you want to use

from typing import Any, Awaitable, Callable

# Location and weather related tools
from .get_location import get_location_info, get_ip_address
from .get_weather import get_weather_alerts

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
    return [get_weather.WEATHER_ALERTS_TOOL_OAI, 
            get_location.GET_LOCATION_TOOL_OAI,
            get_location.GET_IP_ADDRESS_TOOL_OAI]

# Random number tool
# from .random_stuff import get_random_number, RANDOM_NUMBER_TOOL_OAI

# def get_handler(tool_name: str) -> ToolHandler:
#     if tool_name == "get_random_number":
#         return get_random_number
#     raise ValueError(f"Unknown tool name: {tool_name}")

# def get_tools() -> list[dict[str, Any]]:
#     return [RANDOM_NUMBER_TOOL_OAI]
```

The tool descriptions and functions are defined in `tools/get_location.,py`, 
`tools/get_weather.py` and `tools/random_stuff.py` files. Each of these files contains:
- data structures for function arguments
- tool definitions (in `json` form)
- the function definitions.

`tools/get_location.py`
```python
# get_location.py

from typing import Any
import requests
from pydantic import BaseModel, Field
from helpers import tool_helpers

# For the location finder we use Pydantic to create a structure that encapsulates the input parameter 
# (an IP address). 
# This is used for both the location finding function and to craft the tool definitions that 
# are passed to the OpenAI Responses API.
class GetLocationRequest(BaseModel):
    ipaddress: str = Field(description="An IP address")

# Build the tool definitions for the OpenAI Responses API. 
GET_LOCATION_TOOL_OAI: dict[str, Any] = tool_helpers.oai_responses_tool_from_model(
    "get_location_info",
    "Get the location information for an IP address. This includes the city, state, and country.",
    GetLocationRequest)

GET_IP_ADDRESS_TOOL_OAI: dict[str, Any] = tool_helpers.oai_responses_tool_from_model(
    "get_ip_address",
    "Get the IP address of the current machine.",
    None)

# The functions
def get_ip_address() -> str:
    response = requests.get("https://icanhazip.com")
    response.raise_for_status()
    return response.text.strip()

def get_location_info(req: GetLocationRequest) -> str:
    response = requests.get(f"http://ip-api.com/json/{req.ipaddress}")
    response.raise_for_status()
    result = response.json()
    return f"{result['city']}, {result['regionName']}, {result['country']}"
```

See files in github for more tool definitions.

## Create the Worker

The worker is the process that dispatches work to the various parts of the agent implementation - the orchestrator and the activities for the LLM and tool invocations.

*File: worker.py*

```python
import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from workflows.agent import AgentWorkflow
from activities import openai_responses, tool_invoker
from temporalio.contrib.pydantic import pydantic_data_converter

from concurrent.futures import ThreadPoolExecutor


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    worker = Worker(
        client,
        task_queue="tool-invoking-agent-python-task-queue",
        workflows=[
            AgentWorkflow,
        ],
        activities=[
            openai_responses.create,
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

*File:start_workflow.py*
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

    # Submit the the agent workflow for execution
    result = await client.execute_workflow(
        AgentWorkflow.run,
        query,
        id=f"agentic-loop-id-{uuid.uuid4()}",
        task_queue="tool-invoking-agent-python-task-queue",
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main()) 
```


## Running the app

In the terminal where you run the agent worker, set an OpenAI API key:

```
export OPENAI_API_KEY=sk...
```

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
uv run python -m start_workflow "can I please have a random number?"
```