# Tool calling

## Overview

This example shows how function calling (also known as tool calling) works with the Open AI API and Temporal. Tool calling allows the model to make decisions on when functions should be invoked, and structure the data that is needed for the function call.

Tools are supplied to the [`responses` API](https://platform.openai.com/docs/api-reference/responses/create) through the [`tools` parameter](https://platform.openai.com/docs/api-reference/responses/create#responses-create-tools). We supply the description of the tools in `json` using an OpenAI API to generate the correct structure. 

[!WARNING]
The API used to generate the tools json is an internal function and may therefore change in the future. There currently is no public API to generate the tool definition from a Pydantic model or a function signature.

Being external API calls, invoking the LLM and invoking the function are each done within a Temporal Activity. 

This example lays the foundation for the core agentic pattern where the LLM makes the decision on functions/tools to invoke, the agent calls the function/tool and the response from the call is sent back to the LLM for interpretation.

This recipe highlights these key design decisions:

- A generic activity for invoking an LLM API; that is, instructions and other responses arguments are passed into the activity making it appropriate for use in a variety of differnt use cases. Similarly, the result from the responses API call is returned out of the activity so that it is usable in a variety of different use cases.
- We have intentionally not implemented the agentic loop so as to focus on how tool details are made available to the LLM and how functions are invoked. We do, take the tool output and have the LLM interpret it in a manner consistent with the AI agent pattern.
- Retries are handled by Temporal and not by the underlying libraries such as the OpenAI client. This is important because if you leave the client retires on they can interfere with correct and durable error handling and recovery.

## Create the Activity for LLM invocations

We create wrapper for the `create` method of the `AsyncOpenAI` client object.
This is a generic activity that invokes the OpenAI LLM.

We set `max_retries=0` on when creating the `AsyncOpenAI` client.
This moves the responsibility for retries from the OpenAI client to Temporal.

In this implementation, we allow for the model, instructions and input to be passed in, as well as the list of tools.

TODO: Insert code block

## Create the Activity for the tool invocation

TODO: Add description

TODO: Insert code block

## Create the Workflow

TODO: Add description

TODO: Insert code block

*File: workflows/get_weather_workflow.py*


## Create the Worker

TODO: Add description

TODO: Insert code block

*File: worker.py*
```python

```

## Running

Start the Temporal Dev Server:

```bash
temporal server start-dev
```

Run the worker:

```bash
uv run python -m worker
```

Start execution:

```bash
uv run python -m start_workflow "Tell me about recursion in programming."
```

```bash
uv run python -m start_workflow "Are there any weather alerts in California?"
```