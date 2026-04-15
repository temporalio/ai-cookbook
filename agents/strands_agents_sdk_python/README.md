<!--
description: Build a durable AI agent with Strands Agents SDK and AWS Bedrock that uses an agentic loop to intelligently choose and execute tools
tags: [agents, python, strands, bedrock, aws]
priority: 750
-->

# Durable Agent with Strands Agents SDK and AWS Bedrock

This recipe demonstrates how to build a durable AI agent using the [Strands Agents SDK](https://strandsagents.com/) with AWS Bedrock's Claude models. The agent uses an **agentic loop pattern** where the LLM can iteratively call tools and use their results to formulate a final answer.

Key patterns:

- **Agentic loop**: LLM decides to call tools or return final answer, sees tool results, repeats until done
- **Tools as Activities**: Each tool is a Temporal Activity with its own retry/timeout configuration
- **Durable execution**: Temporal manages state and reliability for long-running agent operations

## Prerequisites

1. **AWS Bedrock access**: Request access to Claude Sonnet 4 in the [Bedrock console](https://console.aws.amazon.com/bedrock/)
2. **AWS credentials**: Run `aws configure` or set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
3. **Dependencies**: `pip install temporalio strands-agents strands-agents-tools boto3 requests`

## Create the Activities

*File: activities/tool_activities.py*

```python
from datetime import datetime
import os
from temporalio import activity
import requests
from models.requests import WeatherRequest

@activity.defn
async def get_time_activity() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

@activity.defn
async def get_weather_activity(request: WeatherRequest) -> str:
    response = requests.get(f"https://wttr.in/{request.city}?format=%C+%t", timeout=10)
    return f"{request.city}: {response.text.strip()}"

@activity.defn
async def list_files_activity() -> str:
    files = [f for f in os.listdir('.') if f.endswith('.py')]
    return f"Python files: {', '.join(files[:5])}"
```

*File: activities/strands_agent.py*

```python
import json
import re
from temporalio import activity
from strands import Agent
from strands.models.bedrock import BedrockModel, BotocoreConfig
from models.requests import AgentRequest
from models.orchestrator import AgentResponse
from helpers.prompts import AGENT_SYSTEM_PROMPT

def extract_json(text: str) -> dict:
    """Extract JSON from text that may contain extra content."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError("No valid JSON found in response")

@activity.defn
async def agent_activity(request: AgentRequest) -> AgentResponse:
    # Disable retries in Strands - Temporal handles retries
    config = BotocoreConfig(retries={'max_attempts': 0})
    model = BedrockModel(model_id=request.model_id, config=config)
    agent = Agent(model=model, system_prompt=AGENT_SYSTEM_PROMPT)

    conversation = "\n\n".join([
        f"{msg['role']}: {msg['content']}" for msg in request.messages
    ])
    result = agent(conversation)
    result_text = result.content if hasattr(result, 'content') else str(result)

    try:
        return AgentResponse(**extract_json(result_text))
    except (json.JSONDecodeError, ValueError):
        return AgentResponse(tool_calls=[], final_answer=result_text, reasoning="Parsing failed")
```

## Create the Workflow

Activities are called by string name to avoid importing non-deterministic code into the workflow sandbox.

*File: workflows/agent.py*

```python
from datetime import timedelta
from temporalio import workflow
from models.requests import AgentRequest, WeatherRequest

@workflow.defn
class StrandsAgentWorkflow:
    @workflow.run
    async def run(self, user_input: str) -> str:
        messages = [{"role": "user", "content": user_input}]

        for iteration in range(10):
            response = await workflow.execute_activity(
                "agent_activity",
                AgentRequest(messages=messages),
                start_to_close_timeout=timedelta(seconds=30)
            )

            if response.get("tool_calls"):
                tool_results = []
                for tool_call in response["tool_calls"]:
                    result = await self._execute_tool(tool_call["tool_name"], tool_call.get("parameters", {}))
                    tool_results.append(f"{tool_call['tool_name']}: {result}")
                messages.append({"role": "assistant", "content": f"Called tools: {' | '.join(tool_results)}"})
                continue

            if response.get("final_answer"):
                return response["final_answer"]

        return "Agent exceeded maximum iterations"

    async def _execute_tool(self, tool_name: str, parameters: dict) -> str:
        if tool_name == "get_time":
            return await workflow.execute_activity("get_time_activity", start_to_close_timeout=timedelta(seconds=10))
        elif tool_name == "get_weather":
            return await workflow.execute_activity("get_weather_activity", WeatherRequest(**parameters), start_to_close_timeout=timedelta(seconds=10))
        elif tool_name == "list_files":
            return await workflow.execute_activity("list_files_activity", start_to_close_timeout=timedelta(seconds=10))
        return f"Unknown tool: {tool_name}"
```

## Create the Worker

*File: worker.py*

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.agent import StrandsAgentWorkflow
from activities.strands_agent import agent_activity
from activities.tool_activities import get_time_activity, get_weather_activity, list_files_activity

async def main():
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    worker = Worker(
        client,
        task_queue="strands-agent-task-queue",
        workflows=[StrandsAgentWorkflow],
        activities=[agent_activity, get_time_activity, get_weather_activity, list_files_activity],
        activity_executor=ThreadPoolExecutor(max_workers=10),
    )
    print("Worker started, task queue: strands-agent-task-queue")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
```

## Running

Start the Temporal dev server:

```bash
temporal server start-dev
```

Run the worker (set AWS credentials first):

```bash
export AWS_REGION=us-east-1
python worker.py
```

Start the client:

```bash
python start_workflow.py
```

## Example Interactions

```
================================================================================
Strands Agent Chat (type 'exit' or 'quit' to end)
================================================================================

You: What time is it?

Agent: The current time is 2026-01-30 14:30:15.
--------------------------------------------------------------------------------

You: What's the weather in London?

Agent: The weather in London is Partly cloudy with a temperature of 12°C.
--------------------------------------------------------------------------------

You: exit

Goodbye!
```

## Troubleshooting

**Model access error**: Request access to Claude Sonnet 4 in the [Bedrock console](https://console.aws.amazon.com/bedrock/).

**Credentials not found**: Run `aws configure` or set environment variables.

**Inference profile error**: Change model ID in `models/requests.py` from `us.anthropic.claude-sonnet-4-20250514-v1:0` to `anthropic.claude-sonnet-4-20250514-v1:0`.

## Learn More

- [Strands Agents Documentation](https://strandsagents.com/latest/documentation/)
- [AWS Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/)
- [Temporal Python SDK](https://docs.temporal.io/dev-guide/python)