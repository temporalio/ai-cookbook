<!--
description: Build a durable AI agent in Python with Temporal and the Strands Agents SDK plugin, combining an MCP server tool with an Activity-backed tool that calls a live HTTP feed.
tags: [agents, python, bedrock]
priority: 750
-->

# Durable agent with MCP and Activity-backed tools using the Strands Agents SDK

This recipe builds a durable AI agent using the [Strands Agents SDK Integration for Temporal](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/strands). The `StrandsPlugin` makes every model call, tool call, and MCP interaction run as a durable Temporal Activity. The Workflow creates an agent and invokes it.

The agent acts as an AWS assistant with two kinds of tools:

- **An MCP tool**: the [AWS Documentation MCP server](https://github.com/awslabs/mcp/tree/main/src/aws-documentation-mcp-server), run locally with `uvx`, lets the agent search and read AWS documentation.
- **A non-deterministic, Activity-backed tool**: `get_recent_aws_announcements` fetches the live AWS "What's New" RSS feed. Because it makes a network call, it is a Temporal Activity wrapped with `activity_as_tool`, which gives it Durable Execution, retries, and timeouts.

It uses the Strands default Bedrock model. As of `strands-agents` 1.42.0 that model is Claude Sonnet 4.6, resolved through the `global.anthropic.claude-sonnet-4-6` inference profile.

## Prerequisites

1. **AWS Bedrock access**: Request access to Claude Sonnet 4.6 in the [Bedrock console](https://console.aws.amazon.com/bedrock/). To use a different model, pass a configured `BedrockModel` to `StrandsPlugin`.
2. **AWS credentials and region**: See [Strands' Amazon Bedrock guide](https://strandsagents.com/docs/user-guide/concepts/model-providers/amazon-bedrock/) for credential setup and model configuration. Strands falls back to `us-west-2` when `AWS_REGION` is unset, so set the region where you have model access.
3. **`uvx`**: Required to run the AWS Documentation MCP server (ships with [`uv`](https://docs.astral.sh/uv/)).
4. **A running Temporal dev server**: `temporal server start-dev`.

## Create the Activity-backed tool

A non-deterministic tool (live HTTP call) is defined as a Temporal Activity. The blocking request is offloaded with `asyncio.to_thread`, and retries are left to Temporal. The docstring becomes the tool description the model sees.

*File: activities/tools.py*

```python
import asyncio
import xml.etree.ElementTree as ET

from temporalio import activity

WHATS_NEW_FEED = "https://aws.amazon.com/about-aws/whats-new/recent/feed/"

@activity.defn
async def get_recent_aws_announcements(limit: int = 5) -> list[dict]:
    """Fetch the most recent AWS 'What's New' announcements from the live RSS feed.

    Use this when the user asks what is new or recently launched in AWS. Returns a
    list of {title, link, published} for the latest service launches and updates.
    """
    import requests

    response = await asyncio.to_thread(requests.get, WHATS_NEW_FEED, timeout=10)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    return [
        {
            "title": item.findtext("title", ""),
            "link": item.findtext("link", ""),
            "published": item.findtext("pubDate", ""),
        }
        for item in root.findall(".//item")[:limit]
    ]
```

## Create the Workflow

The Workflow creates a `TemporalAgent`, the replacement for the Strands `Agent` that is safe to use inside a Workflow, and gives it two tools: the AWS Documentation MCP server referenced by name through `TemporalMCPClient`, and the Activity wrapped with `activity_as_tool`. `invoke_async` drives the agentic loop, so there is no manual loop to maintain.

*File: workflows/aws_assistant_workflow.py*

```python
from datetime import timedelta

from temporalio import workflow
from temporalio.contrib.strands import TemporalAgent, TemporalMCPClient
from temporalio.contrib.strands.workflow import activity_as_tool

from activities.tools import get_recent_aws_announcements

SYSTEM_PROMPT = (
    "You are an AWS expert assistant. Use the AWS documentation tools to answer "
    "questions about AWS services, and the announcements tool to report recent "
    "launches. Cite documentation links when they are relevant."
)

@workflow.defn
class AWSAssistantWorkflow:
    def __init__(self) -> None:
        aws_docs = TemporalMCPClient(
            server="aws-docs",
            cache_tools=True,
            start_to_close_timeout=timedelta(seconds=60),
        )

        self.agent = TemporalAgent(
            start_to_close_timeout=timedelta(seconds=120),
            system_prompt=SYSTEM_PROMPT,
            tools=[
                aws_docs,
                activity_as_tool(
                    get_recent_aws_announcements,
                    start_to_close_timeout=timedelta(seconds=30),
                ),
            ],
        )

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await self.agent.invoke_async(prompt)
        return str(result)
```

## Create the Worker

The Worker registers the `StrandsPlugin` on the client. The plugin installs the Pydantic Data Converter, registers the model and MCP Activities, and uses the default `BedrockModel()` because no `models` are configured. MCP servers are registered by name through `mcp_clients`, each as a factory that launches the server (here, the AWS Documentation MCP server over stdio with `uvx`).

*File: worker.py*

```python
import asyncio

from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient
from temporalio.client import Client
from temporalio.contrib.strands import StrandsPlugin
from temporalio.worker import Worker

from activities.tools import get_recent_aws_announcements
from workflows.aws_assistant_workflow import AWSAssistantWorkflow

TASK_QUEUE = "strands-aws-assistant-task-queue"

def make_aws_docs_client() -> MCPClient:
    """Factory for the AWS Documentation MCP server, run locally via uvx."""
    return MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command="uvx",
                args=["awslabs.aws-documentation-mcp-server@latest"],
            )
        )
    )

async def main():
    plugin = StrandsPlugin(mcp_clients={"aws-docs": make_aws_docs_client})
    client = await Client.connect("localhost:7233", plugins=[plugin])

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[AWSAssistantWorkflow],
        activities=[get_recent_aws_announcements],
    )
    print(f"Worker started, task queue: {TASK_QUEUE}")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
```

## Create the Workflow Starter

The starter connects a client configured with the same plugin, so the Data Converters match, prompts for a question, and executes the Workflow.

*File: start_workflow.py*

```python
import asyncio

from temporalio.client import Client
from temporalio.common import WorkflowIDConflictPolicy
from temporalio.contrib.strands import StrandsPlugin

from workflows.aws_assistant_workflow import AWSAssistantWorkflow

TASK_QUEUE = "strands-aws-assistant-task-queue"

async def main():
    client = await Client.connect("localhost:7233", plugins=[StrandsPlugin()])

    user_input = input("Ask the AWS assistant a question: ")

    result = await client.execute_workflow(
        AWSAssistantWorkflow.run,
        user_input,
        id="strands-aws-assistant",
        task_queue=TASK_QUEUE,
        id_conflict_policy=WorkflowIDConflictPolicy.TERMINATE_EXISTING,
    )
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Running

Start the Temporal dev server:

```bash
temporal server start-dev
```

In a new terminal, install dependencies:

```bash
uv sync
```

Run the Worker, with AWS credentials and region configured as described in the prerequisites:

```bash
uv run worker.py
```

In another terminal, start the Workflow:

```bash
uv run start_workflow.py
```

## Example interactions

Try questions that exercise both tools:

- "What did AWS launch recently, and how do I enable S3 bucket versioning?"
- "Summarize the latest AWS announcements."
- "How do I configure a Lambda function URL?"

The agent decides which tools to use. Open the [Temporal UI](http://localhost:8233) to see the model invocation, the `get_recent_aws_announcements` Activity, and the `aws-docs` MCP list-tools and call-tool operations recorded as Activities in the Event History.

## Troubleshooting

**Credentials not found**: See [Strands' Amazon Bedrock guide](https://strandsagents.com/docs/user-guide/concepts/model-providers/amazon-bedrock/).

**Access denied or model not found**: Confirm you have access to Claude Sonnet 4.6 in the Bedrock console for the region you are using, and that `AWS_REGION` names that region. Strands defaults to `us-west-2` when the region is unset.

**`uvx: command not found`**: Install [`uv`](https://docs.astral.sh/uv/); `uvx` runs the AWS Documentation MCP server.

## Learn more

- [Temporal Strands Agents Plugin](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/strands)
- [Strands' Amazon Bedrock guide](https://strandsagents.com/docs/user-guide/concepts/model-providers/amazon-bedrock/)
- [Temporal Strands Agents Samples (samples-python)](https://github.com/temporalio/samples-python/tree/main/strands_plugin)
- [Strands Agents Documentation](https://strandsagents.com/latest/documentation/)
