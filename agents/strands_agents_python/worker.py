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
    # The plugin registers the model invocation and MCP activities, installs the
    # Pydantic data converter, and (since no `models` are given) uses the default
    # BedrockModel(). MCP servers are registered by name via `mcp_clients`.
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
