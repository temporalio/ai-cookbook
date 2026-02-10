"""Temporal worker for documentation agent."""

import asyncio
from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.worker import Worker
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin

from workflow import DocumentationAgent, load_docs

load_dotenv()


async def main():
    """Start the Temporal worker."""
    # API key validation happens in agents module
    # Connect to Temporal with PydanticAI plugin
    client = await Client.connect(
        "localhost:7233",
        plugins=[PydanticAIPlugin()],
    )

    # Create and run worker
    # Note: Tool activities are auto-registered by PydanticAIPlugin
    worker = Worker(
        client,
        task_queue="docs-agent-queue",
        workflows=[DocumentationAgent],
        activities=[load_docs],
    )

    print("Worker started. Waiting for workflows...")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
