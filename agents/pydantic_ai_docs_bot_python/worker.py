"""Temporal worker for Q&A workflows."""

import asyncio
import os
from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.worker import Worker
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin

from workflow import QAWorkflow, load_docs

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
    worker = Worker(
        client,
        task_queue="docs-qa-queue",
        workflows=[QAWorkflow],
        activities=[load_docs],
    )

    print("Worker started. Waiting for questions...")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
