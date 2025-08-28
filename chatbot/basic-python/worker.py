from __future__ import annotations

import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from workflows.hello_world_workflow import HelloWorldAgent
from activities import openai_responses


async def main():
    # Create client connected to server at the given address
    client = await Client.connect(
        "localhost:7233",
    )

    worker = Worker(
        client,
        task_queue="hello-world-python-task-queue",
        workflows=[
            HelloWorldAgent,
        ],
        activities=[
            openai_responses.create,
        ],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
