import asyncio
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.classify_workflow import ClassifyContentWorkflow
from activities.classify import classify


async def main():
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    worker = Worker(
        client,
        task_queue="guardrails-hard-rules-task-queue",
        workflows=[ClassifyContentWorkflow],
        activities=[classify],
    )
    print("Worker started, ctrl+c to exit.")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
