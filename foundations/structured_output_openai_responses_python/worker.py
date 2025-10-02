import asyncio

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.clean_data_workflow import CleanDataWorkflow
from activities import invoke_model


async def main():
    # Connect to Temporal server with pydantic data converter for our data classes
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    # Create worker with all workflows and activities
    worker = Worker(
        client,
        task_queue="clean-data-task-queue",
        workflows=[
            CleanDataWorkflow,
        ],
        activities=[
            invoke_model.invoke_model,
        ],
    )

    print("Starting Clean Data Worker...")

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
