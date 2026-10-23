import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.envconfig import ClientConfig
from temporalio.worker import Worker

from activities.classify import classify
from workflows.classify_workflow import ClassifyContentWorkflow


async def main():
    config = ClientConfig.load_client_connect_config()
    config.setdefault("target_host", "localhost:7233")
    client = await Client.connect(**config, data_converter=pydantic_data_converter)
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
