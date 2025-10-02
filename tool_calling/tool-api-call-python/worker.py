import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from workflows.get_weather_workflow import ToolCallingWorkflow
from activities import openai_responses, get_weather_alerts
from temporalio.contrib.pydantic import pydantic_data_converter


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    worker = Worker(
        client,
        task_queue="tool-calling-python-task-queue",
        workflows=[
            ToolCallingWorkflow,
        ],
        activities=[
            openai_responses.create,
            get_weather_alerts.get_weather_alerts,
        ],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
