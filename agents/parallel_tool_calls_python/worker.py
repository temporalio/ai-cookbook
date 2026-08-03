import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from activities.llm_call import call_llm
from activities.tools import get_time, get_weather
from workflows.recipe_workflow import ParallelToolAgentWorkflow


async def main() -> None:
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    worker = Worker(
        client,
        task_queue="parallel-tool-calls-task-queue",
        workflows=[ParallelToolAgentWorkflow],
        activities=[call_llm, get_weather, get_time],
    )
    print("Worker started, ctrl+c to exit.")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
