import asyncio

from activities.llm_call import call_llm
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker
from workflows.recipe_workflow import RecipeWorkflow


async def main() -> None:
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    worker = Worker(
        client,
        task_queue="RECIPE_SLUG-task-queue",
        workflows=[RecipeWorkflow],
        activities=[call_llm],
    )
    print("Worker started, ctrl+c to exit.")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
