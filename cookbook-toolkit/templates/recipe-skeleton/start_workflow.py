import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from workflows.recipe_workflow import RecipeWorkflow


async def main() -> None:
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    result = await client.execute_workflow(
        RecipeWorkflow.run,
        "Tell me about durable execution.",
        id="RECIPE_SLUG-example",
        task_queue="RECIPE_SLUG-task-queue",
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
