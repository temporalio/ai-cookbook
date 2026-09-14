import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.recipe_workflow import ParallelToolAgentWorkflow


async def main() -> None:
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    result = await client.execute_workflow(
        ParallelToolAgentWorkflow.run,
        "What is the weather in Seattle and the current time in America/Los_Angeles?",
        id="parallel-tool-calls-example",
        task_queue="parallel-tool-calls-task-queue",
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
