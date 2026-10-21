import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.recipe_workflow import SupervisorAgentWorkflow


async def main() -> None:
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    result = await client.execute_workflow(
        SupervisorAgentWorkflow.run,
        "How many words are in the phrase 'durable execution is delightful'? Delegate the counting to a subagent.",
        id="subagent-supervisor-example",
        task_queue="subagent-supervisor-task-queue",
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
