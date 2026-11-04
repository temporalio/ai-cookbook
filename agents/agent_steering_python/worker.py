import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from activities.llm_call import call_llm
from workflows.agent_steering_workflow import SteerableAgentWorkflow


async def main() -> None:
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    worker = Worker(
        client,
        task_queue="agent-steering-task-queue",
        workflows=[SteerableAgentWorkflow],
        activities=[call_llm],
    )
    print("Worker started, ctrl+c to exit.")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
