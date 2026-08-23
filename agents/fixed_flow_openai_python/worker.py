import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from activities.run_agent_stage import run_agent_stage
from workflows.fixed_flow_workflow import FixedFlowWorkflow

TASK_QUEUE = "fixed-flow-openai-task-queue"


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[FixedFlowWorkflow],
        activities=[run_agent_stage],
    )
    print("Worker started, ctrl+c to exit.")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
