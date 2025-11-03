import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from workflows.agent import AgentWorkflow
from activities import openai_responses, tool_invoker
from temporalio.contrib.pydantic import pydantic_data_converter

from concurrent.futures import ThreadPoolExecutor


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    worker = Worker(
        client,
        task_queue="tool-invoking-agent-python-task-queue",
        workflows=[
            AgentWorkflow,
        ],
        activities=[
            openai_responses.create,
            tool_invoker.dynamic_tool_activity,
        ],
        activity_executor=ThreadPoolExecutor(max_workers=10),
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
