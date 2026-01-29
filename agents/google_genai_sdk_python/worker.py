import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from workflows.agent import AgentGeminiWorkflow
from activities import gemini_responses, tool_invoker
from temporalio.contrib.pydantic import pydantic_data_converter

from concurrent.futures import ThreadPoolExecutor


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    worker = Worker(
        client,
        task_queue="tool-invoking-agent-gemini-task-queue",
        workflows=[
            AgentGeminiWorkflow,
        ],
        activities=[
            gemini_responses.create,
            tool_invoker.dynamic_tool_activity,
        ],
        activity_executor=ThreadPoolExecutor(max_workers=10),
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
