import asyncio
from concurrent.futures import ThreadPoolExecutor
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.agent import StrandsAgentWorkflow
from activities.strands_agent import agent_activity
from activities.tool_activities import get_time_activity, get_weather_activity, list_files_activity


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    worker = Worker(
        client,
        task_queue="strands-agent-task-queue",
        workflows=[StrandsAgentWorkflow],
        activities=[
            agent_activity,
            get_time_activity,
            get_weather_activity,
            list_files_activity,
        ],
        activity_executor=ThreadPoolExecutor(max_workers=10),
    )

    print("Worker started, task queue: strands-agent-task-queue")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())