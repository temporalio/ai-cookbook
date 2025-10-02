import asyncio
from datetime import timedelta

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.contrib.openai_agents import OpenAIAgentsPlugin, ModelActivityParameters


from workflows.hello_world_workflow import HelloWorldAgent
from activities.tools import get_weather, calculate_circle_area


async def worker_main():
    # Use the plugin to configure Temporal for use with OpenAI Agents SDK
    client = await Client.connect(
        "localhost:7233",
        plugins=[
            OpenAIAgentsPlugin(
                model_params=ModelActivityParameters(
                    start_to_close_timeout=timedelta(seconds=30)
                )
            ),
        ],
    )

    worker = Worker(
        client,
        task_queue="hello-world-openai-agent-task-queue",
        workflows=[HelloWorldAgent],
        activities=[get_weather, calculate_circle_area],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(worker_main())