import asyncio
from datetime import timedelta

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.contrib.openai_agents import OpenAIAgentsPlugin, ModelActivityParameters

from workflows.streaming_agent_workflow import StreamingAgent
from activities.tools import MODEL_EVENTS_TOPIC, get_weather, calculate_circle_area


async def worker_main():
    # Use the plugin to configure Temporal for use with OpenAI Agents SDK.
    # streaming_topic enables Runner.run_streamed: the model activity
    # publishes every raw OpenAI stream event to this Workflow Streams topic.
    client = await Client.connect(
        "localhost:7233",
        plugins=[
            OpenAIAgentsPlugin(
                model_params=ModelActivityParameters(
                    start_to_close_timeout=timedelta(seconds=30),
                    streaming_topic=MODEL_EVENTS_TOPIC,
                )
            ),
        ],
    )

    worker = Worker(
        client,
        task_queue="streaming-openai-agent-task-queue",
        workflows=[StreamingAgent],
        activities=[get_weather, calculate_circle_area],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(worker_main())
