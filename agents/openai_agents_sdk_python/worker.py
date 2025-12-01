import asyncio
from datetime import timedelta
from pathlib import Path

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.envconfig import ClientConfig
from temporalio.contrib.openai_agents import OpenAIAgentsPlugin, ModelActivityParameters


from workflows.hello_world_workflow import HelloWorldAgent
from activities.tools import get_weather, calculate_circle_area


async def worker_main():
    # Use the plugin to configure Temporal for use with OpenAI Agents SDK
    config_dir = Path(__file__).parent.parent.parent
    config_file = config_dir / "config.toml"
    if not config_file.exists():
        config_file = config_dir / "config.toml.example"
    connect_config = ClientConfig.load_client_connect_config(
        config_file=str(config_file)
    )
    client = await Client.connect(
        **connect_config,
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